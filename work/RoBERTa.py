# -*- coding: utf-8 -*-
"""
对抗性数据改写在欺诈对话检测中的应用
Adversarial Data Rewriting for Fraudulent Dialogue Detection

包含模块：
1. DataProcessor: 处理 CSV 数据，清洗 dialogue_content。
2. FraudClassifier: 基于 RoBERTa 的分类模型训练。
3. AdversarialAttacker: 基于 MaskedLM 的上下文感知改写/攻击。
4. ExperimentRunner: 执行训练、攻击和再训练流程。
"""

import pandas as pd
import numpy as np
import torch
import re
import random
import gc
import subprocess
import sys
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    DataCollatorWithPadding
)


# 检查并安装必要的依赖
def check_dependencies():
    try:
        # 尝试导入accelerate
        import accelerate
        print(f"已安装 accelerate 版本: {accelerate.__version__}")
        # 检查版本是否满足要求
        version_tuple = tuple(map(int, accelerate.__version__.split('.')))
        if version_tuple < (0, 26, 0):
            print("accelerate 版本过低，需要更新...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "accelerate>=0.26.0"])
            print("accelerate 更新成功")
    except ImportError:
        print("未找到 accelerate 库，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate>=0.26.0"])
        print("accelerate 安装成功")

    # 导入transformers的Trainer和TrainingArguments
    global Trainer, TrainingArguments
    from transformers import Trainer, TrainingArguments


# 设置随机种子以保证结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# ==========================================
# 1. 数据处理模块 (Data Processing)
# ==========================================

class DataProcessor:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def load_and_clean(self, file_path):
        """
        读取CSV并清洗文本。
        解析 'specific_dialogue_content' 去除元数据，提取纯文本。
        将 'is_fraud' 转换为 0/1 标签。
        """
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

        # 去除空值
        df = df.dropna(subset=['specific_dialogue_content', 'is_fraud'])

        # 标签转换: True -> 1, False -> 0
        # 处理可能存在的字符串大小写问题
        df['label'] = df['is_fraud'].astype(str).apply(lambda x: 1 if x.lower() == 'true' else 0)

        # 文本清洗函数
        def clean_dialogue(text):
            # 去除 "音频内容：" 头部
            text = re.sub(r'^音频内容：\s*', '', str(text))
            # 去除首尾引号和空白
            text = text.replace('"', '').strip()
            # 简单规范化换行
            text = re.sub(r'\n+', '\n', text)
            return text

        df['text'] = df['specific_dialogue_content'].apply(clean_dialogue)

        return df[['text', 'label', 'fraud_type']]

    def get_datasets(self):
        train_df = self.load_and_clean(self.train_path)
        test_df = self.load_and_clean(self.test_path)

        # 简单统计
        print(f"训练集大小: {len(train_df)}")
        print(f"测试集大小: {len(test_df)}")
        print("训练集标签分布:\n", train_df['label'].value_counts())

        return train_df, test_df


# ==========================================
# 2. 对抗攻击/改写模块 (Adversarial Rewriter)
# ==========================================

class AdversarialRewriter:
    """
    实现论文中的 Contextual Interaction Attack 简化版。
    利用 Masked LM (RoBERTa) 对 'left' (诈骗者) 的话语进行关键词替换。
    """

    def __init__(self, model_name="hfl/chinese-roberta-wwm-ext", device='cuda'):
        print("正在加载 Masked LM 用于对抗样本生成...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 移除dtype参数，避免与混合精度训练冲突
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.device = device
        self.mlm_model.eval()

    def rewrite(self, text, label, replace_ratio=0.15):
        """
        对单条对话进行改写。
        仅针对 label=1 (Fraud) 的样本进行攻击，且仅修改 'left:' 开头的句子。
        """
        # 如果不是欺诈样本，或者是短文本，则不进行改写
        if label == 0 or len(text) < 10:
            return text

        lines = text.split('\n')
        new_lines = []

        for line in lines:
            # 核心策略：只攻击 'left' (诈骗发起者)，保持 'right' (受害者) 不变以维持事实基准
            if line.strip().startswith("left:"):
                content = line.replace("left:", "").strip()

                # 简单的分词 (字级别 for Chinese RoBERTa)
                tokens = list(content)
                if len(tokens) <= 3:
                    new_lines.append(line)
                    continue

                # 随机选择要 Mask 的位置 (模拟重要性采样)
                # 在完整实现中，这里应使用梯度计算重要性
                num_to_mask = max(1, int(len(tokens) * replace_ratio))
                mask_indices = random.sample(range(len(tokens)), num_to_mask)

                original_tokens = tokens.copy()

                # 构建 Masked Input
                for idx in mask_indices:
                    # 避免替换标点符号
                    if tokens[idx] not in ["，", "。", "？", "！", "："]:
                        tokens[idx] = self.tokenizer.mask_token

                masked_text = "".join(tokens)

                # 使用 MLM 预测
                inputs = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.mlm_model(**inputs)
                    logits = outputs.logits

                # 获取预测结果
                # 修复索引对齐问题
                final_tokens = original_tokens.copy()

                # 寻找 mask token 在 input_ids 中的位置
                mask_positions = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)

                # 处理每个mask位置
                for pos_idx, (batch_idx, token_idx) in enumerate(zip(*mask_positions)):
                    if pos_idx < len(mask_indices):
                        # 获取Top-K预测
                        probs = logits[batch_idx, token_idx].softmax(dim=0)
                        top_k_indices = torch.topk(probs, 5).indices

                        # 选择一个不是原字的词
                        char_replaced = False
                        for token_id in top_k_indices:
                            pred_char = self.tokenizer.decode([token_id])
                            # 过滤掉特殊 token 和 原字
                            if pred_char not in self.tokenizer.all_special_tokens and pred_char != original_tokens[
                                mask_indices[pos_idx]]:
                                final_tokens[mask_indices[pos_idx]] = pred_char
                                char_replaced = True
                                break

                        # 如果没有找到合适的替换词，保持原字符不变
                        if not char_replaced:
                            continue

                new_line = f"left: {''.join(final_tokens)}"
                new_lines.append(new_line)
            else:
                new_lines.append(line)

        return "\n".join(new_lines)


# ==========================================
# 3. 模型训练与评估模块
# ==========================================

class FraudDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def run_pipeline():
    # 检查并安装必要的依赖
    check_dependencies()

    # 添加环境变量以禁用symlinks警告
    import os
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    # 禁用torch.compile以避免依赖Triton
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    # 1. 加载数据
    processor = DataProcessor(r'H:\University_work\NLP\final_essay_work\Dataset\train_result.csv',
                              r'H:\University_work\NLP\final_essay_work\Dataset\test_result.csv')
    train_df, test_df = processor.get_datasets()

    # 2. 初始化 Tokenizer 和 模型
    MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 使用dynamic padding替代fixed length padding
    def tokenize_function(texts):
        return tokenizer(texts, padding=False, truncation=True, max_length=256)

    # 3. 准备基线数据
    print("正在处理基线数据...")
    train_encodings = tokenize_function(train_df['text'].tolist())
    test_encodings = tokenize_function(test_df['text'].tolist())

    train_dataset = FraudDataset(train_encodings, train_df['label'].tolist())
    test_dataset = FraudDataset(test_encodings, test_df['label'].tolist())

    # 4. 训练基线模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    # 移除dtype参数，避免与fp16=True设置冲突
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    ).to(device)

    # 参数配置 - 移除torch_compile参数
    training_args = TrainingArguments(
        output_dir='./results_baseline',
        num_train_epochs=3,
        per_device_train_batch_size=16,  # 针对RTX 4060优化
        per_device_eval_batch_size=32,  # 针对RTX 4060优化
        eval_strategy="epoch",  # 修复参数名称
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=2e-5,
        logging_steps=50,
        disable_tqdm=False,
        fp16=True,  # 启用混合精度训练
        gradient_accumulation_steps=2,  # 梯度累积
        metric_for_best_model="f1",
        greater_is_better=True
    )

    # 使用动态padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator  # 添加动态padding
    )

    print("开始训练基线模型 (Baseline Training)...")
    trainer.train()

    print("基线模型评估结果:")
    baseline_metrics = trainer.evaluate()
    print(baseline_metrics)

    # 清理内存
    torch.cuda.empty_cache()
    gc.collect()

    # 5. 生成对抗样本 (Adversarial Attack)
    print("开始生成对抗样本 (Adversarial Generation)...")
    rewriter = AdversarialRewriter(model_name=MODEL_NAME, device=device)

    # 对测试集进行攻击以评估鲁棒性
    test_df['adv_text'] = test_df.apply(lambda x: rewriter.rewrite(x['text'], x['label']), axis=1)

    # 评估基线模型在对抗数据上的表现
    adv_test_encodings = tokenize_function(test_df['adv_text'].tolist())
    adv_test_dataset = FraudDataset(adv_test_encodings, test_df['label'].tolist())

    print("在对抗测试集上评估基线模型:")
    adv_metrics = trainer.predict(adv_test_dataset).metrics
    print(adv_metrics)

    # 清理内存
    torch.cuda.empty_cache()
    gc.collect()

    # 6. 对抗性训练 (Adversarial Training)
    print("准备对抗性训练数据 (Data Augmentation)...")
    # 对训练集进行增强
    train_df_aug = train_df.copy()
    train_df_aug['text'] = train_df_aug.apply(lambda x: rewriter.rewrite(x['text'], x['label']), axis=1)

    # 合并原始数据与对抗数据
    combined_train_df = pd.concat([train_df, train_df_aug]).sample(frac=1).reset_index(drop=True)
    print(f"增强后训练集大小: {len(combined_train_df)}")

    combined_encodings = tokenize_function(combined_train_df['text'].tolist())
    combined_dataset = FraudDataset(combined_encodings, combined_train_df['label'].tolist())

    # 清理内存
    del rewriter
    torch.cuda.empty_cache()
    gc.collect()

    # 重新初始化 Trainer 进行微调 - 同样移除torch_compile参数
    aug_training_args = TrainingArguments(
        output_dir='./results_robust',
        num_train_epochs=3,
        per_device_train_batch_size=16,  # 针对RTX 4060优化
        per_device_eval_batch_size=32,  # 针对RTX 4060优化
        eval_strategy="epoch",  # 修复参数名称
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=2e-5,
        fp16=True,  # 启用混合精度训练
        gradient_accumulation_steps=2,  # 梯度累积
        metric_for_best_model="f1",
        greater_is_better=True
    )

    aug_trainer = Trainer(
        model=model,  # 继续微调
        args=aug_training_args,
        train_dataset=combined_dataset,
        eval_dataset=test_dataset,  # 仍在原始测试集上监控
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    print("开始对抗性训练 (Adversarial Training)...")
    aug_trainer.train()

    # 清理内存
    torch.cuda.empty_cache()
    gc.collect()

    # 7. 最终评估
    print("最终评估：鲁棒模型在对抗测试集上的表现")
    final_adv_metrics = aug_trainer.predict(adv_test_dataset).metrics
    print(final_adv_metrics)

    return baseline_metrics, adv_metrics, final_adv_metrics


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback

        traceback.print_exc()  # 打印详细的错误堆栈
