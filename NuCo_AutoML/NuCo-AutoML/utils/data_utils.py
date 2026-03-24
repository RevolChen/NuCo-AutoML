import os
import pandas as pd
import numpy as np
from PIL import Image
import json


def load_data(file_path):
    """通用数据加载函数"""
    try:
        # 优先尝试 UTF-8
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # 解决 Windows 乱码问题
        return pd.read_csv(file_path, encoding='utf-8-sig')


def get_data_sample(df, n_samples=5):
    """
    从 DataFrame 中随机抽取 N 行，并转化为 JSON 字符串。
    用于给大模型做 In-context Learning。
    """
    # 随机采样，如果数据量不够则取全部
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)

    # 转化为 JSON 字符串 (orient='records' 生成列表形式)
    # force_ascii=False 保证中文正常显示
    return sample_df.to_json(orient='records', force_ascii=False)


def separate_target(df, target_col):
    """
    分离特征和目标列。
    MI-LLM 不需要看到 Target 列。
    """
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不在数据集中，请检查列名拼写。")

    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def get_data_statistics(df):
    """
    计算 DataFrame 的基础统计信息，供 AFE-LLM 参考。
    返回一个字典，Key 为列名，Value 为统计描述字符串。
    """
    stats = {}
    total_rows = len(df)

    for col in df.columns:
        # 计算缺失率
        missing_count = df[col].isnull().sum()
        missing_rate = missing_count / total_rows

        # 计算唯一值数量 (Cardinality)
        unique_count = df[col].nunique()
        unique_ratio = unique_count / total_rows

        # 获取少量样本值 (前3个非空值)
        sample_values = df[col].dropna().head(3).tolist()

        # 组装描述信息
        desc = {
            "missing_rate": f"{missing_rate:.2%}",
            "unique_count": unique_count,
            "unique_ratio": f"{unique_ratio:.2%}",
            "dtype": str(df[col].dtype),
            "samples": sample_values
        }
        stats[col] = desc
    return stats

def get_dataset_meta_features(df, target_col, modality_map):
    """
    提取数据集的元特征 (Meta-Features)，用于辅助 MS-LLM 选择模型。
    """
    meta = {}

    # 基础规模
    total_rows = len(df)
    meta['row_count'] = total_rows

    # 判断数据规模等级
    if total_rows < 2000:
        meta['size_category'] = "Small (Low Resource)"
    elif total_rows < 50000:
        meta['size_category'] = "Medium"
    else:
        meta['size_category'] = "Large (Big Data)"

    # 任务类型 (根据目标列推断)
    if target_col in df.columns:
        y = df[target_col]
        num_classes = y.nunique()
        if pd.api.types.is_float_dtype(y) or num_classes > 20:
            meta['task_type'] = "Regression"
        elif num_classes == 2:
            meta['task_type'] = "Binary Classification"
        else:
            meta['task_type'] = f"Multi-class Classification ({num_classes} classes)"

        # 检查类别不平衡
        if meta['task_type'] != "Regression":
            class_counts = y.value_counts(normalize=True)
            min_class_ratio = class_counts.min()
            if min_class_ratio < 0.05:
                meta['imbalance_status'] = f"Highly Imbalanced (Min class: {min_class_ratio:.1%})"
            else:
                meta['imbalance_status'] = "Balanced"

    # 模态特定特征
    # -- 文本长度分析 --
    text_cols = [c for c, m in modality_map.items() if m == 'Text']
    if text_cols:
        # 抽样计算平均长度
        sample_texts = df[text_cols[0]].dropna().astype(str).head(100)
        avg_len = sample_texts.apply(lambda x: len(x.split())).mean()
        meta['text_avg_length'] = f"{int(avg_len)} words"
        meta['text_context'] = "Long Document" if avg_len > 400 else "Short/Medium Sentence"

    # -- 表格特征分析 --
    cat_cols = [c for c, m in modality_map.items() if m == 'Categorical']
    if cat_cols:
        meta['categorical_features_count'] = len(cat_cols)
        # 简单判断是否高基数
        high_card_cols = [c for c in cat_cols if df[c].nunique() > 50]
        if high_card_cols:
            meta['tabular_complexity'] = "High Cardinality (Complex Categorical)"
        else:
            meta['tabular_complexity'] = "Low Cardinality (Standard)"

    # -- 图像模态约束分析 --
    img_cols = [c for c, m in modality_map.items() if m == 'Image_Path']
    if img_cols:
        img_col_name = img_cols[0]
        # 采样前 10 张存在的图片进行分析
        sample_paths = df[img_col_name].dropna().head(10).tolist()

        resolutions = []
        modes = []  # RGB, L (Grayscale), etc.

        for p in sample_paths:
            try:
                # 假设路径是相对路径，可能需要拼接 base_dir，这里假设是绝对路径或相对运行目录
                if os.path.exists(p):
                    with Image.open(p) as img:
                        resolutions.append(img.size)  # (width, height)
                        modes.append(img.mode)
            except Exception:
                continue

        if resolutions:
            # 计算平均分辨率
            avg_w = np.mean([r[0] for r in resolutions])
            avg_h = np.mean([r[1] for r in resolutions])
            mean_res = (avg_w + avg_h) / 2

            meta['image_resolution_avg'] = f"{int(avg_w)}x{int(avg_h)}"

            # 判断分辨率等级
            if mean_res < 64:
                meta['image_context'] = "Tiny Image (Low Res)"
            elif mean_res > 500:
                meta['image_context'] = "High Resolution"
            else:
                meta['image_context'] = "Standard Resolution"

            # 判断颜色通道
            # 只要有一张是 RGB 就算 RGB，全是 L 才是 Grayscale
            if any('RGB' in m for m in modes):
                meta['image_channels'] = "RGB (3-channel)"
            elif all('L' in m for m in modes):
                meta['image_channels'] = "Grayscale (1-channel)"
            else:
                meta['image_channels'] = "Mixed/Other"
        else:
            meta['image_context'] = "Unknown (Files not found)"

    return meta