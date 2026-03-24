"""
AutoM3L Model Zoo (Model Cards) - Ultimate Performance & Feasibility Edition
定义系统支持的各模态候选模型。
"""

MODEL_ZOO = {
    "Tabular_Model": {
        # --- 梯度提升树 (GBDT - 工业界首选) ---
        "XGBoost": "Gradient Boosting. Select if 'row_count' is Medium/High. The industrial standard. Excellent accuracy for general tabular tasks and supports GPU acceleration.",
        "LightGBM": "Light Gradient Boosting. Select if 'row_count' is Huge (>50k) or 'preference' is 'Efficiency'. Extremely fast training speed and low memory usage. Handles large-scale data perfectly.",
        "CatBoost": "Categorical Boosting. CRITICAL: Select if 'tabular_complexity' is 'High Cardinality' or data is 'Imbalanced'. Best-in-class handling of categorical features without manual preprocessing.",
        # --- 深度表格学习 (Deep Learning - 多模态融合首选) ---
        "FT-Transformer": "Feature Tokenizer + Transformer. Select if 'row_count' > 5k and task is Multimodal. The SOTA deep learning model that learns embeddings compatible with Text/Image encoders.",
        "TabTransformer": "Transformer-based embeddings. Select if 'categorical_features_count' is High. Very effective for high-cardinality categorical data using self-attention.",
        "ResNet-Tabular": "Adapted ResNet. Select if 'row_count' is Low/Medium (<5k) or for a robust Deep Learning baseline. Simpler than Transformers, harder to overfit on small data.",
        # --- 强力集成 (Ensemble - 小数据救星) ---
        "ExtraTrees": "Extremely Randomized Trees. Select if 'row_count' is Low (<2k) or data is noisy. Reduces variance compared to Random Forest. Very stable and hard to overfit on small datasets."
    },

    "Text_Model": {
        # --- 顶级精度 (High Accuracy) ---
        "microsoft/deberta-v3-large": "Decoding-enhanced BERT. Select if 'preference' is 'Accuracy' and text is 'Short/Medium'. Current SOTA for NLU tasks.",
        "roberta-large": "Robust BERT. Select if 'text_context' is 'Short/Medium' and dataset is large. The go-to robust model.",
        "microsoft/mpnet-base": "Masked and Permuted Pre-training. Select for semantic similarity tasks or stable convergence.",
        # --- 效率与长文本 (Efficiency & Long Context) ---
        "google/electra-large-discriminator": "Sample-efficient pre-training. Select if compute resources are limited but high accuracy is needed.",
        "allenai/longformer-base-4096": "Sparse Attention Transformer. CRITICAL: Select ONLY if 'text_context' is 'Long Document' (>512 words). Handles sequences up to 4096 tokens.",
        "google/bigbird-roberta-base": "Google's sparse attention mechanism. Alternative to Longformer. Select ONLY if 'text_context' is 'Long Document'.",
        # --- 参数效率 (Parameter Efficient) ---
        "albert-xxlarge-v2": "A Lite BERT. Select if memory is constrained. Uses parameter sharing to reduce memory footprint."
    },

    "Image_Model": {
        # --- 稳健基线 (Robust Baselines - CNNs) ---
        "tf_efficientnetv2_s": "Modern CNN. Select if 'preference' is 'Efficiency' or 'Speed'. Works well with standard 224x224 input.",
        "convnext_tiny": "Modern ConvNet. Select for 'Low/Medium' resource datasets. Performance comparable to ResNet-50 but with modern design.",
        # --- 视觉 Transformer (ViTs - 224x224 Safe) ---
        "swin_base_patch4_window7_224": "Swin Transformer V1. Select if 'preference' is 'Accuracy' and 'row_count' is Medium/High. Uses `patch4_window7_224`. Captures complex textures.",
        "vit_base_patch16_224": "Standard ViT. Select if 'row_count' is High (>5k). Pretrained on ImageNet-21k at 224x224. Needs data to shine.",
        "deit3_base_patch16_224": "Data-efficient Image Transformers. Select if 'row_count' is Low/Medium but you want Transformer architecture.",
        # --- 高性能/新架构 (High Performance) ---
        "eva02_base_patch14_224": "Next-gen ViT. Select if 'preference' is 'Max Accuracy'. Must use `eva02_base_patch14_224`. Extremely high performance.",
        # --- 多模态对齐 (Multimodal Aligned) ---
        "openai/clip-vit-large-patch14": "OpenAI's CLIP. Select ONLY if Text and Image are semantically aligned (e.g., Captioning logic). Best for cross-modal correlations."    },

    "Fusion_Model": {
        # --- 深度交互融合 (Deep Interaction) ---
        "Transformer_Fusion": "Self-Attention Fusion. Select if 'row_count' is High. Projects all modalities into a shared sequence. Captures complex cross-modal dependencies.",
        "Cross_Attention": "Asymmetric attention. Select if one modality (e.g., Text) dominates or explains the other (e.g., Image). Highly effective for specific cross-modal tasks.",

        # --- 特征调制与加权 (Modulation & Gating) ---
        "FiLM": "Feature-wise Modulation. Select for efficient conditioning. Uses one modality to predict scale/shift parameters for another. Compact and effective.",
        "Gated_Fusion": "Gating Mechanism. Select if some modalities might be noisy or irrelevant. Uses learnable gating to dynamically weight importance.",

        # --- 基础基线 (Baseline) ---
        "Concat_MLP": "Simple Concatenation. Select if 'row_count' is Low or as a robust baseline. Simple concatenation followed by dense layers. Low risk of overfitting."
    }
}