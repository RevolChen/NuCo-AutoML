import json
from openai import OpenAI
from config.settings import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME
from config.model_zoo import MODEL_ZOO


class MSLLM:
    def __init__(self, model_name=None):
        self.model_name = model_name if model_name else MODEL_NAME
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    def _analyze_active_modalities(self, modality_map):
        """
        分析当前数据集包含哪些模态类型。
        返回: 模态集合 (e.g., {'Tabular', 'Text', 'Image'})
        """
        active_modalities = set()

        # 遍历 MI 输出的列模态映射
        for col, modality in modality_map.items():
            if modality == 'Target':
                continue

            if modality in ['Numerical', 'Categorical']:
                active_modalities.add('Tabular')
            elif modality == 'Text':
                active_modalities.add('Text')
            elif modality == 'Image_Path':
                active_modalities.add('Image')

        return list(active_modalities)

    def _build_prompt(self, active_modalities, user_preference, meta_features):
        """
        构建 MS-LLM 提示词 (数据感知版)
        """
        model_zoo_str = json.dumps(MODEL_ZOO, indent=2)
        active_mod_str = str(active_modalities)
        meta_feat_str = json.dumps(meta_features, indent=2, ensure_ascii=False)

        is_multimodal = len(active_modalities) >= 2

        if is_multimodal:
            fusion_instruction = "Multimodal Detected, You MUST select ONE 'Fusion_Model'. For other modalities, you CAN select multiple models (list of strings) for ensemble learning if beneficial."
        else:
            fusion_instruction = "Unimodal Detected, You MUST NOT output 'Fusion_Model'. You CAN select multiple models (list of strings) for ensemble learning if beneficial."

        system_text = """
You are an Expert AI Architect acting as the MS-LLM module.
Your goal is to synthesize data characteristics and user preferences to architect the optimal model configuration.

*** REASONING GUIDELINES (Use these principles to weigh your choices) ***

1. DATA SIZE PRINCIPLES:
   - Low Resource (< 2k rows): Large-scale Transformers are prone to overfitting. Prefer **robust ensemble methods** (e.g., 'ExtraTrees') or **lightweight deep baselines** (e.g., 'ResNet-Tabular').
   - High Resource (> 50k rows): **Data-hungry architectures** (like 'FT-Transformer', 'ViT', 'DeBERTa') tend to shine. **Gradient Boosting** (LightGBM) is also valuable for speed.

2. MODALITY SPECIFIC LOGIC:
   - Text Context: Standard BERT models truncate at 512 tokens. For "Long Documents", you MUST select **Sparse Attention** models (e.g., 'longformer', 'bigbird'). For standard text, prioritize **SOTA NLU** models (e.g., 'deberta', 'roberta').
   - Tabular Complexity: **Tree-based ensembles** are robust defaults. For 'High Cardinality' or 'Imbalanced' data, prioritize models with **native categorical handling** (CatBoost) or **learned embeddings** (TabTransformer).
   - Image Architecture: All inputs are standardized to 224x224.
     - "Stability & Speed": **Modern CNNs** (e.g., 'efficientnet', 'convnext') are easier to optimize and faster.
     - "Max Accuracy": **Hierarchical/Advanced ViTs** (e.g., 'swin', 'eva02') generally outperform CNNs on complex tasks but are heavier.
     - "Transfer Learning": **Standard/Distilled ViTs** (e.g., 'vit_base', 'deit') are excellent for general feature extraction.

3. MULTIMODAL SYNERGY:
   - If Text and Image describe the same entity (e.g., captioning), select **alignment-based models** (e.g., 'openai/clip...') to capture cross-modal semantics.

4. USER PREFERENCE GUIDELINES:
   - "Efficiency/Speed": Prioritize **lightweight/distilled** variants (e.g., 'tiny', 'small', 'albert').
   - "Accuracy": Prioritize **high-capacity/large** variants (e.g., 'large', 'base', 'xxlarge').

5. ENSEMBLE STRATEGY:
   - You are encouraged to select MULTIPLE models for a single modality if it improves robustness or performance (Ensemble Learning).
   - For example, combining a Tree-based model and a Deep Learning model for Tabular data.
   - Or combining a CNN and a ViT for Image data.

6. CONSTRAINT MAPPING:
   - **CRITICAL**: Your selection MUST come strictly from the provided MODEL_ZOO keys.
   - **Analyze the 'description' field**: The keys are technical IDs (e.g., 'swin_base...'). You MUST read the description to verify if the model fits the 'size_category', 'context', or 'preference'.
"""

        user_text = f"""
=== DATASET META-FEATURES (Analyze this carefully!) ===
{meta_feat_str}

=== TASK CONFIGURATION ===
- Active Modalities: {active_mod_str}
- User Preference: "{user_preference}"
- Task Logic: {fusion_instruction}

=== AVAILABLE MODEL ZOO ===
{model_zoo_str}

Requirement:
Based on the 'size_category', 'text_context', 'image_context', 'preference' and so on, select the most suitable models.
Generate a strictly valid JSON configuration.
- If 'Tabular' is active -> Key: "Tabular_Model" (Value can be a string OR a list of strings)
- If 'Text' is active    -> Key: "Text_Model" (Value can be a string OR a list of strings)
- If 'Image' is active   -> Key: "Image_Model" (Value can be a string OR a list of strings)
- "Fusion_Model" key depends on the Task Logic above (YES if >=2 modalities, NO if 1).
Output a strict JSON mapping Modality -> Model Name(s).

Example Output(Scenario: Tabular + Text + Image with Ensemble):
{{
    "Tabular_Model": ["ExtraTrees", "XGBoost"],
    "Text_Model": "allenai/longformer-base-4096",
    "Image_Model": ["convnext_tiny", "swin_base_patch4_window7_224"],
    "Fusion_Model": "Concat_MLP"
}}

Example Output (Scenario: Tabular only with Ensemble):
{{
    "Tabular_Model": ["XGBoost", "CatBoost", "ResNet"]
}}

Please generate the JSON configuration:
"""
        return system_text, user_text

    def select_models(self, modality_input, meta_features, preference="Prioritize Accuracy"):
        """
        执行模型选择
        Args:
            modality_input: 已经加载好的模态字典 (dict)
            meta_features: 元特征
            preference: 用户偏好描述 (字符串)
        """
        print(f">>> MS-LLM: 正在分析模态并选择最佳模型 (偏好: {preference})...")

        # 加载模态字典
        if isinstance(modality_input, dict):
            modality_map = modality_input
        else:
            print("错误: modality_input 必须是字典。")
            return None

        # 分析当前有哪些模态
        active_mods = self._analyze_active_modalities(modality_map)
        print(f"[*] 检测到活跃模态: {active_mods}")

        if not active_mods:
            print("错误: 未检测到有效特征模态。")
            return None

        # 构建 Prompt
        system_prompt, user_prompt = self._build_prompt(active_mods, preference, meta_features)

        # 调用 LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1  # 稍微有点温度，允许根据 preference 灵活选择
            )

            selection_config = json.loads(response.choices[0].message.content)
            return selection_config

        except Exception as e:
            print(f"MS-LLM 错误: {e}")
            return None