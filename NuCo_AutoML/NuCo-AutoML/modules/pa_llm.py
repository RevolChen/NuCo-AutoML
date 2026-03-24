import os
import json
import pandas as pd
from openai import OpenAI
from config.settings import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME
import shutil


class PALLM:
    def __init__(self, model_name=None):
        self.model_name = model_name if model_name else MODEL_NAME
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    def _determine_task_type(self, data_path, target_col):
        """
        读取数据前几行，基于目标列自动推断任务类型。
        返回: 'Classification' 或 'Regression'
        """
        try:
            # 只读前1000行，加速分析
            df = pd.read_csv(data_path, nrows=100)
            if target_col not in df.columns:
                raise ValueError(f"目标列 {target_col} 不在数据集中")

            y = df[target_col]
            unique_count = y.nunique()
            dtype = y.dtype

            # 逻辑判断：
            # 1. 如果是浮点数且唯一值很多 -> 回归
            # 2. 如果是字符串/Object -> 分类
            # 3. 如果是整数，但唯一值很少 (<20) -> 分类
            # 4. 如果是整数，但唯一值很多 -> 回归

            if pd.api.types.is_float_dtype(dtype):
                return 'Regression'
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                return 'Classification'
            elif pd.api.types.is_integer_dtype(dtype):
                if unique_count < 20:
                    return 'Classification'
                else:
                    return 'Regression'
            else:
                # 默认兜底
                return 'Classification'

        except Exception as e:
            print(f"[PA-LLM] 任务类型推断失败: {e}，默认使用 Classification")
            return 'Classification'

    def generate_code(self, data_path, test_path, modality_map, model_config, target_col, meta_features, output_path=None, seed=42):
        """
        核心功能：读取 pipeline_skeleton.py 模板，注入配置，生成 pipeline.py
        """
        # 1. 向 LLM 请求搜索空间建议
        hpo_prompt = f"""
        你是一个 AutoML 专家。请根据以下数据集的元特征和模态信息，为 Optuna HPO 建议最佳的搜索空间和一组初始参数。
        
        数据集信息:
        - 样本量: {meta_features.get('sample_size', '未知')}
        - 特征总数: {meta_features.get('feature_count', '未知')}
        - 模态分布: {json.dumps(modality_map, ensure_ascii=False)}
        - 任务类型: {self._determine_task_type(data_path, target_col)}
        - 选定模型: {json.dumps(model_config, ensure_ascii=False)}

        请输出一个 JSON 对象，包含三个键：
        1. "SEARCH_SPACE_CODE": 一段 Python 代码字符串，包含 trial.suggest_* 的字典内容（但不包含 batch_size）。
           必须包含键: 'lr', 'dropout', 'tab_hidden', 'fusion_dim'。
           注意：
           - 对于融合模型，建议学习率 'lr' 的上限不要超过 1e-3 (0.001) 以防止梯度爆炸。
        2. "BATCH_SIZE_LIST": 一个整数列表，代表 batch_size 的搜索空间，如 [16, 32, 64]。请根据数据量和模型大小给出合理的候选。
        3. "BEST_INIT_PARAMS": 一个字典，包含你认为最可能成功的初始参数值（除了 batch_size）。

        示例输出格式:
        {{
            "SEARCH_SPACE_CODE": "'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True), 'dropout': trial.suggest_float('dropout', 0.1, 0.4), ...",
            "BATCH_SIZE_LIST": [16, 32, 64],
            "BEST_INIT_PARAMS": {{"lr": 1e-4, "dropout": 0.2, "tab_hidden": 512, "fusion_dim": 1024}}
        }}
        只输出 JSON，不要有其他解释。
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": hpo_prompt}],
                response_format={"type": "json_object"}
            )
            hpo_config = json.loads(response.choices[0].message.content)
            search_space_code = hpo_config.get("SEARCH_SPACE_CODE", "")
            batch_size_list = hpo_config.get("BATCH_SIZE_LIST", [16, 32, 64])
            best_init_params = json.dumps(hpo_config.get("BEST_INIT_PARAMS", {}))
        except Exception as e:
            print(f"[PA-LLM] 获取 HPO 建议失败: {e}，使用默认配置")
            search_space_code = "'lr': trial.suggest_float('lr', 1e-5, 3e-4, log=True), 'dropout': trial.suggest_float('dropout', 0.1, 0.4), 'tab_hidden': trial.suggest_categorical('tab_hidden', [256, 512, 1024]), 'fusion_dim': trial.suggest_categorical('fusion_dim', [512, 1024, 2048])"
            batch_size_list = [16, 32, 64]
            best_init_params = '{"lr": 1e-4, "dropout": 0.2, "tab_hidden": 512, "fusion_dim": 1024}'

        # 2. 确定模板路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        template_path = os.path.join(project_root, "templates", "pipeline_skeleton.py")

        final_output_path = output_path
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

        if not os.path.exists(template_path):
            raise FileNotFoundError(f"未找到模板文件: {template_path}")

        # 推断任务类型
        task_type = self._determine_task_type(data_path, target_col)
        
        # 准备注入的数据
        abs_data_path = os.path.abspath(data_path).replace("\\", "/")
        abs_test_path = os.path.abspath(test_path).replace("\\", "/")
        modality_map_json = json.dumps(modality_map, ensure_ascii=False)
        model_config_json = json.dumps(model_config, ensure_ascii=False)

        # 3. 读取并替换模板
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()

        filled_code = template_content.replace("{data_path}", abs_data_path) \
            .replace("{test_path}", abs_test_path) \
            .replace("{target_col}", target_col) \
            .replace("{task_type}", task_type) \
            .replace("{modality_map_json}", modality_map_json) \
            .replace("{model_config_json}", model_config_json) \
            .replace("{seed}", str(seed)) \
            .replace("{search_space_code}", search_space_code) \
            .replace("{batch_size_list}", str(batch_size_list)) \
            .replace("{best_init_params}", best_init_params)

        with open(final_output_path, 'w', encoding='utf-8') as f:
            f.write(filled_code)

        return final_output_path