import json
import pandas as pd
from openai import OpenAI
from config.settings import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME


class AFELLM:
    def __init__(self, model_name=None):
        self.model_name = model_name if model_name else MODEL_NAME
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    def _build_prompt(self, columns_stats, modality_map, target_col):
        """
        构建 AFE-LLM 的提示词。
        结合了论文中 AFE-LLM_filter 和 AFE-LLM_imputed 的逻辑。
        """

        # 准备数据描述字符串
        data_desc_str = json.dumps(columns_stats, indent=2, ensure_ascii=False)
        modality_str = json.dumps(modality_map, indent=2, ensure_ascii=False)

        # System Prompt
        system_text = """
You are an Expert Machine Learning Engineer acting as the AFE-LLM module.
Your task is to analyze dataset statistics and modalities to generate a strictly optimized data cleaning plan.

You must balance two goals:
1. REMOVE NOISE: Drop columns that lead to overfitting (IDs) or have no signal (Names, excessive missing).
2. PRESERVE SIGNAL: Keep unstructured data (Images, Text) even if they have high cardinality.

=== 1. CRITICAL FILTERING RULES (DROP) ===
You MUST add a column to "drop_columns" if it meets ANY of the following conditions:

[Rule A: High Cardinality IDs] -> Risk of Overfitting
- Condition: The column name contains "ID", "Id", "Key", "Code", or "Index".
- AND: The 'unique_ratio' is high (> 0.5) or 'unique_count' is very large.
- Example: "PetID", "RescuerID", "Transaction_Code".

[Rule B: Irrelevant Names] -> Noise
- Condition: The column name indicates a proper name (e.g., "Name", "StudentName").
- Reason: Names are usually unique labels without generalizable patterns.

[Rule C: High Missing Rate] -> Low Information
- Condition: 'missing_rate' > 40% (0.4).
- EXCEPTION: If the column is 'Text' (Modality) or has sparse but valuable info, you may keep it.

[Rule D: Single Value] -> Zero Variance
- Condition: 'unique_count' is 1.

[*** CRITICAL EXCEPTIONS ***] -> DO NOT DROP
- NEVER DROP the Target Column.
- NEVER DROP columns with modality "Image_Path" or "Text" (Description), even if 'unique_ratio' is 1.0 (100%). These contain rich features for downstream models.

=== 2. IMPUTATION RULES ===
For columns NOT dropped, decide how to fill missing values:
- Numerical: Use "mean" (default) or "median" (if skewed).
- Categorical: Use "new_category" (fill with 'Unknown') if cardinality is high; use "mode" if low.
- Text/Image_Path: Use "fill_empty" (fill with empty string).
"""

        # User Prompt
        user_text = f"""
Current Task Configuration:
- Target Column: "{target_col}" (KEEP THIS)

Column Modalities:
{modality_str}

Data Statistics:
{data_desc_str}

=== 2. VALID IMPUTATION STRATEGIES (STRICT ENFORCEMENT) ===
For any column that is NOT dropped, you MUST choose one strategy from the list below.
DO NOT invent new strategies (e.g., do not use "zero", "nearest", "average").

Allowed values:
1. "mean"         : Use for Numerical columns (normal distribution).
2. "median"       : Use for Numerical columns (skewed distribution).
3. "mode"         : Use for Categorical columns (fill with most frequent).
4. "new_category" : Use for Categorical columns (fill with 'Unknown'/-1).
5. "fill_empty"   : Use for Text or Image_Path columns (fill with empty string).

Task:
1. Identify columns to DROP based on Filtering Rules.
2. Assign an ALLOWED imputation strategy for every remaining column with missing values.

Output Format (Strict JSON):
{{
    "drop_columns": ["col_1", "col_2"],
    "imputation_strategies": {{
        "col_3": "mean",
        "col_4": "new_category",
        "col_5": "fill_empty"
    }}
}}

Return ONLY the valid JSON.
"""
        return system_text, user_text

    def generate_plan(self, df, modality_map, target_col):
        """
        调用 LLM 生成清洗计划
        """
        from utils.data_utils import get_data_statistics

        # 计算统计信息
        stats = get_data_statistics(df)

        # 构建 Prompt
        system_prompt, user_prompt = self._build_prompt(stats, modality_map, target_col)

        print(f">>> AFE-LLM: 正在分析数据统计信息并生成清洗策略 ({self.model_name})...")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0  # 保持确定性
            )

            plan_json = json.loads(response.choices[0].message.content)
            return plan_json

        except Exception as e:
            print(f"AFE-LLM 错误: {e}")
            return None

    def execute_plan(self, df, plan, target_col):
        """
        执行清洗计划 (Pandas 操作)
        """
        df_clean = df.copy()

        # 执行删除 (Filtering)
        drop_cols = plan.get("drop_columns", [])
        # 安全检查：防止 LLM 抽风把 Target 删了
        if target_col in drop_cols:
            print(f"警告: LLM 建议删除目标列 '{target_col}'，已强制保留。")
            drop_cols.remove(target_col)

        # 仅删除实际存在的列
        real_drop_cols = [c for c in drop_cols if c in df_clean.columns]
        if real_drop_cols:
            df_clean.drop(columns=real_drop_cols, inplace=True)
            print(f"[*] 已删除列: {real_drop_cols}")

        # 执行填补 (Imputation)
        strategies = plan.get("imputation_strategies", {})

        for col, strategy in strategies.items():
            if col not in df_clean.columns:
                continue

            # 如果该列没有缺失值，跳过
            if df_clean[col].isnull().sum() == 0:
                continue

            if strategy == 'mean':
                # 仅数值型有效
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    val = df_clean[col].mean()
                    df_clean[col] = df_clean[col].fillna(val)

            elif strategy == 'median':
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(val)

            elif strategy == 'mode':
                if not df_clean[col].mode().empty:
                    val = df_clean[col].mode()[0]
                    df_clean[col] = df_clean[col].fillna(val)

            elif strategy == 'new_category':
                # 填充为 'Unknown' 或 -1
                fill_val = 'Unknown' if df_clean[col].dtype == 'object' else -1
                df_clean[col] = df_clean[col].fillna(fill_val)

            elif strategy == 'fill_empty':
                df_clean[col] = df_clean[col].fillna("")

        print(f"[*] 缺失值填补完成。")
        return df_clean