import json
from openai import OpenAI
from config.settings import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME
from utils.data_utils import get_data_sample


class MILLM:
    def __init__(self, model_name=None):
        # 允许初始化时覆盖默认模型
        self.model_name = model_name if model_name else MODEL_NAME
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    def _build_prompt(self, columns, sample_data_str):
        # System Prompt: 定义角色和模态标准
        system_text = f"""
You are an Expert Machine Learning Engineer acting as the MI-LLM module.
Your task is to infer the data modality for each column in a provided dataset subset.

Based on the column names and sample values, categorize each column into one of the following types:
1. Numerical: Continuous or discrete numbers (e.g., 12, 3.5, 100).
2. Categorical: Finite set of labels, classes, or IDs (e.g., 'Male', 'Red', 101, 'US').
3. Text: Long unstructured natural language sentences (e.g., 'This is a cute cat...').
4. Image_Path: File system paths pointing to images. IMPORTANT: Fields may contain multiple paths separated by semicolons (';').

Return ONLY a valid JSON object mapping column names to their modalities. No markdown, no explanations.
"""

        # Few-Shot Examples (包含你要求的 Categorical 和 多图路径 案例)
        # 这里设计了通用的案例，让 LLM 学会识别分号分隔的路径
        few_shot_text = r"""
Examples:

Input:
[
    {"Age": 5, "Type": "Cat", "Desc": "Very cute", "Photos": "D:\\data\\img1.jpg"},
    {"Age": 2, "Type": "Dog", "Desc": "Active", "Photos": "D:\\data\\img2.jpg;D:\\data\\img3.jpg"}
]
Output:
{
    "Age": "Numerical",
    "Type": "Categorical",
    "Desc": "Text",
    "Photos": "Image_Path"
}

Input:
[
    {"Price": 200.5, "Is_Sold": 1, "Comments": "Good product", "Img": "/home/user/a.png;/home/user/b.png"}
]
Output:
{
    "Price": "Numerical",
    "Is_Sold": "Categorical",
    "Comments": "Text",
    "Img": "Image_Path"
}
"""

        # User Input (真实数据)
        user_text = f"""
Now, infer the modalities for the following new dataset.

Column Names: {columns}

Sample Data:
{sample_data_str}

Remember:
- Treat columns with multiple file paths (separated by ';') as 'Image_Path'.
- Distinguish 'Numerical' (quantitative) from 'Categorical' (IDs or finite labels).
"""
        return system_text, few_shot_text + user_text

    def infer(self, df_features, n_samples=5):
        """
        主调用函数
        Args:
            df_features: 不包含目标列的 DataFrame
        Returns:
            dict: {列名: 模态类型}
        """
        # 准备数据采样
        columns = df_features.columns.tolist()
        sample_str = get_data_sample(df_features, n_samples=n_samples)

        # 构建提示词
        system_prompt, user_prompt = self._build_prompt(columns, sample_str)

        print(f">>> MI-LLM: 正在调用大模型 ({self.model_name}) 进行模态推理...")
        print(f">>> 采样样本数: {n_samples}")

        try:
            # 调用 API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},  # 强制 JSON 格式
                temperature=0.0  # 设置为 0 确保结果最稳定
            )

            # 解析结果
            result_content = response.choices[0].message.content
            modality_map = json.loads(result_content)

            return modality_map

        except Exception as e:
            print(f"MI-LLM 错误: {e}")
            return {}