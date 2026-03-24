import os

# 请替换为实际 Key，或者设置环境变量
# 建议使用 gpt-4o 或 gpt-3.5-turbo (json mode)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-8FnHBq5zuIyDAKI2lM6RAhJ0TyPMM70MqW5dSj2cgLjK8wlT")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai-proxy.org/v1") # 如果用中转需修改
MODEL_NAME = "gpt-3.5-turbo"  # 论文使用了强大的 LLM，建议用 gpt-4 或 4o 以保证 JSON 格式稳定

# 这里定义系统支持的模态类型，供 Prompt 使用
SUPPORTED_MODALITIES = [
    "Numerical",    # 数值型 (年龄, 价格, 计数)
    "Categorical",  # 类别型 (颜色, 品种, 性别, ID)
    "Text",         # 文本型 (描述, 评论)
    "Image_Path"    # 图像型 (绝对路径, 支持分号分隔的多图)
]