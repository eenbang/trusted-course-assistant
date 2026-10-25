import os

from dotenv import load_dotenv
from openai import OpenAI


# 从项目根目录的 .env 文件读取环境变量。
load_dotenv()


# 读取 API 密钥。若没配置，就给出清楚的报错。
api_key = os.getenv("SILICONFLOW_API_KEY")
if not api_key:
    raise RuntimeError(
        "找不到 SILICONFLOW_API_KEY。请检查项目根目录里的 .env 文件。"
    )


# 创建 API 客户端：使用你首个项目中的 SiliconFlow 接口地址。
client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
)


# 沿用你首个项目中使用的模型名称。
MODEL = "deepseek-ai/DeepSeek-V3.2"


def ask_llm(messages: list[dict[str, str]]) -> str:
    """把对话消息发给模型，并返回模型生成的文字。"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )

    # choices[0] 是模型返回的第一个回答；
    # message.content 是这个回答的文本。
    return response.choices[0].message.content or ""