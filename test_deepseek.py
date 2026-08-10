import os
from openai import OpenAI

# 从 .env 读取 Key（如果有保存过）
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

api_key = os.environ.get("DEEPSEEK_API_KEY", "你的key填这里")
if not api_key or api_key == "你的key填这里":
    api_key = input("请输入 DeepSeek API Key: ")

# 测试官方示例的 base_url（不带 /v1）
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

try:
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "你是一个测试助手"},
            {"role": "user", "content": "请回复“连接成功”"}
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )
    print("✅ 成功！回复内容：", response.choices[0].message.content)
except Exception as e:
    print("❌ 调用失败，错误信息：", e)