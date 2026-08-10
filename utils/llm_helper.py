from openai import OpenAI
import streamlit as st

MODEL_CONFIGS = {
    "DeepSeek-V4-Flash": ("https://api.deepseek.com", "deepseek-v4-flash"),
}

def call_llm(system_prompt, user_prompt, model_choice, api_key, temperature=0.3, max_tokens=800):
    if not api_key or not model_choice:
        st.warning("缺少 API Key 或模型选择，跳过 AI 生成")
        return None

    config = MODEL_CONFIGS.get(model_choice)
    if not config:
        st.warning(f"不支持的模型：{model_choice}")
        return None

    base_url, model_name = config
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        # 最简请求，不加任何扩展参数
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # 把完整错误显示在页面上，方便排查
        st.error(f"大模型调用失败：{e}")
        return None