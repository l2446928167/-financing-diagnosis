from openai import OpenAI
import streamlit as st
import httpx

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
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(60.0, connect=10.0)
        )
        request_params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        # DeepSeek-V4-Flash 是普通对话模型，不需要 reasoning_effort 和 thinking 参数
        # 如需使用推理模型（deepseek-reasoner），请单独配置
        response = client.chat.completions.create(**request_params)

        # 防止 content 为 None（推理模型可能只在 reasoning_content 有内容）
        content = response.choices[0].message.content
        if content is None:
            st.warning("模型返回内容为空，请检查模型配置是否正确。")
            return None
        return content.strip()
    except Exception as e:
        st.error(f"大模型调用失败：{e}")
        return None
