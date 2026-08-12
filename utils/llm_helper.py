from openai import OpenAI
import streamlit as st
import httpx
import time

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
        response = client.chat.completions.create(**request_params)

        content = response.choices[0].message.content
        if content is None:
            # 有些模型在thinking模式下content为空，尝试读reasoning_content
            reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
            if reasoning:
                return reasoning.strip()
            st.warning("模型返回内容为空，请检查模型配置。")
            return None
        return content.strip()
    except Exception as e:
        st.error(f"大模型调用失败：{type(e).__name__} – {e}")
        return None


def test_api_connection(api_key):
    """测试API连接，返回 (成功布尔, 消息字符串)"""
    if not api_key:
        return False, "API Key 为空"
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=httpx.Timeout(30.0, connect=10.0)
        )
        start = time.time()
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": "You are a helper."},
                {"role": "user", "content": "回复OK"}
            ],
            max_tokens=10
        )
        elapsed = round(time.time() - start, 2)
        content = response.choices[0].message.content
        if content:
            return True, f"连接成功！返回：{content}（耗时{elapsed}秒）"
        else:
            return False, f"连接成功但返回为空（耗时{elapsed}秒），可能模型配置有误"
    except Exception as e:
        return False, f"连接失败：{type(e).__name__} – {e}"
