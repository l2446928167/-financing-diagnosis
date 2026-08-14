"""
config.py — 全局配置（对话式 v4.0）

API Key 加载策略（满足「用户打开即用、无需填 Key、密钥对前端透明」需求）：
  统一入口见 utils/llm_gateway._load_key()，优先级：
  1) Streamlit Secrets（部署 / Streamlit Cloud 推荐，对所有用户透明）；
  2) 环境变量 DEEPSEEK_API_KEY（Docker / 云厂商注入）；
  3) 项目根目录 .env（本地开发，已被 .gitignore 忽略，不会进 git）。

  若三者皆空，AI 提取 / 生成类功能不可用，但规则引擎与本地检索问答仍正常工作。
  密钥只在服务端进程内加载，浏览器 / 会话（st.session_state）永远不持有明文 Key。

部署启用 AI 功能（二选一，均无需用户参与）：
  - Streamlit Cloud：在 App 的 Secrets 里写  DEEPSEEK_API_KEY = "sk-xxx"
  - 自托管 / Docker：在启动前  export DEEPSEEK_API_KEY="sk-xxx"
本地开发：在项目根目录创建 .env 写入  DEEPSEEK_API_KEY=sk-xxx（参考 .env.example）
"""
import os

import streamlit as st


def _load_key() -> str:
    """加载 DeepSeek API Key，优先级：Secrets → 环境变量 → .env。"""
    try:
        v = st.secrets.get("DEEPSEEK_API_KEY")
        if v:
            return v
    except Exception:
        pass
    v = os.environ.get("DEEPSEEK_API_KEY")
    if v:
        return v
    try:
        from dotenv import load_dotenv
        load_dotenv()
        v = os.environ.get("DEEPSEEK_API_KEY")
        if v:
            return v
    except Exception:
        pass
    return ""


# 模型标识（与 utils/llm_helper.MODEL_CONFIGS 对应）
MODEL_NAME = "DeepSeek-V4-Flash"

# API Key：统一由 _load_key() 在部署侧配置（Secrets / 环境变量 / .env），界面绝不暴露输入框。
# 保留该导出名以兼容仍直接引用 config.DEEPSEEK_API_KEY 的旧代码；正式调用请走 utils.llm_gateway.gateway()。
DEEPSEEK_API_KEY = _load_key()

# 数据持久化目录（会话与跨期企业数据）
DATA_DIR = "data_store"
