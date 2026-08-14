"""
config.py — 全局配置（对话式 v4.0）

API Key 加载策略（满足「不再在界面填写」要求）：
  1) 优先读取环境变量 DEEPSEEK_API_KEY；
  2) 其次读取项目根目录 .env（python-dotenv 已加载）；
  3) 若两者皆空，则 AI 提取 / 生成类功能不可用，但规则引擎与本地检索问答仍正常工作。

如需在部署环境启用 AI 功能，在启动前执行：
    export DEEPSEEK_API_KEY="sk-xxx"
或在项目根目录创建 .env 写入：
    DEEPSEEK_API_KEY=sk-xxx
"""
import os

# 模型标识（与 utils/llm_helper.MODEL_CONFIGS 对应）
MODEL_NAME = "DeepSeek-V4-Flash"

# API Key：仅从环境变量 / .env 获取，绝不在界面暴露输入框
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# 数据持久化目录（会话与跨期企业数据）
DATA_DIR = "data_store"
