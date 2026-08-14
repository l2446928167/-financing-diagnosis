"""
utils/llm_gateway.py — 服务端统一 LLM 网关（多用户共享凭证架构）

设计目标（对应需求：用户打开即用、无需填 Key、密钥对前端透明、多用户共享一套凭证）：
  1. 密钥只在服务端进程内加载一次，存于单例，绝不落到前端 / 浏览器 / git。
  2. 所有对话调用都汇聚到 gateway().chat()，由网关统一：
        - 读取密钥（st.secrets → 环境变量 → .env，优先级递减）
        - 限流（令牌桶，保护共享配额不被单个用户刷爆）
        - 并发控制（信号量，避免瞬时大量请求打满配额）
        - 按 user_id 记账（审计 / 后续可按用户做成本归因）
  3. 前端与会话（st.session_state）永远不持有明文密钥；多用户共用网关里的同一套凭证，
     隔离的是“对话历史”，不是“密钥”。

注意：call_llm 仍保留 api_key 形参，网关调用时传入自己的 self.key（传入的 api_key 会被忽略），
因此对现有调用点（如 generate_diagnosis_text / llm_extract_metrics）几乎零改动即可接入。
"""

import os
import time
import threading

import streamlit as st


def _load_key() -> str:
    """加载 DeepSeek API Key，优先级：Streamlit secrets → 环境变量 → .env。

    部署到 Streamlit Cloud / Docker 时，把密钥写在 Secrets 或环境变量里即可，
    对所有用户透明、无需任何人填写。
    """
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


class LLMGateway:
    def __init__(self, rate_limit: int = 20, max_concurrency: int = 6):
        self.key = _load_key()
        self.available = bool(self.key)           # 服务端是否已配置密钥
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(max_concurrency)   # 并发上限，保护共享配额
        self._rate_limit = rate_limit            # 令牌桶：每秒最多请求数
        self._ts: list = []                       # 最近请求时间戳
        self._usage: dict = {}                    # user_id -> 调用次数（审计）

    # ---- 令牌桶限流（每秒最多 self._rate_limit 次）----
    def _throttle(self):
        with self._lock:
            now = time.time()
            self._ts = [t for t in self._ts if now - t < 1.0]
            guard = 0
            while len(self._ts) >= self._rate_limit and guard < 100:
                sleep_for = max(0.0, 1.0 - (now - self._ts[0]))
                if sleep_for <= 0:
                    break
                time.sleep(sleep_for)
                now = time.time()
                self._ts = [t for t in self._ts if now - t < 1.0]
                guard += 1
            self._ts.append(time.time())

    def chat(self, system_prompt, user_prompt, model_choice=None, api_key=None,
             user_id="anon", temperature=0.3, max_tokens=800):
        """统一对话入口。api_key 形参保留以兼容旧调用点，但一律使用网关自身的 self.key。"""
        if not self.available:
            return None
        self._throttle()
        with self._lock:
            self._usage[user_id] = self._usage.get(user_id, 0) + 1
        with self._sem:
            from utils.llm_helper import call_llm
            return call_llm(system_prompt, user_prompt, model_choice, self.key,
                            temperature=temperature, max_tokens=max_tokens)

    def test(self):
        """连通性自检，返回 (成功布尔, 消息字符串)。"""
        from utils.llm_helper import test_api_connection
        return test_api_connection(self.key)

    def usage(self):
        """返回各用户调用次数快照（审计用）。"""
        with self._lock:
            return dict(self._usage)


_gw = None


def gateway() -> LLMGateway:
    """进程内单例。首次调用时（已在 Streamlit 运行时内）才读取 secrets，时机安全。"""
    global _gw
    if _gw is None:
        _gw = LLMGateway()
    return _gw
