# 对话式 UI 重构 + PDF 报告改进说明（v3.0）

> 适用项目：`l2446928167/-financing-diagnosis`（AI + 小微企业融资诊断 Streamlit 应用）
> 交付内容：将「四标签页切换任务」改造为「对话式交互」，并修复 / 丰富 PDF 诊断报告。

---

## 一、改动概述

| 类别 | 文件 | 改动 |
|------|------|------|
| 对话式 UI | `app.py` | 移除 `st.tabs`，改为 `st.chat_message` + `st.chat_input` 对话流；状态机驱动；结果以助手消息返回 |
| 视觉系统 | `utils/ui_style.py` | 重写配色 token、字体（Inter + Noto Sans SC）、消息气泡 / 输入 / 按钮 / 表单样式 |
| 主题 | `.streamlit/config.toml` | 主色 / 背景 / 文字色同步 |
| PDF 报告 | `modules/report_generator.py` | 新增 ML 双轨 / SHAP / 差距分析 / 完整指标明细四节；**修复表格溢出**；规范化排版；自包含中文字体 |
| 工程补全 | `modules/*`、`utils/llm_helper.py`、`requirements.txt`、`models/`、`runtime.txt` | 从仓库 `main` 同步完整运行所需模块，使工程可直接 `streamlit run` |

> 算法模块 `modules/diagnosis.py`、`product_matching.py`、`gap_analysis.py`、`data_input.py`、`ml_model.py` 与 `utils/vector_store.py` **未改动逻辑**，仅随工程一并补齐以便本地运行。

---

## 二、UI 对话式重构

### 2.1 架构（状态机 + 消息历史）
- `session_state.stage ∈ {init, need_confirm, diagnosed}` 驱动主流程：
  - `init`：引导上传 / 手动录入；
  - `need_confirm`：渲染「指标确认 + 补充信息」live 表单（不污染对话历史）；
  - `diagnosed`：诊断完成，可自然衔接产品匹配 / 问答 / 报告。
- `session_state.messages` 保存完整对话历史；每类结果（诊断 / 产品 / 问答 / 报告）以**助手消息**返回，可在同一对话中连续推进，无需切换标签页。
- 政策 / 产品问答随时可用：自由文本默认走 RAG 溯源问答（`_classify` 命中诊断 / 产品 / 报告关键词才走对应任务）。

### 2.2 交互流程（参考主流 AI 助手）
1. 左侧边栏：极简控制（API Key、文件上传、清空对话）。
2. 上传财务文件或点「手动录入指标」→ 对话中引导确认指标。
3. 确认后点「开始金融健康诊断」→ 对话返回 8 维评分 + 双轨结论 + SHAP + 风险 / 建议。
4. 对话中继续：「查看产品匹配」「问政策问题」「生成 PDF 报告」一键衔接。

### 2.3 配色 token（现代金融风，清晰层次）
| 角色 | 颜色 | 用途 |
|------|------|------|
| 主色 / 强调 | `#2F54EB`（geekblue） | 按钮、链接、表头、分节标题、 active 态 |
| 背景 | `#F5F6F8` | 页面底色 |
| 表面 | `#FFFFFF` | 卡片 / 气泡 |
| 正文 | `#1F2733` | 主文字 |
| 次级文字 | `#5B6573` | 说明 / 注释 |
| 边框 | `#E6E8EC` | 分隔线 / 网格 |
| 健康 / 关注 / 高风险 | `#2BA471` / `#D98B1F` / `#E5484D` | 评分灯色、风险标识 |

### 2.4 字体方案
- 引入 **Inter**（西文）+ **Noto Sans SC**（中文）via Google Fonts，字号 / 字重与界面层级协调：标题 14–18px / 正文 9.5–14px / 注释 7.5–9px，提升阅读体验。
- 对话气泡、输入框圆角、按钮 primary 态统一主色，整体干净、聚焦对话本身。

---

## 三、PDF 报告改进（要求 #6）

### 3.1 修复表格溢出 bug（关键）
原报告产品匹配表的「差距说明」等长文本列使用**原始字符串**塞入单元格，超过 A4 列宽即溢出 / 截断。
**修复**：所有表格单元格一律用 `Paragraph` 包裹（含 `差距说明`、`准入条件`、`相关产品` 等），reportlab 按列宽自动换行，彻底消除溢出。

### 3.2 新增评价 / 诊断维度（依勾选）
1. **ML 双轨对照节**：违约概率（`ml_proba`）+ 规则卡 × ML 双轨结论（`ml_conclusion`）；模型不可用时诚实标注为单轨规则卡结论。
2. **SHAP 归因节**：各因子对违约概率的贡献方向与强度（正值推高 / 负值拉低，红绿标识），附「合成数据方法论演示」诚实声明。
3. **差距分析与行动方案表**：基于 `gap_analysis` 输出，按「性价比 = 解锁产品数 ÷ 难度分」排序，含当前值 / 目标值 / 难度 / 影响产品数 / 相关产品 / 预计时间。
4. **诊断指标明细表**：列出诊断用到的**全部指标取值**，提升透明度与可追溯性。

### 3.3 规范化排版
- 统一字体、分节标题（主色 `#2F54EB`）、**彩色表头**、统一网格线与内边距、斑马纹行底色。
- 标题块（报告名 + 生成时间 + 总体评分）置于首页顶部，**未做独立封面页**。
- 依你的选择：**未加页码、未加页脚、未做独立封面**，仅保留结尾一段免责声明（普通段落，非页脚）。

### 3.4 中文字体自包含
报告改用 reportlab 内置 **`STSong-Light`（Adobe CJK）** 字体渲染中文，**不再依赖外部 `fonts/*.ttf` 文件**，在 Streamlit Cloud 等标准 reportlab 安装下即可正确显示中文（缺失时回退 Helvetica）。

---

## 四、运行与部署

### 依赖
`requirements.txt` 已包含：`streamlit / pandas / openpyxl / pdfplumber / reportlab / openai / python-dotenv / httpx / xgboost`。
（本次同步的 `requirements.txt`、`runtime.txt` 来自仓库 `main`。）

### 启动
```bash
pip install -r requirements.txt
streamlit run app.py
```
- **DeepSeek API Key**：在左侧边栏填写可启用 AI 指标提取 / 诊断总结 / 产品推荐 / 溯源问答；不填也能用规则引擎诊断与检索问答（诚实空返回）。

### 模块清单（本次同步 / 改动）
- 改动：`app.py`、`utils/ui_style.py`、`.streamlit/config.toml`、`modules/report_generator.py`
- 同步（逻辑未改）：`modules/{diagnosis,product_matching,gap_analysis,data_input,ml_model,__init__}.py`、`utils/llm_helper.py`、`models/*`、`knowledge/*`

---

## 五、验证与限制

- ✅ 全部模块 `py_compile` 通过；`generate_pdf` 新增形参 `ml_proba/ml_conclusion/shap_contribs/gap_result` 与 `app.py do_report()` 实际传参一致（AST 核对）。
- ✅ `app.py` 所有 widget `key` 唯一（无重复，动态 `metrics_editor_<rev>` 按版本号区分）。
- ⚠️ 本机沙箱未安装 `reportlab` 且无可用的 `pip`，故 PDF 仅做静态结构与 API 校验，未做真实渲染；请在部署环境 `pip install reportlab` 后实测生成。
- ⚠️ 用户原始仓库含 `fonts/`（fzxbs/simhei/kaiti/fangsong.ttf）用于更细腻的排版；本报告为自包含改用了 `STSong-Light`，若需恢复原字体可在 `report_generator.py` 中重新注册 TTF 并替换 `BASE`。
