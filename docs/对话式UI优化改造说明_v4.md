# 对话式融资诊断助手 — UI 与报告优化改造说明（v4.0）

> 基于 Streamlit 的「AI + 小微企业融资诊断」项目。本次改造聚焦 7 项要求：移除侧边栏与界面填 Key、聊天框直传文件、对话持久化与企业数据跨期对比、修复乱码、精简说明文案、政策改为信号分析模型并融入诊断、去除内部架构表述。
> 算法模块（`modules/diagnosis.py`、`product_matching.py`、`gap_analysis.py`、`data_input.py`、`ml_model.py`、`utils/vector_store.py`）**均未改动**，仅新增 `modules/policy_signal.py`、`utils/persistence.py`、`config.py`，并重写 `app.py` / 调整 `report_generator.py`。

---

## 一、需求对照与实现

### 1. 彻底移除侧边栏，API Key 改为代码/环境变量加载
- 删除整个 `st.sidebar` 区块（品牌、Key 输入、文件上传、清空、版本）。
- 新增 `config.py`：`DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")`，由**环境变量或 `.env`** 提供；`app.py` 启动时注入 `session_state.api_key`，界面不再出现任何 Key 输入框。
- 部署启用 AI 功能：`export DEEPSEEK_API_KEY="sk-xxx"` 或项目根 `.env` 写入同款变量。未配置时规则引擎与本地检索问答仍可用。

### 2. 对话框支持上传文件（Excel / PDF / CSV）
- 采用主流 AI 助手式交互：`st.chat_input(accept_file=True, file_type=["pdf","csv","xlsx","xls"])`。
- 用户在输入框「回形针」处选文件，可带/不带文字一并发送；上传后自动解析（沿用 `data_input.parse_financial_data` / `auto_extract_metrics`），并在对话中引导确认指标。
- **基于文件内容问答**：新增 `_file_qa()`——上传后可直接就文件提问（如「营收是多少」「应收账款账龄如何」），有 Key 时由大模型基于文件内容溯源作答，无 Key 时本地检索相关行并摘录；非文件类问题自动回落到政策/产品检索库。

### 3. 对话持久化 + 企业数据保存 + 跨期自动对比
- 新增 `utils/persistence.py`：
  - **会话持久化**：每次对话落盘到 `data_store/conversations/{id}.json`（消息、诊断态、功能态），刷新/重启不丢；顶部提供「对话历史」下拉 +「新建对话」按钮，可「继续历史对话」或「新建」。
  - **企业数据快照**：每次完成诊断，把核心指标（营收/净利/资产/负债/现金流/应收/存货/流动比率/资产负债率/增长率/利率等）写入 `data_store/enterprise/{企业名}.json`（按时间追加）。
  - **跨期对比**：同一企业再次上传，自动与最近一次快照比较，输出关键指标变化表（上次/本次/变化/变化%）与「改善 / 恶化 / 持平」趋势结论，并以助手消息播报——满足「记录每周经营流水/财务数据变化、分析演变趋势」。

### 4. 修复对话框乱码
- 全量清理 Emoji：`app.py`（页面图标、验证结果 ✅/⚠️/ℹ️、欢迎 🤝）、`data_input.py`（📊📝🔧📄💡⚠️）、`product_matching.py`（🟢🟡 匹配度）、`diagnosis.py`（红黄绿交通灯）、`report_generator.py`（无 Emoji，但同步去架构词）。
- 验证结果：静态扫描确认全部 `.py` 零 Emoji。
- 字体栈：`ui_style.py` 维持 `Inter / Noto Sans SC` + 系统中文兜底（PingFang SC / 微软雅黑 / 苹方），避免缺失字体导致方块。

### 5. 去除冗余说明文案
- 精简欢迎语，删除「所有任务都在对话里完成，无需切换页面」及逐项功能罗列；顶部副标题改为一句话。
- 删除侧边栏各类提示文案（已随侧边栏移除）。

### 6. 政策模块改为「政策信号分析模型」并融入整体分析
- **删除**独立「了解政策」搜索按钮（含 init / diagnosed 两个入口）。
- 新增 `modules/policy_signal.py`：
  - 内置**分行业时间序列政策库**（新能源 / 科技 / 制造 + 通用宏观），覆盖税收、补贴、信贷、法规、产业规划维度，每条带时间戳与情感/强度。
  - `compute_policy_signal(行业, 时点)`：对「近 12 个月」与「前 12 个月」两个窗口聚合，输出**政策景气指数（0–100）**、景气等级（利好/中性/承压）、趋势（上行/平稳/下行）、对经营稳定性的定性影响、近期政策摘编。指数随时点与行业自然变化，体现「政策随时间变化」。
  - `collect_latest_policies()` 为**实时采集扩展点**（默认返回 None，由内置数据兜底），接入官方政策 API / 爬虫后无需改计算逻辑。
- **融入诊断**：诊断结果消息新增「行业政策环境」区块；PDF 新增「行业政策环境」章节；政策不再单独成窗，而是作为整体分析的输入。

### 7. 去除内部架构表述
- 界面不再出现「规则引擎 × 违约 ML 双轨」「规则卡」「ML 轨道」等词：
  - 诊断卡片标题「双轨结论（规则卡 × ML）」→ **综合结论**；「ML 违约概率」→ **违约风险概率**。
  - 结论文本由 `_clean_conclusion()` 统一映射为中性对外表述（如「综合评估为低风险，建议优先推荐」），不暴露内部双轨逻辑。
  - SHAP 说明改为「以下为风险归因分析，仅供对照参考」；收尾声明改为「以上诊断由智能分析引擎生成，仅供参考，不构成金融建议」。
  - PDF 同步：章节「双轨诊断对照（规则引擎 × 违约 ML）」→「诊断结论与风险概率」；「AI 工具生成」→「智能诊断工具生成」。

---

## 二、文件变更清单

| 文件 | 变更 |
|---|---|
| `config.py` | **新增**：API Key 环境变量/配置加载、模型名、数据目录 |
| `utils/persistence.py` | **新增**：会话持久化、企业快照、跨期对比 |
| `modules/policy_signal.py` | **新增**：分行业政策信号量化模型 |
| `app.py` | **重写 v4.0**：去侧边栏、聊天上传、持久化、趋势、政策融入、去乱码/去架构词、文件问答 |
| `modules/report_generator.py` | 调整：去架构词 + 新增 `policy_result` 形参与「行业政策环境」章节 |
| `modules/data_input.py` | 清理提取流程中的 Emoji |
| `modules/product_matching.py` | 匹配度去 Emoji（完全匹配 / 差距匹配） |
| `modules/diagnosis.py` | 交通灯去 Emoji（绿/黄/红） |
| `utils/ui_style.py` | 维持（字体兜底稳健） |

---

## 三、运行与部署

```bash
pip install -r requirements.txt
# 可选：启用 AI 功能
export DEEPSEEK_API_KEY="sk-xxx"
streamlit run app.py
```

- 数据目录 `data_store/` 首次运行自动创建（已在 `.gitignore` 思路中排除；打包时不含运行时数据）。
- 实测说明：本机开发环境无 `reportlab` 且无法 `pip`，故 PDF 仅做静态校验（语法/签名/排版逻辑），请在部署环境实测生成。

## 四、验证

- ✅ 全部模块 `py_compile` 通过；`generate_pdf` 新形参（`ml_proba/ml_conclusion/shap_contribs/gap_result/policy_result`）与 `app.py` 调用一致（AST 核对）。
- ✅ `app.py` 所有 widget key 无重复；无残留 `st.sidebar` / 架构词。
- ✅ 全量 `.py` Emoji 扫描为零。
- ✅ `policy_signal.compute_policy_signal` 与 `persistence.compare_snapshots` 运行期测试输出合理（行业景气指数随行业/时点变化；跨期对比正确给出「改善/恶化/持平」）。

## 五、已知限制与后续

- 政策数据为**面向演示维护的结构化样本**；接入真实官方数据源后，只需实现 `collect_latest_policies()` 的归一化，计算逻辑无需改动。
- 企业跨期对比按「企业名称（选填）」分组；同名不同主体需用户区分。
- 会话为本地文件存储，多用户/多机场景需替换为数据库或对象存储。
