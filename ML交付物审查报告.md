# ML 双轨违约模型交付物审查报告（ml-v0.1-dual-track）

审查对象：`ml-v0.1-dual-track_1db9295a.zip`
审查时间：2026-08-13
审查方法：静态代码审查 + 纯标准库实测（本机无 pip / 无 numpy·xgboost·sklearn·matplotlib，且 Bash 沙箱无外网，**无法端到端复跑 `train_ml.py` 或加载 `xgb_default.json`**）+ 产物文件核对。

---

## 一、整体评价（先肯定）

这是一份**质量相当高**的 ML 交付物，清晰吸收了此前对 A–E 方案的修正 1–4 与 P1/P2 全部条目：

- ✅ **修正1（同源派生）**：`generate_statements` 先产原始报表，`statement_to_ml_features` 与 `statement_to_rules_input` 从同一张表派生，原始表直喂 `diagnose()`，零比率反推。
- ✅ **修正2（分歧来源收敛）**：全样本固定"法院执行=无、征信=良好"，分歧纯来自行业周期+交互+拐点，双轨对照干净。
- ✅ **修正3（指标体系分离）**：AUC/KS/PR-AUC/Brier 仅报 XGB/LR；规则卡作阈值分类器（总分<4 拒）报 P/R/F1 + 分箱单调性表。
- ✅ **修正4（部署 wheel）**：xgboost 3.4.0 为 py3-none 通用 wheel，已记录版本。
- ✅ **P1**：早停+正则、`pred_contribs` 自洽断言、`pred_interactions` 版本门控、隐藏因子泄漏双断言、子集头条实验、案例文案自动生成、LR 基线定位——全部到位。
- ✅ **P2**：校准曲线、5% 标签噪声鲁棒性、字体降级兜底、懒加载降级、顶部固定披露段——到位且诚实。
- ✅ **披露段**：合成数据、分歧率受控、需真实数据重训、指标仅合成测试集有效——评委关切点已主动覆盖。

**实测结论**：我用纯标准库复刻验证了三个部署命门不变量（见 §二），全部 PASS；`xgb_default.json` 为合法 XGBoost 3.x booster JSON（含 learner/version，215KB）；`sensitivity.csv` 16 组扫描数据完整。

---

## 二、实测结果（纯标准库，无需依赖）

| 测试 | 内容 | 结果 |
|---|---|---|
| T1 | 训练端 `statement_to_ml_features` vs 推理端 `statement_to_features` 12 维输出逐元素一致（3 样例） | **PASS** |
| T2 | `dual_track_conclusion` 真值表（10 组边界，含 None/中间态）无死分支 | **PASS** |
| T3 | 归因自洽恒等式 `sigmoid(Σcontribs) == p`（数学核对） | **PASS** |
| F1 | `xgb_default.json` 结构合法（learner/version 键） | PASS |
| F2 | `sensitivity.csv` 16 组 λ 扫描数据完整 | PASS |

> 测试脚本：`ml_review/test_pure.py`（纯标准库，可作为 CI 回归基线，建议纳入仓库）。

T1 同时暴露一个**鲁棒性差异**：推理端对缺失字段返回 0 不崩，训练端直接索引会 `KeyError`（见 §三 P1-3）。

---

## 三、改进清单（按优先级）

### P0 · 部署/复现阻断级（建议必改）

**P0-1  `train_ml.py` 硬依赖 `modules/diagnosis.py`，但包内未包含**
- 现象：`scripts/train_ml.py:30` `from modules.diagnosis import diagnose`；压缩包只有 `modules/ml_model.py`，无 `diagnosis.py`。
- 影响：脱离主仓库无法独立复现/运行，评审或队友拿到这个 zip 直接 `ImportError`。
- 建议：① 包内补一个轻量 `diagnosis.py` 占位（仅暴露 `diagnose()` 接口签名 + mock 评分），使脚本可独立跑通"方法论演示"；或 ② 在 README 明确"**必须放置于主项目根目录**运行"。推荐①+②。

**P0-2  字体 `fonts/simhei.ttf` 缺失**
- 现象：README 称"fonts/simhei.ttf 存在已核实"，但压缩包无 `fonts/` 目录。`setup_font` 有降级（不中断），但图表中文会乱码。
- 影响：交付的 `cases/*.png` 若在他处重跑会产生乱码图；当前包内 png 是训练机产物，但复现性受损。
- 建议：打包字体文件，或图表统一用英文标签，或 README 声明"需自备字体/接受英文"。

### P1 · 正确性与稳健性

**P1-1  `assert` 用于运行时关键不变量（部署 `-O` 下失效）**
- 现象：`ml_model.py:234` `assert HIDDEN_FEATURE not in inp`、`explain_contribs` 自洽断言、`leakage_assertion` 的 `assert np.array_equal(...)`。
- 影响：`python -O` 会剥离所有 assert，隐藏因子泄漏保护、SHAP 自洽保护全部失效——偏偏这些是不变量最关键处。
- 建议：改为显式 `if not cond: raise RuntimeError(...)`。例如：
  ```python
  if HIDDEN_FEATURE in inp:
      raise RuntimeError("隐藏因子泄漏进规则卡输入！")
  if not np.allclose(p_from_shap, proba, atol=1e-5):
      raise RuntimeError("SHAP 贡献求和与模型概率不一致！")
  ```

**P1-2  模型版本未校验**
- 现象：`xgb_default.json` 由 xgboost 3.4.0 训练，`feature_meta.json` 记录了 `xgboost_version`，但 `load_model()` 加载时不比对运行环境版本。
- 影响：跨 xgboost 主版本 load 可能失败或行为漂移（尤其是 1.x/2.x/3.x JSON booster 格式）。
- 建议：`load_model` 中比对 `xgb.__version__` 与 `meta["xgboost_version"]`，不一致则打印警告并降级为规则卡。

**P1-3  训练/推理特征函数鲁棒性不一致（T1 暴露）**
- 现象：推理端 `statement_to_features` 用 `.get(x,0)` + 零除保护；训练端 `statement_to_ml_features` 直接索引 `s["营业收入"]`，缺失/零值会 `KeyError`/`ZeroDivisionError`。
- 影响：两函数对异常输入行为不同，且推理端对"零营收"返回 0、训练端会崩——边界口径未对齐。
- 建议：两函数统一带保护写法（都加 `.get`/零除保护），并把 `test_pure.py` 的 T1 纳入回归固化一致性。

**P1-4  `runtime.txt`"无需"结论有部署风险**
- 现象：README/报告称"py3-none 通用 wheel，无需 runtime.txt；实证 runtime.txt 被现行构建系统忽略"。
- 影响：这是经验性假设。Streamlit Community Cloud 官方支持 `runtime.txt` 指定 Python；平台升级或换部署目标时"被忽略"假设可能反转，导致默认 Python 3.14 与 wheel 实测偏差。
- 建议：保留 `runtime.txt=python-3.12`（双保险），或明确声明"已实测 3.14 wheel 可用并锁定 xgboost==3.4.0"，而非依赖"被忽略"。

### P2 · 方法论与呈现打磨

**P2-1  XGB 与 LR 的 AUC 几乎持平（0.9436 vs 0.947）**
- 现象：标签由线性分量主导（`z_lin` 最大系数 2.6），非线性/隐藏分量贡献相对小，故 LR 几乎追平 XGB。非线性子集 XGB 抓 72/77 vs LR 65/77，差距有限。
- 影响：削弱"树模型 > 线性"的论证力度。
- 建议：① 适度提升交互/拐点/隐藏分量权重，使 XGB 在非线性子集明显领先 LR；或 ② 在报告更突出"双轨价值不在 AUC 绝对值，而在规则卡盲区的个案捕获（ML 增量 137/178）"。

**P2-2  规则卡 Recall=0.0024 / F1=0.0047 过低**
- 现象：阈值<4 拒贷导致合成数据上几乎不拒（分箱 [2,4) 仅 1 例）。虽设计使然并已说明，但作为"基准对照"略显刻意。
- 建议：补充规则卡不同阈值下的 P/R 曲线，或调整叙事为"规则卡=稳定性基线、ML=灵敏度增量"，更平衡。

**P2-3  `sensitivity.csv` 报告"16 组全部落带"不准确**
- 现象：实际有 2 组低于目标带下限 10%（λ=2.0/0.5 → 9.46%；λ=1.5/0.5 → 10.04%）。
- 建议：修正措辞为"绝大多数落在 10–20% 带"，或把 `DIVERGENCE_BAND` 下限放宽到 9%。

**P2-4  `dual_track_conclusion` 与 `divergence_stats` 阈值口径不一致**
- 现象：结论里 `strong_reject` 用 `proba>0.60`，分歧统计用 `ml_low=0.35`；二者用途不同（结论分级 vs 分歧统计）不算 bug，但易混淆。
- 建议：加注释说明"结论三级（低/中/高）与分歧统计二分（低/高）口径分离"。

**P2-5  校准曲线/噪声实验仅覆盖 XGB**
- 建议：补全 LR 校准对照图与 LR 噪声对照，使双模型对比更完整。

**P2-6  README 文件名中文乱码**
- 现象：`说明_README.md` 在 zip 内变为 `Φ»┤µÿÄ_README.md`（打包时非 UTF-8 文件名编码）。
- 建议：重打包统一用 ASCII 名 `README.md`，避免跨平台乱码。

**P2-7  工程整洁度**
- `__import__("time").time()` 多次调用（建议顶部 `import time`）；缺 `requirements.txt`（建议锁定 `xgboost>=1.6, numpy, pandas, scikit-learn, matplotlib`）与 `modules/__init__.py`。

---

## 四、建议优先级与下一步

| 优先级 | 条目 | 动作 |
|---|---|---|
| **P0** | P0-1 依赖 diagnosis.py、P0-2 字体缺失 | 补全依赖/占位或明确运行前提，否则无法独立复现 |
| **P1** | P1-1 assert→raise、P1-2 版本校验、P1-3 特征函数统一、P1-4 runtime.txt 双保险 | 提升部署稳健性，建议全部纳入 |
| **P2** | P2-1~P2-7 | 打磨方法论论证与呈现，答辩前完成即可 |

**结论**：核心方法与代码质量已达竞赛可用水平，命门不变量实测通过。P0 两项关乎"能否独立复现/呈现"，建议优先修；P1 四项关乎"线上稳健性"，强烈建议修；P2 为答辩加分项。

**本环境无法实跑的限制说明**：本机无 pip、无 numpy/xgboost/sklearn/matplotlib，且 Bash 沙箱无外网，故 `train_ml.py` 全量训练与 `xgb_default.json` 加载未能端到端复跑；上述判断基于静态审查 + 纯标准库实测 + 产物核对。在具备依赖的环境（主项目根目录 + `pip install -r requirements.txt`）下，建议运行 `python scripts/train_ml.py` 做最终复现确认。
