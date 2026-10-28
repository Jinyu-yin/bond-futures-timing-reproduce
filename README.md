# Bond Futures Timing Reproduction

> 复现一份卖方研报里的**国债期货择时策略**：从因子筛选、信号合成到净值出图，全流程脚本化，并把"黑箱"拆成可以逐层验证的环节。

![python](https://img.shields.io/badge/python-3.10%2B-3776ab) ![style](https://img.shields.io/badge/style-paper--like-8b5cf6)

---

## 项目定位

这份工作解决的是研报复现里最尴尬的一类问题：

> 研报给了一条漂亮的净值曲线，但**没说清楚参数怎么来的**。

于是这里的做法是——不猜参数，而是**把"最像研报"定义成一个可优化的目标**，用搜索去找。围绕这个思路，仓库里沉淀了三个阶段：

| 阶段 | 做什么 | 关键脚本 |
|---|---|---|
| **A. 自动找参数** | 在参数网格里搜"最像研报"的组合 | `optimize_public_paper_style_fast.py` / `optimize_public_paper_target.py` → `rank_most_paper_like.py` |
| **B. 局部精修** | 围绕 Top-5 参数做邻域夹逼 | `second_stage_refine.py` → `third_stage_refine.py` → `fourth_stage_robust.py` |
| **C. 因子诊断** | 判断哪类因子在拖累、哪类值得加强 | `diagnose_factor_buckets.py` / `diagnose_public_divergence.py` |

---

## 策略逻辑：V1 与 V2

标的为两个中证指数：

- **5–10 年期国债活跃券指数（931018）** — 做多腿
- **中证短融指数（H11014）** — 空头 / 对冲腿

### V1：固定规则

看国债与短融的日收益差值，差值为正满仓国债、为负满仓短融。因子表只做展示，不参与决策。

### V2：动态选因子

这是核心升级——**规则由历史数据决定，而不是人拍脑袋**：

```
每 12 个月复盘一次
   └─ 回溯过去 3 年数据
        ├─ 逐个宏观指标 × 预测周期(1/5/10/22 日) 打分
        ├─ 粗筛：把指标转成看多/看空信号，跑模拟，不赚钱的直接淘汰
        ├─ 精筛：按盈利排序，相关系数 > 0.75 的视为重复，剔除
        ├─ 上限 20 个指标
        └─ 生成信号 → 固定使用一整年
```

信号合成有两套：

- **Static**：入选指标等权平均（全员平等投票）；
- **Dynamic**：滚动 60 天观察，近期持续预判错误的指标**暂停投票权**，只有当下有效的才参与合成。

最后做限幅，把信号强制约束在 `[-3, 3]`，防止极端值导致仓位跳变。

> 详细的中文白话说明见 [`V2方法说明.md`](V2方法说明.md)，逐步命令见 [`docs/original-run-guide.md`](docs/original-run-guide.md)。

---

## 目录结构

```
bond-futures-timing-reproduce/
├── README.md
├── V2方法说明.md                        # V2 策略的完整白话说明 ⭐
├── docs/original-run-guide.md           # 原始逐步运行指南
├── paper_style_bond_futures_timing.py   # 研报风格主策略
├── build_public_macro_dataset.py        # 公开宏观数据替代版
├── gov_cp_*.py                          # 国债/短融择时各环节（权重、净收益、指标、HMM）
├── optimize_public_*.py                 # 参数优化（fast / target）
├── second_stage_refine.py               # 二阶段局部精修
├── third_stage_refine.py
├── fourth_stage_robust.py / _pretty.py
├── rank_most_paper_like.py              # 按"像研报"程度排序
├── diagnose_factor_buckets.py           # 因子桶诊断
├── diagnose_public_divergence.py        # 公开数据 vs 研报的偏离诊断
├── plot_*.py                            # 各类出图脚本
├── build_ls_gov_cp_report*.py           # HTML/PDF 简报生成
├── reports/gov_cp_ls/                   # 已生成的 HTML 简报（含多种变体）
└── *.csv                                # 净值序列、参数网格结果、诊断表
```

---

## 快速开始

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl jinja2
```

按推荐顺序执行（每一步都有落盘产物，可断点续跑）：

```bash
# 1. 构建公开数据替代版
python build_public_macro_dataset.py

# 2. 基础回测 + 出图
python run_public_paper_style.py
python plot_public_paper_style_figures.py

# 3. 阈值对比
python run_public_threshold_comparison.py
python plot_public_threshold_figures.py

# 4. 参数优化（最耗时）
python optimize_public_paper_style_fast.py
python optimize_public_paper_target.py
python rank_most_paper_like.py

# 5. 局部精修
python second_stage_refine.py
python third_stage_refine.py

# 6. 因子诊断
python diagnose_factor_buckets.py
python diagnose_public_divergence.py
python plot_public_fast_target_results.py
```

---

## 关键输出

| 文件 | 含义 |
|---|---|
| `public_ranked_most_paper_like.csv` | 全参数组合按"像研报"排序（主结果表） |
| `public_fast_optimization_results.csv` | fast 目标下的参数网格 |
| `public_paper_target_optimization_results.csv` | paper-target 目标下的参数网格 |
| `public_second_stage_refine_results.csv` | 局部精修结果 |
| `factor_bucket_diagnosis.csv` | 四类因子桶的贡献诊断 |
| `public_divergence_diagnosis.csv` | 与研报的偏离归因 |
| `reports/gov_cp_ls/report.html` | 可直接阅读的 HTML 简报 |

---

## 多空组合的几种构造

`build_public_macro_dataset.py` 会在 `Factors` 中额外写入几条**多空参考**序列（前缀 `5_`，不参与默认四因子分组，全部使用中证口径、不使用中债指数）：

| 序列 | 含义 |
|---|---|
| `5_long_gov_510y_csi931018_close` | 中证 5–10 年国债活跃券指数收盘 |
| `5_short_cp_csi_close` | 中证短融指数 H11014 收盘 |
| `5_ls_gov510y_minus_shortcp_nav` | 等额 1:1 多国债 / 空短融，日末再平衡 |
| `5_hedge_rollbeta60_ls_nav` | 滚动 60 日 β 对冲（用前一日 β 乘空头腿） |
| `5_strat_longonly_nav` | 买入并持有国债活跃券 |
| `5_strat_longshort_nav` | 等价于等额多空组合 |

---

## 重要的诚实声明

- **本仓库不含原始研报 PDF**，也不含由其渲染的逐页截图。`figures/` 与 `doc_assets/` 中的图表是本策略的产出，已随仓库提供，也可由 `plot_*.py` 重新生成。
- **数据口径不同**：研报用的可能是 Wind / iFinD 全量宏观库，这里用公开数据替代，因此**绝对值不可直接对比**，但方法框架可直接复用。
- **无交易成本**：所有净值均为毛收益，未计手续费、冲击成本与保证金占用。真实收益会明显低于图示。
- **参数搜索有过度拟合风险**：以"像研报"为目标做优化，本质是在拟合一条已知曲线。请把 `public_ranked_most_paper_like.csv` 当作**敏感性分析**看，而不是当作最优参数。
- 时间点对齐：宏观指标的**发布时点**与数据**所属期**需严格区分，否则会引入前视偏差。

---

## 复用建议

如果换成 Wind / iFinD 的原始数据与精确发布时间，整套框架可以直接复用：

1. 替换 `build_public_macro_dataset.py` 的数据加载层；
2. 保留 `optimize_*` / `*_refine` 的搜索与诊断逻辑；
3. 用 `diagnose_factor_buckets.py` 重新判断因子桶权重。
