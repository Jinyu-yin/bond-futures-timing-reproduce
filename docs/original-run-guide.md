# `just_reference.pdf` 复现与出图

## 1) PDF 逐页导出（对照研报原图）

```bash
py reproduce/render_just_reference_pages.py
```

## 2) 增强版回测

```bash
py just_reference_reproduction_v2.py
py reproduce/plot_just_reference_figures.py
```

## 3) 研报风格示意脚本

```bash
py reproduce/paper_style_bond_futures_timing.py
py reproduce/plot_paper_style_figures.py
```

## 4) 公开数据替代版基础流程

```bash
py reproduce/build_public_macro_dataset.py
py reproduce/run_public_paper_style.py
py reproduce/plot_public_paper_style_figures.py
py reproduce/run_public_threshold_comparison.py
py reproduce/plot_public_threshold_figures.py
```

`build_public_macro_dataset.py` 会在 `Factors` 里额外写入三组「多空参考」序列（前缀 `5_`，不参与 `paper_style_bond_futures_timing` 默认的四类因子分组；**均为中证 CSI 口径，不使用中债指数**）：

- `5_long_gov_510y_csi931018_close`：中证 **5–10 年期国债活跃券**指数（**931018**）收盘（做多/中长期国债参考）
- `5_short_cp_csi_close`：中证短融指数 **H11014** 收盘（短融端参考，多空里常作空头或对冲标的）
- `5_ls_gov510y_minus_shortcp_nav`：日收益为「国债活跃券日收益 − 短融日收益」的累计净值（起点归一为 1），**等额 1:1、日末再平衡** 的简化多空对冲示意
- `5_hedge_rollbeta60_ls_nav`：**滚动 60 日 β** 对冲示意净值，日收益为 rL 减「前一日估计的 β」乘以 rS（减轻短融方向暴露；仍无费用/保证金）
- `5_strat_longonly_nav`：**LongOnly** 策略净值（买入并持有 931018，起点归一）
- `5_strat_longshort_nav`：**LongShort** 策略净值（与 `5_ls_gov510y_minus_shortcp_nav` 相同：等额多国债、空短融，日末再平衡）

简报（HTML + PDF，含图与指标表）：`py reproduce/build_ls_gov_cp_report.py` → `reproduce/reports/gov_cp_ls/report.html` 与 `report.pdf`

## 5) 方案 A：自动找“最像研报”的参数

先跑两套优化：

```bash
py reproduce/optimize_public_paper_style_fast.py
py reproduce/optimize_public_paper_target.py
```

然后统一排序：

```bash
py reproduce/rank_most_paper_like.py
```

输出：

- `reproduce/public_fast_optimization_results.csv`
- `reproduce/public_paper_target_optimization_results.csv`
- `reproduce/public_ranked_most_paper_like.csv`

## 6) 方案 B：二阶段局部精细优化

围绕最像研报的前 5 组参数，在局部邻域再做一次更细搜索：

```bash
py reproduce/second_stage_refine.py
```

输出：

- `reproduce/public_second_stage_refine_results.csv`
- `reproduce/paper_style_nav_public_second_stage_best.csv`

## 7) 方案 C：因子桶诊断（决定哪些类该加强/削弱/剔除）

```bash
py reproduce/diagnose_factor_buckets.py
```

输出：

- `reproduce/factor_bucket_diagnosis.csv`

## 8) 综合诊断与对比图

```bash
py reproduce/diagnose_public_divergence.py
py reproduce/plot_public_fast_target_results.py
```

输出：

- `reproduce/public_divergence_diagnosis.csv`
- `reproduce/figures/fig_09_public_fast_vs_target_vs_base.png`
- `reproduce/figures/public_fast_target_compare_metrics.csv`
- `reproduce/figures/public_fast_top10.csv`
- `reproduce/figures/public_paper_target_top10.csv`

## 推荐执行顺序

```bash
py reproduce/optimize_public_paper_style_fast.py
py reproduce/optimize_public_paper_target.py
py reproduce/rank_most_paper_like.py
py reproduce/second_stage_refine.py
py reproduce/diagnose_factor_buckets.py
py reproduce/diagnose_public_divergence.py
py reproduce/plot_public_fast_target_results.py
```

## 说明

- A：先从 fast / paper-target 两套结果里自动挑“最像研报”的组合；
- B：再围绕这些组合做局部夹逼优化；
- C：最后从四类因子桶角度判断哪些类在拖累、哪些类值得强化；
- 如果后续替换成 Wind / iFinD 的原始数据与更精确的发布时间，整套框架可以直接复用。
