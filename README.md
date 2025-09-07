````markdown
# TSFM-MRE: Minimal Reproducible Experiment for Time-Series Foundation Models in Finance
[English](#english) · [简体中文](#简体中文)


![Build](https://img.shields.io/github/actions/workflow/status/TonyLu786/tsfm-mre/ci.yml?branch=main&label=CI)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen)
![Reproducibility](https://img.shields.io/badge/reproducibility-one--click-success)

---

## English

### 1) Overview
**TSFM-MRE** is a compact, **audit-ready** pipeline to evaluate time-series foundation models (TSFMs) on yield-curve forecasting and risk backtesting. It runs **offline by default** using synthetic H.15-like data and outputs clean artifacts (Parquet/CSV/PNG/HTML) suitable for peer review and regulatory audit.

**What you get end-to-end**
- Data → preprocessing (business-day alignment, robust scaling) → **distributional forecasts** → **VaR/ES backtesting** → Basel traffic-light → HTML report.
- Strict **rolling-origin** evaluation with multiple horizons (1/5/10/20 business days).
- Default **AR(1)** baseline; optional tiny Transformer (if PyTorch is installed).
- Reproducibility: fixed seeds, deterministic configs, hashed artifacts.

---

### 2) Table of Contents
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Results & Report](#results--report)
- [Continuous Integration (CI)](#continuous-integration-ci)
- [Governance & Compliance](#governance--compliance)
- [Contributing](#contributing)
- [Security](#security)
- [Citation](#citation)
- [License](#license)
- [Publish to GitHub](#publish-to-github)
- [FAQ / Troubleshooting](#faq--troubleshooting)

---

### 3) Quickstart
**Option A — One-click runner (cross-platform)**
```bash
python run_all.py
````

**Option B — Makefile**

```bash
make all
```

**Option C — Shell script**

```bash
bash ./run_all.sh
```

This will:

1. Ingest synthetic H.15-like data (`DGS3MO`, `DGS2`, `DGS10`)
2. Preprocess & align business days; fit robust z-score scalers
3. Produce distributional forecasts for horizons 1/5/10/20
4. Run VaR backtests (99%, h=1) with Kupiec/Christoffersen diagnostics
5. Plot exceptions & rolling Basel traffic-light thresholds
6. Build an HTML report at `outputs/report/index.html`

> **Optional online mode** (requires internet & `pandas_datareader`):
>
> ```bash
> python src/data_ingest.py --outdir data/raw --mode fred --start 2003-01-01 --end 2025-06-30
> ```

---

### 4) Installation

**pip**

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**conda**

```bash
conda env create -f environment.yml
conda activate tsfm-mre
```

Minimal dependencies: `pandas`, `numpy`, `scipy`, `matplotlib`, `pyarrow`, `statsmodels`, `pyyaml`
Optional: `pandas_datareader` (FRED mode), `torch` (tiny transformer)

---

### 5) Repository Structure

```
tsfm-mre/
  src/
    data_ingest.py    # synthetic/FRED ingestion
    preprocess.py     # business-day alignment + robust scaling + splits
    forecast.py       # AR(1) baseline + optional tiny Transformer
    risk.py           # VaR/ES backtesting + Basel traffic-light
    plots.py          # exception overlays & rolling counts
    report.py         # bundles summary HTML report
  data/               # generated data (raw/interim)
  outputs/            # forecasts / risk / figures / report artifacts
  .github/            # CI workflow, issue/PR templates (optional but recommended)
  Makefile
  run_all.py
  run_all.sh
  requirements.txt
  environment.yml
  LICENSE
```

---

### 6) Configuration

Command-line flags let you change dates, splits, horizons, and risk windows. Defaults match the paper’s MRE:

* **Dates:** `2003-01-01` → `2025-06-30`
* **Splits:** `train=2003-01-01 → valid=2013-01-02 → test=2019-01-02`
* **Horizons:** `1, 5, 10, 20` (business days)
* **Risk window:** `250` (Basel convention)

Examples:

```bash
# Change date range
python src/data_ingest.py --outdir data/raw --mode synthetic --start 2005-01-03 --end 2024-12-31

# Change horizons & context window for forecasting
python src/forecast.py --indir data/interim --outdir outputs/forecasts --model ar1 --horizons 1 5 10 --context 60

# VaR backtest at 2.5% for h=5
python src/risk.py --indir data/interim --pred outputs/forecasts/pred_ar1.parquet \
  --outdir outputs/risk --alpha 0.025 --horizon 5 --window 250
```

---

### 7) Results & Report

Artifacts after a successful run:

* `outputs/forecasts/pred_*.parquet` — distributional predictions per series × horizon
* `outputs/risk/risk_summary.csv` — Kupiec/Christoffersen p-values, exception counts, traffic-light zone
* `outputs/risk/var_backtest.csv` — realized vs VaR alignment and exceedance indicators
* `outputs/figures/*.png` — per-series backtest plots & rolling exception counts
* `outputs/report/index.html` — portable summary report (open in a browser)

---

### 8) Continuous Integration (CI)

A ready-to-use workflow (recommended path: `.github/workflows/ci.yml`) can:

* set up Python (3.10 recommended),
* install dependencies,
* run `python run_all.py` entirely **offline** (synthetic data path),
* upload report artifacts (HTML, figures, risk tables, forecasts).


---

### 9) Governance & Compliance

* **Reproducibility**: fixed random seeds; hashed outputs; deterministic configs.
* **Risk alignment**: implements 99% VaR backtesting + Christoffersen/Kupiec diagnostics; plots Basel traffic-light thresholds.
* **Data policy**: offline synthetic data by default; if using public data (FRED), keep source & timestamps in your commit messages or metadata.

---

### 10) Contributing

We welcome issues, feature requests, and PRs. Suggested flow:

1. Fork and create a feature branch: `git checkout -b feat/<short-name>`
2. Keep PRs small and focused; include validation notes.
3. Ensure `python run_all.py` succeeds offline before requesting review.

(Optionally add `CONTRIBUTING.md`, PR/Issue templates, and a `CODE_OF_CONDUCT.md`.)

---

### 11) Security

If you discover a vulnerability or have sensitive concerns, please open a private channel (e.g., [github.com/TonyLu786](github.com/TonyLu786)) or use GitHub’s **Security Policy** page if available. Avoid attaching proprietary datasets/logs in public issues.

---

### 12) Citation

If this repository helps your research, please cite it:

```bibtex
@software{tsfm_mre_2025,
  title        = {TSFM-MRE: Minimal Reproducible Experiment for Time-Series Foundation Models in Finance},
  author       = {Juntong Lu},
  year         = {2025},
  url          = {https://github.com/TonyLu786/tsfm-mre},
  note         = {Offline-capable, audit-ready TSFM evaluation pipeline}
}
```

---

### 13) License

This project is distributed under the **MIT License**. See `LICENSE` for details.

---

### 14) Publish to GitHub

```bash
git init
git checkout -b main
git add .
git commit -m "Initial commit: TSFM-MRE"
git remote add origin git@github.com:TonyLu786/tsfm-mre.git  # or https://github.com/TonyLu786/tsfm-mre.git
git push -u origin main
```

---

### 15) FAQ / Troubleshooting

* **CI fails due to missing system packages**
  Ensure `pyarrow`, `statsmodels`, and a recent `pip` are installed. Try:

  ```bash
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```
* **Plots or report are empty**
  Check that `outputs/forecasts/pred_*.parquet` exists and paths passed to `plots.py` / `report.py` are correct.
* **Use FRED online mode**
  Install `pandas_datareader` and ensure network access; then rerun the data step with `--mode fred`.

---

## 简体中文

### 1）项目简介

**TSFM-MRE** 提供一个精简、**可审计**、**可复现**的实验管线：在收益率曲线任务上评估时间序列基础模型（TSFM），默认**离线**使用合成 H.15 类数据，最终输出 Parquet/CSV/PNG/HTML 等制品，便于学术审稿与合规审计。

**端到端流程**

* 数据 → 预处理（工作日对齐、稳健标准化）→ **分布式预测** → **VaR/ES 回测** → Basel 交通灯 → HTML 报告
* **滚动起点**评估，地平线：1/5/10/20 个工作日
* 默认 **AR(1)** 基线；如已安装 PyTorch，可使用简版 Transformer
* 固定随机种子、可复现配置、制品哈希

---

### 2）目录

* [快速上手](#快速上手)
* [环境安装](#环境安装)
* [目录结构](#目录结构)
* [配置说明](#配置说明)
* [输出与报告](#输出与报告)
* [持续集成（CI）](#持续集成ci-1)
* [治理与合规](#治理与合规)
* [贡献方式](#贡献方式)
* [安全策略](#安全策略)
* [学术引用](#学术引用)
* [许可协议](#许可协议)
* [发布到 GitHub](#发布到-github)
* [常见问题](#常见问题)

---

### 3）快速上手

**方式 A — 跨平台一键运行**

```bash
python run_all.py
```

**方式 B — Makefile**

```bash
make all
```

**方式 C — Shell 脚本**

```bash
bash ./run_all.sh
```

运行后将依次完成：数据摄取 → 预处理 → 分布式预测 → VaR 回测 → 图表 → HTML 报告（`outputs/report/index.html`）。

> **可选在线模式（需联网）**：
>
> ```bash
> python src/data_ingest.py --outdir data/raw --mode fred --start 2003-01-01 --end 2025-06-30
> ```

---

### 4）环境安装

**pip**

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**conda**

```bash
conda env create -f environment.yml
conda activate tsfm-mre
```

核心依赖：`pandas`、`numpy`、`scipy`、`matplotlib`、`pyarrow`、`statsmodels`、`pyyaml`
可选：`pandas_datareader`（FRED 模式）、`torch`（简版 Transformer）

---

### 5）目录结构

```
tsfm-mre/
  src/（核心脚本：数据、预处理、预测、风控回测、出图、报告）
  data/（运行时生成的数据）
  outputs/（预测、回测、图表、报告）
  .github/（CI 与 issue/PR 模板，推荐）
  Makefile / run_all.py / run_all.sh / requirements.txt / environment.yml / LICENSE
```

---

### 6）配置说明

可通过命令行参数调整日期范围、数据切分、地平线与回测窗口。默认：

* 日期：`2003-01-01` → `2025-06-30`
* 切分：`train=2003-01-01 → valid=2013-01-02 → test=2019-01-02`
* 地平线：`1, 5, 10, 20`
* 回测窗口：`250`

示例：

```bash
python src/data_ingest.py --outdir data/raw --mode synthetic --start 2005-01-03 --end 2024-12-31
python src/forecast.py --indir data/interim --outdir outputs/forecasts --model ar1 --horizons 1 5 10 --context 60
python src/risk.py --indir data/interim --pred outputs/forecasts/pred_ar1.parquet --outdir outputs/risk --alpha 0.025 --horizon 5 --window 250
```

---

### 7）输出与报告

* `outputs/forecasts/pred_*.parquet`：分布式预测
* `outputs/risk/risk_summary.csv`：Kupiec/Christoffersen 统计量、异常计数、交通灯分区
* `outputs/risk/var_backtest.csv`：实际值与 VaR 对齐、越界标记
* `outputs/figures/*.png`：回测图与 250 日滚动异常数
* `outputs/report/index.html`：汇总报告

---

### 8）持续集成（CI）

推荐添加 `.github/workflows/ci.yml`：

* 设置 Python（建议 3.10）
* 安装依赖
* 运行 `python run_all.py`（默认走离线合成数据路径）
* 上传 `outputs/**` 为构建工件（Artifact）


---

### 9）治理与合规

* **可复现**：固定随机种子、制品哈希、确定性配置
* **风险口径**：99% VaR 回测，Christoffersen/Kupiec 诊断，Basel 交通灯阈值可视化
* **数据政策**：默认使用离线合成数据；如启用公共数据（FRED），请在提交信息或元数据中保留来源与时间戳

---

### 10）贡献方式

欢迎提 issue、feature request 与 PR。建议流程：

1. 建立分支：`git checkout -b feat/<short-name>`
2. PR 尽量聚焦与小步提交，附上验证说明
3. 在本地先确保 `python run_all.py` 离线运行成功

（可根据需要新增 `CONTRIBUTING.md`、PR/Issue 模板与 `CODE_OF_CONDUCT.md`。）

---

### 11）安全策略

如发现安全漏洞或敏感问题，建议**私密**沟通（例如 [github.com/TonyLu786](github.com/TonyLu786)），或使用 GitHub Security Policy 页面（若已启用）。请勿在公共 issue 中上传含敏感信息的数据或日志。

---

### 12）学术引用

```bibtex
@software{tsfm_mre_2025,
  title        = {TSFM-MRE: Minimal Reproducible Experiment for Time-Series Foundation Models in Finance},
  author       = {Juntong Lu},
  year         = {2025},
  url          = {https://github.com/TonyLu786/tsfm-mre},
  note         = {Offline-capable, audit-ready TSFM evaluation pipeline}
}
```

---

### 13）许可协议

遵循 **MIT License**（见 `LICENSE`）。

---

### 14）发布到 GitHub

```bash
git init
git checkout -b main
git add .
git commit -m "Initial commit: TSFM-MRE"
git remote add origin git@github.com:TonyLu786/tsfm-mre.git  # 或 https://github.com/TonyLu786/tsfm-mre.git
git push -u origin main
```

---

### 15）常见问题

* **CI 缺少依赖**：确保 `pyarrow`、`statsmodels` 与最新 `pip` 已安装。
* **报告为空**：确认 `pred_*.parquet` 已生成，`plots.py`/`report.py` 的输入路径正确。
* **启用 FRED**：安装 `pandas_datareader` 且确保网络可用，再按 FRED 模式重跑数据步骤。

---

```
::contentReference[oaicite:0]{index=0}
```
