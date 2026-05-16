# ⚽ Football Predictor v5.0

**XGBoost × Poisson Ensemble × Kelly Staking — 全流程足球预测与回测系统**

从数据管道、特征工程、模型训练、概率集成到严谨回测，构建了一套完整的足球预测系统。覆盖五大联赛 12 个赛季，21,511 场比赛，xG 覆盖率 99.3%。模型概率校准逼近博彩公司水平（ECE < 0.03），大小球 Under 市场发现疑似 edge。

---

## 目录

- [项目概览](#项目概览)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [Pipeline 架构](#pipeline-架构)
- [使用详解](#使用详解)
- [模型架构](#模型架构)
- [性能指标](#性能指标)
- [回测发现](#回测发现)
- [诚实评估](#诚实评估)
- [文件结构](#文件结构)
- [开源协议](#开源协议)

---

## 项目概览

| 维度 | 指标 |
|------|------|
| 数据规模 | 21,511 场比赛，64 个特征列 |
| 覆盖范围 | 英超、西甲、德甲、意甲、法甲 |
| 时间跨度 | 2014-15 至 2025-26（12 个赛季） |
| xG 覆盖率 | 99.27% |
| 数据来源 | Football-Data.co.uk + Understat |
| 模型 | XGBoost 分类器 + 2 个 Poisson 回归器 + 集成 |
| 输出 | 胜平负概率、主客队进球期望、大小球概率 |

---

## 环境要求

- **Python 3.9+**
- 核心依赖：`pandas` `numpy` `xgboost` `scipy` `scikit-learn` `tqdm`
- 可选：`matplotlib`（输出图表）

```bash
pip install -r requirements.txt
```

macOS 用户还需安装 XGBoost 的 OpenMP 运行时：

```bash
brew install libomp
```

---

## 快速开始

### 一键运行全流程

```bash
# 阶段 1: 数据管道（下载原始数据 → 合并）
python run_pipeline.py

# 阶段 2: 特征工程（原始数据 → 64维特征矩阵）
python features.py

# 阶段 3: 模型训练（XGBoost 分类 + Poisson 回归）
python train.py

# 阶段 3.5: 集成（Poisson 推导胜平负 + 加权集成）
python ensemble.py

# 阶段 4: 胜平负回测（校准、value bet、模拟下注）
python backtest.py

# 阶段 5: 大小球回测（Poisson 推导 Over/Under + 回测）
python over_under.py

# 阶段 5.5: 稳健性检验（Bootstrap、Walk-Forward 等6项）
python robustness_check.py
```

### 5 分钟测试

```python
import pandas as pd
import xgboost as xgb
import json

# 加载特征
features = pd.read_parquet("data/processed/features.parquet")

# 加载模型
with open("models/feature_columns.json") as f:
    cols = json.load(f)

clf = xgb.Booster()
clf.load_model("models/clf_result.json")

# 预测
X = features[cols].tail(10).astype(np.float32)
probs = clf.predict(xgb.DMatrix(X, missing=np.nan))
print(probs)  # [[P(H), P(D), P(A)], ...]
```

---

## Pipeline 架构

```
                          ┌─────────────────────┐
                          │  Football-Data.co.uk │
                          │     60 CSV files     │
                          └──────────┬──────────┘
                                     │
                                     ▼
┌──────────────┐    ┌─────────────────────────┐    ┌──────────────────────┐
│   Understat  │───▶│  data/processed/         │◀───│  team_name_mapping   │
│ 60 JSON files│    │  matches.parquet         │    │       .json          │
└──────────────┘    │  21,515 rows × 32 cols   │    └──────────────────────┘
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  features.py             │
                    │  滚动窗口 · 主客场分离    │
                    │  H2H · 时间特征 · 市场    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  features.parquet        │
                    │  21,511 rows × 64 cols   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
          ┌──────────────────┐    ┌──────────────────┐
          │  XGBClassifier   │    │  2× XGBRegressor  │
          │  multi:softprob  │    │  count:poisson    │
          │  H/D/A 三分类     │    │  主客队进球数回归   │
          └────────┬─────────┘    └────────┬─────────┘
                   │                       │
                   └───────────┬───────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │  ensemble.py             │
                    │  加权集成 + Poisson 推导   │
                    │  w_xgb=0.65, w_poisson=0.35│
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
          ┌──────────────────┐    ┌──────────────────┐
          │  胜平负回测        │    │  大小球回测        │
          │  backtest.py      │    │  over_under.py    │
          │  ROI: -2.5%       │    │  Under ROI: +4.4% │
          └──────────────────┘    └────────┬─────────┘
                                           │
                                           ▼
                                ┌──────────────────┐
                                │  稳健性检验        │
                                │  robustness_check │
                                │  6 tests → RED    │
                                └──────────────────┘
```

---

## 使用详解

### 阶段 1: 数据管道 (`run_pipeline.py`)

```bash
python run_pipeline.py
```

- 从 Football-Data.co.uk 下载 60 个 CSV（5 联赛 × 12 赛季）
- 从 Understat API 拉取 xG 数据（60 次请求，速率限制 1.5s/次）
- 标准化球队名称（`data/team_name_mapping.json`，含 230+ 条映射）
- 输出 `data/processed/matches.parquet`

已下载的文件会被跳过，支持断点续传。

### 阶段 2: 特征工程 (`features.py`)

```bash
python features.py
```

生成 **64 列特征矩阵**，包含：

| 特征组 | 列数 | 示例列名 | 说明 |
|--------|------|----------|------|
| ID 列 | 5 | `date`, `league`, `season`, `home_team`, `away_team` | — |
| 主队滚动 5 | 8 | `h_roll5_goals_for`, `h_roll5_win_rate`, ... | min_periods=3 |
| 主队滚动 10 | 8 | `h_roll10_*` | min_periods=5 |
| 客队滚动 5 | 8 | `a_roll5_*` | — |
| 客队滚动 10 | 8 | `a_roll10_*` | — |
| 主场特化 | 5 | `h_venue5_*` | 仅主队主场 |
| 客场特化 | 5 | `a_venue5_*` | 仅客队客场 |
| 时间特征 | 4 | `rest_days_home`, `matchday_away`, ... | — |
| 交锋特征 | 3 | `h2h5_win_rate`, `h2h5_avg_goal_diff`, ... | — |
| 市场特征 | 6 | `mkt_b365_p_home`, `mkt_avg_p_draw`, ... | vig 归一化 |
| 目标变量 | 4 | `target_result`, `target_total_goals`, ... | — |

**防泄漏策略**：所有比赛按日期排序后，`groupby(team).shift(1).rolling(window)` 保证第 i 场只能用 [0, i-1] 的历史。

### 阶段 3: 模型训练 (`train.py`)

```bash
python train.py
```

**时间切分**（严格按赛季，禁止随机切分）：

| 数据集 | 赛季 | 场次 |
|--------|------|------|
| 训练集 | 2014-15 → 2022-23 (9季) | 16,333 |
| 验证集 | 2023-24 (1季) | 1,752 |
| 测试集 | 2024-25 → 2025-26 (2季) | 3,426 |

**三个模型**：XGBoost 分类器（胜平负）+ 两个 XGBoost 回归器（主/客队进球，Poisson 目标函数），均使用早停（patience=50）。

输出到 `models/` 目录：`clf_result.json`、`reg_home_goals.json`、`reg_away_goals.json`、`feature_columns.json`、`label_encoder.json`。

### 阶段 3.5: 集成 (`ensemble.py`)

```bash
python ensemble.py
```

- 用两个 Poisson 回归器的预测进球率推导 H/D/A 概率（独立 Poisson 比分模型）
- 与 XGBoost 分类器加权集成（w=0.65，在验证集上搜索）
- 输出 `models/ensemble_config.json`、`data/processed/ensemble_probs.parquet`

### 阶段 4: 胜平负回测 (`backtest.py`)

```bash
python backtest.py
```

在测试集上完成四项分析：校准曲线（ECE + reliability diagram）、AUC 判别、Value Bet 按 edge 分桶、Kelly 模拟下注（3 种策略对比）。

### 阶段 5: 大小球回测 (`over_under.py`)

```bash
python over_under.py
```

- 利用总进球 ~ Poisson(λ_home + λ_away) 推导 Over/Under 2.5 概率
- 从原始 CSV 提取 B365 大小球赔率
- 完整回测：校准、value bet、Kelly 模拟（4 种策略）

### 阶段 5.5: 稳健性检验 (`robustness_check.py`)

```bash
python robustness_check.py
```

对最优策略做 6 项严格检验：Bootstrap、时间稳定性、联赛稳定性、Edge 阈值稳健性、Walk-Forward、风险调整收益。

---

## 模型架构

### XGBoost 超参数

```python
xgb_params = {
    'n_estimators': 2000,       # 配合早停，实际 75-120 轮收敛
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
}
```

### Poisson 推导胜平负

```
P(i, j) = Poisson.pmf(i, λ_home) × Poisson.pmf(j, λ_away)   # 比分概率矩阵

P(H) = Σ P(i, j) for i > j    (上三角)
P(D) = Σ P(i, j) for i = j    (对角线)
P(A) = Σ P(i, j) for i < j    (下三角)
```

截断 max_goals=10，归一化后与 XGBoost 分类概率加权平均。

### 集成权重

在验证集（2023-24 赛季）上网格搜索得到最优 w_xgb = 0.65。

---

## 性能指标

### 分类模型（测试集：2024-25 + 2025-26）

| 指标 | XGBoost | Poisson | Ensemble + Thresh | B365 基准 |
|------|---------|---------|-------------------|-----------|
| Accuracy | 52.7% | 53.2% | 48.9% | 53.5% |
| Log Loss | 0.9754 | 0.9742 | 0.9727 | 0.9693 |
| Draw Recall | 4.6% | 0.0% | **45.1%** | 0.1% |
| Draw F1 | 0.080 | 0.000 | 0.356 | 0.002 |

### 回归模型

| 指标 | 主队进球 | 客队进球 | 总进球 |
|------|---------|---------|--------|
| RMSE | 1.17 | 1.08 | 1.62 |
| MAE | 0.93 | 0.86 | 1.30 |
| Σ 实际 | 5,196 | 4,380 | 9,576 |
| Σ 预测 | 5,392 | 4,376 | 9,768 |

### 校准（ECE，越低越好）

| 方法 | 主胜 | 平局 | 客胜 |
|------|------|------|------|
| XGBoost | 0.0177 | 0.0259 | 0.0148 |
| Poisson | 0.0277 | 0.0222 | 0.0214 |
| Ensemble | 0.0216 | 0.0145 | 0.0105 |
| **B365** | 0.0227 | 0.0063 | 0.0165 |

所有 ECE < 0.03，模型概率校准**优秀**。

### 特征重要性 Top 5（分类模型，按 gain）

| 排名 | 特征 | Gain | 说明 |
|------|------|------|------|
| 1 | `mkt_b365_p_away` | 29.6 | B365 客胜隐含概率 |
| 2 | `mkt_b365_p_home` | 24.9 | B365 主胜隐含概率 |
| 3 | `mkt_b365_p_draw` | 6.5 | B365 平局隐含概率 |
| 4 | `a_roll10_xg_for` | 3.9 | 客队近 10 场平均 xG |
| 5 | `mkt_avg_p_away` | 3.8 | 市场平均客胜隐含概率 |

赔率特征主导，滚动 form 特征起补充作用 — 符合预期。

---

## 回测发现

### 胜平负市场

| 策略 | 期末资金 | 下注数 | 命中率 | ROI | 最大回撤 |
|------|---------|--------|--------|-----|---------|
| Ensemble + 1/4 Kelly | $352 | 1,743 | 39.3% | **-3.4%** | -87.9% |
| Ensemble + Fixed 1% | $552 | 1,848 | 40.5% | **-2.5%** | -75.1% |
| XGBoost + 1/4 Kelly | $124 | 2,093 | 37.9% | **-4.4%** | -96.1% |

**结论：模型与 B365 高度同质化，edge 被 vig 吃掉。胜平负市场无法盈利。**

### 大小球市场（Over/Under 2.5）

| 策略 | 期末资金 | 下注数 | ROI | 最大回撤 |
|------|---------|--------|-----|---------|
| OU-A: 双边 + Kelly | $525 | 659 | **-3.2%** | -72.9% |
| OU-B: 双边 + Fixed 1% | $846 | 660 | **-2.5%** | -31.4% |
| OU-C: Over Only | $525 | 659 | **-3.2%** | -72.9% |
| **OU-D: Under Only** | **$1,447** | 401 | **+4.35%** | -39.4% |

**Under 2.5 方向发现疑似 edge，edge 单调性显著（-18.8% → +14.4%）**。

### 稳健性检验（OU-D 策略）

| 检验 | 结果 |
|------|------|
| 1. Bootstrap 置信区间 | ❌ FAIL — P(ROI>0)=74.9%, 80% CI=[-4.1%, +13.1%] |
| 2. 时间稳定性 | ⚠️ WARN — 单月贡献 78% 总利润 |
| 3. 联赛稳定性 | ✅ PASS — 5/5 联赛全部盈利 |
| 4. Edge 阈值稳健性 | ⚠️ WARN — 去掉 edge≥7% 后 ROI 转负 |
| 5. Walk-Forward | ❌ FAIL — 6月窗口 WF ROI = -2.6% |
| 6. 风险调整收益 | ❌ FAIL — Sharpe=0.48, 破产概率=24% |

**总评：🔴 红灯 — +4.35% 主要来自运气集中 + 后视过拟合，不是 genuine edge。**

---

## 诚实评估

### 模型做对了什么

1. **概率校准优秀**（ECE < 0.03）：模型的概率估计接近真实频率，比大多数业余预测可靠
2. **AUC 与博彩公司持平**（0.73 vs 0.73）：判别能力达到市场水平
3. **进球数预测精准**（RMSE 1.08-1.17，MAE < 1.0）：Poisson 回归校准良好
4. **Under 2.5 方向 edge 单调**：更高模型 edge → 更高 ROI，是全部回测中唯一的正相关信号

### 模型做错了什么

1. **胜平负市场无 edge**：模型与 B365 高度一致（相关性 >0.9），分歧不足以覆盖抽水
2. **平局几乎无法预测**：XGBoost recall 4.6%，Poisson argmax 直接放弃平局
3. **Under edge 在 Walk-Forward 中消失**：说明参数选择存在后视偏差
4. **Kelly 策略最大回撤过大**（39-96%）：说明模型在连续出错时会过度下注

### 如果要继续

| 优先级 | 方向 | 理由 |
|--------|------|------|
| 高 | 引入更多特征源（球队新闻、伤病、转会市场） | 与市场产生差异化 |
| 高 | 扩大数据范围（欧联杯、欧冠、次级联赛） | 增加样本量，发现被忽视的市场 |
| 中 | 尝试 Bivariate Poisson / Dixon-Coles 模型 | 捕捉进球相关性，改善平局预测 |
| 中 | 探索亚洲盘口市场 | 结构不同（2 结果），vig 更低 |
| 低 | 换用 LightGBM / CatBoost | 边际改进，不是瓶颈 |
| 低 | 引入球员级数据 | 对五大联赛边际价值有限 |

---

## 文件结构

```
football-predictor-v5/
├── run_pipeline.py               # 阶段 1: 数据管道入口
├── features.py                   # 阶段 2: 特征工程
├── train.py                      # 阶段 3: 模型训练
├── ensemble.py                   # 阶段 3.5: 集成与推导
├── backtest.py                   # 阶段 4: 胜平负回测
├── over_under.py                 # 阶段 5: 大小球回测
├── robustness_check.py           # 阶段 5.5: 稳健性检验
├── requirements.txt
├── README.md
├── CLAUDE.md                     # AI 辅助开发指南
│
├── data/
│   ├── raw/
│   │   ├── football_data/        # 60 个 Football-Data CSV
│   │   └── understat/            # 60 个 Understat JSON
│   ├── processed/
│   │   ├── matches.parquet       # 原始合并数据 (21,511 × 32)
│   │   ├── features.parquet      # 特征矩阵 (21,511 × 64)
│   │   ├── ensemble_probs.parquet # 集成概率 (3,426 × 22)
│   │   ├── backtest_bets_*.parquet # 胜平负下注记录
│   │   └── ou_bets_*.parquet     # 大小球下注记录
│   └── team_name_mapping.json    # 队名标准化映射 (230+条)
│
├── scrapers/
│   ├── football_data_scraper.py  # FD CSV 下载器
│   ├── understat_scraper.py      # Understat API 爬虫
│   └── merge.py                  # 合并与标准化逻辑
│
├── models/
│   ├── clf_result.json           # XGBoost 分类器
│   ├── reg_home_goals.json       # 主队进球 Poisson 回归器
│   ├── reg_away_goals.json       # 客队进球 Poisson 回归器
│   ├── feature_columns.json      # 特征列顺序
│   ├── label_encoder.json        # 联赛编码映射
│   ├── ensemble_config.json      # 集成权重配置
│   └── training_report.json      # 训练指标汇总
│
├── reports/
│   ├── ensemble_evaluation.md    # 集成评估报告
│   ├── backtest_report.md        # 胜平负回测报告
│   ├── over_under_report.md      # 大小球回测报告
│   └── robustness_report.md      # 稳健性检验报告
│
└── docs/
    └── FEATURES.md               # 特征说明文档
```

---

## 开源协议

```
MIT License

Copyright (c) 2026 james2077

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 免责声明

本项目仅供学习和研究用途。所有预测结果基于统计模型，不构成任何投注、赌博或其他决策建议。足球比赛存在大量不可预测因素，历史规律不保证未来表现。模型回测发现无稳定正期望策略，使用者应自行承担基于本工具输出所做决策的全部风险。
