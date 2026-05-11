# 📊 Bitcoin Sentiment & Trading Strategy Analysis

Analyzing the relationship between Bitcoin's **Fear & Greed Index** and real trader performance on Hyperliquid (crypto derivatives exchange). Applied Kruskal-Wallis, Spearman Rank, transition modeling, and risk-adjusted metrics to uncover patterns invisible to standard linear tests.

**Data:** 211K+ trades × 2,645 daily sentiment readings (2018–2025)  
**Sources:** [Hyperliquid](https://hyperliquid.xyz/) | [Alternative.me Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/)

---

## 📈 Key Results

| Finding | Insight |
|---|---|
| **Kruskal-Wallis H=730, p≈0** | PnL distributions **significantly differ** across sentiment groups — Pearson missed this entirely |
| **Spearman ρ=0.10, p=0.03** | Significant monotonic relationship between sentiment and profitability |
| **Contrarian pattern** | Longs earn **$119/trade** in Extreme Fear; Shorts earn **$176/trade** in Extreme Greed |
| **Fear → Greed transition** | Single most profitable trading shift at **$523/trade** avg with 87% win rate |
| **Extreme Greed Profit Factor = 11.0** | Traders earn **$11 for every $1 lost** — most capital-efficient environment |
| **Reward/Risk = 1.34 (Extreme Greed)** | Average wins significantly larger than average losses during euphoria |
| **Win rate 76–89% across all sentiments** | Traders are consistently skilled regardless of market mood |
| **Lag-1 prediction p=0.115** | Yesterday's sentiment alone **cannot** predict today's PnL |

## 🚀 Strategy Recommendations

1. **Directional Bias** — Weight Longs during Extreme Fear, Shorts during Extreme Greed (contrarian edge)
2. **Trade the Fear → Greed Breakout** — Scale into positions when sentiment flips from scared to optimistic
3. **Optimize Capital by Risk** — Increase position sizing during Extreme Greed (Profit Factor 11.0, best Sharpe)
4. **Don't Use Lag-1 Signals** — Sentiment is a macro filter, not a next-day buy/sell trigger

## 🔬 Methodology

| Step | Notebook | Technique |
|---|---|---|
| Baseline EDA | `bitcoin_sentiment_analysis.ipynb` | Merge trades with sentiment, compute win rates, PnL by group, 10-panel visualization |
| Statistical Tests | `extended_analysis.ipynb` | Kruskal-Wallis H-test, Spearman Rank, Pearson comparison |
| Transition Modeling | `extended_analysis.ipynb` | Day-to-day sentiment shift probabilities & PnL per transition |
| Risk-Adjusted Metrics | `extended_analysis.ipynb` | Reward/Risk ratio, Profit Factor, Sharpe-like ratio, Max Drawdown |
| Lag-1 Prediction | `extended_analysis.ipynb` | Spearman on lagged sentiment vs next-day PnL |
| Consolidated Findings | `findings_and_recommendations.ipynb` | Interview-ready summary with all insights |

## 🔗 Project Structure

```
trade_ai/
├── data/                     # Raw datasets
├── notebooks/
│   ├── bitcoin_sentiment_analysis.ipynb   # Baseline EDA & charts
│   ├── extended_analysis.ipynb            # Advanced statistical tests
│   └── findings_and_recommendations.ipynb # All insights consolidated
├── charts/                   # Generated visualizations
├── requirements.txt
└── README.md
```

---
*Built by Luqman Syed*