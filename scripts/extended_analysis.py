"""
Extended Analysis: Kruskal-Wallis, Spearman, Sentiment Transitions,
Risk-Adjusted Metrics, and Lag-1 Predictive Analysis
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Theme ──
BG, PANEL, GRID, TEXT = '#0f0f14', '#1a1a2e', '#ffffff12', '#e0e0e0'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': PANEL,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': GRID,
    'font.family': 'sans-serif'
})

ORDER = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
COLORS = {
    'Extreme Fear': '#e63946', 'Fear': '#f4a261',
    'Neutral': '#adb5bd', 'Greed': '#52b788', 'Extreme Greed': '#2dc653'
}
palette = [COLORS[s] for s in ORDER]
SHORT = ["Ext. Fear", "Fear", "Neutral", "Greed", "Ext. Greed"]

# ═══════════════════════════════════════════════════════
# LOAD & PREP  (mirrors the notebook)
# ═══════════════════════════════════════════════════════
trader = pd.read_csv("../data/historical_data.csv")
sentiment = pd.read_csv("../data/fear_greed_index.csv")

trader['date'] = pd.to_datetime(trader['Timestamp IST'], dayfirst=True).dt.normalize()
for col in ['Execution Price','Size Tokens','Size USD','Closed PnL','Fee']:
    trader[col] = pd.to_numeric(trader[col], errors='coerce')
sentiment['date'] = pd.to_datetime(sentiment['date'])

merged = pd.merge(
    trader,
    sentiment[['date','value','classification']].drop_duplicates('date'),
    on='date', how='inner'
)
merged.rename(columns={'value': 'sentiment_val'}, inplace=True)

closed_m = merged[merged['Closed PnL'] != 0].copy()
closed_m['win'] = (closed_m['Closed PnL'] > 0).astype(int)
closed_m['classification'] = pd.Categorical(closed_m['classification'], categories=ORDER, ordered=True)

daily = merged.groupby(['date','classification','sentiment_val']).agg(
    avg_pnl=('Closed PnL','mean'), total_vol=('Size USD','sum'),
    trade_count=('Closed PnL','count')
).reset_index()

print("Data loaded. Closed trades:", len(closed_m))

# ═══════════════════════════════════════════════════════
# 1. KRUSKAL-WALLIS + SPEARMAN TESTS
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("  1. ADVANCED STATISTICAL TESTS")
print("="*60)

# Kruskal-Wallis: non-parametric test across sentiment groups
groups = [closed_m[closed_m['classification']==s]['Closed PnL'].dropna().values for s in ORDER]
groups = [g for g in groups if len(g) > 0]
kw_stat, kw_p = stats.kruskal(*groups)
print(f"\n  Kruskal-Wallis H-test (PnL across sentiment groups):")
print(f"    H = {kw_stat:.4f}  |  p = {kw_p:.6f}")
print(f"    {'** Significant (p<0.05): groups differ!' if kw_p<0.05 else 'Not significant'}")

# Spearman rank correlation
sp_r, sp_p = stats.spearmanr(daily['sentiment_val'], daily['avg_pnl'])
print(f"\n  Spearman Rank Correlation (sentiment vs daily PnL):")
print(f"    rho = {sp_r:.4f}  |  p = {sp_p:.4f}")
print(f"    {'** Significant monotonic relationship' if sp_p<0.05 else 'No significant monotonic relationship'}")

# Pearson for comparison
pe_r, pe_p = stats.pearsonr(daily['sentiment_val'], daily['avg_pnl'])
print(f"\n  Pearson (for comparison):")
print(f"    r = {pe_r:.4f}  |  p = {pe_p:.4f}")

# ═══════════════════════════════════════════════════════
# 2. SENTIMENT TRANSITION ANALYSIS
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("  2. SENTIMENT TRANSITION ANALYSIS")
print("="*60)

sent_daily = sentiment.sort_values('date').copy()
sent_daily['prev_class'] = sent_daily['classification'].shift(1)
sent_daily = sent_daily.dropna(subset=['prev_class'])

# Build transition matrix
transitions = pd.crosstab(
    sent_daily['prev_class'], sent_daily['classification'],
    normalize='index'
) * 100

# Reindex to ORDER
transitions = transitions.reindex(index=ORDER, columns=ORDER, fill_value=0)
print("\n  Sentiment Transition Matrix (% probability):")
print(transitions.round(1).to_string())

# Merge transitions with trades
trades_with_trans = pd.merge(
    closed_m, sent_daily[['date','prev_class']].drop_duplicates('date'),
    on='date', how='inner'
)
trades_with_trans['transition'] = trades_with_trans['prev_class'] + ' → ' + trades_with_trans['classification'].astype(str)

# Top transitions by avg PnL
trans_pnl = trades_with_trans.groupby('transition').agg(
    avg_pnl=('Closed PnL','mean'), count=('Closed PnL','count'),
    win_rate=('win','mean')
).query('count >= 50').sort_values('avg_pnl', ascending=False)

print("\n  Top 5 Most Profitable Transitions (min 50 trades):")
for t, row in trans_pnl.head(5).iterrows():
    print(f"    {t:<35} avg=${row['avg_pnl']:>8.2f}  wins={row['win_rate']*100:.1f}%  n={int(row['count'])}")

print("\n  Bottom 5 Transitions:")
for t, row in trans_pnl.tail(5).iterrows():
    print(f"    {t:<35} avg=${row['avg_pnl']:>8.2f}  wins={row['win_rate']*100:.1f}%  n={int(row['count'])}")

# ═══════════════════════════════════════════════════════
# 3. RISK-ADJUSTED METRICS
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("  3. RISK-ADJUSTED METRICS (Reward/Risk Ratio)")
print("="*60)

risk_metrics = []
for s in ORDER:
    subset = closed_m[closed_m['classification'] == s]['Closed PnL']
    if len(subset) == 0:
        continue
    avg = subset.mean()
    std = subset.std()
    wins = subset[subset > 0]
    losses = subset[subset < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
    reward_risk = avg_win / avg_loss if avg_loss > 0 else np.inf
    sharpe_like = avg / std if std > 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.inf
    max_dd = (subset.cumsum() - subset.cumsum().cummax()).min()

    risk_metrics.append({
        'Sentiment': s, 'Avg PnL': avg, 'Std Dev': std,
        'Reward/Risk': reward_risk, 'Sharpe-like': sharpe_like,
        'Profit Factor': profit_factor, 'Max Drawdown': max_dd,
        'Avg Win': avg_win, 'Avg Loss': avg_loss
    })

risk_df = pd.DataFrame(risk_metrics).set_index('Sentiment')
print("\n  Risk-Adjusted Performance by Sentiment:")
print(f"  {'Sentiment':<16} {'Reward/Risk':>12} {'Sharpe-like':>12} {'Profit Factor':>14} {'Max DD':>12}")
print("  " + "-"*68)
for s in ORDER:
    if s in risk_df.index:
        r = risk_df.loc[s]
        pf_str = f"{r['Profit Factor']:.3f}" if r['Profit Factor'] != np.inf else "∞"
        print(f"  {s:<16} {r['Reward/Risk']:>12.3f} {r['Sharpe-like']:>12.4f} {pf_str:>14} ${r['Max Drawdown']:>10,.0f}")

# ═══════════════════════════════════════════════════════
# 4. LAG-1 PREDICTIVE ANALYSIS
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("  4. LAG-1 PREDICTIVE ANALYSIS")
print("="*60)

daily_agg = merged.groupby('date').agg(
    avg_pnl=('Closed PnL','mean'), total_pnl=('Closed PnL','sum'),
    trade_count=('Closed PnL','count')
).reset_index()
daily_agg = pd.merge(daily_agg, sentiment[['date','value','classification']].drop_duplicates('date'),
                     on='date', how='inner').sort_values('date')

# Yesterday's sentiment → today's PnL
daily_agg['prev_sentiment'] = daily_agg['value'].shift(1)
daily_agg['prev_class'] = daily_agg['classification'].shift(1)
daily_agg = daily_agg.dropna(subset=['prev_sentiment'])

lag_corr, lag_p = stats.spearmanr(daily_agg['prev_sentiment'], daily_agg['avg_pnl'])
print(f"\n  Spearman (yesterday's sentiment → today's PnL):")
print(f"    rho = {lag_corr:.4f}  |  p = {lag_p:.4f}")
print(f"    {'** Predictive signal exists!' if lag_p<0.05 else 'No predictive signal from lagged sentiment'}")

lag_by_prev = daily_agg.groupby('prev_class').agg(
    next_day_avg_pnl=('avg_pnl','mean'),
    next_day_median_pnl=('avg_pnl','median'),
    count=('avg_pnl','count')
)
lag_by_prev = lag_by_prev.reindex(ORDER)
print(f"\n  Next-Day Avg PnL by Previous Day's Sentiment:")
print(f"  {'Yesterday':<16} {'Next-Day Avg PnL':>18} {'Median':>10} {'Days':>6}")
print("  " + "-"*52)
for s in ORDER:
    if s in lag_by_prev.index:
        r = lag_by_prev.loc[s]
        print(f"  {s:<16} ${r['next_day_avg_pnl']:>16.2f} ${r['next_day_median_pnl']:>8.2f} {int(r['count']):>6}")

# ═══════════════════════════════════════════════════════
# VISUALIZATION — 2x2 Grid
# ═══════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 16), facecolor=BG)
fig.suptitle("Extended Analysis: Statistical Tests, Transitions, Risk & Lag-1",
             fontsize=18, fontweight='bold', color='#ffffff', y=0.98)
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3,
                       left=0.06, right=0.96, top=0.92, bottom=0.06)

# ── Panel 1: Statistical Test Comparison ──
ax1 = fig.add_subplot(gs[0, 0])
test_names = ['Pearson r', 'Spearman ρ', 'Kruskal-Wallis H']
test_stats = [abs(pe_r), abs(sp_r), min(kw_stat/100, 1.0)]
test_pvals = [pe_p, sp_p, kw_p]
test_colors = ['#e63946' if p > 0.05 else '#2dc653' for p in test_pvals]

bars = ax1.barh(test_names, test_stats, color=test_colors,
                edgecolor='#ffffff18', height=0.5, zorder=3)
for bar, stat, pval, name in zip(bars, [pe_r, sp_r, kw_stat], test_pvals, test_names):
    val_str = f"stat={stat:.4f}, p={pval:.4f}"
    sig = "✓ Sig" if pval < 0.05 else "✗ NS"
    ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f"{val_str}  [{sig}]", va='center', fontsize=9, color='#ffffff')
ax1.set_title("① Statistical Test Results", fontsize=13, fontweight='bold', color='#ffffff', pad=10)
ax1.set_xlabel("Effect Size (normalized)")
ax1.grid(axis='x', zorder=0)
legend_elems = [mpatches.Patch(color='#2dc653', label='Significant (p<0.05)'),
                mpatches.Patch(color='#e63946', label='Not Significant')]
ax1.legend(handles=legend_elems, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

# ── Panel 2: Transition Heatmap ──
ax2 = fig.add_subplot(gs[0, 1])
im = ax2.imshow(transitions.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=60)
ax2.set_xticks(range(len(ORDER))); ax2.set_xticklabels(SHORT, fontsize=8, rotation=30, ha='right')
ax2.set_yticks(range(len(ORDER))); ax2.set_yticklabels(SHORT, fontsize=8)
ax2.set_xlabel("To (next day)"); ax2.set_ylabel("From (current day)")
ax2.set_title("② Sentiment Transition Probabilities (%)", fontsize=13, fontweight='bold', color='#ffffff', pad=10)
for i in range(len(ORDER)):
    for j in range(len(ORDER)):
        val = transitions.values[i, j]
        ax2.text(j, i, f"{val:.0f}%", ha='center', va='center',
                 fontsize=9, fontweight='bold', color='black' if val > 30 else 'white')
cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color=TEXT); cbar.ax.set_ylabel('%', color=TEXT)

# ── Panel 3: Risk-Adjusted Metrics ──
ax3 = fig.add_subplot(gs[1, 0])
x = np.arange(len(ORDER))
w = 0.25
rr_vals = [risk_df.loc[s, 'Reward/Risk'] if s in risk_df.index else 0 for s in ORDER]
pf_vals = [min(risk_df.loc[s, 'Profit Factor'], 5) if s in risk_df.index else 0 for s in ORDER]
sh_vals = [risk_df.loc[s, 'Sharpe-like']*100 if s in risk_df.index else 0 for s in ORDER]

ax3.bar(x - w, rr_vals, w, color='#52b788', edgecolor='#ffffff18', label='Reward/Risk', zorder=3)
ax3.bar(x,     pf_vals, w, color='#4895ef', edgecolor='#ffffff18', label='Profit Factor', zorder=3)
ax3.bar(x + w, sh_vals, w, color='#f4a261', edgecolor='#ffffff18', label='Sharpe×100', zorder=3)
ax3.set_xticks(x); ax3.set_xticklabels(SHORT, fontsize=9, rotation=15, ha='right')
ax3.set_title("③ Risk-Adjusted Metrics by Sentiment", fontsize=13, fontweight='bold', color='#ffffff', pad=10)
ax3.set_ylabel("Value"); ax3.grid(axis='y', zorder=0)
ax3.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

# ── Panel 4: Lag-1 Predictive ──
ax4 = fig.add_subplot(gs[1, 1])
lag_vals = [lag_by_prev.loc[s, 'next_day_avg_pnl'] if s in lag_by_prev.index else 0 for s in ORDER]
lag_colors = ['#2dc653' if v >= 0 else '#e63946' for v in lag_vals]
bars4 = ax4.bar(x, lag_vals, color=lag_colors, edgecolor='#ffffff18', width=0.6, zorder=3)
for i, (bar, v) in enumerate(zip(bars4, lag_vals)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(lag_vals)*0.02 if v>=0 else min(lag_vals)*0.02),
             f"${v:.1f}", ha='center', va='bottom' if v>=0 else 'top',
             fontsize=10, fontweight='bold', color='#ffffff')
ax4.set_xticks(x); ax4.set_xticklabels([f"Yest:\n{s}" for s in SHORT], fontsize=8)
ax4.axhline(0, color='#ffffff55', linewidth=1)
ax4.set_title(f"④ Lag-1: Yesterday's Sentiment → Today's PnL\n(Spearman ρ={lag_corr:.3f}, p={lag_p:.3f})",
              fontsize=13, fontweight='bold', color='#ffffff', pad=10)
ax4.set_ylabel("Next-Day Avg PnL (USD)"); ax4.grid(axis='y', zorder=0)

plt.savefig("../charts/extended_analysis.png", dpi=160, bbox_inches='tight', facecolor=BG)
plt.show()
print("\n✅ Extended analysis chart saved as ../charts/extended_analysis.png")
