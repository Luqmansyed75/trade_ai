import json

with open('notebooks/bitcoin_sentiment_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = []
        for line in cell['source']:
            line = line.replace('pd.read_csv("historical_data.csv")', 'pd.read_csv("../data/historical_data.csv")')
            line = line.replace("pd.read_csv('historical_data.csv')", "pd.read_csv('../data/historical_data.csv')")
            line = line.replace('pd.read_csv("fear_greed_index.csv")', 'pd.read_csv("../data/fear_greed_index.csv")')
            line = line.replace("pd.read_csv('fear_greed_index.csv')", "pd.read_csv('../data/fear_greed_index.csv')")
            line = line.replace('plt.savefig("trading_sentiment_analysis.png"', 'plt.savefig("../charts/trading_sentiment_analysis.png"')
            line = line.replace("plt.savefig('trading_sentiment_analysis.png'", "plt.savefig('../charts/trading_sentiment_analysis.png'")
            source.append(line)
        cell['source'] = source

with open('notebooks/bitcoin_sentiment_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
