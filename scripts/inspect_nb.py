import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('data/bitcoin_sentiment_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Print full code cells
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        print(f"\n{'='*80}")
        print(f"CELL {i} [code]:")
        print('='*80)
        print(source)
