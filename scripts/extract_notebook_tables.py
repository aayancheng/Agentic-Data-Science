import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NB_PATH = Path('notebooks/fraud_eda.ipynb')
OUT_DIR = Path('docs/images')


def save_text_image(text: str, outfile: Path, title: str = None, fontsize: int = 10):
    lines = text.strip('\n').splitlines()
    # Estimate figure size based on text length
    width_chars = max(len(line) for line in lines) if lines else 40
    height_lines = len(lines) + (1 if title else 0)
    # Rough scaling: 8 px per char, 18 px per line at fontsize 10
    fig_w = min(max(width_chars * 0.09, 6), 18)
    fig_h = min(max(height_lines * 0.35, 2), 12)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    y = 1.0
    if title:
        ax.text(0.01, y, title, fontsize=fontsize + 2, fontweight='bold', family='monospace', va='top')
        y -= 0.08
    ax.text(0.01, y, text, fontsize=fontsize, family='monospace', va='top')
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb = json.loads(NB_PATH.read_text())
    all_text = []
    for cell in nb.get('cells', []):
        for out in cell.get('outputs', []):
            text = ''.join(out.get('text', [])) if isinstance(out.get('text'), list) else out.get('text')
            if text:
                all_text.append(text)
    blob = '\n'.join(all_text)

    # 1) Class balance table
    m = re.search(r"Class Label\s+Count\s+Percent\n(?:.*\n){1,4}", blob)
    if m:
        save_text_image(m.group().strip(), OUT_DIR / 'fraud_eda_table_class_balance.png', title='Class Balance')
        print('Wrote class balance table image')

    # 2) Summary stats for Time/Amount
    m = re.search(r"\s*Time\s+Amount\n(?:.*\n){6,9}", blob)
    if m:
        save_text_image(m.group().strip(), OUT_DIR / 'fraud_eda_table_summary_stats.png', title='Summary: Time & Amount')
        print('Wrote summary stats table image')

    # 3) Top hours by fraud rate
    m = re.search(r"Top hours ranked by fraud rate \(%\):\n(?:.*\n){3,12}", blob)
    if m:
        save_text_image(m.group().strip(), OUT_DIR / 'fraud_eda_table_top_hours.png', title='Top Hours by Fraud Rate')
        print('Wrote top hours table image')

    # 4) Best threshold section
    m = re.search(r"Best threshold achieving.*\n(?:.*\n){1,4}", blob)
    if m:
        save_text_image(m.group().strip(), OUT_DIR / 'fraud_eda_table_best_threshold.png', title='Threshold Tuning Snapshot')
        print('Wrote best threshold table image')

    # 5) Confusion matrix
    m = re.search(r"Confusion matrix:\n\[\[.*\n.*\]\]", blob)
    if m:
        save_text_image(m.group().strip(), OUT_DIR / 'fraud_eda_table_confusion_matrix.png', title='Confusion Matrix (Validation)')
        print('Wrote confusion matrix image')

    # 6) Feature importance (LR and RF)
    m = re.search(r"Logistic regression top \|coef\| features:\n(?:.*\n){5,15}\n\nRandom forest feature importances:\n(?:.*\n){5,15}", blob)
    if m:
        save_text_image(m.group().strip(), OUT_DIR / 'fraud_eda_table_feature_importance.png', title='Feature Importance Snapshots')
        print('Wrote feature importance image')

    print('Done.')


if __name__ == '__main__':
    main()

