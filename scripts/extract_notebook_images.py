import json
import base64
from pathlib import Path

NB_PATH = Path('notebooks/fraud_eda.ipynb')
OUT_DIR = Path('docs/images')

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb = json.loads(NB_PATH.read_text())
    count = 0
    for ci, cell in enumerate(nb.get('cells', [])):
        for oi, out in enumerate(cell.get('outputs', [])):
            data = out.get('data') or {}
            img_b64 = data.get('image/png')
            if not img_b64:
                continue
            # Some notebooks store list of strings; join if needed
            if isinstance(img_b64, list):
                img_b64 = ''.join(img_b64)
            img_bytes = base64.b64decode(img_b64)
            count += 1
            path = OUT_DIR / f'fraud_eda_chart_{count:02d}.png'
            path.write_bytes(img_bytes)
            print(f'Wrote {path}')
    if count == 0:
        print('No images found in notebook outputs.')

if __name__ == '__main__':
    main()

