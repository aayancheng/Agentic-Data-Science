"""
Download the 3 datasets into data/raw/. 
- Kaggle CLI is required for the fraud dataset: https://www.kaggle.com/docs/api
- NYC TLC parquet files are public; fetch a recent month by default.
- IMDb Large Movie Review comes from Stanford; download and extract.
"""
import os, subprocess, sys, shutil, tarfile, urllib.request, pathlib

BASE = pathlib.Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

def have(cmd: str) -> bool:
    from shutil import which
    return which(cmd) is not None

def run(cmd, **kw):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)

def kaggle_download():
    tgt = RAW / "credit_card_fraud"
    tgt.mkdir(exist_ok=True, parents=True)
    if not have("kaggle"):
        print("! Kaggle CLI not found. Install: pip install kaggle ; then set KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json")
        return
    run(["kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud", "-p", str(tgt), "--unzip"])

def nyc_taxi_download(year="2024", month="01"):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet"
    tgt_dir = RAW / "nyc_taxi"; tgt_dir.mkdir(exist_ok=True, parents=True)
    tgt = tgt_dir / f"yellow_tripdata_{year}-{month}.parquet"
    print(f"Downloading {url} -> {tgt}")
    urllib.request.urlretrieve(url, tgt)

def imdb_download():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tgt_dir = RAW / "imdb_reviews"; tgt_dir.mkdir(exist_ok=True, parents=True)
    tar_path = tgt_dir / "aclImdb_v1.tar.gz"
    if not tar_path.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, tar_path)
    print("Extracting IMDB tarball...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=tgt_dir)

if __name__ == "__main__":
    kaggle_download()
    nyc_taxi_download()
    imdb_download()
    print("All downloads attempted. Check messages above for any steps you need to complete (e.g., Kaggle auth).")
