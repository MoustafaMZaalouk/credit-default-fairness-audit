"""Download, verify and load the UCI 'Default of Credit Card Clients' data (Taiwan, 2005)."""
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
XLS = "default of credit card clients.xls"
TARGET = "default payment next month"
DATA_DIR = Path("data")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_data(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    if (data_dir / XLS).exists():
        return data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    zpath = data_dir / "credit.zip"
    urllib.request.urlretrieve(URL, zpath)
    got = sha256_of(zpath)
    if got != SHA256:
        zpath.unlink()
        raise RuntimeError(f"checksum mismatch: expected {SHA256}, got {got}")
    with zipfile.ZipFile(zpath) as zf:
        if zf.namelist() != [XLS]:
            raise RuntimeError(f"unexpected archive members: {zf.namelist()}")
        zf.extract(XLS, data_dir)
    return data_dir


def load(data_dir=DATA_DIR):
    return pd.read_excel(Path(data_dir) / XLS, header=1)


def age_group(age):
    return pd.cut(age, [0, 29, 39, 200], labels=["<=29", "30-39", ">=40"])
