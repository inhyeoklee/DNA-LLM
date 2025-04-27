#!/usr/bin/env python3
"""
Usage:
  python data_prep.py             # generate full-vocab data under data/full
  python data_prep.py --filter    # generate 4-base data under data/vocab4
Ensure reference genome is at data/genome.fa (downloaded automatically if missing).
"""
import os, gzip, pickle, requests, numpy as np, argparse

# Ensure root data directory and reference genome
os.makedirs('data', exist_ok=True)
root_gz = 'data/genome.fa.gz'
root_fa = 'data/genome.fa'
if not os.path.isfile(root_fa):
    if not os.path.isfile(root_gz):
        print(f"Downloading reference to {root_gz}...")
        with requests.get(
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
            stream=True
        ) as r:
            r.raise_for_status()
            with open(root_gz, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1_048_576):
                    f.write(chunk)
    with gzip.open(root_gz, 'rb') as f_in, open(root_fa, 'wb') as f_out:
        for chunk in f_in:
            f_out.write(chunk)

# Parse arguments
parser = argparse.ArgumentParser(description="Prepare genome tokenization datasets")
parser.add_argument(
    '--filter', action='store_true',
    help='retain only uppercase A,C,G,T'
)
args = parser.parse_args()

# Choose output subdirectory
mode = 'vocab4' if args.filter else 'full'
base = os.path.join('data', mode)
os.makedirs(base, exist_ok=True)

# Read genome and optionally filter
with open(root_fa, 'r') as f:
    text_all = f.read()
text = text_all if not args.filter else ''.join(c for c in text_all if c in 'ACGT')

# Build vocabulary
chars = sorted(set(text))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for i,c in enumerate(chars)}

# Save metadata
meta = {'vocab_size': len(chars), 'stoi': stoi, 'itos': itos}
with open(os.path.join(base, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# Tokenization: split and save
n = len(text)
train = text[:int(0.9*n)]
val   = text[int(0.9*n):]
del text

def append_bin(path, arr):
    with open(path, 'ab') as f:
        arr.tofile(f)

# Clear old binaries
for split in ('train','val'):
    open(os.path.join(base, f"{split}.bin"), 'wb').close()

# Encode in chunks
for frac in np.arange(0, 1, 0.05):
    t0, t1 = int(len(train)*frac), int(len(train)*(frac+0.05))
    v0, v1 = int(len(val)*frac),   int(len(val)*(frac+0.05))
    arr_t = np.array([stoi[c] for c in train[t0:t1]], dtype=np.uint16)
    arr_v = np.array([stoi[c] for c in val[v0:v1]],   dtype=np.uint16)
    append_bin(os.path.join(base, 'train.bin'), arr_t)
    append_bin(os.path.join(base, 'val.bin'),   arr_v)

print(f"Data ready in {base}: train.bin, val.bin, meta.pkl")
