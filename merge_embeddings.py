# merge_embeddings.py
# ------------------------------------------------------------
# Merge all precomputed batch_*.pt files into one large tensor
# + single ids.json for faster loading in the main app
# ------------------------------------------------------------

import json
import torch
from pathlib import Path
import re

CACHE_DIR = Path("cache_batches")
OUTPUT_DIR = Path("cache")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Collect all batch files ---
batch_files = sorted(CACHE_DIR.glob("batch_*.pt"), key=lambda x: int(re.search(r"(\d+)", x.stem).group()))
if not batch_files:
    print("❌ No batch files found in cache_batches/. Run precompute_embeddings.py first.")
    exit()

print(f"🔄 Found {len(batch_files)} batch files, merging...")

all_embeddings = []
all_ids = []

for bf in batch_files:
    data = torch.load(bf)
    all_embeddings.append(data["embeddings"])
    all_ids.extend(data["ids"])

# --- Concatenate all batches into one tensor ---
final_embeddings = torch.cat(all_embeddings, dim=0)

# --- Save outputs ---
csv_sig = re.search(r"index_(\w+).json", str((CACHE_DIR / "index_*.json"))).group(1) if (CACHE_DIR / f"index_*.json").exists() else "merged"
EMB_FILE = OUTPUT_DIR / f"embeddings_{csv_sig}.pt"
IDS_FILE = OUTPUT_DIR / f"ids_{csv_sig}.json"

torch.save(final_embeddings, EMB_FILE)
with open(IDS_FILE, "w", encoding="utf-8") as f:
    json.dump(all_ids, f)

print(f"✅ Merge complete! Saved:")
print(f"   • {EMB_FILE}  (Tensor size: {final_embeddings.shape})")
print(f"   • {IDS_FILE}  (Total IDs: {len(all_ids)})")
