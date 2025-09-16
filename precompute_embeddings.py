# precompute_embeddings.py
# ------------------------------------------------------------
# Precompute embeddings for TMDB movies
# - Resume-safe using existing cache
# - Batched encoding
# - Full CPU utilization
# - Console progress + ETA + debug logs
# ------------------------------------------------------------

import os, json, time
from pathlib import Path
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# --- Force full CPU utilization ---
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
torch.set_num_threads(os.cpu_count())

# --- Config ---
CSV_PATH = r"C:\Users\santosh\Desktop\python-project\TMDB_movie_dataset_v11.csv"
CACHE_DIR = Path(r"C:\Users\santosh\Desktop\python-project\cache")
CACHE_DIR.mkdir(exist_ok=True)

EMB_FILE = CACHE_DIR / "embeddings_dd625368.pt"
IDS_FILE = CACHE_DIR / "ids_dd625368.json"

print(f"📂 Using cache folder: {CACHE_DIR}")
print(f"📄 Using embeddings file: {EMB_FILE}")
print(f"📄 Using ids file: {IDS_FILE}")

# --- Load CSV ---
print(f"📄 Loading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH, low_memory=False)
df["title"] = df["title"].fillna("").astype(str)
df["overview"] = df.get("overview", "").fillna("").astype(str)
df["id"] = df["id"].astype(str)

# --- Load checkpoint ---
embeddings = torch.empty((0, 384))  # MiniLM-L6-v2 embedding size
ids = []

if EMB_FILE.exists() and IDS_FILE.exists():
    try:
        embeddings = torch.load(EMB_FILE)
        with open(IDS_FILE, "r", encoding="utf-8") as f:
            ids = json.load(f)
        print(f"✅ Loaded checkpoint: {len(ids)} embeddings. Resuming...")
    except Exception:
        print("⚠️ Checkpoint corrupted, starting fresh...")
        embeddings = torch.empty((0, 384))
        ids = []

def save_checkpoint():
    torch.save(embeddings, EMB_FILE)
    with open(IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(ids, f)

# --- Load model ---
print("🔄 Loading SentenceTransformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded successfully. Starting precompute...")

# --- Debug info ---
print(f"🔎 Total rows in CSV: {len(df)}")
print(f"🔎 Already embedded: {len(ids)}")

# --- Precompute embeddings ---
BATCH_SIZE = 64
total = len(df)
already = len(ids)
start_time = time.time()

batch_texts, batch_ids = [], []
skipped = 0
processed_any = False
id_set = set(ids)  # for faster lookup

for row in df.itertuples(index=False):
    mid = str(row.id)
    if mid in id_set:
        skipped += 1
        continue

    batch_texts.append(f"{row.title} {row.overview}")
    batch_ids.append(mid)

    if len(batch_texts) >= BATCH_SIZE:
        processed_any = True
        with torch.inference_mode():
            batch_embs = embedder.encode(batch_texts, convert_to_tensor=True, normalize_embeddings=True)

        embeddings = torch.cat([embeddings, batch_embs], dim=0) if embeddings.numel() > 0 else batch_embs
        ids.extend(batch_ids)
        id_set.update(batch_ids)
        batch_texts, batch_ids = [], []

        save_checkpoint()

        # Progress
        done = len(ids)
        pct = done / total * 100
        elapsed = time.time() - start_time
        eta = (elapsed / (done - already)) * (total - done) if done > already else 0
        print(f"{pct:.2f}% ({done}/{total}) | ETA: {eta/60:.1f} min | Batch time: {elapsed:.2f}s")
        start_time = time.time()  # reset timer

print(f"🔎 Rows skipped (already processed): {skipped}")

# --- Process leftover ---
if batch_texts:
    processed_any = True
    with torch.inference_mode():
        batch_embs = embedder.encode(batch_texts, convert_to_tensor=True, normalize_embeddings=True)
    embeddings = torch.cat([embeddings, batch_embs], dim=0) if embeddings.numel() > 0 else batch_embs
    ids.extend(batch_ids)
    save_checkpoint()
    print(f"✅ Final batch saved. Total embeddings: {len(ids)}")

if not processed_any:
    print("✅ Nothing new to process — you are fully up to date!")

print("🎉 Precomputation complete!")
