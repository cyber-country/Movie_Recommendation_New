# genergenie_flask.py
# ------------------------------------------------------------
# GenerGenie — Flask Version with Embedded Frontend
# ------------------------------------------------------------

import os, json
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import gdown

# --- API Keys ---
TMDB_KEY = os.getenv("TMDB_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

# --- CPU optimization ---
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
torch.set_num_threads(os.cpu_count())

# --- Cache setup ---
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

CSV_PATH = CACHE_DIR / "TMDB_movie_dataset_v11.csv"
EMB_FILE = CACHE_DIR / "embeddings_dd625368.pt"
IDS_FILE = CACHE_DIR / "ids_dd625368.json"
TMDB_IMAGE = "https://image.tmdb.org/t/p/w500"

# --- Google Drive file IDs ---
GDRIVE_FILES = {
    "csv": "1taVn3jeu9R5g4RkYJuuthdjANMZ_29dh",
    "embeddings_pt": "1kJ1ThTtPqkVkBPVTgtm7D7jMvAtPGKDD",
    "ids_json": "1rc0niTdB3RT0YlTRdQehE32U23NV47PW"
}

# --- Download helper ---
def download_from_gdrive(file_id, out_path: Path):
    if not out_path.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {out_path} ...")
        gdown.download(url, str(out_path), quiet=False)
    else:
        print(f"{out_path} already exists, skipping download.")

for key, fid in GDRIVE_FILES.items():
    download_from_gdrive(fid, {"csv": CSV_PATH, "embeddings_pt": EMB_FILE, "ids_json": IDS_FILE}[key])

# --- Load model and embeddings ---
print("Loading model and data...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv(CSV_PATH)
df["title"] = df["title"].fillna("").astype(str)
df["overview"] = df.get("overview","").fillna("").astype(str)
df["id"] = df["id"].astype(int)
df["adult"] = df.get("adult", False).fillna(False).astype(bool)
df["popularity"] = df.get("popularity",0).fillna(0)
df["genres"] = df.get("genres","").fillna("").astype(str)

embeddings = torch.load(EMB_FILE, map_location="cpu")
with open(IDS_FILE,"r",encoding="utf-8") as f:
    ids = json.load(f)
id_to_idx = {str(ids[i]): i for i in range(len(ids))}
print("Model and data loaded!")

# --- GPT helpers ---
def expand_query_with_gpt(query):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Expand this search query into related keywords: {query}"}]
        )
        return resp.choices[0].message.content.strip()
    except:
        return query

def gpt_suggest_movies(query):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Suggest 5 movies similar to: '{query}'. Only list titles."}]
        )
        text = resp.choices[0].message.content.strip()
        return [line.strip(" -•") for line in text.splitlines() if line.strip()]
    except:
        return []

# --- Search function ---
def search_movies(query, top_k=50, hide_adult=True, genre_filter=None, base_candidate_pool=500, max_candidate_pool=5000, threshold=0.37):
    if embeddings.numel() == 0 or len(ids)==0:
        return []

    expanded_query = expand_query_with_gpt(query)
    q_emb = embedder.encode(expanded_query, convert_to_tensor=True, normalize_embeddings=True)

    candidate_pool = base_candidate_pool
    final_results = []

    while candidate_pool <= max_candidate_pool:
        candidates = df.sort_values("popularity",ascending=False).head(candidate_pool)
        if hide_adult: candidates = candidates[candidates["adult"]==False]
        if genre_filter: candidates = candidates[candidates['genres'].str.contains('|'.join(genre_filter),case=False,na=False)]

        rows, emb_rows = [], []
        for _, row in candidates.iterrows():
            mid = str(int(row["id"]))
            idx = id_to_idx.get(mid)
            if idx is None: continue
            rows.append(row)
            emb_rows.append(embeddings[idx])

        if not emb_rows: break

        embs_tensor = torch.stack(emb_rows)
        sims = util.cos_sim(q_emb, embs_tensor)[0]

        pop_values = [r.get("popularity",0) for r in rows]
        max_pop = max(pop_values) if pop_values else 1.0
        norm_pop = torch.tensor([p/max_pop for p in pop_values])

        final_scores = 0.9 * sims + 0.1 * norm_pop
        query_lower = query.lower()
        keyword_bonus = torch.tensor([0.05 if query_lower in r.title.lower() or query_lower in r.overview.lower() else 0.0 for r in rows])
        final_scores += keyword_bonus

        topk_vals, topk_idx = torch.topk(final_scores, k=min(top_k,len(rows)))
        results = [(rows[i], float(topk_vals[j])) for j,i in enumerate(topk_idx)]
        results = [(r,s) for r,s in results if s>=threshold]

        final_results = results
        if len(final_results)>=10: break
        candidate_pool += 500
        threshold *= 0.95

    return final_results

# --- Flask app ---
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🔮 GenerGenie 🔮</title>
<style>
body { background-color:#000; color:#f5f5f5; font-family: Arial, sans-serif; text-align:center; }
.container { max-width:900px; margin:auto; padding:2rem; }
input { padding:0.5rem 1rem; border-radius:8px; border:1px solid #333; background:#141414; color:#fff; width:60%; }
button { padding:0.5rem 1rem; border-radius:8px; border:none; background:#e50914; color:#fff; font-weight:bold; }
.movie { border-bottom:1px solid #333; margin:1rem 0; padding-bottom:1rem; text-align:left; overflow:auto; }
.movie img { width:150px; float:left; margin-right:1rem; }
h1 { color:#e50914; }
h3 { color:#aaa; }
</style>
</head>
<body>
<div class="container">
<h1>🔮 GenerGenie 🔮</h1>
<h3>Ask your movie wish and GenerGenie will grant it...</h3>
<input type="text" id="query" placeholder="🧞‍♂️ Your movie wish here..." />
<button onclick="searchMovies()">Search</button>
<div id="results"></div>
</div>

<script>
async function searchMovies() {
    const query = document.getElementById("query").value;
    const resDiv = document.getElementById("results");
    resDiv.innerHTML = "<p>Searching...</p>";
    const response = await fetch("/search", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({query})
    });
    const results = await response.json();
    resDiv.innerHTML = "";
    if(results.length===0){ resDiv.innerHTML="<p>No results found.</p>"; return; }
    results.forEach(movie => {
        const div = document.createElement("div");
        div.className = "movie";
        div.innerHTML = `<img src="${movie.poster_path}" alt="${movie.title}" />
                         <h3>${movie.title}</h3>
                         <p>${movie.overview}</p>
                         <p>⭐ Score: ${movie.score.toFixed(3)}</p>`;
        resDiv.appendChild(div);
    });
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query","")
    results = search_movies(query)
    if not results:
        suggestions = gpt_suggest_movies(query)
        results = [{"title":t,"overview":"","poster_path":"","score":0} for t in suggestions]
    else:
        results = [{"title":r.title,"overview":r.overview,"poster_path":TMDB_IMAGE+r.poster_path if r.poster_path else "", "score":s} for r,s in results]
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
