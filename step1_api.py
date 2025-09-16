# genergenie_dynamic.py
import os, json, hashlib
from pathlib import Path

import torch
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from openai import OpenAI

# --- Load API keys ---
load_dotenv()
TMDB_KEY = os.getenv("TMDB_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)

# --- CPU optimization ---
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
torch.set_num_threads(os.cpu_count())

# --- Config ---
CSV_PATH = r"C:\Users\santosh\Desktop\python-project\TMDB_movie_dataset_v11.csv"
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

csv_sig = hashlib.md5(CSV_PATH.encode()).hexdigest()[:8]
EMB_FILE = CACHE_DIR / f"embeddings_{csv_sig}.pt"
IDS_FILE = CACHE_DIR / f"ids_{csv_sig}.json"
TMDB_IMAGE = "https://image.tmdb.org/t/p/w500"

# --- Load model ---
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
embedder = load_model()

# --- Load embeddings and dataset ---
@st.cache_resource(show_spinner=True)
def load_embeddings_and_ids():
    if not EMB_FILE.exists() or not IDS_FILE.exists():
        st.error("❌ No embeddings found. Run precompute_embeddings.py first.")
        return torch.empty((0,384)), [], pd.DataFrame()
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["title"] = df["title"].fillna("").astype(str)
    df["overview"] = df.get("overview", "").fillna("").astype(str)
    df["id"] = df["id"].astype(int)
    df["adult"] = df.get("adult", False).fillna(False).astype(bool)
    df["popularity"] = df.get("popularity", 0).fillna(0)
    df["genres"] = df.get("genres", "").fillna("").astype(str)
    embeddings = torch.load(EMB_FILE, map_location="cpu")
    with open(IDS_FILE, "r", encoding="utf-8") as f:
        ids = json.load(f)
    return embeddings, ids, df

embeddings, ids, df = load_embeddings_and_ids()
id_to_idx = {ids[i]: i for i in range(len(ids))}

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

def gpt_rerank(results, query, top_n=10):
    try:
        titles = [r.title for r, s in results[:top_n*2]]
        prompt = f"From these titles: {titles}, rank the top {top_n} that best match the user query '{query}'. Only return titles."
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        ranked_titles = [line.strip(" -•") for line in resp.choices[0].message.content.strip().splitlines() if line.strip()]
        ranked_rows = []
        for t in ranked_titles:
            for r, s in results:
                if r.title == t and (r, s) not in ranked_rows:
                    ranked_rows.append((r, s))
        return ranked_rows
    except:
        return results

def gpt_suggest_movies(query):
    try:
        prompt = f"Suggest 5 movies similar to: '{query}'. Only list titles."
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.choices[0].message.content.strip()
        return [line.strip(" -•") for line in text.splitlines() if line.strip()]
    except:
        return []

# --- Search function with dynamic pool & adaptive threshold ---
def search_movies(query, top_k=50, hide_adult=True, genre_filter=None, base_candidate_pool=500, max_candidate_pool=5000, threshold=0.37):
    if embeddings.numel() == 0 or len(ids) == 0:
        return []

    expanded_query = expand_query_with_gpt(query)
    q_emb = embedder.encode(expanded_query, convert_to_tensor=True, normalize_embeddings=True)

    candidate_pool = base_candidate_pool
    final_results = []

    while candidate_pool <= max_candidate_pool:
        candidates = df.sort_values("popularity", ascending=False).head(candidate_pool)
        if hide_adult:
            candidates = candidates[candidates["adult"] == False]
        if genre_filter:
            candidates = candidates[candidates['genres'].str.contains('|'.join(genre_filter), case=False, na=False)]

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

        pop_values = [r.get("popularity", 0) for r in rows]
        max_pop = max(pop_values) if pop_values else 1.0
        norm_pop = torch.tensor([p/max_pop for p in pop_values])

        final_scores = 0.9 * sims + 0.1 * norm_pop

        query_lower = query.lower()
        keyword_bonus = torch.tensor([0.05 if query_lower in r.title.lower() or query_lower in r.overview.lower() else 0.0 for r in rows])
        final_scores = final_scores + keyword_bonus

        topk_vals, topk_idx = torch.topk(final_scores, k=min(top_k, len(rows)))
        results = [(rows[i], float(topk_vals[j])) for j, i in enumerate(topk_idx)]
        results = [(r, s) for r, s in results if s >= threshold]

        final_results = gpt_rerank(results, query, top_n=top_k)
        if len(final_results) >= 10:  # enough results found
            break
        candidate_pool += 500   # expand candidate pool
        threshold *= 0.95      # slightly lower threshold if too few matches

    return final_results

# --- Streamlit UI ---
st.set_page_config(page_title="🌟 GenerGenie 🌟", layout="wide")

# Dark Netflix theme
st.markdown("""
<style>
body, .stApp { background-color: #000; color: #f5f5f5; }
h1,h2,h3,h4,h5 { color: #e50914; font-family: Arial Black, sans-serif; }
.stTextInput input {background-color: #141414; color:#fff; border:1px solid #333; padding:0.6rem 1rem; border-radius:8px;}
.stButton>button {background-color:#e50914;color:#fff;font-weight:bold;padding:0.6rem 1rem;border-radius:8px;border:none;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center;font-size:3rem;'>🔮 GenerGenie 🔮</h1>
<h3 style='text-align:center;color:#aaa;'>Ask your movie wish and GenerGenie will grant it...</h3>
""", unsafe_allow_html=True)

hide_adult = st.checkbox("🚫 Hide Adult Movies", value=True)
query = st.text_input("🧞‍♂️ Your movie wish here...", "")

# Session state
if "results" not in st.session_state: st.session_state.results=[]
if "visible_count" not in st.session_state: st.session_state.visible_count=10
if "last_query" not in st.session_state: st.session_state.last_query=""

# Homepage trending
if not query:
    st.markdown("<h2>🔥 Trending Movies</h2>", unsafe_allow_html=True)
    trending = df[df["adult"]==False].sort_values("popularity", ascending=False).head(6)
    cols = st.columns(6)
    for i, col in enumerate(cols):
        poster = trending.iloc[i].get("poster_path")
        col.image(TMDB_IMAGE+poster if poster else "https://via.placeholder.com/200x300/000000/FFFFFF?text=No+Image",
                  use_container_width=True)
        col.caption(trending.iloc[i].title)

# Search results
elif query:
    if query != st.session_state.last_query:
        st.session_state.last_query=query
        st.session_state.results=[]
        st.session_state.visible_count=10

        with st.spinner("🧞 As you wish, my master... GenerGenie is searching ✨"):
            genre_keywords={"romantic":["Romance"],"comedy":["Comedy"],"action":["Action"],
                            "sci-fi":["Science Fiction"],"adventure":["Adventure"],"horror":["Horror"]}
            genre_filter=[]
            for k,g in genre_keywords.items():
                if k in query.lower(): genre_filter=g; break

            results = search_movies(query, hide_adult=hide_adult, genre_filter=genre_filter)
            if not results:
                results = search_movies(query, hide_adult=hide_adult, genre_filter=None)
            st.session_state.results = results

    results = st.session_state.results
    visible_count = st.session_state.visible_count

    if not results:
        st.warning("No matches in cache. Asking the Genie (GPT)...")
        suggestions = gpt_suggest_movies(query)
        if suggestions:
            st.markdown("<h2>🪄 Genie Suggestions</h2>", unsafe_allow_html=True)
            for title in suggestions: st.write(f"🎬 {title}")
        else:
            st.error("Nothing found, even the Genie is puzzled!")
    else:
        st.markdown(f"<h2>Results for: {query}</h2>", unsafe_allow_html=True)
        for row, score in results[:visible_count]:
            col1, col2 = st.columns([1,3])
            poster = row.get("poster_path")
            overview = row.get("overview","")
            with col1:
                st.image(TMDB_IMAGE+poster if poster else "https://via.placeholder.com/300x450/000000/FFFFFF?text=No+Image",
                         use_container_width=True)
            with col2:
                st.subheader(row.title)
                st.caption(f"⭐ Popularity: {row.get('popularity',0):.1f} | Score: {score:.3f}")
                st.write(overview if overview else "_No description available._")

        if visible_count < len(results):
            if st.button("Load More"):
                st.session_state.visible_count += 10
                st.rerun()
