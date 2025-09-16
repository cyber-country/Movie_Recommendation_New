import pandas as pd

# Path to your downloaded CSV
CSV_PATH = "TMDB_movie_dataset_v11.csv"
FIXED_CSV_PATH = "TMDB_movie_dataset_v11_fixed.csv"

# Load CSV
df = pd.read_csv(CSV_PATH)

# Clean column names
df.rename(columns=lambda x: x.strip(), inplace=True)

# Add default columns if missing
if "popularity" not in df.columns:
    df["popularity"] = 0.0
if "adult" not in df.columns:
    df["adult"] = False

# Save fixed CSV
df.to_csv(FIXED_CSV_PATH, index=False)
print(f"✅ Fixed CSV saved to: {FIXED_CSV_PATH}")
