
# parfum_model_utils.py : memuat model hasil notebook di backend (tanpa notebook).
# Kebutuhan: pandas, numpy, scikit-learn, joblib.   Contoh:
#   from parfum_model_utils import SimilarityModel, ConceptScorer
#   sim = SimilarityModel("model_similarity.joblib");  sim.search("vanilla, amber, musk", top_n=5)
#   sc = ConceptScorer(".", "../ranking_original_ALL.csv");  sc.score("Extrait", "Unisex", 299000, 100, 2, "bergamot, musk", "Aventus")
import re
from collections import Counter
import numpy as np, pandas as pd, joblib

def note_tokens(s):
    s = re.sub(r"\s+", " ", str(s).lower())
    ph = [p.strip().replace(" ", "_") for p in s.split(",") if p.strip()]
    words = [w for w in re.findall(r"[a-z]+", s) if len(w) > 2 and w not in ("notes", "note", "accord")]
    return ph + words

class SimilarityModel:
    def __init__(self, path):
        d = joblib.load(path); self.vocab, self.idf, self.X, self.variants = d["vocab"], d["idf"], d["X_notes"], d["variants"]
    def vectorize(self, text):
        row = np.zeros(len(self.vocab))
        for t, c in Counter(t for t in note_tokens(text) if t in self.vocab).items(): row[self.vocab[t]] = (1 + np.log(c)) * self.idf[self.vocab[t]]
        n = np.linalg.norm(row); return row / n if n else row
    def search(self, text, top_n=10):
        sims = np.asarray(self.X @ self.vectorize(text)).ravel(); idx = np.argsort(-sims)[:top_n]
        out = self.variants.iloc[idx].copy(); out.insert(0, "skor_mirip", sims[idx].round(3)); return out.reset_index(drop=True)

class ConceptScorer:
    def __init__(self, models_dir=".", ranking_csv=None):
        self.bagus = joblib.load(f"{models_dir}/model_bagus.joblib"); self.top = joblib.load(f"{models_dir}/model_top10000.joblib")
        self.rank = pd.read_csv(ranking_csv, index_col=0) if ranking_csv else None
    def n_pesaing(self, name):
        if self.rank is None or not name: return 0
        c = [n for n in self.rank.index if str(name).lower() in str(n).lower()]
        return int(self.rank.loc[c[0], "n_varian"]) if c else 0
    def score(self, conc="Extrait", gender="Unisex", price_mid=299000, size_max=100, n_sizes=2, notes="", dupe_dari=None):
        names = [dupe_dari] if isinstance(dupe_dari, str) else (dupe_dari or [])
        row = {"conc_short": conc, "gender": gender, "price_mid": price_mid, "size_max": size_max, "n_sizes": n_sizes,
               "n_pesaing": max([self.n_pesaing(n) for n in names] or [0]), "n_perfume": len(names), "is_dupe_i": int(len(names) > 0)}
        for k, rx in self.bagus["fam_rules"].items(): row[f"fam_{k}"] = int(bool(re.search(rx, str(notes).lower())))
        res = {}
        for nm, m in [("peluang_BAGUS", self.bagus), ("peluang_bucket_10000+", self.top)]:
            X = pd.DataFrame([row])[m["features_cat"] + m["features_num"]]; res[nm] = round(float(m["model"].predict_proba(X)[0, 1]), 3)
        res["n_pesaing_di_original"] = row["n_pesaing"]; res["catatan"] = "indikasi kasar; lihat model_card.csv (akurasi brand baru rendah)"
        return res
