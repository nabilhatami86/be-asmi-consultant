"""Memuat data riset terbaru dan mengintegrasikan model machine learning joblib (SimilarityModel & ConceptScorer).
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"

import sys
if str(MODELS_DIR) not in sys.path:
    sys.path.append(str(MODELS_DIR))

try:
    from parfum_model_utils import SimilarityModel, ConceptScorer
except ImportError:
    SimilarityModel = None
    ConceptScorer = None

ALIASES = {"one million": "1 million", "lv": "louis vuitton", "a fternoon": "afternoon", "mfk": "maison francis kurkdjian"}


def norm_name(x: str) -> str:
    x = str(x).lower().replace("&", " and ").replace("-", " ").replace("é", "e").replace("è", "e")
    x = re.sub(r"[^a-z0-9 ]", " ", x)
    for k, v in ALIASES.items():
        x = re.sub(rf"\b{re.escape(k)}\b", v, x)
    return re.sub(r"\s+", " ", x).strip()


def token_key(x: str) -> str:
    return " ".join(sorted(norm_name(x).split()))


def to_records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", force_ascii=False))


PRODUCT_COLS = [
    "id_parfume", "brand", "kelompok", "variant", "sizes", "price_mid", "price_min", "price_max",
    "sold_est", "sold_raw", "gender", "concentrate", "top_notes", "mid_notes", "base_notes",
    "dupe", "dupe_type", "status", "is_best_seller",
]

ORIGINAL_COLS = [
    "line_full", "original_brand", "n_varian_dupe", "n_brand_dupe", "sold_w",
    "opportunity_score", "label", "notes_umum", "kuadran"
]


class Store:
    def __init__(self) -> None:
        xl = pd.ExcelFile(DATA_DIR / "riset_dupe_asmi.xlsx")
        
        var_path = DATA_DIR / "parfume_clean_varian.csv"
        if var_path.exists():
            self.var = pd.read_csv(var_path)
        else:
            self.var = xl.parse("data_bersih_varian")

        # Map correct column names for price and sold if needed
        if "price_median" not in self.var.columns and "price_mid" in self.var.columns:
            self.var["price_median"] = self.var["price_mid"]
        if "sold_total" not in self.var.columns and "sold_est" in self.var.columns:
            self.var["sold_total"] = self.var["sold_est"]
        if "is_best_seller" not in self.var.columns and "is_fav" in self.var.columns:
            self.var["is_best_seller"] = self.var["is_fav"]

        pairs_path = DATA_DIR / "dupe_pairs_clean.csv"
        if pairs_path.exists():
            self.pairs = pd.read_csv(pairs_path)
        else:
            self.pairs = pd.DataFrame()

        master_path = DATA_DIR / "data-master-parfume.csv"
        if master_path.exists():
            self.master = pd.read_csv(master_path, sep=";", encoding="latin-1")
            self.master["price"] = pd.to_numeric(self.master["price"], errors="coerce")
            self.master["sold_shoppe"] = pd.to_numeric(self.master["sold_shoppe"], errors="coerce")
            self.master["all_notes"] = (
                self.master.top_notes.fillna("") + ", " +
                self.master.mid_notes.fillna("") + ", " +
                self.master.base_notes.fillna("")
            ).str.lower()
        else:
            self.master = pd.DataFrame()

        self.rank = xl.parse("ranking_ALL")
        if "ringkasan_keputusan" in xl.sheet_names:
            dec = xl.parse("ringkasan_keputusan")
            if "line_full" in dec.columns and "opportunity_score" in dec.columns:
                self.rank = self.rank.merge(dec[["line_full", "opportunity_score", "label"]], on="line_full", how="left")

        self.sheets = {
            s: xl.parse(s)
            for s in [
                "notes_per_kelompok", "notes_lokal_vs_timteng", "skenario_revenue",
                "ringkasan_keputusan", "model_card",
            ] if s in xl.sheet_names
        }

        self.var["all_notes"] = (
            self.var.top_notes.fillna("") + ", " + self.var.mid_notes.fillna("") + ", " + self.var.base_notes.fillna("")
        ).str.lower()
        self.var["gender"] = self.var["gender"].fillna("Unisex")
        
        self._idx = {str(vid): i for i, vid in enumerate(self.var.id_parfume)}

        try:
            if SimilarityModel and (MODELS_DIR / "model_similarity.joblib").exists():
                self.sim_model = SimilarityModel(str(MODELS_DIR / "model_similarity.joblib"))
            else:
                self.sim_model = None
        except Exception:
            self.sim_model = None

        try:
            rank_csv = DATA_DIR / "ranking_original_ALL.csv"
            if not rank_csv.exists():
                rank_csv = None
            if ConceptScorer and (MODELS_DIR / "model_bagus.joblib").exists():
                self.concept_scorer = ConceptScorer(models_dir=str(MODELS_DIR), ranking_csv=str(rank_csv) if rank_csv and rank_csv.exists() else None)
            else:
                self.concept_scorer = None
        except Exception:
            self.concept_scorer = None

        orig_col = "line_full" if "line_full" in self.rank.columns else "original"
        self._orig = [(str(l), token_key(str(l))) for l in self.rank[orig_col].dropna().unique()]
        self._orig = [(l, k, set(k.split())) for l, k in self._orig]

    def _products(self, idx: list[int], sims: np.ndarray | None = None) -> list[dict]:
        available_cols = [c for c in PRODUCT_COLS if c in self.var.columns]
        rows = to_records(self.var.iloc[idx][available_cols])
        if sims is not None:
            for r, i in zip(rows, idx):
                r["skor_kemiripan"] = round(float(sims[i]), 3)
        return rows

    def match_original(self, q: str, k: int = 3) -> list[tuple[str, float]]:
        qk = token_key(q)
        qt = set(qk.split())
        if not qt:
            return []
        scored = []
        for line, key, toks in self._orig:
            s = SequenceMatcher(None, key, qk).ratio()
            if qt <= toks:
                s = max(s, 0.9 + 0.1 * len(qt) / len(toks))
            scored.append((line, round(s, 3)))
        scored.sort(key=lambda t: -t[1])
        return scored[:k]

    def find_local_dupes(self, original: str, top_n: int = 5) -> dict:
        cands = self.match_original(original)
        if not cands or cands[0][1] < 0.6:
            return {
                "ditemukan": False,
                "pesan": f"Tidak ada original yang cocok dengan '{original}' di data riset.",
                "contoh_nama": [l for l, _, _ in self._orig[:8]],
            }
        line, score = cands[0]
        orig_col = "line_full" if "line_full" in self.rank.columns else "original"
        info_df = self.rank[self.rank[orig_col] == line]
        info = info_df.iloc[0] if not info_df.empty else {}
        
        lab = self.pairs[self.pairs.line_full == line].drop_duplicates("variant_id") if "line_full" in self.pairs.columns else pd.DataFrame()
        if lab.empty and "original_full" in self.pairs.columns:
            lab = self.pairs[self.pairs.original_full == line].drop_duplicates("variant_id")
            
        lab_idx = [self._idx[str(v)] for v in lab.variant_id if str(v) in self._idx] if not lab.empty else []
        labeled = self._products(lab_idx)
        labeled.sort(key=lambda r: -(r.get("sold_est") or r.get("sold_total") or 0))
        
        similar: list[dict] = []
        if self.sim_model is not None:
            res_df = self.sim_model.search(line, top_n=top_n + len(lab_idx))
            similar = to_records(res_df.head(top_n))
        
        stat = {col: info.get(col) for col in ORIGINAL_COLS if col in info}
        return {
            "ditemukan": True,
            "original_cocok": line,
            "skor_kecocokan_nama": score,
            "alternatif_nama": [l for l, _ in cands[1:]] if score < 0.95 else [],
            "statistik_original": stat,
            "produk_lokal_berlabel_dupe": labeled[:15],
            "produk_lokal_notes_mirip": similar,
            "catatan": "Pencarian dupe menggunakan model ML terbaru.",
        }

    def recommend_by_aroma(self, notes: str, gender: str | None = None, max_price: float | None = None,
                           top_n: int = 6) -> dict:
        if self.sim_model is not None:
            res_df = self.sim_model.search(notes, top_n=top_n * 2)
            if gender and "gender" in res_df.columns:
                res_df = res_df[res_df.gender.str.lower().isin([gender.lower(), "unisex"])]
            price_col = "price_median" if "price_median" in res_df.columns else "price_mid"
            if max_price and price_col in res_df.columns:
                res_df = res_df[res_df[price_col] <= max_price]
            res_df = res_df.head(top_n)
            return {"ditemukan": not res_df.empty, "hasil": to_records(res_df)}
        
        return {"ditemukan": False, "pesan": "Model ML SimilarityModel tidak aktif."}

    def list_originals(self, kuadran: str | None = None, sort_by: str = "opportunity", brand: str | None = None,
                       limit: int = 10, q: str | None = None) -> dict:
        d = self.rank.copy()
        if brand and "original_brand" in d.columns:
            d = d[d.original_brand.str.contains(brand, case=False, na=False)]
        orig_col = "line_full" if "line_full" in d.columns else "original"
        if q:
            qt = set(token_key(q).split())
            d = d[d[orig_col].apply(lambda s: qt <= set(token_key(str(s)).split()))]
            
        sorts = {"opportunity": "opportunity_score", "terjual": "sold_w", "pesaing": "n_brand_dupe", "median": "n_varian_dupe"}
        sort_col = sorts.get(sort_by, "opportunity_score")
        if sort_col in d.columns:
            d = d.sort_values(sort_col, ascending=(sort_by == "pesaing"), na_position="last")
            
        cols = [c for c in ORIGINAL_COLS if c in d.columns]
        return {"total_cocok": int(len(d)), "hasil": to_records(d[cols].head(limit))}

    def top_local_products(self, brand: str | None = None, gender: str | None = None, sort: str = "terjual",
                           limit: int = 10) -> dict:
        d = self.var
        if brand and "brand" in d.columns:
            d = d[d.brand.str.contains(brand, case=False, na=False)]
        if gender and "gender" in d.columns:
            d = d[d.gender.str.lower() == gender.lower()]
        col = "sold_est" if "sold_est" in d.columns else "sold_total"
        if sort == "harga_asc":
            col = "price_mid" if "price_mid" in d.columns else "price_median"
        if col in d.columns:
            d = d.sort_values(col, ascending=(sort == "harga_asc"), na_position="last").head(limit)
        idx = [self._idx[str(v)] for v in d.id_parfume if str(v) in self._idx]
        return {"hasil": self._products(idx)}

    def notes_insight(self, layer: str = "top", limit: int = 10) -> dict:
        if "notes_per_kelompok" in self.sheets:
            t = self.sheets["notes_per_kelompok"]
            return {"layer": layer, "hasil": to_records(t.head(limit)), "catatan": "Analisis notes per kelompok dari model ML terbaru."}
        return {"layer": layer, "hasil": [], "catatan": "Data notes tidak tersedia."}

    def price_benchmark(self) -> dict:
        res = {}
        if "skenario_revenue" in self.sheets:
            res["skenario_revenue"] = to_records(self.sheets["skenario_revenue"])
        return res

    def market_overview(self) -> dict:
        price_col = "price_mid" if "price_mid" in self.var.columns else "price_median"
        sold_col = "sold_est" if "sold_est" in self.var.columns else "sold_total"
        b = (self.var.groupby("brand")
             .agg(n_varian=("variant", "count"), terjual=(sold_col, "sum"), harga_median=(price_col, "median"))
             .sort_values("terjual", ascending=False))
        ringkasan = to_records(self.sheets["ringkasan_keputusan"])[0] if "ringkasan_keputusan" in self.sheets and not self.sheets["ringkasan_keputusan"].empty else {}
        return {
            "ringkasan": ringkasan,
            "brand": to_records(b.reset_index()),
            "peringatan": [
                "terjual = angka kumulatif satu kali ambil (snapshot), tidak ada tren waktu",
                "omzet = harga x terjual, estimasi kotor bukan untung",
                "menggunakan model ML dan utilitas model terbaru",
            ],
        }

    def scout_opportunity(self, focus: str = "gap", gender: str | None = None,
                          budget_max: float | None = None, limit: int = 10) -> dict:
        results: dict = {}
        if focus in ("gap", "all"):
            gap = self.rank.sort_values("opportunity_score", ascending=False).head(limit) if "opportunity_score" in self.rank.columns else self.rank.head(limit)
            cols = [c for c in ORIGINAL_COLS if c in gap.columns]
            results["gap_analysis"] = {
                "deskripsi": "Original dengan peluang terbaik (skor tertinggi dari model ML)",
                "total": int(len(self.rank)),
                "top": to_records(gap[cols]),
            }
        if focus in ("whitespace", "all") and "notes_lokal_vs_timteng" in self.sheets:
            ns = self.sheets["notes_lokal_vs_timteng"].head(limit)
            results["white_space"] = {
                "deskripsi": "Analisis perbandingan notes Lokal vs Timteng",
                "hasil": to_records(ns),
            }
        return results

    def search_master_data(self, query: str | None = None, brand: str | None = None,
                           gender: str | None = None, max_price: float | None = None,
                           limit: int = 10) -> dict:
        if self.master.empty:
            return {"ditemukan": False, "pesan": "Data master parfum belum tersedia."}
        d = self.master.copy()
        if brand:
            d = d[d.brand.str.contains(brand, case=False, na=False)]
        if gender:
            d = d[d.gender.str.lower() == gender.lower()]
        if max_price:
            d = d[d.price <= max_price]
        if query:
            q = query.lower()
            mask = (
                d.variant.str.lower().str.contains(q, na=False) |
                d.brand.str.lower().str.contains(q, na=False) |
                d.dupe.str.lower().str.contains(q, na=False) |
                d.all_notes.str.contains(q, na=False)
            )
            d = d[mask]
        if d.empty:
            return {"ditemukan": False, "pesan": "Tidak ada produk yang cocok dengan pencarian.", "total": 0}
        d = d.sort_values("sold_shoppe", ascending=False, na_position="last").head(limit)
        cols = [c for c in ["id_parfume", "variant", "brand", "price", "size", "gender", "concetrate",
                            "top_notes", "mid_notes", "base_notes", "sold_shoppe", "dupe", "status"] if c in d.columns]
        return {"ditemukan": True, "total": int(len(d)), "hasil": to_records(d[cols]), "catatan": "Data dari master parfum."}


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
