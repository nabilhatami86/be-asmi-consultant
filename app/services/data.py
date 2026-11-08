"""Memuat data riset terbaru dan mengintegrasikan model machine learning joblib (SimilarityModel & ConceptScorer).
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
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
    "id_parfume", "variant_id", "brand", "kelompok", "variant", "sizes", "price_mid", "price_min", "price_max",
    "sold_est", "sold_raw", "gender", "concentrate", "top_notes", "mid_notes", "base_notes",
    "dupe", "dupe_type", "status", "is_best_seller",
]

ORIGINAL_COLS = [
    "line_full", "original_brand", "n_varian", "n_kelompok", "sold_lb_w",
    "opportunity_score", "skor_populer", "rank_komposit", "label", "notes_umum", "kuadran"
]


class Store:
    def __init__(self) -> None:
        xl = pd.ExcelFile(DATA_DIR / "riset_dupe_asmi.xlsx")
        
        var_path = DATA_DIR / "parfume_clean_varian.csv"
        if var_path.exists():
            self.var = pd.read_csv(var_path)
        else:
            self.var = xl.parse("data_bersih_varian")

        if "price_median" not in self.var.columns:
            self.var["price_median"] = self.var["price_mid"] if "price_mid" in self.var.columns else self.var.get("price", 0)
        if "sold_total" not in self.var.columns:
            self.var["sold_total"] = self.var["sold_est"] if "sold_est" in self.var.columns else 0
        if "is_best_seller" not in self.var.columns:
            self.var["is_best_seller"] = self.var["is_fav"] if "is_fav" in self.var.columns else False

        pairs_path = DATA_DIR / "dupe_pairs_clean.csv"
        if pairs_path.exists():
            self.pairs = pd.read_csv(pairs_path)
        else:
            self.pairs = pd.DataFrame()

        master_path = DATA_DIR / "data-master-parfume.csv"
        if master_path.exists():
            self.master = pd.read_csv(master_path, sep=";", encoding="latin-1")
            if "price" in self.master.columns:
                self.master["price"] = pd.to_numeric(self.master["price"], errors="coerce")
            if "sold_shoppe" in self.master.columns:
                self.master["sold_shoppe"] = pd.to_numeric(self.master["sold_shoppe"], errors="coerce")
            elif "sold" in self.master.columns:
                self.master["sold_shoppe"] = pd.to_numeric(self.master["sold"], errors="coerce")
            else:
                self.master["sold_shoppe"] = 0
            self.master["all_notes"] = (
                self.master.get("top_notes", pd.Series([""] * len(self.master))).fillna("") + ", " +
                self.master.get("mid_notes", pd.Series([""] * len(self.master))).fillna("") + ", " +
                self.master.get("base_notes", pd.Series([""] * len(self.master))).fillna("")
            ).str.lower()
        else:
            self.master = pd.DataFrame()

        # Juga gabungkan data varian ke self.master agar pencarian master data / umum menemukan produk lokal
        if not self.var.empty:
            local_as_master = self.var[["id_parfume", "brand", "variant", "price_mid", "sizes", "gender", "concentrate", "top_notes", "mid_notes", "base_notes", "sold_raw", "dupe", "status"]].copy()
            local_as_master.rename(columns={"price_mid": "price", "sold_raw": "sold_shoppe", "concentrate": "concetrate", "sizes": "size"}, inplace=True)
            local_as_master["all_notes"] = (
                local_as_master.top_notes.fillna("") + ", " +
                local_as_master.mid_notes.fillna("") + ", " +
                local_as_master.base_notes.fillna("")
            ).str.lower()
            if self.master.empty:
                self.master = local_as_master
            else:
                self.master = pd.concat([self.master, local_as_master], ignore_index=True)

        self.rank = xl.parse("ranking_ALL")
        if "ringkasan_keputusan" in xl.sheet_names:
            dec = xl.parse("ringkasan_keputusan")
            if "line_full" in dec.columns:
                merge_cols = [c for c in ["line_full", "opportunity_score", "label", "skor_populer", "rank_komposit"] if c in dec.columns]
                self.rank = self.rank.merge(dec[merge_cols], on="line_full", how="left", suffixes=("", "_dec"))

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
        for i, row in self.var.iterrows():
            if pd.notna(row.get("variant_id")):
                self._idx[str(row["variant_id"])] = i

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
        
        lab = pd.DataFrame()
        if not self.pairs.empty and "line_full" in self.pairs.columns:
            lab_match = self.pairs[self.pairs.line_full == line]
            if not lab_match.empty and "variant_id" in lab_match.columns:
                lab = lab_match.drop_duplicates("variant_id")
        
        lab_idx = []
        if not lab.empty and "variant_id" in lab.columns:
            for v in lab.variant_id:
                if str(v) in self._idx:
                    lab_idx.append(self._idx[str(v)])
                else:
                    match_v = self.var[self.var.id_parfume == v]
                    if not match_v.empty:
                        lab_idx.append(self._idx[str(match_v.iloc[0].id_parfume)])
                    else:
                        match_v2 = self.var[self.var.variant_id == v]
                        if not match_v2.empty:
                            lab_idx.append(self._idx[str(match_v2.iloc[0].id_parfume)])

        labeled = self._products(lab_idx)
        labeled.sort(key=lambda r: -(r.get("sold_est") or r.get("sold_total") or 0))
        
        similar: list[dict] = []
        if self.sim_model is not None:
            try:
                res_df = self.sim_model.search(line, top_n=top_n + len(lab_idx))
                similar = to_records(res_df.head(top_n))
            except Exception:
                pass
        
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
            try:
                res_df = self.sim_model.search(notes, top_n=top_n * 2)
                if gender and "gender" in res_df.columns:
                    res_df = res_df[res_df.gender.str.lower().isin([gender.lower(), "unisex"])]
                price_col = "price_median" if "price_median" in res_df.columns else "price_mid"
                if max_price and price_col in res_df.columns:
                    res_df = res_df[res_df[price_col] <= max_price]
                res_df = res_df.head(top_n)
                return {"ditemukan": not res_df.empty, "hasil": to_records(res_df)}
            except Exception:
                pass
        
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
            
        if kuadran:
            if "kuadran" in d.columns:
                q_map = {
                    "peluang": "peluang",
                    "terbukti": "RAMAI",
                    "hati_hati": "biasa",
                    "sepi": "sepi"
                }
                kw = q_map.get(kuadran, kuadran)
                d = d[d.kuadran.str.contains(kw, case=False, na=False)]

        d["n_brand_dupe"] = d.get("n_kelompok") if "n_kelompok" in d.columns else d.get("n_varian", 1)
        d["sold_w"] = d.get("sold_lb_w") if "sold_lb_w" in d.columns else 0
        d["opportunity_score"] = d.get("skor_populer") if "skor_populer" in d.columns else d.get("opportunity_score", 0)
        d["ppm_edp"] = d.get("harga_tengah_median") if "harga_tengah_median" in d.columns else None

        sorts = {"opportunity": "opportunity_score", "terjual": "sold_w", "pesaing": "n_brand_dupe", "median": "n_varian"}
        sort_col = sorts.get(sort_by, "sold_w")
        if sort_col in d.columns:
            d = d.sort_values(sort_col, ascending=False if sort_col == "sold_w" else (sort_by == "pesaing"), na_position="last")
            
        cols = ["line_full", "original_brand", "n_brand_dupe", "sold_w", "opportunity_score", "kuadran", "notes_umum", "ppm_edp"]
        existing_cols = [c for c in cols if c in d.columns]
        return {"total_cocok": int(len(d)), "hasil": to_records(d[existing_cols].head(limit))}

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
        price_col = "price_median" if "price_median" in self.var.columns else "price_mid"
        sold_col = "sold_total" if "sold_total" in self.var.columns else "sold_est"
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
            sort_key = "skor_populer" if "skor_populer" in self.rank.columns else "rank_komposit"
            gap = self.rank.sort_values(sort_key, ascending=False).head(limit)
            gap_records = []
            for _, row in gap.iterrows():
                line = row.get("line_full") or row.get("original")
                score = row.get("skor_populer") if pd.notna(row.get("skor_populer")) else row.get("rank_komposit")
                rec = {
                    "line_full": line,
                    "n_brand_dupe": row.get("n_kelompok") or row.get("n_varian"),
                    "sold_w": row.get("sold_lb_w"),
                    "opportunity_score": score,
                    "notes_umum": row.get("notes_umum"),
                }
                if not self.pairs.empty and "line_full" in self.pairs.columns:
                    dupes = self.pairs[self.pairs.line_full == line]
                elif not self.pairs.empty and "original_full" in self.pairs.columns:
                    dupes = self.pairs[self.pairs.original_full == line]
                else:
                    dupes = pd.DataFrame()

                if not dupes.empty:
                    dupe_idxs = []
                    for v in dupes.get("variant_id", []):
                        if str(v) in self._idx:
                            dupe_idxs.append(self._idx[str(v)])
                        else:
                            match_v = self.var[self.var.id_parfume == v]
                            if not match_v.empty:
                                dupe_idxs.append(self._idx[str(match_v.iloc[0].id_parfume)])
                            else:
                                match_v2 = self.var[self.var.variant_id == v]
                                if not match_v2.empty:
                                    dupe_idxs.append(self._idx[str(match_v2.iloc[0].id_parfume)])
                    if dupe_idxs:
                        prods = to_records(self.var.iloc[dupe_idxs][["id_parfume", "brand", "variant", "price_median", "sold_total", "concentrate"]].head(3))
                        rec["contoh_dupe_lokal"] = prods
                    else:
                        rec["contoh_dupe_lokal"] = []
                else:
                    rec["contoh_dupe_lokal"] = []
                gap_records.append(rec)

            results["gap_analysis"] = {
                "deskripsi": "Original dengan peluang terbaik (skor popularitas dari model ML)",
                "total": int(len(self.rank)),
                "top": gap_records,
            }

        if focus in ("whitespace", "all") and "notes_lokal_vs_timteng" in self.sheets:
            ns = self.sheets["notes_lokal_vs_timteng"].head(limit)
            results["white_space"] = {
                "deskripsi": "Analisis perbandingan notes Lokal vs Timteng",
                "hasil": to_records(ns),
            }

        if focus in ("blindspot", "all"):
            brands = self.var.brand.unique().tolist() if "brand" in self.var.columns else []
            blind_spots = []
            sort_key = "skor_populer" if "skor_populer" in self.rank.columns else "rank_komposit"
            top_originals = self.rank.nlargest(20, sort_key) if sort_key in self.rank.columns else self.rank.head(20)
            for br in brands:
                br_pairs = self.pairs[self.pairs.brand == br] if not self.pairs.empty and "brand" in self.pairs.columns else pd.DataFrame()
                covered = set(br_pairs.line_full.dropna().unique()) if not br_pairs.empty and "line_full" in br_pairs.columns else set()
                orig_col = "line_full" if "line_full" in top_originals.columns else "original"
                missing = top_originals[~top_originals[orig_col].isin(covered)]
                if not missing.empty:
                    recs = to_records(missing[[orig_col]].head(5))
                    blind_spots.append({"brand": br, "belum_cover": recs, "total_missing": int(len(missing))})
            blind_spots.sort(key=lambda x: -x["total_missing"])
            results["blind_spot"] = {
                "deskripsi": "Original dengan demand tinggi yang belum di-cover oleh tiap brand",
                "hasil": blind_spots[:limit],
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
