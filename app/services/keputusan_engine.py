# keputusan_engine.py: sistem keputusan jual parfum (spesifikasi: konsentrasi, ml, range harga; target per hari; ML).
# Baca data CSV (folder data/) + model permintaan. Dipakai notebook dan backend.
#   from keputusan_engine import KeputusanEngine
#   eng = KeputusanEngine("data", "output/model_permintaan.joblib")
#   eng.rekomendasi(dupe_dari="Creed Aventus", target_laba_hari=250000, target_balik_modal_hari=180, modal_maks=30_000_000)
import math, os, warnings
import numpy as np, pandas as pd, joblib
from scipy.stats import norm, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")

FITUR = ["pml", "conc_ext", "size_max", "lokal", "age_bulan"]
BATAS = ["Skala penjualan bergantung asumsi faktor brand baru (brand baru jauh lebih lambat dari rata-rata pasar); kalibrasi dengan penjualan nyata lewat kalibrasi().",
         "Model permintaan belajar dari ~109 produk Shopee (terjual = batas bawah, satu snapshot); akurasi pada brand baru rendah-sedang, jadi pakai sebagai rentang, bukan angka pasti.",
         "Harga laris di pasar adalah korelasi (brand kuat juga lebih mahal), bukan bukti bahwa menaikkan harga menaikkan penjualan.",
         "Efek iklan/afiliasi pada volume tidak ada di data; yang dihitung hanya biaya dan titik impasnya."]

def ceil_to(x, step): return math.ceil(round(x / step, 9)) * step

class KeputusanEngine:
    def __init__(self, data_dir="data", model_path=None, faktor_brand=(0.05, 0.15, 0.4), umur_ref_bulan=6, random_state=42):
        rd = lambda f: pd.read_csv(os.path.join(data_dir, f))
        self.A = dict(zip(rd("asumsi.csv").key, rd("asumsi.csv").nilai)); self.opsi = rd("opsi.csv"); self.scn = rd("param_skenario.csv"); self.pl = rd("pricelist_maklon.csv")
        self.mk = rd("data_pasar.csv"); self.mk["pml"] = self.mk.harga_per_ml; self.umur_ref = umur_ref_bulan; self.fb = faktor_brand; self.rs = random_state
        rng = np.random.RandomState(random_state); f = rng.triangular(*faktor_brand, 20000); self.lnf_m, self.lnf_s = float(np.log(f).mean()), float(np.log(f).std())
        if model_path and os.path.exists(model_path):
            try:
                self.dm = joblib.load(model_path)
            except Exception:
                self.dm = self.fit_demand()
                joblib.dump(self.dm, model_path)
        else:
            self.dm = self.fit_demand()
            if model_path: os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True); joblib.dump(self.dm, model_path)

    # ---------- ML #1: model permintaan dari data pasar ----------
    def _train_frame(self):
        t = self.mk[self.mk.sold_lb.notna() & (self.mk.age_bulan <= 24)].copy()
        t["conc_ext"] = (t.conc_short == "Extrait").astype(int); t["lokal"] = (t.origin == "Lokal").astype(int)
        t["spd"] = t.sold_lb / t.age_bulan.clip(lower=3) / 30.44; t["y"] = np.log(t.spd.clip(lower=0.05)); return t
    @staticmethod
    def _gbr(a): return GradientBoostingRegressor(loss="quantile", alpha=a, n_estimators=120, max_depth=2, learning_rate=.05, min_samples_leaf=8, random_state=0)
    def fit_demand(self):
        t = self._train_frame(); return {a: self._gbr(a).fit(t[FITUR], t.y) for a in (.25, .5, .75)}
    def evaluasi_demand(self):
        t = self._train_frame(); pr = np.zeros(len(t))
        for tr, te in GroupKFold(6).split(t, groups=t.kelompok): pr[te] = self._gbr(.5).fit(t.iloc[tr][FITUR], t.y.iloc[tr]).predict(t.iloc[te][FITUR])
        return {"n_train": len(t), "spearman_brand_baru": round(float(spearmanr(pr, t.y)[0]), 3), "MAE_log_model": round(float(np.abs(pr - t.y).mean()), 3), "MAE_log_baseline": round(float(np.abs(t.y - t.y.median()).mean()), 3)}
    def lam_dist(self, pml, conc_ext, size, lokal=1):
        X = pd.DataFrame({"pml": np.atleast_1d(pml), "conc_ext": conc_ext, "size_max": size, "lokal": lokal, "age_bulan": self.umur_ref})
        q = np.sort(np.vstack([self.dm[a].predict(X[FITUR]) for a in (.25, .5, .75)]), axis=0)
        mu = q[1]; sg = np.maximum((q[2] - q[0]) / 1.349, 0.5)
        return mu + self.lnf_m, np.sqrt(sg ** 2 + self.lnf_s ** 2)          # parameter lognormal botol/hari untuk brand baru
    @staticmethod
    def p_ge(x, mu, sg): return 1 - norm.cdf((np.log(np.maximum(x, 1e-9)) - mu) / sg)

    # ---------- ekonomi (sama dengan kalkulator Excel) ----------
    def hpp(self, o, tier):
        A = self.A; ml = int(o.ukuran_ml); tot = A["varian"] * tier
        c = self.pl[(self.pl.ukuran_ml == ml) & (self.pl.moq == tier) & (self.pl.konsentrasi == o.konsentrasi)].harga
        if c.empty or pd.isna(c.iloc[0]): return None
        kem = A[f"botol_{ml}"] + A[f"box_{ml}"] + A[f"shrink_{ml}"]; legal = A["varian"] * (A["bpom_per_aroma"] + A["halal_per_aroma"]) + A["hki_total"]
        return {"total_botol": tot, "cairan": float(c.iloc[0]), "kemasan": kem, "modal_total": tot * (float(c.iloc[0]) + kem) + legal, "hpp_kirim": float(c.iloc[0]) + kem + legal / tot + A["packing_per_pesanan"]}
    @staticmethod
    def _K(s): return s.admin + s.pph + s.program + s.afiliasi + s.iklan + s.retur
    def harga_min(self, hk, s): return ceil_to((hk + s.biaya_tetap) / (1 - self._K(s) - s.target_margin), self.A["pembulatan"])
    def laba(self, P, s, hk): return P * (1 - (self._K(s) - s.program)) - min(P * s.program, s.cap_program) - s.biaya_tetap - hk
    def kontribusi(self, P, s): return P * (1 - (self._K(s) - s.program)) - min(P * s.program, s.cap_program) - s.biaya_tetap - self.A["packing_per_pesanan"]

    def bangun_dataset(self, tiers=(100, 250, 500, 1000), mults=tuple(np.round(np.arange(0.9, 2.01, 0.1), 2)), target_laba_hari=250000, target_balik_modal_hari=180, lead_hari=30, modal_maks=None, kanal=None, lokal=1):
        rows = []; dp = self.A["dp_persen"]; dis = self.A["diskon_coret"]; pb = self.A["pembulatan"]
        for _, o in self.opsi.iterrows():
            for tier in tiers:
                h = self.hpp(o, tier)
                if h is None: continue
                for _, s in self.scn.iterrows():
                    if kanal and s.skenario not in kanal: continue
                    pmin = self.harga_min(h["hpp_kirim"], s)
                    for m in mults:
                        P = ceil_to(pmin * m, pb); tay = ceil_to(P / (1 - dis), pb); pml = tay / o.ukuran_ml
                        lb = self.laba(P, s, h["hpp_kirim"]); c = self.kontribusi(P, s)
                        mu, sg = self.lam_dist(pml, int(o.conc_short == "Extrait"), o.ukuran_ml, lokal); mu, sg = float(mu[0]), float(sg[0])
                        x_laba = target_laba_hari / lb if lb > 0 else np.inf
                        hari_pay = max(target_balik_modal_hari - lead_hari, 1); x_pay = h["modal_total"] / (c * hari_pay) if c > 0 else np.inf
                        rows.append({"opsi": o.opsi, "ukuran_ml": int(o.ukuran_ml), "konsentrasi": o.konsentrasi, "conc_short": o.conc_short, "skenario": s.skenario, "kanal": s.kanal, "tier_per_varian": tier, "total_botol": h["total_botol"],
                            "modal_total": h["modal_total"], "dp": h["modal_total"] * dp, "hpp_kirim": h["hpp_kirim"], "mult_harga": m, "harga_net": P, "harga_tayang": tay, "harga_per_ml": round(pml, 0),
                            "laba_botol": lb, "margin": lb / P, "kontribusi_botol": c, "lam_p50": float(np.exp(mu)), "lam_p10": float(np.exp(mu - 1.2816 * sg)), "lam_p90": float(np.exp(mu + 1.2816 * sg)), "mu": mu, "sigma": sg,
                            "botol_hari_target_laba": x_laba, "botol_hari_target_modal": x_pay, "botol_hari_dibutuhkan": max(x_laba, x_pay),
                            "p_target_laba": float(self.p_ge(x_laba, mu, sg)) if np.isfinite(x_laba) else 0.0, "p_balik_modal": float(self.p_ge(x_pay, mu, sg)) if np.isfinite(x_pay) else 0.0,
                            "p_keduanya": float(self.p_ge(max(x_laba, x_pay), mu, sg)) if np.isfinite(max(x_laba, x_pay)) else 0.0,
                            "hari_habis_p50": h["total_botol"] / float(np.exp(mu)), "layak_modal": True if modal_maks is None else h["modal_total"] <= modal_maks})
        return pd.DataFrame(rows)

    # ---------- pasar pendupe & range harga ----------
    def peers(self, dupe_dari=None, lokal=True):
        m = self.mk[self.mk.origin == "Lokal"] if lokal else self.mk
        if dupe_dari:
            hit = lambda d: d[d.original_list.fillna("").str.lower().str.contains(str(dupe_dari).lower(), regex=False)]
            p = hit(m)
            if len(p) < 6 and lokal: p = hit(self.mk)
            if len(p) >= 6: return p, f"pendupe {dupe_dari} ({len(p)} varian)"
        return m, f"semua pesaing {'lokal' if lokal else ''} ({len(m)} varian)"
    def range_pasar(self, dupe_dari=None, lokal=True):
        p, ket = self.peers(dupe_dari, lokal); out = {}
        for cs in ("EDP", "Extrait"):
            q = p[p.conc_short == cs]
            if len(q) < 5 or q.pml.nunique() < 3: q = (self.mk[self.mk.origin == "Lokal"] if lokal else self.mk); q = q[q.conc_short == cs]      # pendupe terlalu sedikit: pakai seluruh pesaing sekonsentrasi
            laris = q[q.sold_lb >= 5000]; base = laris if len(laris) >= 8 else q
            if len(base) == 0: continue
            out[cs] = {"n_pesaing": int(len(q)), "n_laris": int(len(laris)), "pml_p25": float(base.pml.quantile(.25)), "pml_p50": float(base.pml.median()), "pml_p75": float(base.pml.quantile(.75)),
                       "pml_semua_p50": float(q.pml.median()), "ukuran_median": float(q.size_max.median())}
        return out, ket

    def rekomendasi(self, dupe_dari=None, target_laba_hari=250000, target_balik_modal_hari=180, modal_maks=30_000_000, lead_hari=30, kanal=None, top=4, lokal=True, **kw):
        if kanal is None: kanal = self.scn[(self.scn.afiliasi == 0) & self.scn.skenario.str.contains("Tanpa")].skenario.tolist()      # default: tanpa program/afiliasi (efek program pada volume tidak ada di data); isi kanal=[...] untuk skenario lain
        ds = self.bangun_dataset(target_laba_hari=target_laba_hari, target_balik_modal_hari=target_balik_modal_hari, lead_hari=lead_hari, modal_maks=modal_maks, kanal=kanal, lokal=int(lokal), **kw)
        rp, ket = self.range_pasar(dupe_dari, lokal); ok = ds[ds.layak_modal & (ds.laba_botol > 0)]
        cap = ok.conc_short.map(lambda c: rp[c]["pml_p75"] if c in rp else np.inf)          # jangan ekstrapolasi di atas rentang laris pasar
        ok = ok[(ok.harga_per_ml <= cap) | (ok.mult_harga == ok.mult_harga.min()) & False]
        if ok.empty: ok = ds[ds.layak_modal & (ds.laba_botol > 0) & (ds.mult_harga == 1.0)]
        hasil = []
        for (ml, cs), g in ok.groupby(["ukuran_ml", "conc_short"]):
            top_p = g.p_keduanya.max(); cand = g[g.p_keduanya >= 0.9 * top_p]                                   # banyak harga hampir setara (>=90% peluang terbaik): pilih harga TERENDAH, jangan mendorong harga lebih tinggi dari yang perlu
            best = cand.sort_values(["harga_tayang", "modal_total"]).iloc[0]; sw = g[(g.skenario == best.skenario) & (g.tier_per_varian == best.tier_per_varian)]
            band = sw[sw.p_keduanya >= 0.75 * top_p] if top_p > 0 else sw.iloc[0:0]
            r = rp.get(cs); pasar = None if r is None else {"harga_tayang_low": ceil_to(r["pml_p25"] * ml, 1000), "harga_tayang_mid": ceil_to(r["pml_p50"] * ml, 1000), "harga_tayang_high": ceil_to(r["pml_p75"] * ml, 1000), **r}
            pmin_tay = ceil_to(ceil_to(self.harga_min(best.hpp_kirim, self.scn[self.scn.skenario == best.skenario].iloc[0]), 1) / (1 - self.A["diskon_coret"]), 1000)
            alasan = (f"Peluang mencapai KEDUA target = {best.p_keduanya:.0%} (laba/hari >= Rp{target_laba_hari:,.0f}: {best.p_target_laba:.0%}; balik modal <= {target_balik_modal_hari} hari: {best.p_balik_modal:.0%}). "
                      f"Perlu terjual {best.botol_hari_dibutuhkan:.1f} botol/hari; perkiraan wajar brand baru {best.lam_p50:.1f}/hari (rentang {best.lam_p10:.1f}-{best.lam_p90:.1f}). "
                      f"Harga tayang Rp{best.harga_tayang:,.0f} = Rp{best.harga_per_ml:,.0f}/ml" + (f", rentang laris {ket}: Rp{r['pml_p25']:,.0f}-{r['pml_p75']:,.0f}/ml." if r else ".")
                      + (f" Harga ini {best.harga_tayang / pmin_tay:.1f}x harga minimum aman (hasil korelasi pasar, bukan jaminan): mulai uji di Rp{band.harga_tayang.min():,.0f} dan naikkan bertahap bila penjualan sesuai garis peringatan." if len(band) else ""))
            hasil.append({"konsentrasi": cs, "ukuran_ml": int(ml), "opsi": best.opsi, "skenario": best.skenario, "kanal": best.kanal, "volume_per_varian": int(best.tier_per_varian), "total_botol": int(best.total_botol),
                          "modal_total": float(best.modal_total), "dp": float(best.dp), "harga_net": float(best.harga_net), "harga_tayang": float(best.harga_tayang), "harga_per_ml": float(best.harga_per_ml),
                          "range_harga": {"min_aman_tayang": float(pmin_tay), "pasar_laris": pasar, "mesin_tayang_low": float(band.harga_tayang.min()) if len(band) else None, "mesin_tayang_high": float(band.harga_tayang.max()) if len(band) else None},
                          "laba_botol": float(best.laba_botol), "margin": float(best.margin), "botol_hari_dibutuhkan": float(best.botol_hari_dibutuhkan), "perkiraan_botol_hari_p50": float(best.lam_p50),
                          "p_target_laba": float(best.p_target_laba), "p_balik_modal": float(best.p_balik_modal), "p_keduanya": float(best.p_keduanya), "hari_habis_p50": float(best.hari_habis_p50), "alasan": alasan})
        hasil.sort(key=lambda x: (-x["p_keduanya"], -x["laba_botol"]))
        best_p = hasil[0]["p_keduanya"] if hasil else 0
        realistis = None
        if hasil and best_p < 0.5:
            b = hasil[0]; mu = np.log(b["perkiraan_botol_hari_p50"]); realistis = {"laba_hari_p50": float(b["perkiraan_botol_hari_p50"] * b["laba_botol"]), "catatan": "Target Anda di atas perkiraan median; target realistis = laba/hari pada peluang 50%."}
        catatan = None
        if not hasil: catatan = f"Tidak ada kombinasi yang muat modal_maks Rp{modal_maks:,.0f}; modal minimum di data Rp{ds.modal_total.min():,.0f} (tier terkecil)." if modal_maks is not None and ds.modal_total.min() > modal_maks else "Tidak ada kombinasi dengan laba positif."
        return {"catatan": catatan, "input": {"dupe_dari": dupe_dari, "target_laba_hari": target_laba_hari, "target_balik_modal_hari": target_balik_modal_hari, "modal_maks": modal_maks, "lead_hari": lead_hari, "lokal": lokal},
                "pasar_pembanding": ket, "rekomendasi": hasil[:top], "target_realistis": realistis, "dataset": ds, "batasan": BATAS}

    # ---------- ML #3: rencana harian + garis peringatan ----------
    def rencana_harian(self, plan, hari=180, lead_hari=30):
        s = self.scn[self.scn.skenario == plan["skenario"]].iloc[0]; c = self.kontribusi(plan["harga_net"], s); r = plan["botol_hari_dibutuhkan"]; rows = []; cum = 0.0; kas = -plan["dp"]
        for d in range(0, hari + 1):
            jual = 0.0 if d <= lead_hari else min(r, plan["total_botol"] - cum); cum += jual; kas += jual * c
            if d == lead_hari: kas -= plan["modal_total"] - plan["dp"]
            rows.append({"hari": d, "target_botol_hari": round(jual, 2), "target_botol_kumulatif": round(cum, 1), "target_omzet_net": round(jual * plan["harga_net"]), "batas_iklan_hari": round(jual * plan["harga_net"] * s.iklan), "kas_kumulatif_target": round(kas)})
        return pd.DataFrame(rows)
    def checkpoint(self, plan, target_balik_modal_hari=180, lead_hari=30, n_sim=4000, hari_cek=(7, 14, 30, 60, 90)):
        s = self.scn[self.scn.skenario == plan["skenario"]].iloc[0]; c = self.kontribusi(plan["harga_net"], s); nbe = math.ceil(plan["modal_total"] / c)
        o = self.opsi[self.opsi.opsi == plan["opsi"]].iloc[0]; pml = plan["harga_per_ml"]
        mu, sg = self.lam_dist(pml, int(o.conc_short == "Extrait"), o.ukuran_ml); rng = np.random.RandomState(self.rs); lam = np.exp(rng.normal(mu[0], sg[0], n_sim))
        win = max(target_balik_modal_hari - lead_hari, 0); total = np.minimum(rng.poisson(lam * win), plan["total_botol"]); ok = total >= nbe; out = []
        for d in hari_cek:
            cum = np.random.RandomState(d).binomial(total.astype(int), min(d / max(win, 1), 1.0)); x = cum.reshape(-1, 1)
            if ok.all() or (~ok).all(): thr = {0.5: np.nan, 0.8: np.nan}
            else:
                lr = LogisticRegression(C=100).fit(x, ok); b0, b1 = lr.intercept_[0], lr.coef_[0][0]; thr = {p: (np.log(p / (1 - p)) - b0) / b1 for p in (0.5, 0.8)}
            out.append({"hari_sejak_siap_jual": d, "botol_min_waspada_p50": None if np.isnan(thr[0.5]) else int(math.ceil(max(thr[0.5], 0))), "botol_min_aman_p80": None if np.isnan(thr[0.8]) else int(math.ceil(max(thr[0.8], 0))),
                        "botol_target_lurus": int(math.ceil(nbe * d / max(win, 1)))})
        return {"botol_impas": nbe, "p_balik_modal_awal": float(ok.mean()), "titik": out}

    # ---------- kalibrasi dari penjualan nyata ----------
    def kalibrasi(self, plan, botol_terjual, hari_jual, target_laba_hari=250000, target_balik_modal_hari=180, lead_hari=30):
        o = self.opsi[self.opsi.opsi == plan["opsi"]].iloc[0]; mu, sg = self.lam_dist(plan["harga_per_ml"], int(o.conc_short == "Extrait"), o.ukuran_ml); mu, sg = float(mu[0]), float(sg[0])
        n = max(botol_terjual, 1); lh = np.log(n / hari_jual); prec = 1 / sg ** 2 + n; mu2 = (mu / sg ** 2 + n * lh) / prec; sg2 = float(np.sqrt(1 / prec))
        xl = target_laba_hari / plan["laba_botol"]; xp = plan["modal_total"] / (self.kontribusi(plan["harga_net"], self.scn[self.scn.skenario == plan["skenario"]].iloc[0]) * max(target_balik_modal_hari - lead_hari, 1))
        return {"botol_hari_perkiraan_baru": float(np.exp(mu2)), "botol_hari_sebelumnya": float(np.exp(mu)), "p_keduanya_baru": float(self.p_ge(max(xl, xp), mu2, sg2)), "p_keduanya_lama": float(plan["p_keduanya"])}
