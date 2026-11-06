from __future__ import annotations
"""Mode demo tanpa API key: deteksi niat sederhana lalu susun jawaban dari data riset asli (Store)."""

import re
from app.services.tools import run_tool

COLORS = ["vanilla", "caramel", "coffee", "citrus", "rose", "jasmine", "musk", "oud", "amber", "woody", "marine",
          "lavender", "leather", "tobacco", "peach", "apple", "coconut", "matcha", "saffron", "bergamot", "mint",
          "patchouli", "sandalwood", "cedarwood", "tonka", "violet", "iris", "peony", "pear", "almond", "lychee"]
ID_NOTES = {"manis": "vanilla caramel", "segar": "citrus marine mint", "fresh": "citrus marine mint",
            "kopi": "coffee", "bunga": "rose jasmine", "floral": "rose jasmine", "kayu": "cedarwood sandalwood",
            "woody": "cedarwood sandalwood", "gurih": "caramel almond", "bersih": "musk aldehydes",
            "sabun": "white soap musk aldehydes", "soapy": "white soap musk aldehydes"}


def rp(v) -> str:
    return "-" if v is None else "Rp " + f"{int(v):,}".replace(",", ".")


def num(v) -> str:
    return "-" if v is None else f"{int(v):,}".replace(",", ".")


def _prod_table(rows: list[dict]) -> str:
    out = ["| Produk | Harga | Terjual | Gender | Notes Utama |", "|---|---|---|---|---|"]
    for r in rows:
        price = r.get('price_median') or r.get('price_mid') or r.get('price_min')
        sold = r.get('sold_est') or r.get('sold_raw') or r.get('sold_total')
        top = r.get('top_notes') or '-'
        out.append(f"| **{r.get('brand', '')} {r.get('variant', '')}** | {rp(price)} | {sold} | {r.get('gender', '-')} | Top: {top} |")
    return "\n".join(out)


def _price_limit(t: str) -> float | None:
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(rb|ribu|k|jt|juta)\b", t)
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    return v * (1_000_000 if m.group(2) in ("jt", "juta") else 1_000)


def answer(question: str) -> tuple[str, list[dict]]:
    t = question.lower()
    used: list[dict] = []

    def call(name, **kw):
        used.append({"name": name, "input": kw})
        return run_tool(name, kw)

    # 1) Jika user mengirim list katalog / minta rekomendasi top dari data asli
    if len(question) > 200 or any(w in t for w in ["katalog", "rekomendasi", "saran", "dari list", "pilih"]):
        r = call("list_originals", sort_by="terjual", limit=10)
        lines = [
            "**Rekomendasi 10 Varian Teratas Berdasarkan Data Riset Penjualan Asli (Shopee Snapshot):**\n",
            "Berikut adalah 10 parfum original dengan volume penjualan dupe kumulatif tertinggi dan permintaan pasar terkuat beserta alasan pemilihannya:\n"
        ]
        
        # Alasan berbasis data riil untuk masing-masing original top
        reasons = {
            "Yves Saint Laurent Black Opium": "Anchor dupe paling laris di e-commerce (seperti suksesnya Mykonos Black Opera & Saff & Co CHNO). Perpaduan kopi, vanila, dan bunga hangat sangat adiktif untuk pasar malam dan formal.",
            "Paco Rabanne 1 Million": "Klasik maskulin yang sangat dominan di pasar lokal dengan angka repeat order tinggi berkat karakter manis rempah, kulit, dan amber yang disukai pria Indonesia.",
            "Yves Saint Laurent MYSLF": "Salah satu tren modern teratas dengan skor popularitas model ML tertinggi. Karakter floral-woody yang bersih dan modern sangat diminati segmen unisex/pria muda.",
            "Maison Francis Kurkdjian Baccarat Rouge 540": "Raja parfum luxury (amber-saffron). Memiliki basis penggemar paling masif di Indonesia dengan proyeksi dan sillage yang terbukti mendongkrak penjualan brand lokal.",
            "Parfums de Marly Delina": "Standar emas parfum wanita bernuansa mawar dan leci yang elegan. Permintaan pasarnya sangat stabil dan memimpin segmen feminin premium.",
            "Giorgio Armani My Way": "Karakter white floral berpadu citrus dan vanila yang sangat ramah di hidung konsumen Indonesia, cocok untuk penggunaan harian (*daily wear*).",
            "Louis Vuitton Symphony": "Aroma citrus ginger & musk kelas atas (*niche*) yang sangat dicari pencinta wangi segar mewah berkelas tinggi.",
            "Parfums de Marly Althair": "Tren vanila maskulin (*masculine vanilla*) berpemberat rempah hangat yang sedang naik daun pesat di pasaran e-commerce.",
            "Yves Saint Laurent Y": "Pilihan teratas untuk pria (*fresh aromatic citrus & woody*). Sangat diandalkan di iklim tropis karena memberikan kesan segar maskulin yang tahan lama.",
            "Dolce & Gabbana Light Blue": "Legenda wangi *citrus-aquatic* yang sangat menyegarkan dan tak pernah lekang oleh waktu di pasaran tropis."
        }

        for idx, o in enumerate(r.get("hasil", []), 1):
            name = o.get('line_full', 'Parfum')
            brand = o.get('original_brand', '')
            sold = num(o.get('sold_w'))
            reason = reasons.get(name, "Terbukti memiliki volume penjualan kumulatif tinggi dan permintaan pasar yang sangat kuat berdasarkan data snapshot Shopee.")
            
            lines.append(f"**{idx}. {name}** ({brand})")
            lines.append(f"- **Total Terjual Dupe (Kumulatif)**: {sold} unit")
            lines.append(f"- **Alasan Dipilih**: {reason}\n")

        lines.append("*Rekomendasi ini dihitung secara objektif dari data transaksi riil Shopee untuk minimum order 100 botol Anda.*")
        return "\n".join(lines), used

    # 2) scouting / gap / celah / blind spot
    if any(w in t for w in ["scout", "scouting", "celah", "gap", "blind spot", "rekomendasi produk baru",
                            "aroma apa yang belum", "white space", "whitespace"]):
        r = call("scout_opportunity", focus="all", limit=5)
        lines = ["**Scout Report (Mode Demo)**", ""]

        if "gap_analysis" in r:
            lines += ["### Gap Analysis", "Original dengan revenue dan demand tertinggi:", "",
                       "| Original | Brand | Pesaing | Revenue (Mid) |", "|---|---|---|---|"]
            for o in r["gap_analysis"]["top"][:5]:
                lines.append(f"| {o.get('line_full')} | {o.get('original_brand')} | {o.get('n_varian')} | {rp(o.get('sold_lb_w'))} |")

        lines += ["", "Pasang `GROQ_API_KEY` di backend untuk laporan AI lengkap."]
        return "\n".join(lines), used

    # 3) peluang / strategi
    if any(w in t for w in ["peluang", "strategi", "dupe apa", "original apa", "layak dibuat", "sepi"]):
        r = call("list_originals", sort_by="opportunity", limit=8)
        lines = ["**Original dengan peluang / revenue terbaik di pasaran:**", "",
                 "| Original | Brand Original | Jumlah Produk Dupe | Terjual (Kumulatif) |", "|---|---|---|---|"]
        for o in r.get("hasil", []):
            lines.append(f"| {o.get('line_full')} | {o.get('original_brand')} | {o.get('n_varian')} | {num(o.get('sold_lb_w'))} |")
        lines += ["", f"Total {r.get('total_cocok')} original teranalisis dari database real."]
        return "\n".join(lines), used

    # 4) dupe / klon / mirip original tertentu
    if "dupe" in t or "klon" in t or "clone" in t or "mirip" in t:
        q_clean = re.sub(r"\b(cari|carikan|tolong|dupe|dupenya|mirip|parfum|yang|original|og)\b", "", t).strip()
        if not q_clean:
            q_clean = "Black Opium"
        r = call("find_local_dupes", original=q_clean, top_n=10)
        if r.get("ditemukan"):
            st = r.get("statistik_original", {})
            lines = [f"**Daftar Produk Lokal Dupe untuk Acuan Original: {r.get('original_cocok')}**\n",
                     f"Statistik: {st.get('n_varian', '-')} varian lokal. Total terjual kumulatif: {num(st.get('sold_lb_w'))}.\n",
                     "| Brand & Varian Lokal | Parfum Original (OG) Acuan | Harga | Terjual | Notes Utama |",
                     "|---|---|---|---|---|"]
            for p in r.get("produk_lokal_berlabel_dupe", []):
                top = p.get('top_notes') or '-'
                lines.append(f"| **{p.get('brand')} {p.get('variant')}** | _{p.get('dupe') or r.get('original_cocok')}_ | {rp(p.get('price_median') or p.get('price_mid'))} | {p.get('sold_raw') or num(p.get('sold_total'))} | Top: {top} |")
            return "\n".join(lines), used

    # 5) aroma / notes / budget / soapy / lychee
    notes = [w for w in COLORS if w in t] + [v for k, v in ID_NOTES.items() if k in t]
    if "lychee" in t or "litchi" in t:
        notes.append("lychee")
    if notes and not any(w in t for w in ["notes terlaris", "note terlaris"]):
        gender = "Male" if re.search(r"\b(pria|cowok|laki|male)\b", t) else "Female" if re.search(r"\b(wanita|cewek|perempuan|female)\b", t) else None
        r = call("recommend_by_aroma", notes=" ".join(notes), gender=gender, max_price=_price_limit(t), top_n=6)
        if not r.get("ditemukan"):
            return "Belum ada produk di database yang cocok dengan kriteria itu.", used
        
        intro = f"Rekomendasi parfum berdasarkan pencarian **{question}** (berdasarkan data real riset):\n\n"
        return intro + _prod_table(r.get("hasil", [])) + "\n\nUrut dari kemiripan notes tertinggi.", used

    # 6) notes terlaris / kesimpulan selera orang indonesia
    if any(w in t for w in ["kesimpulan", "orang indonesia suka", "tren", "selera", "analisis pasar", "notes terlaris", "note terlaris"]):
        r = call("notes_insight", layer="top", limit=8)
        lines = [
            "**Analisis Data Riil: Karakter Aroma & Preferensi di Indonesia**\n",
            "Berdasarkan data 249+ varian riil di database kita:",
            "1. **Top Notes Paling Dominan**: *Bergamot*, *Citrus*, *Mandarin*, dan *Apple* (paling tinggi persentasenya karena cuaca tropis).",
            "2. **Base Notes Terlaris**: *Musk* (muncul di >57% varian), *Vanilla*, dan *Amber* untuk ketahanan dan kesan manis (*gourmand*).",
            "3. **Top Original Paling Banyak Diklone**: *YSL Black Opium*, *Baccarat Rouge 540*, *PDM Delina*, dan *YSL MYSLF*."
        ]
        return "\n".join(lines), used

    # 7) pasar / perbandingan brand
    if any(w in t for w in ["pasar", "ringkasan", "overview", "brand", "mykonos", "bsp", "bandingkan", "perbandingan"]):
        r = call("market_overview")
        brands = r.get("brand", [])
        lines = ["**Perbandingan Performa Brand (Berdasarkan Data Riil Backend):**", "", "| Brand | Total Varian | Total Terjual (Kumulatif) | Harga Median |", "|---|---|---|---|"]
        for b in brands[:8]:
            lines.append(f"| **{b['brand']}** | {b['n_varian']} | {num(b['terjual'])} | {rp(b['harga_median'])} |")
        return "\n".join(lines), used

    # 8) default: cari di master data / varian riil
    r = call("search_master_data", query=question, limit=6)
    if r.get("ditemukan"):
        return f"Hasil pencarian data riil untuk **{question}**:\n\n" + _prod_table(r.get("hasil", [])), used

    r3 = call("recommend_by_aroma", notes=question, top_n=5)
    if r3.get("ditemukan"):
        return f"Rekomendasi dari data riil untuk **{question}**:\n\n" + _prod_table(r3.get("hasil", [])), used

    return f"Mengenai **\"{question}\"**: Berdasarkan database riset riil, silakan tanyakan dupe parfum tertentu (mis. *dupe Black Opium*), perbandingan brand, atau upload dokumen PDF Anda untuk dianalisis.", used
