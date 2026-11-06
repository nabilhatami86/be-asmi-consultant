"""Mode demo tanpa API key: deteksi niat sederhana lalu susun jawaban dari data."""
from __future__ import annotations

import re

from .tools import run_tool

COLORS = ["vanilla", "caramel", "coffee", "citrus", "rose", "jasmine", "musk", "oud", "amber", "woody", "marine",
          "lavender", "leather", "tobacco", "peach", "apple", "coconut", "matcha", "saffron", "bergamot", "mint",
          "patchouli", "sandalwood", "cedarwood", "tonka", "violet", "iris", "peony", "pear", "almond"]
ID_NOTES = {"manis": "vanilla caramel", "segar": "citrus marine mint", "fresh": "citrus marine mint",
            "kopi": "coffee", "bunga": "rose jasmine", "floral": "rose jasmine", "kayu": "cedarwood sandalwood",
            "woody": "cedarwood sandalwood", "gurih": "caramel almond", "bersih": "musk aldehydes",
            "sabun": "white soap musk aldehydes", "soapy": "white soap musk aldehydes"}


def rp(v) -> str:
    return "-" if v is None else "Rp " + f"{int(v):,}".replace(",", ".")


def num(v) -> str:
    return "-" if v is None else f"{int(v):,}".replace(",", ".")


def _prod_table(rows: list[dict]) -> str:
    out = ["| Produk | Harga | Terjual | Gender | Notes & Karakter Aroma |", "|---|---|---|---|---|"]
    for r in rows:
        price = r.get('price_median') or r.get('price_mid') or r.get('price_min')
        sold = r.get('sold_est') or r.get('sold_raw') or r.get('sold_total')
        top = r.get('top_notes') or '-'
        mid = r.get('mid_notes') or '-'
        base = r.get('base_notes') or '-'
        notes_str = f"Top: {top} | Mid: {mid} | Base: {base}"
        out.append(f"| **{r.get('brand', '')} {r.get('variant', '')}** | {rp(price)} | {sold} | {r.get('gender', '-')} | {notes_str} |")
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

    # 1) scouting / gap / celah / blind spot
    if any(w in t for w in ["scout", "scouting", "celah", "gap", "blind spot", "rekomendasi produk baru",
                            "aroma apa yang belum", "white space", "whitespace"]):
        r = call("scout_opportunity", focus="all", limit=5)
        lines = ["**Scout Report (Mode Demo)**", ""]

        if "gap_analysis" in r:
            lines += ["### Gap Analysis", "Original dengan revenue dan demand tertinggi:", "",
                       "| Original | Brand | Pesaing | Revenue (Mid) |", "|---|---|---|---|"]
            for o in r["gap_analysis"]["top"][:5]:
                lines.append(f"| {o.get('original')} | {o.get('original_brand')} | {o.get('n_produk')} | {rp(o.get('rev_mid_w'))} |")

        lines += ["", "Pasang `GROQ_API_KEY` untuk laporan AI lengkap."]
        return "\n".join(lines), used

    # 2) peluang / strategi
    if any(w in t for w in ["peluang", "strategi", "dupe apa", "original apa", "layak dibuat", "sepi"]):
        r = call("list_originals", sort_by="opportunity", limit=8)
        lines = ["**Original dengan peluang / revenue terbaik di pasaran:**", "",
                 "| Original | Brand Original | Jumlah Produk Dupe | Revenue Est. |", "|---|---|---|---|"]
        for o in r.get("hasil", []):
            lines.append(f"| {o.get('original')} | {o.get('original_brand')} | {o.get('n_produk')} | {rp(o.get('rev_mid_w'))} |")
        lines += ["", f"Total {r.get('total_cocok')} original teranalisis."]
        return "\n".join(lines), used

    # 3) aroma / notes / budget / soapy / lychee
    notes = [w for w in COLORS if w in t] + [v for k, v in ID_NOTES.items() if k in t]
    if "lychee" in t or "litchi" in t:
        notes.append("lychee")
    if notes and not any(w in t for w in ["notes terlaris", "note terlaris"]):
        gender = "Male" if re.search(r"\b(pria|cowok|laki|male)\b", t) else "Female" if re.search(r"\b(wanita|cewek|perempuan|female)\b", t) else None
        r = call("recommend_by_aroma", notes=" ".join(notes), gender=gender, max_price=_price_limit(t), top_n=6)
        if not r.get("ditemukan"):
            return "Belum ada produk yang cocok dengan kriteria itu. Coba longgarkan harga atau ganti notes.", agent_fallback_desc(question, used)
        
        intro = f"Rekomendasi parfum berdasarkan pencarian **{question}**:\n\n*Karakter & Deskripsi Aroma:* Perpaduan notes di bawah memberikan sensasi wangi yang segar, menonjolkan nuansa buah dan kesegaran khas yang dicari.\n\n"
        return intro + _prod_table(r.get("hasil", [])) + "\n\nUrut dari kemiripan notes tertinggi berdasarkan model riset.", used

    # 4) notes terlaris
    if "notes" in t or "note" in t or "sabun" in t or "soapy" in t:
        layer = "base" if "base" in t else "mid" if "mid" in t else "top"
        r = call("notes_insight", layer=layer, limit=8)
        lines = [f"**Notes {layer} teratas yang sering muncul di produk bestseller/berkualitas:**", "",
                 "| Note | Jumlah Produk | Lift Bagus |", "|---|---|---|"]
        for x in r.get("hasil", []):
            lines.append(f"| {x.get('note')} | {x.get('n_produk')} | {x.get('lift_bagus')} |")
        return "\n".join(lines), used

    # 5) harga
    if any(w in t for w in ["harga", "price", "per ml", "ukuran"]):
        r = call("price_benchmark")
        lines = ["**Benchmark dan Ringkasan Kategori**", ""]
        if "rev_kategori_konsentrasi" in r:
            lines += ["| Brand / Kategori | EDP | Extrait | Total |", "|---|---|---|---|"]
            for x in r["rev_kategori_konsentrasi"][:10]:
                lines.append(f"| {x.get('group')} | {x.get('EDP')} | {x.get('Extrait')} | {x.get('total')} |")
        return "\n".join(lines), used

    # 6) terlaris
    if any(w in t for w in ["terlaris", "laris", "best seller", "bestseller"]):
        gender = "Male" if re.search(r"\b(pria|cowok|laki|male)\b", t) else "Female" if re.search(r"\b(wanita|cewek|perempuan|female)\b", t) else None
        r = call("top_local_products", gender=gender, sort="terjual", limit=10)
        return "**Produk lokal terlaris (snapshot Shopee):**\n\n" + _prod_table(r.get("hasil", [])), used

    # 7) pasar
    if any(w in t for w in ["pasar", "ringkasan", "overview", "brand", "mykonos", "bsp", "bandingkan", "perbandingan"]):
        r = call("market_overview")
        brands = r.get("brand", [])
        
        # Cek apakah user menyebut brand tertentu (termasuk typo seperti mykonosh)
        target_brands = [b for b in brands if b["brand"].lower() in t or (b["brand"].lower() == "mykonos" and ("mykonosh" in t or "mikonos" in t))]
        
        if target_brands or any(w in t for w in ["banding", "perbandingan", "brand", "mykonos", "bsp"]):
            lines = ["**Perbandingan Performa Brand (Berdasarkan Data Riset Snapshot Shopee):**", ""]
            show_brands = target_brands if target_brands else brands[:6]
            lines += ["| Brand | Total Varian | Total Terjual (Kumulatif) | Harga Median |", "|---|---|---|---|"]
            for b in show_brands:
                lines.append(f"| **{b['brand']}** | {b['n_varian']} | {num(b['terjual'])} | {rp(b['harga_median'])} |")
            lines += ["", "*Catatan: Omzet dan penjualan adalah estimasi kotor dari snapshot Shopee, bukan keuntungan bersih.*"]
            return "\n".join(lines), used

        s = r.get("ringkasan", {})
        lines = [f"**Ringkasan pasar**: {len(brands)} brand aktif.", "", "| Brand | Varian | Terjual |", "|---|---|---|"]
        lines += [f"| {b['brand']} | {b['n_varian']} | {num(b['terjual'])} |" for b in brands[:8]]
        return "\n".join(lines), used

    # 8) default: anggap nama original
    cleaned = re.sub(r"\b(dupe|dupenya|mirip|yang|apa|apakah|ada|parfum|lokal|untuk|dari|cari|carikan|tolong|sama|seperti)\b", " ", t)
    r = call("find_local_dupes", original=cleaned.strip() or question, top_n=4)
    if r.get("ditemukan"):
        st = r.get("statistik_original", {})
        lines = [f"**{r.get('original_cocok')}** diduga jadi acuan di riset (jumlah produk dupe: {st.get('n_produk', '-')}, "
                 f"total toko: {st.get('n_toko', '-')}).", "",
                 _prod_table(r.get("produk_lokal_berlabel_dupe", [])[:8])]
        if r.get("produk_lokal_notes_mirip"):
            lines += ["", "**Notes mirip (dugaan, bukan label):**", "", _prod_table(r.get("produk_lokal_notes_mirip", [])[:4])]
        return "\n".join(lines), used

    # Jika pertanyaan umum / tidak cocok dengan intent data, gunakan fallback cerdas atau jawab sebagai asisten parfum
    if any(w in t for w in ["halo", "hai", "pagi", "siang", "malam", "terima kasih", "thanks"]):
        return "Halo! Ada yang bisa saya bantu seputar riset parfum lokal, rekomendasi aroma, atau dunia parfum?", used

    r = call("search_master_data", query=question, limit=6)
    if r.get("ditemukan"):
        return f"Hasil pencarian terkait **{question}**:\n\n" + _prod_table(r.get("hasil", [])), used

    r3 = call("recommend_by_aroma", notes=question, top_n=5)
    if r3.get("ditemukan"):
        return f"Rekomendasi berdasarkan **{question}**:\n\n" + _prod_table(r3.get("hasil", [])), used

    # Jika mode demo / rule-based menemui pertanyaan general di luar database, berikan jawaban informatif
    return (f"Mengenai **\"{question}\"**: Dalam industri parfum, hal ini berkaitan dengan karakteristik aroma, proyeksi, dan pemilihan notes yang sesuai dengan preferensi personal (seperti fresh, sweet, woody, atau gourmand).\n\n"
            "Tips: Coba tanyakan dupe parfum tertentu (mis. *dupe Baccarat*), rekomendasi aroma (*parfum vanilla di bawah 200rb*), atau perbandingan brand lokal untuk analisis riset yang lebih mendalam."), used
