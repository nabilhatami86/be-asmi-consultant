"""Tools yang bisa dipanggil LLM. Semua jawaban berasal dari data riset."""
from __future__ import annotations

from typing import Any

from .data import get_store

# ---- OpenAI-format tool definitions (compatible with Groq) ----
TOOLS_OPENAI: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "find_local_dupes",
            "description": "Cari produk parfum LOKAL yang dilabeli dupe dari satu parfum original (mis. 'Black Opium', 'Dior Sauvage'), "
                           "plus statistik original itu (jumlah pesaing, terjual, kuadran peluang) dan produk lokal lain dengan notes mirip.",
            "parameters": {"type": "object", "properties": {
                "original": {"type": "string", "description": "Nama parfum original, boleh sebagian"},
                "top_n": {"type": "integer", "default": 5}}, "required": ["original"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_by_aroma",
            "description": "Rekomendasi produk lokal berdasarkan notes/aroma yang diinginkan (bahasa Inggris, mis. 'vanilla, caramel, coffee'). "
                           "Bisa difilter gender dan harga maksimum (rupiah).",
            "parameters": {"type": "object", "properties": {
                "notes": {"type": "string"}, "gender": {"type": "string", "enum": ["Male", "Female", "Unisex"]},
                "max_price": {"type": "number"}, "top_n": {"type": "integer", "default": 6}}, "required": ["notes"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_originals",
            "description": "Daftar parfum original yang jadi acuan dupe, beserta skor peluang. kuadran: peluang = sepi pesaing tapi laris; "
                           "terbukti = ramai & laris; hati_hati = ramai tapi kurang laris; sepi = sepi & kurang laris. "
                           "Pakai untuk pertanyaan strategi produk (original apa yang layak dibuat dupe-nya).",
            "parameters": {"type": "object", "properties": {
                "kuadran": {"type": "string", "enum": ["peluang", "terbukti", "hati_hati", "sepi"]},
                "sort_by": {"type": "string", "enum": ["opportunity", "terjual", "pesaing", "median"], "default": "opportunity"},
                "brand": {"type": "string", "description": "Filter brand original"},
                "limit": {"type": "integer", "default": 10}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "top_local_products",
            "description": "Produk lokal terlaris atau termurah/termahal, bisa difilter brand lokal atau gender.",
            "parameters": {"type": "object", "properties": {
                "brand": {"type": "string"}, "gender": {"type": "string", "enum": ["Male", "Female", "Unisex"]},
                "sort": {"type": "string", "enum": ["terjual", "harga_asc", "harga_desc"], "default": "terjual"},
                "limit": {"type": "integer", "default": 10}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notes_insight",
            "description": "Notes (top/mid/base) yang lebih sering muncul di produk terlaris (lift > 1).",
            "parameters": {"type": "object", "properties": {
                "layer": {"type": "string", "enum": ["top", "mid", "base"], "default": "top"},
                "limit": {"type": "integer", "default": 10}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "price_benchmark",
            "description": "Benchmark harga per konsentrasi (EDP/Extrait) dan ukuran (ml), plus harga per ml.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "market_overview",
            "description": "Ringkasan pasar: total listing/varian/brand, terjual total, brand terbesar, dan peringatan kualitas data.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_master_data",
            "description": "Cari data detail dari master parfum (274 produk). Bisa cari berdasarkan brand, variant, notes, dupe, "
                           "gender, atau kata kunci bebas. Mengembalikan info lengkap termasuk harga, ukuran, notes, dan penjualan Shopee.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Kata kunci pencarian (nama varian, brand, notes, dupe, dsb)"},
                "brand": {"type": "string", "description": "Filter berdasarkan brand"},
                "gender": {"type": "string", "enum": ["Male", "Female", "Unisex"]},
                "max_price": {"type": "number", "description": "Harga maksimum (rupiah)"},
                "limit": {"type": "integer", "default": 10}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scout_opportunity",
            "description": "SCOUTING: Analisis mendalam untuk menemukan peluang produk baru ASMI. "
                           "focus='gap' = original dengan demand tinggi tapi pesaing sedikit; "
                           "'whitespace' = notes/aroma yang jarang dipakai tapi ada di bestseller; "
                           "'blindspot' = original laris yang belum di-cover per brand; "
                           "'rekomendasi' = rekomendasi produk baru berbasis data; "
                           "'all' = semua analisis sekaligus. "
                           "Gunakan tool ini saat user bertanya soal: scouting, peluang baru, rekomendasi produk, "
                           "celah pasar, gap market, aroma apa yang belum digarap, blind spot pesaing.",
            "parameters": {"type": "object", "properties": {
                "focus": {"type": "string", "enum": ["gap", "whitespace", "blindspot", "rekomendasi", "all"],
                          "default": "all", "description": "Jenis analisis scouting"},
                "gender": {"type": "string", "enum": ["Male", "Female", "Unisex"],
                           "description": "Filter gender produk (opsional)"},
                "budget_max": {"type": "number", "description": "Budget maksimum harga produk (rupiah, opsional)"},
                "limit": {"type": "integer", "default": 10}}, "required": []},
        },
    },
]

# Legacy Anthropic-style list (kept for backward compat with fallback)
TOOLS: list[dict[str, Any]] = [
    {
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "input_schema": t["function"]["parameters"],
    }
    for t in TOOLS_OPENAI
]


def run_tool(name: str, args: dict[str, Any]) -> Any:
    s = get_store()
    args = {k: v for k, v in (args or {}).items() if v is not None}
    if "limit" in args:
        args["limit"] = max(1, min(int(args["limit"]), 25))
    if "top_n" in args:
        args["top_n"] = max(1, min(int(args["top_n"]), 15))
    if name == "find_local_dupes":
        return s.find_local_dupes(**args)
    if name == "recommend_by_aroma":
        return s.recommend_by_aroma(**args)
    if name == "list_originals":
        return s.list_originals(**args)
    if name == "top_local_products":
        sort = args.get("sort")
        if sort == "harga_desc":
            r = s.top_local_products(**{**args, "sort": "harga_asc", "limit": 200})
            r["hasil"] = sorted(r["hasil"], key=lambda x: -(x["price_median"] or 0))[: args.get("limit", 10)]
            return r
        return s.top_local_products(**args)
    if name == "notes_insight":
        return s.notes_insight(**args)
    if name == "price_benchmark":
        return s.price_benchmark()
    if name == "market_overview":
        return s.market_overview()
    if name == "search_master_data":
        return s.search_master_data(**args)
    if name == "scout_opportunity":
        return s.scout_opportunity(**args)
    return {"error": f"tool tidak dikenal: {name}"}
