"""Konsultan berbasis Groq (OpenAI-compatible) dengan tool-use dan kemampuan analisis PDF/teks panjang."""
from __future__ import annotations

import json
import os

from openai import AsyncOpenAI

from app.services.tools import TOOLS_OPENAI, run_tool

MODEL = "openai/gpt-oss-20b"
MAX_STEPS = 6

SYSTEM = """Kamu adalah "ASMI Consultant", Konsultan Senior AI dan Pakar Riset Pasar Industri Parfum Global & Lokal Indonesia. 
Kamu dibekali akses penuh ke database riset kuantitatif 249+ varian parfum lokal, analisis komparatif merek, matriks dupe original dunia, serta model Machine Learning.

PRINSIP KERJA & KEMAMPUAN SUPER (POWERFUL AGENT):
1. **Analisis Pasar Mendalam & Berbasis Data**: Gunakan data kuantitatif dari tool secara presisi (penjualan Shopee, skor popularitas model ML, estimasi omzet, struktur notes). Jangan mengarang angka statistik.
2. **Keahlian Sensoris & Karakter Aroma (Perfumer Expert)**: Berikan penjelasan deskriptif, elegan, dan edukatif mengenai profil wangi, karakteristik notes (fresh, soapy/clean, gourmand, woody, floral), dan kemiripan dupe.
3. **Analisis Dokumen & List Dupe (PDF / Teks Panjang)**: Jika pengguna melampirkan teks dokumen atau isi file PDF (seperti daftar dupe, formula, atau riset), baca dan analisa dengan teliti. Berikan masukan strategis, validasi kesesuaian market Indonesia, atau evaluasi formulanya.
4. **Pengetahuan Global Tanpa Batas**: Jika ditanya di luar database lokal (sejarah parfum, tips layering, rekomendasi niche/designer dunia), jawab secara luas, mendalam, dan sangat profesional.
5. **Gaya Bahasa**: Profesional, tajam, strategis, ramah, dan memikat dalam Bahasa Indonesia."""


def enabled() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ.get("GROQ_API_KEY") or os.environ.get("BACKUP_GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )


async def ask_groq(messages: list[dict]) -> tuple[str, list[dict]]:
    client = _get_client()
    convo: list[dict] = [{"role": "system", "content": SYSTEM}]
    convo += [{"role": m["role"], "content": m["content"]} for m in messages]
    used: list[dict] = []

    for _ in range(MAX_STEPS):
        resp = await client.chat.completions.create(
            model=MODEL,
            max_tokens=1800,
            messages=convo,
            tools=TOOLS_OPENAI,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return (msg.content or "").strip(), used

        msg_dict = msg.model_dump(exclude_none=True)
        if "tool_calls" in msg_dict:
            for tc_dict in msg_dict["tool_calls"]:
                if "function" in tc_dict and "arguments" in tc_dict["function"]:
                    try:
                        args = json.loads(tc_dict["function"]["arguments"])
                        cleaned = {k: v for k, v in args.items() if v is not None}
                        tc_dict["function"]["arguments"] = json.dumps(cleaned, ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError):
                        pass
        convo.append(msg_dict)

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            used.append({"name": fn_name, "input": fn_args})

            try:
                out = run_tool(fn_name, fn_args)
            except Exception as e:
                out = {"error": str(e)}

            convo.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(out, ensure_ascii=False),
            })

    return "Maaf, pertanyaannya butuh terlalu banyak langkah. Coba dipersempit.", used
