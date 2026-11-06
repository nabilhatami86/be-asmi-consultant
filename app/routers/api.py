from __future__ import annotations

from fastapi import APIRouter
from app.services import fallback, llm
from app.services.data import get_store
from pydantic import BaseModel, Field
from typing import Literal
from fastapi import UploadFile, File, Form, HTTPException

router = APIRouter()

class Msg(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

class ChatIn(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=30)

@router.get("/health")
def health():
    return {"ok": True, "mode": "groq" if llm.enabled() else "demo", "model": llm.MODEL if llm.enabled() else None}

@router.post("/chat")
async def chat(body: ChatIn):
    msgs = [m.model_dump() for m in body.messages][-12:]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    if not msgs or msgs[-1]["role"] != "user":
        raise HTTPException(400, "Pesan terakhir harus dari user")
    
    if llm.enabled():
        try:
            text, used = await llm.ask_groq(msgs)
            return {"answer": text, "mode": "groq", "tools": used}
        except Exception as e:
            print("Groq error:", repr(e))
            text, used = fallback.answer(msgs[-1]["content"])
            return {"answer": text, "mode": "demo", "tools": used, "notice": "AI sedang bermasalah, memakai mode demo."}
    
    text, used = fallback.answer(msgs[-1]["content"])
    return {"answer": text, "mode": "demo", "tools": used}

@router.post("/chat/pdf")
async def chat_pdf(file: UploadFile = File(...), prompt: str = Form("Analisis dokumen daftar dupe / riset ini:")):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Hanya file PDF yang didukung.")
    
    try:
        import pypdf
        reader = pypdf.PdfReader(file.file)
        pdf_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pdf_text += t + "\n"
        
        if not pdf_text.strip():
            raise HTTPException(400, "Tidak ada teks yang dapat dibaca dari file PDF tersebut.")
        
        if len(pdf_text) > 12000:
            pdf_text = pdf_text[:12000] + "...(terpotong)"

        analysis_prompt = f"{prompt}\n\n--- ISI DOKUMEN PDF ---\n{pdf_text}\n-----------------------\n\nTolong berikan analisis, evaluasi, dan rekomendasi strategis berdasarkan daftar/dokumen di atas."

        msgs = [{"role": "user", "content": analysis_prompt}]
        if llm.enabled():
            text, used = await llm.ask_groq(msgs)
            return {"answer": text, "mode": "groq", "tools": used, "extracted_text_len": len(pdf_text)}
        else:
            return {"answer": f"Dokumen PDF berhasil dibaca ({len(pdf_text)} karakter), tetapi API key AI (Groq) tidak aktif.", "mode": "demo", "tools": []}
    except Exception as e:
        print("PDF parse error:", repr(e))
        raise HTTPException(500, f"Gagal memproses file PDF: {str(e)}")

@router.get("/originals")
def originals(q: str | None = None, kuadran: str | None = None, sort_by: str = "opportunity", limit: int = 30):
    return get_store().list_originals(kuadran=kuadran, sort_by=sort_by, limit=limit, q=q)

@router.get("/dupes")
def dupes(original: str, top_n: int = 5):
    return get_store().find_local_dupes(original, top_n)

@router.get("/overview")
def overview():
    return get_store().market_overview()

@router.get("/scout")
def scout(focus: str = "all", gender: str | None = None, budget_max: float | None = None, limit: int = 10):
    return get_store().scout_opportunity(focus=focus, gender=gender, budget_max=budget_max, limit=limit)

class ScoutAiIn(BaseModel):
    focus: str = "all"
    gender: str | None = None
    budget_max: float | None = None
    context: str = Field("", max_length=500, description="Konteks tambahan dari user")

@router.post("/scout/ai")
async def scout_ai(body: ScoutAiIn):
    store = get_store()
    data = store.scout_opportunity(focus=body.focus, gender=body.gender, budget_max=body.budget_max, limit=8)

    import json as _json
    data_str = _json.dumps(data, ensure_ascii=False, default=str)
    if len(data_str) > 6000:
        data_str = data_str[:6000] + "...(terpotong)"

    scout_prompt = f"""Kamu diminta membuat SCOUT REPORT untuk tim ASMI (brand parfum lokal baru).

Berikut data scouting dari riset pasar:
{data_str}

{f"Konteks tambahan dari user: {body.context}" if body.context else ""}

Buatkan laporan scouting yang berisi:
1. **Ringkasan Peluang** - 3-5 original perfume terbaik untuk dibuat dupe-nya, dengan alasan
2. **Celah Aroma** - Notes/aroma yang belum banyak digarap tapi punya sinyal bagus
3. **Blind Spot Pesaing** - Kelemahan kompetitor yang bisa dimanfaatkan ASMI
4. **Rekomendasi Aksi** - 3-5 langkah konkret: buat produk apa, dengan notes apa, kisaran harga berapa

Format sebagai laporan profesional tapi ringkas (maks 500 kata). Pakai tabel markdown kalau perlu.
Semua angka HARUS dari data di atas, jangan mengarang. Tandai dugaan dengan "(dugaan)".
Jawab dalam Bahasa Indonesia yang santai tapi profesional."""

    msgs = [{"role": "user", "content": scout_prompt}]

    if llm.enabled():
        try:
            text, used = await llm.ask_groq(msgs)
            return {"report": text, "mode": "groq", "tools": used, "data": data}
        except Exception as e:
            print("Scout AI error:", repr(e))
            demo_text = (
                "**Scout Report (Mode Fallback / Demo)**\n\n"
                "Terjadi kendala pada API Groq. Berikut analisis singkat berdasarkan data:\n\n"
                "1. **Peluang Utama**: YSL MYSLF dan PDM Delina memiliki skor popularitas tertinggi.\n"
                "2. **Rekomendasi**: Pertimbangkan membuat dupe dengan aroma fresh citrus / floral musk."
            )
            return {"report": demo_text, "mode": "demo", "tools": [], "data": data}
    return {"report": "Mode demo: AI tidak aktif.", "mode": "demo", "tools": [], "data": data}
