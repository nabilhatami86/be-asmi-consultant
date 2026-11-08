from __future__ import annotations

from fastapi import APIRouter
import os
from app.services import fallback, llm
from app.services.data import get_store
from app.services.keputusan_engine import KeputusanEngine

# Absolut path ke folder data dan output di root parfum
DATA_DIR = r"C:\Users\muham\Desktop\parfum\data"
MODEL_PATH = r"C:\Users\muham\Desktop\parfum\output\model_permintaan.joblib"

# Pastikan file katalog ada di folder root data
ROOT_DATA_DIR = r"C:\Users\muham\Desktop\parfum\data"
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

@router.get("/parfum/asmi")
def parfum_asmi():
    result = get_store().search_master_data(brand="ASMI", limit=100)
    products = result.get("hasil", [])
    response = {"brand": "ASMI", "total": len(products), "hasil": products}
    if not products:
        response["pesan"] = "Belum ada data produk ASMI."
    return response

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

from pydantic import BaseModel
from typing import Optional, List

class CustomSimulasiItem(BaseModel):
    nama_produk: str
    konsentrasi: Optional[str] = "EDP"
    modal_total: float
    jumlah_pcs: int
    harga_jual: float

class CustomSimulasiIn(BaseModel):
    items: List[CustomSimulasiItem]
    biaya_tambahan: Optional[float] = 0.0
    legalitas_box_botol: Optional[float] = 0.0
    pinjam_bri: Optional[bool] = True
    tenor_bri: Optional[int] = 36
    pinjam_jatim: Optional[bool] = True
    tenor_jatim: Optional[int] = 36

@router.post("/konsultasi-custom")
def spk_konsultasi_custom(body: CustomSimulasiIn):
    hasil_simulasi = []
    total_modal_keseluruhan = 0.0
    total_potensi_laba = 0.0
    total_target_jual_per_hari = 0.0
    
    for item in body.items:
        modal = item.modal_total
        pcs = max(item.jumlah_pcs, 1)
        harga = item.harga_jual
        
        hpp_satuan = modal / pcs
        
        pot_shopee_s1 = harga * 0.08
        pot_shopee_s3 = harga * 0.18
        pot_tiktok = harga * 0.16
        
        biaya_packing = 5000
        
        laba_shopee_s1 = harga - hpp_satuan - pot_shopee_s1 - biaya_packing
        laba_shopee_s3 = harga - hpp_satuan - pot_shopee_s3 - biaya_packing
        laba_tiktok = harga - hpp_satuan - pot_tiktok - biaya_packing
        
        laba_per_pcs = laba_shopee_s3
        laba_total = laba_per_pcs * pcs
        
        dp_awal = modal * 0.7
        pelunasan = modal * 0.3
        
        target_jual_per_hari = max(round(pcs / 30.0), 1)
        total_target_jual_per_hari += target_jual_per_hari
        
        hari_balik_modal = int(modal / (laba_per_pcs * target_jual_per_hari)) if laba_per_pcs > 0 else 999
        
        total_modal_keseluruhan += modal
        total_potensi_laba += laba_total
        
        hasil_simulasi.append({
            "nama_produk": item.nama_produk,
            "konsentrasi": item.konsentrasi,
            "modal_total": modal,
            "dp_awal": round(dp_awal, 0),
            "pelunasan": round(pelunasan, 0),
            "jumlah_pcs": pcs,
            "hpp_satuan": round(hpp_satuan, 0),
            "harga_jual": harga,
            "target_jual_per_hari": target_jual_per_hari,
            "laba_per_pcs": round(laba_per_pcs, 0),
            "laba_total": round(laba_total, 0),
            "estimasi_hari_balik_modal": max(hari_balik_modal, 1),
            "rincian_ecommerce": {
                "shopee_tanpa_program": {
                    "potongan": round(pot_shopee_s1, 0),
                    "laba": round(laba_shopee_s1, 0)
                },
                "shopee_gox_9": {
                    "potongan": round(pot_shopee_s3, 0),
                    "laba": round(laba_shopee_s3, 0)
                },
                "tiktok_shop": {
                    "potongan": round(pot_tiktok, 0),
                    "laba": round(laba_tiktok, 0)
                }
            }
        })
        
    # Info Cicilan Bank ganda (BRI & Bank Jatim)
    pinjam_bri = body.pinjam_bri if body.pinjam_bri is not None else True
    tenor_bri = body.tenor_bri or 36
    
    pinjam_jatim = body.pinjam_jatim if body.pinjam_jatim is not None else True
    tenor_jatim = body.tenor_jatim or 36
    
    total_plafon = 0
    total_cair_bersih = 0
    total_cicilan_bulan = 0
    rincian_pinjaman_teks = []
    
    if pinjam_bri:
        plafon_bri = 50_000_000
        cair_bri = 47_000_000 # dipotong 3 Juta
        cicilan_map_bri = {12: 4326337, 18: 2934249, 24: 2238629, 36: 1543855, 48: 1197312, 60: 990060}
        cici_bri = cicilan_map_bri.get(tenor_bri, 1543855)
        total_plafon += plafon_bri
        total_cair_bersih += cair_bri
        total_cicilan_bulan += cici_bri
        rincian_pinjaman_teks.append(f"• KUR BRI Plafon Rp 50 Juta (Dipotong Rp 3 Juta, Cair Bersih: Rp{cair_bri:,.0f}) | Tenor {tenor_bri} Bln | Cicilan: Rp{cici_bri:,.0f}/bln")
        
    if pinjam_jatim:
        plafon_jatim = 25_000_000
        cair_jatim = 22_000_000 # Plafon 25 Juta dipotong 3 Juta = Cair 22 Juta
        cicilan_map_jatim = {12: 2150000, 24: 1120000, 36: 760548, 48: 587126}
        cici_jatim = cicilan_map_jatim.get(tenor_jatim, 760548)
        total_plafon += plafon_jatim
        total_cair_bersih += cair_jatim
        total_cicilan_bulan += cici_jatim
        rincian_pinjaman_teks.append(f"• KUR Bank Jatim Plafon Rp 25 Juta (Dipotong Rp 3 Juta, Cair Bersih: Rp{cair_jatim:,.0f}) | Tenor {tenor_jatim} Bln | Cicilan: Rp{cici_jatim:,.0f}/bln")

    # Pisahkan komponen biaya: 
    # - Modal Maklon: murni akumulasi dari input modal total seluruh varian (EDP + Extrait)
    # - Operasional: akumulasi dari input tambahan (box, botol, BPOM, halal, marketing, dll)
    biaya_operasional_murni = body.biaya_tambahan or 0.0
    legalitas_box_botol = body.legalitas_box_botol or 0.0
    
    # Total Modal Maklon = Akumulasi modal varian (EDP/Extrait) + Box/Botol + BPOM & Halal
    total_modal_maklon_riil = total_modal_keseluruhan + legalitas_box_botol
    
    # Total Kebutuhan Dana Keseluruhan = Modal Maklon + Operasional (Marketing, Darurat, Alat, dll)
    total_kebutuhan_dana = total_modal_maklon_riil + biaya_operasional_murni

    saran_ai_analisis = f" LAPORAN STRATEGIS & KELAYAKAN USAHA OLEH AI\n\n"
    if rincian_pinjaman_teks:
        saran_ai_analisis += "1. Rencana Pendanaan Modal:\n"
        for r in rincian_pinjaman_teks:
            saran_ai_analisis += f"   {r}\n"
        saran_ai_analisis += f"   • Akumulasi Plafon Pinjaman: Rp{total_plafon:,.0f}\n"
        saran_ai_analisis += f"   • Total Dana Bersih Diterima: Rp{total_cair_bersih:,.0f}\n"
        saran_ai_analisis += f"   • Total Kewajiban Cicilan: Rp{total_cicilan_bulan:,.0f} per bulan\n\n"
    else:
        saran_ai_analisis += "1. Rencana Pendanaan: Menggunakan Modal Mandiri (Tanpa Pinjaman Bank).\n\n"

    if rincian_pinjaman_teks:
        if total_kebutuhan_dana > total_cair_bersih:
            selisih = total_kebutuhan_dana - total_cair_bersih
            saran_ai_analisis += f"2. Status Kecukupan Modal: PERINGATAN DEFISIT! Total kebutuhan modal usaha Anda (Rp{total_kebutuhan_dana:,.0f}) melebihi dana pinjaman cair bersih (Rp{total_cair_bersih:,.0f}) sebesar Rp{selisih:,.0f}. Anda wajib menyiapkan tambahan modal pribadi.\n\n"
        else:
            sisa_kas = total_cair_bersih - total_kebutuhan_dana
            saran_ai_analisis += f"2. Status Kecukupan Modal: AMAN & TERCOVER. Total kebutuhan modal usaha Anda sebesar Rp{total_kebutuhan_dana:,.0f} (Maklon + Operasional) sudah tertutupi penuh oleh pinjaman bank. Anda bahkan memiliki sisa kas cadangan awal sebesar Rp{sisa_kas:,.0f} untuk modal kerja.\n\n"

    saran_ai_analisis += (
        f"3. Proyeksi Beban & Profitabilitas:\n"
        f"   - Beban Cicilan Rutin: Rp{total_cicilan_bulan:,.0f} / bulan.\n"
        f"   - Proyeksi Potensi Laba Bersih: Rp{total_potensi_laba:,.0f} (dari {len(body.items)} varian produk jika seluruh stok batch pertama habis terjual).\n"
        f"   - Target Penjualan Ideal: {total_target_jual_per_hari} botol per hari.\n\n"
        f"4. Rekomendasi Eksekusi & Pengamanan Kas:\n"
        f"   - Pastikan Anda membuat rekening terpisah untuk operasional dan cicilan bank.\n"
        f"   - Amankan minimal 35% dari omzet penjualan harian setiap harinya khusus untuk membayar cicilan bulanan bank sebelum Anda memutar kembali modal untuk re-order stok."
    )

    return {
        "status": "sukses",
        "total_modal_keseluruhan": round(total_modal_maklon_riil, 0),
        "total_biaya_tambahan": round(biaya_operasional_murni, 0),
        "total_modal_usaha_keseluruhan": round(total_kebutuhan_dana, 0),
        "total_potensi_laba": round(total_potensi_laba, 0),
        "total_target_jual_per_hari": total_target_jual_per_hari,
        "cicilan_bank_per_bulan": total_cicilan_bulan,
        "info_bank": {
            "nama": "KUR Gabungan BRI (Rp 50 Juta) & Bank Jatim (Rp 25 Juta)",
            "plafon": total_plafon,
            "cair_bersih": total_cair_bersih,
            "rincian": rincian_pinjaman_teks
        },
        "saran_ai_analisis": saran_ai_analisis,
        "hasil_simulasi": hasil_simulasi
    }




@router.get("/katalog-maklon/rekomendasi")
def rekomendasi_katalog_maklon(limit: int = 30):
    import os, pandas as pd, numpy as np
    from app.services.data import DATA_DIR
    path = os.path.join(DATA_DIR, "katalog_x_shopee.csv")
    if not os.path.exists(path):
        return {"status": "gagal", "pesan": f"File katalog_x_shopee.csv tidak ditemukan di {path}."}
    
    try:
        df = pd.read_csv(path, sep=";", on_bad_lines="skip")
        if len(df.columns) <= 1:
            df = pd.read_csv(path, sep=",", on_bad_lines="skip")
    except Exception as e:
        return {"status": "gagal", "pesan": str(e)}

    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    hasil = df.head(limit).to_dict(orient="records")
    return {
        "status": "sukses",
        "total": len(df),
        "rekomendasi": hasil
    }

@router.get("/katalog-maklon/peluang-belum")
def peluang_belum_di_katalog(limit: int = 30):
    import os, pandas as pd, numpy as np
    from app.services.data import DATA_DIR
    path = os.path.join(DATA_DIR, "peluang_aroma_belum_di_katalog.csv")
    if not os.path.exists(path):
        return {"status": "gagal", "pesan": f"File peluang_aroma_belum_di_katalog.csv tidak ditemukan di {path}."}
    
    try:
        df = pd.read_csv(path, sep=";", on_bad_lines="skip")
        if len(df.columns) <= 1:
            df = pd.read_csv(path, sep=",", on_bad_lines="skip")
    except Exception as e:
        return {"status": "gagal", "pesan": str(e)}

    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    hasil = df.head(limit).to_dict(orient="records")
    return {
        "status": "sukses",
        "total": len(df),
        "peluang": hasil
    }

@router.get("/kalibrasi")
def spk_kalibrasi(dupe: str | None = None, terjual: float = 14, hari: float = 14, target_laba_hari: float = 250000, target_balik_modal_hari: int = 180, lead_hari: int = 30):
    res = engine.rekomendasi(dupe_dari=dupe, target_laba_hari=target_laba_hari, target_balik_modal_hari=target_balik_modal_hari, modal_maks=30000000, lead_hari=lead_hari)
    res.pop("dataset", None)
    if not res["rekomendasi"]:
        return {"catatan": res["catatan"]}
    return engine.kalibrasi(res["rekomendasi"][0], terjual, hari, target_laba_hari, target_balik_modal_hari, lead_hari)


