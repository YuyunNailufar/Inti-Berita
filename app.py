import os
import re
import math
import json
import requests
import numpy as np
from collections import defaultdict
from pathlib import Path
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TextIteratorStreamer
import io
import PyPDF2
from docx import Document
from threading import Thread
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib import colors

load_dotenv()  # baca file .env di folder yang sama dengan app.py

app = Flask(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
# Pakai Path(__file__).resolve().parent agar path selalu relatif ke file ini,
# bukan ke working directory saat server dijalankan.
MODEL_DIR = str(Path(__file__).resolve().parent / "best_mt5_model")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")

# Length presets
# max_new_tokens dibuat sedikit lebih besar dari kebutuhan ideal, supaya model
# punya ruang menyelesaikan kalimat terakhir sebelum dipotong oleh token limit.
# trim_to_complete_sentence() lalu memotong sisa kalimat yang tidak lengkap.
LENGTH_PRESETS = {
    1: {"max_new_tokens": 110, "min_new_tokens": 30,  "extractive_sentences": 2},
    2: {"max_new_tokens": 190, "min_new_tokens": 60,  "extractive_sentences": 3},
    3: {"max_new_tokens": 280, "min_new_tokens": 100, "extractive_sentences": 5},
}

# ─── Model Loading ────────────────────────────────────────────────────────────
print(f"Loading mT5 model dari: {MODEL_DIR}")
try:
    if not Path(MODEL_DIR).exists():
        raise FileNotFoundError(f"Folder model tidak ditemukan: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✓ Model berhasil dimuat pada {device}")
except Exception as e:
    import traceback
    print(f"[ERROR] Model load gagal: {e}")
    traceback.print_exc()
    tokenizer = None
    model = None
    device = "cpu"


# ─── TextRank (Extractive) ────────────────────────────────────────────────────
def preprocess_sentence(sentence: str) -> list[str]:
    sentence = sentence.lower()
    sentence = re.sub(r"[^\w\s]", "", sentence)
    return sentence.split()


def build_similarity_matrix(sentences: list[str]) -> np.ndarray:
    n = len(sentences)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            words_i = set(preprocess_sentence(sentences[i]))
            words_j = set(preprocess_sentence(sentences[j]))
            if not words_i or not words_j:
                continue
            intersection = words_i & words_j
            denom = math.log(len(words_i) + 1) + math.log(len(words_j) + 1)
            sim_matrix[i][j] = len(intersection) / denom if denom else 0
    return sim_matrix


def textrank_summarize(text: str, num_sentences: int = 3) -> tuple[str, list[str]]:
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if len(sentences) <= num_sentences:
        return " ".join(sentences), sentences

    sim_matrix = build_similarity_matrix(sentences)

    # Power iteration (PageRank)
    scores = np.ones(len(sentences)) / len(sentences)
    damping = 0.85
    for _ in range(100):
        new_scores = np.zeros(len(sentences))
        for i in range(len(sentences)):
            col_sum = sim_matrix[:, i].sum()
            if col_sum == 0:
                new_scores[i] = (1 - damping) / len(sentences)
            else:
                new_scores[i] = (1 - damping) / len(sentences) + damping * np.dot(
                    scores, sim_matrix[:, i] / col_sum
                )
        if np.allclose(scores, new_scores, atol=1e-6):
            break
        scores = new_scores

    ranked_indices = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    top_indices = sorted(ranked_indices[:num_sentences])
    selected = [sentences[i] for i in top_indices]
    return " ".join(selected), selected


# ─── mT5 (Abstractive) ───────────────────────────────────────────────────────
def trim_to_complete_sentence(text: str) -> str:
    """Potong teks supaya berakhir di kalimat lengkap terakhir (., !, ?).
    Dipakai baik di jalur non-streaming maupun streaming agar konsisten.
    Hanya memotong kalau memang ada kalimat lengkap di awal (last_end > 20);
    kalau tidak ada sama sekali, biarkan teks apa adanya — lebih baik
    daripada mengembalikan string kosong."""
    text = text.strip()
    if text and text[-1] not in '.!?':
        last_end = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_end > 20:
            text = text[:last_end + 1]
    return text


def mt5_summarize(text: str, max_new_tokens: int = 120, min_new_tokens: int = 40) -> str:
    if model is None or tokenizer is None:
        return "[Model tidak tersedia. Pastikan folder best_mt5_model ada dan valid.]"

    prefix = "summarize: "
    inputs = tokenizer(
        prefix + text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=4,
            length_penalty=2.0,       # dorong model hasilkan kalimat lebih lengkap
            no_repeat_ngram_size=2,   # turunkan dari 3 agar tidak terlalu ketat
            early_stopping=True,
            forced_eos_token_id=tokenizer.eos_token_id,
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return trim_to_complete_sentence(summary)

def mt5_summarize_stream(text: str, max_new_tokens: int = 120, min_new_tokens: int = 40):
    if model is None or tokenizer is None:
        yield "[Model tidak tersedia. Pastikan folder best_mt5_model ada dan valid.]"
        return

    prefix = "summarize: "
    inputs = tokenizer(
        prefix + text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        num_beams=1,
        length_penalty=2.0,
        no_repeat_ngram_size=2,
        early_stopping=True,
        forced_eos_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text


# ─── ROUGE Score (lightweight, no dependencies) ──────────────────────────────
def tokenize_rouge(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def get_ngrams(tokens: list[str], n: int) -> defaultdict:
    ngrams = defaultdict(int)
    for i in range(len(tokens) - n + 1):
        ngrams[tuple(tokens[i : i + n])] += 1
    return ngrams


def rouge_n(hypothesis: str, reference: str, n: int) -> dict:
    hyp_tokens = tokenize_rouge(hypothesis)
    ref_tokens = tokenize_rouge(reference)
    hyp_ngrams = get_ngrams(hyp_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)
    overlap = sum(min(hyp_ngrams[k], ref_ngrams[k]) for k in hyp_ngrams)
    precision = overlap / max(sum(hyp_ngrams.values()), 1)
    recall = overlap / max(sum(ref_ngrams.values()), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def rouge_l(hypothesis: str, reference: str) -> dict:
    hyp_tokens = tokenize_rouge(hypothesis)
    ref_tokens = tokenize_rouge(reference)
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / max(n, 1)
    recall = lcs / max(m, 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def compute_metrics(hypothesis: str, reference: str) -> dict:
    r1 = rouge_n(hypothesis, reference, 1)
    r2 = rouge_n(hypothesis, reference, 2)
    rl = rouge_l(hypothesis, reference)
    return {
        "rouge1": r1["f1"],
        "rouge2": r2["f1"],
        "rougeL": rl["f1"],
        "precision": r1["precision"],
        "recall": r1["recall"],
    }


# ─── Web Scraping ─────────────────────────────────────────────────────────────
def scrape_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; IntiBerita/1.0)"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove nav, footer, ads
    for tag in soup(["script", "style", "nav", "footer", "aside", "header", "figure"]):
        tag.decompose()
    # Try article body first
    article = soup.find("article") or soup.find("div", class_=re.compile(r"content|body|article", re.I))
    if article:
        paragraphs = article.find_all("p")
    else:
        paragraphs = soup.find_all("p")
    text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs if len(p.get_text()) > 40)
    return text.strip()


# ─── Document Extraction ──────────────────────────────────────────────────────
def extract_text_from_pdf(file_stream) -> str:
    try:
        reader = PyPDF2.PdfReader(file_stream)
        text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
        return "\n".join(text).strip()
    except Exception as e:
        print(f"[ERROR] PDF extraction failed: {e}")
        return ""

def extract_text_from_docx(file_stream) -> str:
    try:
        doc = Document(file_stream)
        text = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(text).strip()
    except Exception as e:
        print(f"[ERROR] DOCX extraction failed: {e}")
        return ""


# ─── News API ─────────────────────────────────────────────────────────────────
def search_news(query: str) -> list[dict]:
    if not NEWS_API_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "id",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        data = resp.json()
        articles = data.get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "source": a.get("source", {}).get("name", ""),
                "publishedAt": a.get("publishedAt", ""),
                "description": a.get("description", ""),
            }
            for a in articles
            if a.get("url")
        ]
    except Exception:
        return []


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah."}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "File tidak valid."}), 400
    
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file.stream)
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(file.stream)
    else:
        return jsonify({"error": "Format file tidak didukung. Harap unggah PDF atau DOCX."}), 400
    
    if not text or len(text) < 50:
        return jsonify({"error": "Gagal mengekstrak teks atau teks terlalu pendek."}), 400
        
    word_count = len(text.split())
    read_time = max(1, round(word_count / 200))
    return jsonify({"text": text, "wordCount": word_count, "readTime": read_time})

@app.route("/api/summarize", methods=["POST"])
def summarize():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    length_level = int(data.get("length", 2))  # 1, 2, 3
    focus_keywords = data.get("focusKeywords", False)
    method = data.get("method", "both")  # both | abstractive | extractive

    if not text or len(text) < 50:
        return jsonify({"error": "Teks terlalu pendek (min. 50 karakter)."}), 400

    preset = LENGTH_PRESETS.get(length_level, LENGTH_PRESETS[2])
    orig_words = len(text.split())
    reference = " ".join(text.split()[:100])

    # Keyword extraction (hanya jika dibutuhkan oleh abstractive)
    top_kw = []
    input_text = text
    if focus_keywords and method in ("both", "abstractive"):
        words = tokenize_rouge(text)
        freq = defaultdict(int)
        stopwords_id = {
            "yang","dan","di","ke","dari","dengan","untuk","ini","itu","pada",
            "adalah","akan","telah","dalam","oleh","tidak","ada","juga","sudah",
            "saat","bisa","agar","serta","karena","tetapi","namun","seperti",
            "setelah","sebelum","antara","lebih","sangat","hanya","tersebut",
            "mereka","kami","kita","saya","anda","dia","ia","nya","an",
        }
        for w in words:
            if w not in stopwords_id and len(w) > 3:
                freq[w] += 1
        top_kw = sorted(freq, key=freq.get, reverse=True)[:8]
        input_text = "Kata kunci: " + ", ".join(top_kw) + ". " + text

    result = {"keywords": top_kw, "originalWordCount": orig_words}

    # ── Abstractive (hanya jika method = both atau abstractive)
    if method in ("both", "abstractive"):
        abstractive_text = mt5_summarize(
            input_text,
            max_new_tokens=preset["max_new_tokens"],
            min_new_tokens=preset["min_new_tokens"],
        )
        abs_words = len(abstractive_text.split())
        result["abstractive"] = {
            "text": abstractive_text,
            "wordCount": abs_words,
            "compression": round((1 - abs_words / max(orig_words, 1)) * 100, 1),
            "metrics": compute_metrics(abstractive_text, reference),
        }

    # ── Extractive (hanya jika method = both atau extractive)
    if method in ("both", "extractive"):
        extractive_text, extractive_sentences = textrank_summarize(
            text, num_sentences=preset["extractive_sentences"]
        )
        ext_words = len(extractive_text.split())
        result["extractive"] = {
            "text": extractive_text,
            "sentences": extractive_sentences,
            "wordCount": ext_words,
            "compression": round((1 - ext_words / max(orig_words, 1)) * 100, 1),
            "metrics": compute_metrics(extractive_text, reference),
        }

    return jsonify(result)

@app.route("/api/summarize-stream", methods=["POST"])
def summarize_stream():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    length_level = int(data.get("length", 2))
    focus_keywords = data.get("focusKeywords", False)
    method = data.get("method", "both")

    if not text or len(text) < 50:
        return jsonify({"error": "Teks terlalu pendek (min. 50 karakter)."}), 400

    preset = LENGTH_PRESETS.get(length_level, LENGTH_PRESETS[2])
    orig_words = len(text.split())
    reference = " ".join(text.split()[:100])

    top_kw = []
    input_text = text
    if focus_keywords and method in ("both", "abstractive"):
        words = tokenize_rouge(text)
        freq = defaultdict(int)
        stopwords_id = {
            "yang","dan","di","ke","dari","dengan","untuk","ini","itu","pada",
            "adalah","akan","telah","dalam","oleh","tidak","ada","juga","sudah",
            "saat","bisa","agar","serta","karena","tetapi","namun","seperti",
            "setelah","sebelum","antara","lebih","sangat","hanya","tersebut",
            "mereka","kami","kita","saya","anda","dia","ia","nya","an",
        }
        for w in words:
            if w not in stopwords_id and len(w) > 3:
                freq[w] += 1
        top_kw = sorted(freq, key=freq.get, reverse=True)[:8]
        input_text = "Kata kunci: " + ", ".join(top_kw) + ". " + text

    def generate():
        init_data = {"type": "init", "keywords": top_kw, "originalWordCount": orig_words}
        
        if method in ("both", "extractive"):
            extractive_text, extractive_sentences = textrank_summarize(text, num_sentences=preset["extractive_sentences"])
            ext_words = len(extractive_text.split())
            init_data["extractive"] = {
                "text": extractive_text,
                "sentences": extractive_sentences,
                "wordCount": ext_words,
                "compression": round((1 - ext_words / max(orig_words, 1)) * 100, 1),
                "metrics": compute_metrics(extractive_text, reference),
            }
            
        yield f"data: {json.dumps(init_data)}\n\n"

        if method in ("both", "abstractive"):
            full_abs_text = ""
            for chunk in mt5_summarize_stream(input_text, preset["max_new_tokens"], preset["min_new_tokens"]):
                full_abs_text += chunk
                chunk_data = {"type": "chunk", "text": chunk}
                yield f"data: {json.dumps(chunk_data)}\n\n"
            
            full_abs_text = trim_to_complete_sentence(full_abs_text)

            abs_words = len(full_abs_text.split())
            final_data = {
                "type": "done",
                "abstractive": {
                    "text": full_abs_text,
                    "wordCount": abs_words,
                    "compression": round((1 - abs_words / max(orig_words, 1)) * 100, 1),
                    "metrics": compute_metrics(full_abs_text, reference)
                }
            }
            yield f"data: {json.dumps(final_data)}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/scrape", methods=["POST"])
def scrape():
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL tidak boleh kosong."}), 400
    try:
        text = scrape_url(url)
        if not text:
            return jsonify({"error": "Tidak dapat mengekstrak teks dari URL ini."}), 400
        word_count = len(text.split())
        read_time = max(1, round(word_count / 200))
        return jsonify({"text": text, "wordCount": word_count, "readTime": read_time})
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Gagal mengakses URL: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500


@app.route("/api/search-news", methods=["POST"])
def search_news_api():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Kata kunci tidak boleh kosong."}), 400
    if not NEWS_API_KEY:
        return jsonify({"error": "NEWS_API_KEY belum dikonfigurasi di server."}), 503
    articles = search_news(query)
    return jsonify({"articles": articles})


@app.route("/api/word-count", methods=["POST"])
def word_count():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    words = len(text.split()) if text else 0
    read_time = max(1, round(words / 200))
    return jsonify({"wordCount": words, "readTime": read_time})


@app.route("/api/export-pdf", methods=["POST"])
def export_pdf():
    """
    Generate PDF asli di server (pakai reportlab) dan kirim sebagai file
    download langsung — TIDAK lewat dialog print browser sama sekali.
    Ini berbeda dari window.print() di frontend yang selalu memunculkan
    dialog OS/browser; endpoint ini menghasilkan file biner PDF yang
    langsung di-download begitu response diterima.
    """
    data = request.get_json(force=True)
    abstractive = data.get("abstractive")
    extractive = data.get("extractive")
    keywords = data.get("keywords") or []
    method = data.get("method", "both")

    if not abstractive and not extractive:
        return jsonify({"error": "Tidak ada data ringkasan untuk diekspor."}), 400

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=18, spaceAfter=12,
        textColor=colors.black,
    )
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], fontSize=13, spaceAfter=6,
        textColor=colors.black,
    )
    body_style = ParagraphStyle(
        "BodyJustify", parent=styles["Normal"], fontSize=11, leading=16,
        alignment=TA_JUSTIFY, spaceAfter=10,
    )
    meta_style = ParagraphStyle(
        "Meta", parent=styles["Normal"], fontSize=9, textColor=colors.grey,
        spaceAfter=14,
    )

    story = []
    story.append(Paragraph("Laporan Hasil Ringkasan Otomatis", title_style))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.black))
    story.append(Spacer(1, 14))

    if keywords:
        kw_text = "Kata kunci: " + ", ".join(keywords)
        story.append(Paragraph(kw_text, meta_style))

    # ── Abstraktif
    if abstractive and method in ("both", "abstractive"):
        story.append(Paragraph("Abstraktif &middot; mT5-small", heading_style))
        story.append(Paragraph(abstractive.get("text", ""), body_style))
        m = abstractive.get("metrics", {})
        meta_line = (
            f"Jumlah kata: {abstractive.get('wordCount', '—')} &nbsp;|&nbsp; "
            f"Kompresi: -{abstractive.get('compression', '—')}% &nbsp;|&nbsp; "
            f"ROUGE-1: {m.get('rouge1', '—')} &nbsp;ROUGE-2: {m.get('rouge2', '—')} "
            f"&nbsp;ROUGE-L: {m.get('rougeL', '—')}"
        )
        story.append(Paragraph(meta_line, meta_style))
        story.append(Spacer(1, 10))

    # ── Ekstraktif
    if extractive and method in ("both", "extractive"):
        story.append(Paragraph("Ekstraktif &middot; TextRank", heading_style))
        sentences = extractive.get("sentences") or [extractive.get("text", "")]
        for i, s in enumerate(sentences, 1):
            story.append(Paragraph(f"{i}. {s}", body_style))
        m = extractive.get("metrics", {})
        meta_line = (
            f"Jumlah kata: {extractive.get('wordCount', '—')} &nbsp;|&nbsp; "
            f"Kompresi: -{extractive.get('compression', '—')}% &nbsp;|&nbsp; "
            f"ROUGE-1: {m.get('rouge1', '—')} &nbsp;ROUGE-2: {m.get('rouge2', '—')} "
            f"&nbsp;ROUGE-L: {m.get('rougeL', '—')}"
        )
        story.append(Paragraph(meta_line, meta_style))

    # ── Tabel perbandingan (hanya jika kedua metode ada)
    if method == "both" and abstractive and extractive:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Perbandingan Metrik", heading_style))
        am = abstractive.get("metrics", {})
        em = extractive.get("metrics", {})
        table_data = [
            ["Metrik", "Abstraktif (mT5)", "Ekstraktif (TextRank)"],
            ["ROUGE-1", str(am.get("rouge1", "—")), str(em.get("rouge1", "—"))],
            ["ROUGE-2", str(am.get("rouge2", "—")), str(em.get("rouge2", "—"))],
            ["ROUGE-L", str(am.get("rougeL", "—")), str(em.get("rougeL", "—"))],
        ]
        table = Table(table_data, colWidths=[140, 160, 160])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#141b2b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f3ff")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(table)

    doc.build(story)
    buffer.seek(0)

    return Response(
        buffer.getvalue(),
        mimetype="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=laporan_ringkasan.pdf"
        },
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)