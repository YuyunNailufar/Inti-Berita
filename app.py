import os
import re
import math
import json
import requests
import numpy as np
from collections import defaultdict
from flask import Flask, request, jsonify, render_template
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "best_mt5_model")
NEWS_API_KEY = "440da492f3384e9b8f6aa5d8be8c8ae4"

# Length presets
LENGTH_PRESETS = {
    1: {"max_new_tokens": 60,  "min_new_tokens": 20, "extractive_sentences": 2},
    2: {"max_new_tokens": 120, "min_new_tokens": 40, "extractive_sentences": 3},
    3: {"max_new_tokens": 200, "min_new_tokens": 70, "extractive_sentences": 5},
}

# ─── Model Loading ────────────────────────────────────────────────────────────
print("Loading mT5 model…")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device}")
except Exception as e:
    print(f"[WARN] Model load failed: {e}")
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
            length_penalty=1.5,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.strip()


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
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RingkasKilat/1.0)"}
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


@app.route("/api/summarize", methods=["POST"])
def summarize():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    length_level = int(data.get("length", 2))  # 1, 2, 3
    focus_keywords = data.get("focusKeywords", False)

    if not text or len(text) < 50:
        return jsonify({"error": "Teks terlalu pendek (min. 50 karakter)."}), 400

    preset = LENGTH_PRESETS.get(length_level, LENGTH_PRESETS[2])

    # If focusKeywords, extract top keywords and prepend to text for better context
    if focus_keywords:
        words = tokenize_rouge(text)
        freq = defaultdict(int)
        stopwords_id = {
            "yang","dan","di","ke","dari","dengan","untuk","ini","itu","pada",
            "adalah","akan","telah","dalam","oleh","tidak","ada","juga","sudah",
            "saat","bisa","agar","serta","karena","tetapi","namun","seperti",
            "setelah","sebelum","antara","lebih","sangat","hanya","tersebut",
            "mereka","kami","kita","saya","anda","dia","ia","nya","nya","an",
        }
        for w in words:
            if w not in stopwords_id and len(w) > 3:
                freq[w] += 1
        top_kw = sorted(freq, key=freq.get, reverse=True)[:8]
        keyword_prefix = "Kata kunci: " + ", ".join(top_kw) + ". "
        input_text = keyword_prefix + text
    else:
        input_text = text
        top_kw = []

    # Abstractive
    abstractive_text = mt5_summarize(
        input_text,
        max_new_tokens=preset["max_new_tokens"],
        min_new_tokens=preset["min_new_tokens"],
    )

    # Extractive
    extractive_text, extractive_sentences = textrank_summarize(
        text, num_sentences=preset["extractive_sentences"]
    )

    # Metrics (compare both summaries against each other as proxy, since no reference)
    # Also compute self-metrics vs original (first 512 chars as reference)
    reference = " ".join(text.split()[:100])  # first 100 words as reference proxy
    abs_metrics = compute_metrics(abstractive_text, reference)
    ext_metrics = compute_metrics(extractive_text, reference)

    # Word count
    abs_words = len(abstractive_text.split())
    ext_words = len(extractive_text.split())
    orig_words = len(text.split())
    compression_abs = round((1 - abs_words / max(orig_words, 1)) * 100, 1)
    compression_ext = round((1 - ext_words / max(orig_words, 1)) * 100, 1)

    return jsonify({
        "abstractive": {
            "text": abstractive_text,
            "wordCount": abs_words,
            "compression": compression_abs,
            "metrics": abs_metrics,
        },
        "extractive": {
            "text": extractive_text,
            "sentences": extractive_sentences,
            "wordCount": ext_words,
            "compression": compression_ext,
            "metrics": ext_metrics,
        },
        "keywords": top_kw,
        "originalWordCount": orig_words,
    })


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


if __name__ == "__main__":
    app.run(debug=True, port=5000)