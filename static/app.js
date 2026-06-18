/**
 * app.js — RingkasKilat Frontend Logic
 * Handles: tab switching, input modes, summarize API, scrape API,
 *          news search API, result rendering, metrics bars, toast, copy, export.
 */

// ─── BASELINE METRICS (dari evaluasi training) ───────────────────────────────
const BASELINE = {
  textrank: { rouge1: 0.6623, rouge2: 0.5953, rougeL: 0.6293, bertscore: 0.8432 },
  mt5:      { rouge1: 0.5555, rouge2: 0.4684, rougeL: 0.5121, bertscore: 0.8113 },
};

// ─── STATE ────────────────────────────────────────────────────────────────────
const state = {
  activeTab: "paste",       // paste | url | search
  method: "both",           // both | abstractive | extractive
  lengthLevel: 2,           // 1 | 2 | 3
  focusKeywords: true,
  isLoading: false,
  lastResult: null,
};

// ─── DOM REFS ─────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const DOM = {
  // Tabs
  tabBtns:       () => document.querySelectorAll(".tab-btn"),
  panelPaste:    () => $("panel-paste"),
  panelUrl:      () => $("panel-url"),
  panelSearch:   () => $("panel-search"),

  // Paste panel
  textarea:      () => $("main-textarea"),
  wordCountEl:   () => $("word-count"),
  readTimeEl:    () => $("read-time"),

  // URL panel
  urlInput:      () => $("url-input"),
  urlFetchBtn:   () => $("url-fetch-btn"),
  urlStatus:     () => $("url-status"),

  // Search panel
  searchInput:   () => $("search-input"),
  searchBtn:     () => $("search-btn"),
  searchResults: () => $("search-results"),

  // Config
  slider:        () => $("length-slider"),
  sliderLabel:   () => $("slider-label"),
  toggleKw:      () => $("toggle-keywords"),
  methodBothBtn: () => $("method-both"),
  methodAbsBtn:  () => $("method-abs"),
  methodExtBtn:  () => $("method-ext"),

  // Action
  summarizeBtn:  () => $("summarize-btn"),
  scanLine:      () => $("scan-line"),

  // Results
  resultsSection: () => $("results-section"),
  absCard:        () => $("abs-card"),
  extCard:        () => $("ext-card"),
  absText:        () => $("abs-text"),
  extList:        () => $("ext-list"),
  absBadges:      () => $("abs-badges"),
  extBadges:      () => $("ext-badges"),
  absWordCount:   () => $("abs-word-count"),
  extWordCount:   () => $("ext-word-count"),
  absCompression: () => $("abs-compression"),
  extCompression: () => $("ext-compression"),
  keywordBadges:  () => $("keyword-badges"),
  keywordSection: () => $("keyword-section"),
  compTable:      () => $("comparison-table"),
  newBtn:         () => $("new-btn"),

  // Copy / Export
  copyAbsBtn:    () => $("copy-abs"),
  copyExtBtn:    () => $("copy-ext"),
  exportBtn:     () => $("export-btn"),
};

// ─── UTILS ────────────────────────────────────────────────────────────────────
function showToast(msg, type = "success") {
  const existing = document.querySelector(".toast");
  if (existing) existing.remove();

  const icon = type === "success" ? "check_circle" : "error";
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.innerHTML = `<span class="material-symbols-outlined text-[18px]">${icon}</span>${msg}`;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

function setLoading(on) {
  state.isLoading = on;
  const btn = DOM.summarizeBtn();
  if (!btn) return;
  if (on) {
    btn.disabled = true;
    btn.innerHTML = `
      <svg class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
      </svg>
      Memproses…`;
    DOM.scanLine()?.classList.add("active");
  } else {
    btn.disabled = false;
    btn.innerHTML = `<span class="material-symbols-outlined">auto_awesome</span>Ringkas Sekarang`;
    DOM.scanLine()?.classList.remove("active");
  }
}

function getInputText() {
  return (DOM.textarea()?.value || "").trim();
}

const SLIDER_LABELS = { 1: "Pendek", 2: "Sedang", 3: "Panjang" };

// ─── TAB SWITCHING ────────────────────────────────────────────────────────────
function initTabs() {
  DOM.tabBtns().forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      state.activeTab = tab;

      // Style active tab
      DOM.tabBtns().forEach((b) => {
        b.classList.remove("bg-white", "shadow-sm", "text-primary");
        b.classList.add("text-on-surface-variant");
      });
      btn.classList.add("bg-white", "shadow-sm", "text-primary");
      btn.classList.remove("text-on-surface-variant");

      // Show/hide panels
      ["paste", "url", "search"].forEach((t) => {
        const panel = $(`panel-${t}`);
        if (panel) panel.classList.toggle("hidden", t !== tab);
      });
    });
  });
}

// ─── TEXTAREA WORD COUNT ──────────────────────────────────────────────────────
function initTextarea() {
  const ta = DOM.textarea();
  if (!ta) return;
  ta.addEventListener("input", debounce(updateWordCount, 300));
}

function updateWordCount() {
  const text = getInputText();
  const words = text ? text.split(/\s+/).filter(Boolean).length : 0;
  const readTime = Math.max(1, Math.round(words / 200));
  if (DOM.wordCountEl()) DOM.wordCountEl().textContent = `${words} Kata`;
  if (DOM.readTimeEl())  DOM.readTimeEl().textContent  = `${readTime} Menit Baca`;
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// ─── METHOD RADIO ─────────────────────────────────────────────────────────────
function initMethodRadio() {
  const radios = document.querySelectorAll('input[name="method"]');
  const labels = {
    both:        DOM.methodBothBtn,
    abstractive: DOM.methodAbsBtn,
    extractive:  DOM.methodExtBtn,
  };

  function updateHighlight(selectedValue) {
    Object.entries(labels).forEach(([value, getLabel]) => {
      const label = getLabel();
      if (!label) return;
      if (value === selectedValue) {
        label.classList.add("border-primary", "bg-primary/5");
        label.classList.remove("border-outline-variant/30");
      } else {
        label.classList.remove("border-primary", "bg-primary/5");
        label.classList.add("border-outline-variant/30");
      }
    });
  }

  // Set initial state (both is checked by default)
  updateHighlight("both");

  radios.forEach((radio) => {
    radio.addEventListener("change", () => {
      state.method = radio.value;
      updateHighlight(radio.value);
    });
  });
}

// ─── SLIDER ───────────────────────────────────────────────────────────────────
function initSlider() {
  const slider = DOM.slider();
  if (!slider) return;
  slider.addEventListener("input", () => {
    state.lengthLevel = parseInt(slider.value);
    if (DOM.sliderLabel()) DOM.sliderLabel().textContent = SLIDER_LABELS[state.lengthLevel];
  });
}

// ─── KEYWORD TOGGLE ───────────────────────────────────────────────────────────
function initToggle() {
  const toggle = DOM.toggleKw();
  if (!toggle) return;
  toggle.addEventListener("click", () => {
    state.focusKeywords = !state.focusKeywords;
    toggle.classList.toggle("on", state.focusKeywords);
  });
}

// ─── URL FETCH ────────────────────────────────────────────────────────────────
function initUrlFetch() {
  DOM.urlFetchBtn()?.addEventListener("click", fetchUrl);
  DOM.urlInput()?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") fetchUrl();
  });
}

async function fetchUrl() {
  const url = (DOM.urlInput()?.value || "").trim();
  if (!url) { showToast("Masukkan URL terlebih dahulu.", "error"); return; }

  const btn = DOM.urlFetchBtn();
  const status = DOM.urlStatus();
  btn.disabled = true;
  btn.textContent = "Mengambil…";
  if (status) { status.textContent = ""; status.className = ""; }

  try {
    const res = await fetch("/api/scrape", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Gagal mengambil URL.");

    // Isi url-textarea dan tampilkan preview box
    const urlTextarea = $("url-textarea");
    if (urlTextarea) {
      urlTextarea.value = data.text;
      urlTextarea.removeAttribute("readonly");
    }
    $("url-preview-box")?.classList.remove("hidden");
    const urlWordCount = $("url-word-count");
    const urlReadTime  = $("url-read-time");
    if (urlWordCount) urlWordCount.textContent = `${data.wordCount} kata`;
    if (urlReadTime)  urlReadTime.textContent  = `${data.readTime} menit baca`;

    // Juga isi main textarea untuk fallback summarize
    DOM.textarea().value = data.text;
    updateWordCount();
    if (status) {
      status.textContent = `✓ Berhasil — ${data.wordCount} kata diambil`;
      status.className = "text-secondary text-label-sm font-medium";
    }
    showToast(`${data.wordCount} kata berhasil diambil dari URL.`);
  } catch (err) {
    if (status) {
      status.textContent = err.message;
      status.className = "text-error text-label-sm font-medium";
    }
    showToast(err.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Ambil Teks";
  }
}

// ─── NEWS SEARCH ──────────────────────────────────────────────────────────────
function initNewsSearch() {
  DOM.searchBtn()?.addEventListener("click", searchNews);
  DOM.searchInput()?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") searchNews();
  });
}

async function searchNews() {
  const query = (DOM.searchInput()?.value || "").trim();
  if (!query) { showToast("Masukkan kata kunci pencarian.", "error"); return; }

  const btn = DOM.searchBtn();
  const container = DOM.searchResults();
  btn.disabled = true;
  btn.textContent = "Mencari…";
  if (container) container.innerHTML = renderNewsSkeletons();

  try {
    const res = await fetch("/api/search-news", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Gagal mencari berita.");
    renderNewsResults(data.articles);
  } catch (err) {
    if (container) container.innerHTML = `<p class="text-error text-label-md p-2">${err.message}</p>`;
    showToast(err.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Cari";
  }
}

function renderNewsSkeletons() {
  return Array(3).fill(0).map(() => `
    <div class="news-card space-y-2">
      <div class="skeleton h-4 w-3/4"></div>
      <div class="skeleton h-3 w-full"></div>
      <div class="skeleton h-3 w-1/2"></div>
    </div>`).join("");
}

function renderNewsResults(articles) {
  const container = DOM.searchResults();
  if (!container) return;
  if (!articles.length) {
    container.innerHTML = `<p class="text-on-surface-variant text-label-md p-2 text-center">Tidak ada berita ditemukan.</p>`;
    return;
  }
  container.innerHTML = articles.map((a) => `
    <div class="news-card" data-url="${a.url}">
      <p class="font-label-md text-label-md text-on-surface line-clamp-2 mb-1">${a.title}</p>
      <p class="text-xs text-on-surface-variant line-clamp-2 mb-2">${a.description || ""}</p>
      <div class="flex items-center justify-between">
        <span class="text-[10px] text-on-surface-variant/70">${a.source} · ${formatDate(a.publishedAt)}</span>
        <button class="use-news-btn text-primary text-xs font-semibold hover:underline flex items-center gap-1">
          <span class="material-symbols-outlined text-[14px]">open_in_new</span>Gunakan
        </button>
      </div>
    </div>`).join("");

  // Attach click handlers
  container.querySelectorAll(".use-news-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const card = btn.closest("[data-url]");
      const url = card?.dataset.url;
      if (url) {
        DOM.urlInput().value = url;
        // Switch to URL tab
        document.querySelector('[data-tab="url"]')?.click();
        fetchUrl();
      }
    });
  });
}

function formatDate(iso) {
  if (!iso) return "";
  try { return new Date(iso).toLocaleDateString("id-ID", { day: "numeric", month: "short", year: "numeric" }); }
  catch { return ""; }
}

// ─── SUMMARIZE ────────────────────────────────────────────────────────────────
function initSummarize() {
  DOM.summarizeBtn()?.addEventListener("click", doSummarize);
  // Tombol "Ringkas Sekarang" di panel URL
  $("summarize-btn-url")?.addEventListener("click", () => {
    const urlText = $("url-textarea")?.value?.trim();
    if (urlText) {
      DOM.textarea().value = urlText;
      updateWordCount();
    }
    doSummarize();
  });
  DOM.newBtn()?.addEventListener("click", resetUI);
}

async function doSummarize() {
  const text = getInputText();
  if (!text || text.length < 50) {
    showToast("Teks terlalu pendek (min. 50 karakter).", "error");
    return;
  }
  if (state.isLoading) return;

  setLoading(true);
  showSkeletonResults();

  try {
    const res = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        length: state.lengthLevel,
        focusKeywords: state.focusKeywords,
        method: state.method,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Gagal memproses ringkasan.");

    state.lastResult = data;
    renderResults(data);

    DOM.resultsSection()?.scrollIntoView({ behavior: "smooth", block: "start" });
    showToast("Ringkasan berhasil dibuat!");
  } catch (err) {
    showToast(err.message, "error");
    hideSkeletonResults();
  } finally {
    setLoading(false);
  }
}

// ─── SKELETON ─────────────────────────────────────────────────────────────────
function showSkeletonResults() {
  const section = DOM.resultsSection();
  if (!section) return;
  section.classList.remove("hidden");

  const m = state.method;

  // Show/hide cards berdasarkan metode dipilih
  DOM.absCard()?.classList.toggle("hidden", m === "extractive");
  DOM.extCard()?.classList.toggle("hidden", m === "abstractive");

  if (m !== "extractive" && DOM.absText()) DOM.absText().innerHTML = `
    <div class="space-y-2">
      <div class="skeleton h-4 w-full"></div>
      <div class="skeleton h-4 w-5/6"></div>
      <div class="skeleton h-4 w-4/6"></div>
    </div>`;

  if (m !== "abstractive" && DOM.extList()) DOM.extList().innerHTML = `
    <div class="space-y-3">
      ${Array(3).fill(0).map(() => `
        <div class="flex gap-3">
          <div class="skeleton h-6 w-6 rounded-full flex-shrink-0"></div>
          <div class="skeleton h-4 w-full rounded"></div>
        </div>`).join("")}
    </div>`;
}

function hideSkeletonResults() {
  DOM.resultsSection()?.classList.add("hidden");
}

// ─── RENDER RESULTS ───────────────────────────────────────────────────────────
function renderResults(data) {
  const { abstractive, extractive, keywords } = data;
  const m = state.method;

  // ── Show/hide cards sesuai metode
  DOM.absCard()?.classList.toggle("hidden", m === "extractive");
  DOM.extCard()?.classList.toggle("hidden", m === "abstractive");

  // ── Abstractive text (hanya kalau ada di response)
  if (abstractive && DOM.absText()) {
    DOM.absText().textContent = abstractive.text;
    DOM.absText().classList.add("fade-in");
    renderMetricBadges(DOM.absBadges(), abstractive.metrics, "abs");
    if (DOM.absWordCount())   DOM.absWordCount().textContent   = `${abstractive.wordCount} kata`;
    if (DOM.absCompression()) DOM.absCompression().textContent = `−${abstractive.compression}%`;
  }

  // ── Extractive sentences list (hanya kalau ada di response)
  if (extractive && DOM.extList()) {
    DOM.extList().innerHTML = (extractive.sentences || [extractive.text]).map((s, i) => `
      <div class="sentence-item fade-in" style="animation-delay:${i * 80}ms">
        <span class="sentence-number">${i + 1}</span>
        <p class="font-body-md text-body-md text-on-surface leading-relaxed">${s}</p>
      </div>`).join("");
    renderMetricBadges(DOM.extBadges(), extractive.metrics, "ext");
    if (DOM.extWordCount())   DOM.extWordCount().textContent   = `${extractive.wordCount} kata`;
    if (DOM.extCompression()) DOM.extCompression().textContent = `−${extractive.compression}%`;
  }

  // ── Keywords
  if (keywords?.length && DOM.keywordSection()) {
    DOM.keywordSection().classList.remove("hidden");
    DOM.keywordBadges().innerHTML = keywords.map((kw) =>
      `<span class="px-3 py-1 rounded-full bg-primary/10 text-primary text-label-sm font-semibold">${kw}</span>`
    ).join("");
  }

  // ── Comparison table — hanya tampil kalau keduanya ada
  const compTableWrapper = DOM.compTable()?.closest(".bg-white.rounded-xl.border");
  if (m === "both" && abstractive && extractive) {
    renderComparisonTable(abstractive.metrics, extractive.metrics);
    compTableWrapper?.classList.remove("hidden");
  } else {
    compTableWrapper?.classList.add("hidden");
  }

  DOM.resultsSection()?.classList.remove("hidden");
}

function renderMetricBadges(container, metrics, type) {
  if (!container || !metrics) return;

  // ROUGE badges
  const badges = [
    { label: "R-1", value: metrics.rouge1 },
    { label: "R-2", value: metrics.rouge2 },
    { label: "R-L", value: metrics.rougeL },
  ];

  // Also show baseline comparison tooltip
  const baseline = type === "abs" ? BASELINE.mt5 : BASELINE.textrank;

  container.innerHTML = badges.map((b) => {
    const bKey = b.label === "R-1" ? "rouge1" : b.label === "R-2" ? "rouge2" : "rougeL";
    const diff = b.value - baseline[bKey];
    const diffStr = diff >= 0 ? `+${diff.toFixed(3)}` : diff.toFixed(3);
    const diffColor = diff >= 0 ? "text-secondary" : "text-error";
    return `
      <div class="metric-badge" title="Baseline: ${baseline[bKey]}">
        <span class="label">${b.label}</span>
        <span class="value">${b.value?.toFixed(3) ?? "—"}</span>
        <span class="text-[9px] ${diffColor} font-bold">${diffStr}</span>
      </div>`;
  }).join("");
}

function renderComparisonTable(absMetrics, extMetrics) {
  const table = DOM.compTable();
  if (!table) return;

  const rows = [
    { label: "ROUGE-1",   abs: absMetrics.rouge1, ext: extMetrics.rouge1, baseAbs: BASELINE.mt5.rouge1,   baseExt: BASELINE.textrank.rouge1 },
    { label: "ROUGE-2",   abs: absMetrics.rouge2, ext: extMetrics.rouge2, baseAbs: BASELINE.mt5.rouge2,   baseExt: BASELINE.textrank.rouge2 },
    { label: "ROUGE-L",   abs: absMetrics.rougeL, ext: extMetrics.rougeL, baseAbs: BASELINE.mt5.rougeL,   baseExt: BASELINE.textrank.rougeL },
    { label: "Precision", abs: absMetrics.precision, ext: extMetrics.precision, baseAbs: null, baseExt: null },
    { label: "Recall",    abs: absMetrics.recall,    ext: extMetrics.recall,    baseAbs: null, baseExt: null },
  ];

  table.innerHTML = `
    <table class="w-full text-sm border-collapse">
      <thead>
        <tr class="border-b border-outline-variant/30">
          <th class="text-left py-2 px-3 font-label-md text-label-md text-on-surface-variant">Metrik</th>
          <th class="text-center py-2 px-3 font-label-md text-label-md text-primary">mT5 (Abstraktif)</th>
          <th class="text-center py-2 px-3 font-label-md text-label-md text-on-surface-variant">TextRank (Ekstraktif)</th>
          <th class="text-center py-2 px-3 font-label-md text-label-md text-on-surface-variant/60 hidden md:table-cell">Winner</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map((r) => {
          const absVal = r.abs ?? 0;
          const extVal = r.ext ?? 0;
          const absWins = absVal >= extVal;
          return `
            <tr class="border-b border-outline-variant/20 hover:bg-surface-container-low transition-colors">
              <td class="py-2 px-3 font-label-md text-label-md text-on-surface">${r.label}</td>
              <td class="py-2 px-3 text-center ${absWins ? "comparison-winner font-bold text-secondary" : "comparison-loser text-on-surface-variant"}">
                ${absVal.toFixed(4)}
                ${r.baseAbs != null ? `<span class="text-[9px] block text-on-surface-variant/60">base: ${r.baseAbs}</span>` : ""}
              </td>
              <td class="py-2 px-3 text-center ${!absWins ? "comparison-winner font-bold text-secondary" : "comparison-loser text-on-surface-variant"}">
                ${extVal.toFixed(4)}
                ${r.baseExt != null ? `<span class="text-[9px] block text-on-surface-variant/60">base: ${r.baseExt}</span>` : ""}
              </td>
              <td class="py-2 px-3 text-center hidden md:table-cell">
                <span class="text-xs font-bold ${absWins ? "text-primary" : "text-on-surface-variant"}">
                  ${absWins ? "mT5 ✓" : "TextRank ✓"}
                </span>
              </td>
            </tr>`;
        }).join("")}
      </tbody>
    </table>`;
}

// ─── COPY / EXPORT ────────────────────────────────────────────────────────────
function initCopyExport() {
  DOM.copyAbsBtn()?.addEventListener("click", () => {
    const text = DOM.absText()?.textContent?.trim();
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => showToast("Ringkasan abstraktif disalin!"));
  });

  DOM.copyExtBtn()?.addEventListener("click", () => {
    const items = DOM.extList()?.querySelectorAll(".sentence-item p");
    if (!items?.length) return;
    const text = [...items].map((el, i) => `${i + 1}. ${el.textContent.trim()}`).join("\n");
    navigator.clipboard.writeText(text).then(() => showToast("Ringkasan ekstraktif disalin!"));
  });

  DOM.exportBtn()?.addEventListener("click", exportTxt);
}

function exportTxt() {
  if (!state.lastResult) return;
  const { abstractive, extractive, keywords } = state.lastResult;
  const lines = [
    "═══════════════════════════════════════════════",
    "  RINGKASKILAT — Hasil Ringkasan Otomatis",
    "═══════════════════════════════════════════════",
    "",
    "📌 RINGKASAN ABSTRAKTIF (mT5)",
    "───────────────────────────────",
    abstractive.text,
    "",
    `Kata: ${abstractive.wordCount} | Kompresi: −${abstractive.compression}%`,
    `ROUGE-1: ${abstractive.metrics.rouge1} | ROUGE-2: ${abstractive.metrics.rouge2} | ROUGE-L: ${abstractive.metrics.rougeL}`,
    "",
    "📋 RINGKASAN EKSTRAKTIF (TextRank)",
    "───────────────────────────────",
    ...(extractive.sentences || [extractive.text]).map((s, i) => `${i + 1}. ${s}`),
    "",
    `Kata: ${extractive.wordCount} | Kompresi: −${extractive.compression}%`,
    `ROUGE-1: ${extractive.metrics.rouge1} | ROUGE-2: ${extractive.metrics.rouge2} | ROUGE-L: ${extractive.metrics.rougeL}`,
    "",
    keywords?.length ? `🔑 Kata Kunci: ${keywords.join(", ")}` : "",
    "",
    "═══════════════════════════════════════════════",
    `Diekspor: ${new Date().toLocaleString("id-ID")}`,
  ].filter((l) => l !== undefined).join("\n");

  const blob = new Blob([lines], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `ringkasan_${Date.now()}.txt`;
  a.click();
  showToast("File berhasil diunduh!");
}

// ─── RESET ────────────────────────────────────────────────────────────────────
function resetUI() {
  DOM.textarea().value = "";
  updateWordCount();
  DOM.resultsSection()?.classList.add("hidden");
  state.lastResult = null;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ─── ROUGE BARS ANIMATION (on scroll into view) ───────────────────────────────
function initRougeBarAnimation() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.querySelectorAll(".rouge-bar-fill[data-width]").forEach((bar) => {
          bar.style.width = bar.dataset.width;
        });
      }
    });
  }, { threshold: 0.3 });

  document.querySelectorAll(".rouge-bars-section").forEach((el) => observer.observe(el));
}

// ─── BASELINE STATS SECTION ───────────────────────────────────────────────────
function renderBaselineStats() {
  const container = $("baseline-stats");
  if (!container) return;

  const metrics = [
    { label: "ROUGE-1", tr: BASELINE.textrank.rouge1, mt5: BASELINE.mt5.rouge1 },
    { label: "ROUGE-2", tr: BASELINE.textrank.rouge2, mt5: BASELINE.mt5.rouge2 },
    { label: "ROUGE-L", tr: BASELINE.textrank.rougeL, mt5: BASELINE.mt5.rougeL },
    { label: "BERTScore-F1", tr: BASELINE.textrank.bertscore, mt5: BASELINE.mt5.bertscore },
  ];

  container.innerHTML = metrics.map((m) => `
    <div class="space-y-2">
      <div class="flex justify-between items-center">
        <span class="text-label-sm font-label-sm text-on-surface-variant uppercase tracking-wider">${m.label}</span>
        <div class="flex gap-3">
          <span class="text-[11px] font-bold text-on-surface-variant">TextRank: ${m.tr}</span>
          <span class="text-[11px] font-bold text-primary">mT5: ${m.mt5}</span>
        </div>
      </div>
      <div class="rouge-bar-track">
        <div class="rouge-bar-fill bg-on-surface-variant/30" data-width="${m.tr * 100}%" style="width:0%"></div>
      </div>
      <div class="rouge-bar-track">
        <div class="rouge-bar-fill bg-primary" data-width="${m.mt5 * 100}%" style="width:0%"></div>
      </div>
    </div>`).join("");

  // Observe for animation
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.querySelectorAll(".rouge-bar-fill[data-width]").forEach((bar) => {
          setTimeout(() => { bar.style.width = bar.dataset.width; }, 200);
        });
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.4 });

  observer.observe(container);
}

// ─── INIT ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initTextarea();
  initSlider();
  initToggle();
  initMethodRadio();
  initUrlFetch();
  initNewsSearch();
  initSummarize();
  initCopyExport();
  renderBaselineStats();
  initRougeBarAnimation();

  // Hide results section on load
  DOM.resultsSection()?.classList.add("hidden");
  DOM.keywordSection()?.classList.add("hidden");
});