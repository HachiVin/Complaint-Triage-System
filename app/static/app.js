const $ = (id) => document.getElementById(id);
let LABELS = { aspect: [], sentiment: [], urgency: [] };
let trendChart = null;
let currentUrgency = "kritis";

const nameOf = (task, key) => (LABELS[task].find((l) => l.key === key) || { name: key }).name;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x == null ? "–" : `${(x * 100).toFixed(1)}%`);

async function getJSON(url, opts) {
  const res = await fetch(url, opts);
  const body = res.status === 204 ? null : await res.json().catch(() => null);
  if (!res.ok) throw new Error((body && body.detail) || `HTTP ${res.status}`);
  return body;
}

function params() {
  const p = new URLSearchParams({ weeks: $("weeks").value });
  if ($("app").value) p.set("app", $("app").value);
  return p;
}

function renderAspects(aspects) {
  const rows = aspects.map((a) => ({ ...a, neg: Math.round(a.reviews * a.negative_share) }))
    .filter((a) => a.neg > 0).sort((a, b) => b.neg - a.neg);
  const max = Math.max(1, ...rows.map((r) => r.neg));
  $("aspects").innerHTML = rows.map((r) => `
    <li><span>${esc(nameOf("aspect", r.aspect))}</span>
      <span class="track"><span class="fill" style="width:${(100 * r.neg) / max}%"></span></span>
      <span class="num">${r.neg}${r.kritis ? `<span class="kr">${r.kritis} kritis</span>` : ""}</span></li>`).join("")
    || `<li class="muted">Belum ada keluhan pada periode ini.</li>`;
}

function renderTrend(weekly) {
  const weeks = [...new Set(weekly.map((w) => w.week))].sort();
  const totals = {};
  weekly.forEach((w) => (totals[w.aspect] = (totals[w.aspect] || 0) + w.reviews));
  const top = Object.entries(totals).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([k]) => k);
  const palette = ["#165c4b", "#c2593f", "#5b7fb5", "#a08a2d"];
  const datasets = top.map((aspect, i) => ({
    label: nameOf("aspect", aspect), borderColor: palette[i], backgroundColor: palette[i], tension: 0, pointRadius: 2,
    data: weeks.map((wk) => (weekly.find((w) => w.week === wk && w.aspect === aspect) || { reviews: 0 }).reviews),
  }));
  if (!window.Chart) return;
  const muted = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim();
  const cfg = {
    type: "line",
    data: { labels: weeks.map((w) => w.slice(5)), datasets },
    options: { responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10, color: muted } } },
      scales: { y: { beginAtZero: true, ticks: { precision: 0, color: muted } }, x: { ticks: { color: muted } } } },
  };
  if (trendChart) trendChart.destroy();
  trendChart = new Chart($("trend"), cfg);
}

async function loadOverview() {
  const o = await getJSON(`/api/overview?${params()}`);
  if (!o.reviews) { $("period").textContent = "Belum ada data."; return; }
  $("period").textContent = `${o.weeks} minggu sampai ${o.period_end} (tanggal ulasan terbaru)`;
  $("k-reviews").textContent = o.reviews.toLocaleString("id-ID");
  $("k-neg").textContent = pct(o.negative_share);
  $("k-kritis").textContent = o.kritis;
  const d = o.kritis - o.kritis_previous_period;
  $("k-kritis-delta").textContent = `${d > 0 ? "+" : ""}${d} dari ${o.weeks} minggu sebelumnya`;
  $("k-kritis-delta").classList.toggle("up", d > 0);
  $("k-tinggi").textContent = o.tinggi;
  $("k-rating").textContent = o.avg_rating ?? "–";
  $("k-review").textContent = pct(o.needs_review_share);
  renderAspects(o.aspects);
  renderTrend(o.weekly_negative);
}

async function loadQueue() {
  const p = params(); p.delete("weeks"); p.set("urgency", currentUrgency); p.set("limit", "50");
  const rows = await getJSON(`/api/queue?${p}`);
  const options = (sel) => LABELS.urgency.map((u) => `<option value="${u.key}" ${u.key === sel ? "selected" : ""}>${esc(u.name)}</option>`).join("");
  $("queue").querySelector("tbody").innerHTML = rows.map((r) => `
    <tr>
      <td class="nowrap">${esc(r.review_date || "–")}</td>
      <td>${esc(nameOf("aspect", r.aspect))}</td>
      <td>${r.rating ?? "–"}</td>
      <td class="text">${esc(r.text)} ${r.needs_review ? '<span class="flag">· perlu dicek</span>' : ""}</td>
      <td class="nowrap">${esc(r.app_version || "–")}</td>
      <td><select data-review="${r.review_id}" aria-label="Ubah urgensi">${options(r.urgency)}</select></td>
    </tr>`).join("") || `<tr><td colspan="6" class="muted">Tidak ada ulasan dengan urgensi ini.</td></tr>`;
}

async function refresh() {
  try { await Promise.all([loadOverview(), loadQueue()]); }
  catch (e) { $("period").textContent = `Gagal memuat data: ${e.message}`; }
}

document.addEventListener("change", async (ev) => {
  const sel = ev.target;
  if (sel.matches("select[data-review]")) {
    sel.disabled = true;
    try {
      await getJSON("/api/correct", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review_id: Number(sel.dataset.review), task: "urgency", label: sel.value }) });
      sel.title = "Koreksi tersimpan";
    } catch (e) { alert(`Gagal menyimpan: ${e.message}`); }
    sel.disabled = false;
  }
});

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
  currentUrgency = t.dataset.urgency;
  loadQueue();
}));
$("app").addEventListener("change", refresh);
$("weeks").addEventListener("change", refresh);

document.querySelectorAll(".suggest .chip").forEach((c) => c.addEventListener("click", () => { $("question").value = c.textContent; }));
$("ask-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const q = $("question").value.trim();
  if (!q) return;
  const btn = $("ask-btn"); btn.disabled = true; btn.textContent = "Menganalisis...";
  const box = $("answer"); box.hidden = false; box.classList.remove("err"); box.textContent = "…";
  try {
    const r = await getJSON("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q }) });
    box.textContent = r.ok ? r.answer : `Agent gagal menjawab: ${r.error || "tidak ada jawaban"}`;
    if (!r.ok) box.classList.add("err");
    const sql = $("sql"); sql.hidden = !r.queries.length;
    sql.querySelector("pre").textContent = r.queries.join("\n\n");
  } catch (e) { box.classList.add("err"); box.textContent = e.message; }
  btn.disabled = false; btn.textContent = "Tanya";
});

$("cls-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = $("cls-text").value.trim();
  if (!text) return;
  try {
    const r = await getJSON("/api/classify", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    const row = (k, v) => `<dt>${k}</dt><dd>${v}</dd>`;
    $("cls-out").innerHTML =
      row("Aspek", `${esc(r.aspect.name)} <span class="muted">(${pct(r.aspect.confidence)})</span>`) +
      row("Sentimen", `${esc(r.sentiment.name)} <span class="muted">(${pct(r.sentiment.confidence)})</span>`) +
      row("Urgensi", `<span class="badge ${r.urgency.label}">${esc(r.urgency.name)}</span> <span class="muted">${r.urgency.source === "rules" ? "(aturan)" : `(${pct(r.urgency.confidence)})`}</span>`) +
      (r.matched_phrases.length ? row("Pemicu", esc(r.matched_phrases.join(", "))) : "") +
      (r.needs_review ? row("Catatan", "Keyakinan rendah, perlu dicek manusia") : "");
    $("cls-out").hidden = false;
  } catch (e) { $("cls-out").innerHTML = `<dt>Error</dt><dd>${esc(e.message)}</dd>`; $("cls-out").hidden = false; }
});

async function init() {
  try {
    LABELS = await getJSON("/api/labels");
    const h = await getJSON("/health");
    $("model").textContent = h.model_version;
    const notes = [];
    if (h.demo_model) notes.push("Model demo dilatih dengan ulasan fiktif.");
    if (h.database === "not_configured") notes.push("Database belum dikonfigurasi: hanya klasifikasi yang aktif.");
    if (!h.agent) {
      notes.push("Agent belum aktif (butuh LLM_API_KEY).");
      ["question", "ask-btn"].forEach((id) => ($(id).disabled = true));
      document.querySelectorAll(".suggest .chip").forEach((c) => (c.disabled = true));
      $("ask-form").insertAdjacentHTML("afterend", '<p class="off">Fitur ini aktif setelah LLM_API_KEY diisi di .env.</p>');
    }
    if (notes.length) { $("demo-note").textContent = notes.join(" "); $("demo-note").hidden = false; }
    if (h.database === "not_configured") return;
    const apps = await getJSON("/api/apps");
    $("app").innerHTML = `<option value="">Semua</option>` + apps.map((a) => `<option>${esc(a)}</option>`).join("");
    if (apps.some((a) => /demo|kitapay/i.test(a))) {
      $("demo-note").textContent = ["Data demo: ulasan dan aplikasi fiktif (DemoBank Mobile, KitaPay).", ...notes].join(" ");
      $("demo-note").hidden = false;
    }
    await refresh();
  } catch (e) { $("period").textContent = `Gagal memuat: ${e.message}`; }
}
init();
