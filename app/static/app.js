// Complaint Triage dashboard. Plain JS, no build step.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (n) => Number(n ?? 0).toLocaleString("id-ID");
const pct = (x, d = 1) => (x == null ? "–" : `${(x * 100).toLocaleString("id-ID", { maximumFractionDigits: d })}%`);
const fmtDate = (iso) => (iso ? new Date(`${iso}T00:00:00`).toLocaleDateString("id-ID", { day: "numeric", month: "short", year: "numeric" }) : "–");

const state = {
  labels: { aspect: [], sentiment: [], urgency: [] },
  health: null,
  overview: null,
  queue: { urgency: "kritis", aspect: "", limit: 30 },
  chart: null,
  asking: false,
};
const nameOf = (task, key) => (state.labels[task].find((l) => l.key === key) || { name: key }).name;
const descOf = (task, key) => (state.labels[task].find((l) => l.key === key) || { description: "" }).description;

async function getJSON(url, opts) {
  const res = await fetch(url, opts);
  const body = res.status === 204 ? null : await res.json().catch(() => null);
  if (!res.ok) throw new Error((body && body.detail) || `Server menjawab ${res.status}`);
  return body;
}
const postJSON = (url, data) => getJSON(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

function scopeParams() {
  const p = new URLSearchParams();
  if ($("app").value) p.set("app", $("app").value);
  return p;
}

function setStatus(text, kind = "") {
  const el = $("status");
  el.textContent = text || "";
  el.className = `status ${kind}`;
  el.hidden = !text;
}

// ---- routing -------------------------------------------------------------------------------------

const VIEWS = ["ringkasan", "antrean", "tanya", "coba"];
function showView() {
  const view = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "ringkasan";
  for (const v of VIEWS) $(`view-${v}`).hidden = v !== view;
  document.querySelectorAll(".tabs a").forEach((a) => {
    if (a.dataset.view === view) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  if (view === "antrean") loadQueue();
  if (view === "ringkasan" && state.chart) state.chart.resize();
  window.scrollTo({ top: 0 });
}
window.addEventListener("hashchange", showView);

// ---- overview ------------------------------------------------------------------------------------

function renderTriage(o) {
  const normal = Math.max(0, o.reviews - o.kritis - o.tinggi);
  const parts = [
    { key: "kritis", n: o.kritis, label: "Kritis", text: "uang atau akun nasabah terdampak" },
    { key: "tinggi", n: o.tinggi, label: "Tinggi", text: "tidak bisa memakai layanan utama" },
    { key: "normal", n: normal, label: "Normal", text: "saran, pujian, keluhan ringan" },
  ];
  $("triage-bar").innerHTML = parts.filter((p) => p.n > 0).map((p) => {
    const grow = Math.max(p.n / Math.max(o.reviews, 1), 0.06);
    const inner = `${num(p.n)} ${p.label.toLowerCase()}`;
    return p.key === "normal"
      ? `<span class="seg-normal" style="flex:${grow}" title="${esc(p.label)}">${inner}</span>`
      : `<a class="seg-${p.key}" style="flex:${grow}" href="#antrean" data-urgency="${p.key}" title="Buka antrean ${p.label.toLowerCase()}">${inner}</a>`;
  }).join("");
  const d = o.kritis - o.kritis_previous_period;
  const delta = o.kritis_previous_period
    ? `<span class="delta ${d > 0 ? "up" : "down"}">${d > 0 ? "naik" : "turun"} ${num(Math.abs(d))} dari ${o.weeks} minggu sebelumnya</span>`
    : "";
  $("triage-legend").innerHTML = parts.map((p) => `
    <div><strong><i class="dot ${p.key}"></i> ${p.label}: ${pct(p.n / Math.max(o.reviews, 1))}</strong>
      ${esc(p.text)}${p.key === "kritis" && delta ? `<br>${delta}` : ""}</div>`).join("");
  $("nav-kritis").textContent = o.kritis ? num(o.kritis) : "";
}

function renderAspects(aspects) {
  const rows = aspects.map((a) => ({ ...a, neg: Math.round(a.reviews * a.negative_share) }))
    .filter((a) => a.neg > 0).sort((a, b) => b.neg - a.neg);
  const max = Math.max(1, ...rows.map((r) => r.neg));
  $("aspects").innerHTML = rows.map((r) => `
    <li><button type="button" data-aspect="${esc(r.aspect)}" aria-label="${esc(nameOf("aspect", r.aspect))}: ${r.neg} ulasan negatif, ${r.kritis} kritis. Buka antrean.">
      <span>${esc(nameOf("aspect", r.aspect))}</span>
      <span class="track"><span class="neg" style="width:${(100 * r.neg) / max}%"></span><span class="kr" style="width:${(100 * r.kritis) / max}%"></span></span>
      <span class="n">${num(r.neg)}</span><span class="k">${r.kritis ? `${num(r.kritis)} kritis` : ""}</span>
    </button></li>`).join("") || `<li class="empty">Tidak ada ulasan negatif di periode ini.</li>`;
}

function renderTrend(weekly, periodEnd) {
  if (!window.Chart) return;
  const weeks = [...new Set(weekly.map((w) => w.week))].sort();
  const totals = {};
  weekly.forEach((w) => (totals[w.aspect] = (totals[w.aspect] || 0) + w.reviews));
  const top = Object.entries(totals).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([k]) => k);
  const palette = ["#9dcbff", "#b9a3f5", "#6fd3be", "#d6dde6"];
  const last = weeks[weeks.length - 1];
  const partial = last && (new Date(periodEnd) - new Date(last)) / 86400000 < 6;
  const label = (w) => new Date(`${w}T00:00:00`).toLocaleDateString("id-ID", { day: "numeric", month: "short" }) + (partial && w === last ? "*" : "");
  const datasets = top.map((aspect, i) => ({
    label: nameOf("aspect", aspect), borderColor: palette[i], backgroundColor: palette[i],
    borderWidth: i === 0 ? 2.5 : 1.75, pointRadius: 2.5, tension: 0,
    data: weeks.map((wk) => (weekly.find((w) => w.week === wk && w.aspect === aspect) || { reviews: 0 }).reviews),
  }));
  const muted = "#95a4b7", grid = "#223246";
  const cfg = {
    type: "line", data: { labels: weeks.map(label), datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false, interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "bottom", labels: { color: muted, boxWidth: 10, boxHeight: 10, font: { family: "Schibsted Grotesk" } } },
        tooltip: { backgroundColor: "#1e2c3d", borderColor: "#2a3b50", borderWidth: 1, titleColor: "#e7edf4", bodyColor: "#e7edf4" },
      },
      scales: {
        y: { beginAtZero: true, ticks: { precision: 0, color: muted }, grid: { color: grid } },
        x: { ticks: { color: muted }, grid: { display: false } },
      },
    },
  };
  if (state.chart) state.chart.destroy();
  state.chart = new Chart($("trend"), cfg);
  $("trend-note").textContent = "Ulasan negatif per minggu, empat aspek terbanyak." + (partial ? " Tanda * = minggu terakhir belum penuh." : "");
}

function renderVersions(versions) {
  const max = Math.max(1, ...versions.map((v) => v.kritis / Math.max(v.reviews, 1)));
  $("versions").innerHTML = versions.map((v) => `
    <tr><td>${esc(v.app_version)}</td><td class="num">${num(v.reviews)}</td><td class="num">${pct(v.negative_share, 0)}</td>
      <td class="num">${num(v.kritis)}</td>
      <td class="bar"><div class="mini" title="${pct(v.kritis / Math.max(v.reviews, 1))} kritis"><i style="width:${(100 * v.kritis / Math.max(v.reviews, 1)) / max}%"></i></div></td></tr>`).join("")
    || `<tr><td colspan="5" class="empty">Belum ada versi dengan cukup ulasan.</td></tr>`;
}

async function loadOverview() {
  const p = scopeParams(); p.set("weeks", $("weeks").value);
  const o = await getJSON(`/api/overview?${p}`);
  state.overview = o;
  if (!o.reviews) { $("period-title").textContent = "Belum ada data ulasan"; return; }
  $("period-title").textContent = `${o.weeks} minggu terakhir: ${num(o.reviews)} ulasan`;
  $("period-sub").textContent = `Sampai ${fmtDate(o.period_end)}, tanggal ulasan terbaru di data${o.app ? ` aplikasi ${o.app}` : ""}.`;
  $("facts").innerHTML = [
    ["Sentimen negatif", pct(o.negative_share)],
    ["Rata-rata rating", o.avg_rating == null ? "–" : o.avg_rating.toLocaleString("id-ID")],
    ["Perlu dicek manual", pct(o.needs_review_share)],
  ].map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  renderTriage(o);
  renderAspects(o.aspects);
  renderTrend(o.weekly_negative, o.period_end);
  renderVersions(o.versions);
}

// ---- queue ---------------------------------------------------------------------------------------

async function loadQueue() {
  if (!state.health || state.health.database !== true) return;
  const { urgency, aspect, limit } = state.queue;
  document.querySelectorAll(".segmented button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.urgency === urgency)));
  $("queue-aspect").value = aspect;
  $("urgency-explain").textContent = `${nameOf("urgency", urgency)}: ${descOf("urgency", urgency)}`;
  const p = scopeParams(); p.set("urgency", urgency); p.set("limit", String(limit));
  if (aspect) p.set("aspect", aspect);
  $("queue").innerHTML = `<li class="empty" style="padding:16px 4px">Memuat antrean…</li>`;
  let rows;
  try { rows = await getJSON(`/api/queue?${p}`); }
  catch (e) { $("queue").innerHTML = `<li class="empty" style="padding:16px 4px">Antrean tidak bisa dimuat: ${esc(e.message)}</li>`; return; }
  $("queue-sub").textContent = `${aspect ? `${nameOf("aspect", aspect)}, ` : ""}ulasan ${nameOf("urgency", urgency).toLowerCase()} terbaru. Klik ulasan untuk melihat detailnya.`;
  $("queue").innerHTML = rows.map((r, i) => queueRow(r, i)).join("")
    || `<li class="empty" style="padding:16px 4px">Tidak ada ulasan ${esc(nameOf("urgency", urgency).toLowerCase())} untuk filter ini.</li>`;
  $("queue-more").hidden = rows.length < limit || limit >= 200;
}

function queueRow(r, i) {
  const conf = Math.round((r.urgency_conf ?? 0) * 100);
  const canCorrect = state.health && state.health.corrections;
  const options = state.labels.urgency.map((u) => `<option value="${u.key}" ${u.key === r.urgency ? "selected" : ""}>${esc(u.name)}</option>`).join("");
  const askAbout = `Berapa ulasan ${nameOf("urgency", r.urgency).toLowerCase()} tentang ${nameOf("aspect", r.aspect).toLowerCase()} dalam 4 minggu terakhir, dan versi aplikasi mana yang paling banyak?`;
  return `
  <li>
    <button type="button" class="q-row" aria-expanded="false" aria-controls="qd-${i}">
      <span class="q-tick ${r.urgency}" aria-hidden="true"></span>
      <span>
        <span class="q-text">${esc(r.text)}</span>
        <span class="q-meta">
          <span>${fmtDate(r.review_date)}</span><span>${esc(nameOf("aspect", r.aspect))}</span>
          ${r.rating ? `<span>Rating ${r.rating}</span>` : ""}${r.app_version ? `<span>Versi ${esc(r.app_version)}</span>` : ""}
        </span>
      </span>
      <span class="q-side">Yakin ${conf}%${r.needs_review ? `<br><span class="flag">perlu dicek</span>` : ""}</span>
    </button>
    <div class="q-detail" id="qd-${i}" hidden>
      <div><h3>Urgensi</h3>${esc(nameOf("urgency", r.urgency))}, keyakinan ${conf}%<div class="meter ${r.urgency}"><i style="width:${conf}%"></i></div></div>
      <div><h3>Aspek dan sentimen</h3>${esc(nameOf("aspect", r.aspect))}, ${esc(nameOf("sentiment", r.sentiment).toLowerCase())}</div>
      <div><h3>ID ulasan</h3>${r.review_id}${r.needs_review ? `<br><span class="flag">Model kurang yakin, perlu dicek</span>` : ""}</div>
      <div><h3>Tindak lanjut</h3><a href="#tanya" data-ask="${esc(askAbout)}">Tanya data tentang keluhan sejenis</a></div>
      ${canCorrect ? `<div><h3>Koreksi urgensi</h3><select data-review="${r.review_id}" aria-label="Koreksi urgensi">${options}</select></div>` : ""}
    </div>
  </li>`;
}

// ---- ask -----------------------------------------------------------------------------------------

function md(text) {
  // Minimal, safe Markdown for agent answers: paragraphs, lists, tables, **bold**, `code`.
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");
  const lines = String(text || "").replace(/\r/g, "").split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*\|.*\|\s*$/.test(line)) {
      const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(lines[i++]);
      const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const body = rows.filter((r) => !/^\s*\|?\s*:?-{2,}/.test(r));
      const [head, ...rest] = body;
      out.push(`<div class="table-scroll"><table><thead><tr>${cells(head).map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${
        rest.map((r) => `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
      continue;
    }
    if (/^\s*([-*]|\d+[.)])\s+/.test(line)) {
      const ordered = /^\s*\d/.test(line);
      const items = [];
      while (i < lines.length && /^\s*([-*]|\d+[.)])\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*([-*]|\d+[.)])\s+/, ""));
      out.push(`<${ordered ? "ol" : "ul"}>${items.map((it) => `<li>${inline(it)}</li>`).join("")}</${ordered ? "ol" : "ul"}>`);
      continue;
    }
    if (line.trim()) {
      const para = [];
      while (i < lines.length && lines[i].trim() && !/^\s*(\||[-*]\s|\d+[.)]\s)/.test(lines[i])) para.push(lines[i++].replace(/^#+\s*/, ""));
      out.push(`<p>${para.map(inline).join("<br>")}</p>`);
      continue;
    }
    i++;
  }
  return out.join("");
}

async function ask(question) {
  question = question.trim();
  if (!question || state.asking) return;
  if (!state.health || !state.health.agent) return;
  state.asking = true;
  $("ask-btn").disabled = true;
  const item = document.createElement("article");
  item.className = "qa";
  item.innerHTML = `<p class="qa-q">${esc(question)}</p><div class="qa-a"><p class="thinking"><span class="pulse"></span>Agent sedang menulis dan menjalankan query. Biasanya 5–15 detik.</p></div>`;
  $("answers").prepend(item);
  try {
    const r = await postJSON("/api/ask", { question });
    const sql = r.queries.length
      ? `<details><summary>Lihat ${r.queries.length} query SQL yang dijalankan</summary>${r.queries.map((q) => `<pre>${esc(q)}</pre>`).join("")}</details>` : "";
    item.querySelector(".qa-a").innerHTML = r.ok ? md(r.answer) : `<p>Agent tidak berhasil menjawab: ${esc(r.error || "tanpa jawaban")}. Coba ulangi dengan kalimat lain.</p>`;
    if (!r.ok) item.classList.add("error");
    item.insertAdjacentHTML("beforeend", `${sql}<div class="qa-meta"><span>Dijawab dalam ${(r.latency_ms / 1000).toLocaleString("id-ID", { maximumFractionDigits: 1 })} detik</span><span>${r.steps} langkah agent</span></div>`);
  } catch (e) {
    item.classList.add("error");
    item.querySelector(".qa-a").innerHTML = `<p>${esc(e.message)}</p>`;
  }
  state.asking = false;
  $("ask-btn").disabled = false;
}

function goAsk(question) {
  $("question").value = question;
  if (location.hash !== "#tanya") location.hash = "#tanya";
  ask(question);
}

// ---- try the model -------------------------------------------------------------------------------

async function classify(text) {
  text = text.trim();
  if (!text) return;
  const out = $("cls-out");
  out.innerHTML = `<p class="thinking"><span class="pulse"></span>Mengklasifikasi…</p>`;
  try {
    const r = await postJSON("/api/classify", { text });
    const meter = (x, cls = "") => `<div class="meter ${cls}"><i style="width:${Math.round(x * 100)}%"></i></div>`;
    const u = r.urgency;
    out.innerHTML = `
      <div class="verdict"><span class="tag ${u.label}">${esc(u.name)}</span><p>${esc(descOf("urgency", u.label))}</p></div>
      <div class="result-grid">
        <div><h3>Aspek</h3><span class="val">${esc(r.aspect.name)}</span> <span class="muted">${pct(r.aspect.confidence, 0)} yakin</span>${meter(r.aspect.confidence)}</div>
        <div><h3>Sentimen</h3><span class="val">${esc(r.sentiment.name)}</span> <span class="muted">${pct(r.sentiment.confidence, 0)} yakin</span>${meter(r.sentiment.confidence)}</div>
        <div><h3>Urgensi</h3><span class="val">${esc(u.name)}</span> <span class="muted">${u.source === "rules" ? "dari aturan kata kunci" : `${pct(u.confidence, 0)} yakin`}</span>${meter(u.confidence, u.label)}</div>
        <div><h3>Kata pemicu (aturan)</h3>${r.matched_phrases.length ? esc(r.matched_phrases.join(", ")) : "<span class='muted'>tidak ada</span>"}</div>
      </div>
      <p class="result-note">${r.needs_review ? "Salah satu tebakan di bawah 55% yakin, jadi ulasan ini akan ditandai perlu dicek manusia. " : ""}Model ${esc(r.model_version)}, ${r.latency_ms.toLocaleString("id-ID")} ms.</p>`;
  } catch (e) { out.innerHTML = `<p class="empty">Tidak bisa mengklasifikasi: ${esc(e.message)}</p>`; }
}

// ---- events --------------------------------------------------------------------------------------

document.addEventListener("click", (ev) => {
  const t = ev.target.closest("button, a");
  if (!t) return;
  if (t.matches("#triage-bar a[data-urgency]")) { state.queue = { ...state.queue, urgency: t.dataset.urgency, aspect: "", limit: 30 }; }
  else if (t.matches(".aspects button")) {
    state.queue = { ...state.queue, aspect: t.dataset.aspect, limit: 30 };
    if (location.hash === "#antrean") loadQueue(); else location.hash = "#antrean";
  }
  else if (t.matches(".segmented button")) { state.queue = { ...state.queue, urgency: t.dataset.urgency, limit: 30 }; loadQueue(); }
  else if (t.matches(".q-row")) {
    const open = t.getAttribute("aria-expanded") === "true";
    t.setAttribute("aria-expanded", String(!open));
    $(t.getAttribute("aria-controls")).hidden = open;
  }
  else if (t.matches("[data-ask]")) { ev.preventDefault(); goAsk(t.dataset.ask); }
  else if (t.matches("#suggest .chip")) { goAsk(t.textContent); }
  else if (t.matches("[data-sample]")) { $("cls-text").value = t.dataset.sample; classify(t.dataset.sample); }
  else if (t.id === "queue-more") { state.queue.limit = Math.min(200, state.queue.limit + 30); loadQueue(); }
  else if (t.id === "intro-close") { $("intro").hidden = true; try { localStorage.setItem("ct-intro-closed", "1"); } catch { /* private mode */ } }
});

document.addEventListener("change", async (ev) => {
  const el = ev.target;
  if (el.id === "app" || el.id === "weeks") { await refresh(); if (location.hash === "#antrean") loadQueue(); }
  else if (el.id === "queue-aspect") { state.queue = { ...state.queue, aspect: el.value, limit: 30 }; loadQueue(); }
  else if (el.matches("select[data-review]")) {
    el.disabled = true;
    try { await postJSON("/api/correct", { review_id: Number(el.dataset.review), task: "urgency", label: el.value }); el.title = "Koreksi tersimpan"; }
    catch (e) { setStatus(`Koreksi tidak tersimpan: ${e.message}`, "error"); }
    el.disabled = false;
  }
});

$("askbar").addEventListener("submit", (ev) => { ev.preventDefault(); const q = $("askbar-input").value; $("askbar-input").value = ""; goAsk(q); });
$("ask-form").addEventListener("submit", (ev) => { ev.preventDefault(); ask($("question").value); });
$("question").addEventListener("keydown", (ev) => { if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); ask($("question").value); } });
$("cls-form").addEventListener("submit", (ev) => { ev.preventDefault(); classify($("cls-text").value); });
document.addEventListener("keydown", (ev) => {
  if (ev.key !== "/" || ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  ev.preventDefault();
  if (location.hash === "#tanya") $("question").focus(); else { location.hash = "#ringkasan"; $("askbar-input").focus(); }
});

// ---- start ---------------------------------------------------------------------------------------

async function refresh() {
  try { await loadOverview(); }
  catch (e) { setStatus(`Ringkasan tidak bisa dimuat: ${e.message}`, "error"); }
}

function disableAgent(reason) {
  for (const id of ["askbar-input", "question", "ask-btn"]) $(id).disabled = true;
  document.querySelectorAll("#askbar button, #suggest .chip").forEach((b) => (b.disabled = true));
  $("ask-hint").textContent = reason;
  $("askbar").querySelector(".hint").textContent = reason;
}

async function init() {
  showView();
  try { if (!localStorage.getItem("ct-intro-closed")) $("intro").hidden = false; } catch { $("intro").hidden = false; }
  // A serverless demo sleeps when nobody uses it; say so instead of showing a blank page.
  const slow = setTimeout(() => setStatus("Menyalakan server. Saat pertama dibuka setelah sepi, ini bisa sampai 1 menit."), 2500);
  try {
    state.labels = await getJSON("/api/labels");
    state.health = await getJSON("/health");
  } catch (e) {
    clearTimeout(slow);
    setStatus(`Server belum bisa dihubungi: ${e.message}. Muat ulang halaman sebentar lagi.`, "error");
    return;
  }
  clearTimeout(slow);
  setStatus("");
  const h = state.health;
  const batch = (h.dashboard_models || []).join(", ");
  $("model-line").textContent = batch && batch !== h.model_version
    ? `Klasifikasi langsung: ${h.model_version}. Data dasbor: ${batch}.` : `Model: ${h.model_version}.`;
  $("queue-aspect").innerHTML = `<option value="">Semua aspek</option>` + state.labels.aspect.map((a) => `<option value="${a.key}">${esc(a.name)}</option>`).join("");
  if (!h.agent) disableAgent("Agent belum aktif di server ini (butuh LLM_API_KEY).");
  if (h.database !== true) {
    setStatus("Database belum terhubung: hanya Coba model yang bisa dipakai.", "error");
    return;
  }
  const apps = await getJSON("/api/apps");
  $("app").innerHTML = (apps.length > 1 ? `<option value="">Semua aplikasi</option>` : "") + apps.map((a) => `<option>${esc(a)}</option>`).join("");
  await refresh();
  if (location.hash === "#antrean") loadQueue();
}
init();