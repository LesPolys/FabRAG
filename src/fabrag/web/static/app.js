/* FabRAG frontend — vanilla JS, no framework.
 *
 * Three patterns to see here:
 *  - render functions build DOM from the API's JSON (one per result kind);
 *  - a single floating <img> implements hover-zoom for every card on the
 *    page (cheaper and simpler than a preview node per card);
 *  - /api/ask streams over SSE: EventSource fires 'grounding' once (the
 *    citations render before the first token) and then 'token' events that
 *    append to the answer as the model thinks.
 */
"use strict";

const $ = (id) => document.getElementById(id);

/* ---------- tabs ---------- */
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("tab-" + btn.dataset.tab).classList.add("active");
  });
});

/* ---------- hover zoom + lightbox (shared by all card images) ---------- */
const zoom = $("zoom");
const lightbox = $("lightbox");

function attachCardImageEvents(frame, card) {
  if (!card.image_url) return;
  frame.addEventListener("mousemove", (e) => {
    zoom.src = card.image_url;
    zoom.classList.toggle("horizontal", !!card.horizontal);
    zoom.classList.remove("hidden");
    // keep the preview on-screen, biased away from the cursor
    const w = 320, h = 320 / (450 / 628);
    let x = e.clientX + 24, y = Math.min(Math.max(e.clientY - h / 2, 8), innerHeight - h - 8);
    if (x + w > innerWidth - 8) x = e.clientX - w - 24;
    zoom.style.left = x + "px";
    zoom.style.top = y + "px";
  });
  frame.addEventListener("mouseleave", () => zoom.classList.add("hidden"));
  frame.addEventListener("click", () => {
    const img = lightbox.querySelector("img");
    img.src = card.image_url;
    img.classList.toggle("horizontal", !!card.horizontal);
    zoom.classList.add("hidden");
    lightbox.classList.remove("hidden");
  });
}
lightbox.addEventListener("click", () => lightbox.classList.add("hidden"));
addEventListener("keydown", (e) => { if (e.key === "Escape") lightbox.classList.add("hidden"); });

/* ---------- renderers ---------- */
function renderCardCell(card) {
  const cell = document.createElement("div");
  cell.className = "card-cell" + (card.horizontal ? " horizontal" : "");
  const frame = document.createElement("div");
  frame.className = "frame";
  if (card.image_url) {
    const img = document.createElement("img");
    img.loading = "lazy";          // big grids: only fetch what scrolls into view
    img.src = card.image_url;
    img.alt = card.name;
    frame.appendChild(img);
  } else {
    const ph = document.createElement("div");
    ph.className = "noimg";
    ph.textContent = card.name;
    frame.appendChild(ph);
  }
  if (card.copies && card.copies > 1) {
    const n = document.createElement("div");
    n.className = "copies";
    n.textContent = card.copies + "x";
    cell.appendChild(n);
  }
  const cap = document.createElement("div");
  cap.className = "caption";
  cap.textContent = card.name + (card.pitch ? ` · p${card.pitch}` : "");
  cell.append(frame, cap);
  attachCardImageEvents(frame, card);
  return cell;
}

function renderRuleHit(rule) {
  const div = document.createElement("div");
  div.className = "rule-hit";
  const cite = rule.url
    ? `<a href="${rule.url}" target="_blank" rel="noopener">${rule.citation}</a>`
    : rule.citation;
  div.innerHTML =
    `<span class="cite">${cite}</span>` +
    `<span class="where">${rule.chapter !== rule.section ? rule.section : ""}</span>` +
    `<div class="body"></div>`;
  div.querySelector(".body").textContent =
    rule.text.length > 420 ? rule.text.slice(0, 420) + " …" : rule.text;
  return div;
}

function renderResults(results, gridEl, rulesEl) {
  gridEl.replaceChildren();
  rulesEl.replaceChildren();
  for (const r of results) {
    if (r.kind === "card") gridEl.appendChild(renderCardCell(r));
    else rulesEl.appendChild(renderRuleHit(r));
  }
}

/* ---------- search ---------- */
$("search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const params = new URLSearchParams({
    q: $("search-q").value,
    source: $("search-source").value,
    k: $("search-k").value,
  });
  const hero = $("search-hero").value.trim();
  if (hero) params.set("hero", hero);
  $("search-status").textContent = "searching…";
  try {
    const resp = await fetch("/api/search?" + params);
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    const data = await resp.json();
    $("search-status").textContent = data.results.length ? "" : "nothing matched";
    renderResults(data.results, $("search-grid"), $("search-rules"));
  } catch (err) {
    $("search-status").textContent = "error: " + err.message;
  }
});

/* ---------- ask (SSE streaming) ---------- */
let askSource = null;
$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  if (askSource) askSource.close();
  const params = new URLSearchParams({
    q: $("ask-q").value,
    source: $("ask-source").value,
  });
  const answer = $("ask-answer");
  answer.innerHTML = '<span class="cursor">▍</span>';
  $("ask-grounding-title").classList.add("hidden");
  $("ask-grid").replaceChildren();
  $("ask-rules").replaceChildren();

  let text = "";
  askSource = new EventSource("/api/ask?" + params);
  askSource.addEventListener("grounding", (ev) => {
    const data = JSON.parse(ev.data);
    if (data.results.length) $("ask-grounding-title").classList.remove("hidden");
    renderResults(data.results, $("ask-grid"), $("ask-rules"));
  });
  askSource.addEventListener("token", (ev) => {
    text += JSON.parse(ev.data).t;
    answer.innerHTML = "";
    answer.append(document.createTextNode(text), cursorNode());
  });
  askSource.addEventListener("done", () => {
    askSource.close();
    answer.textContent = text;
  });
  askSource.onerror = () => {
    askSource.close();
    if (!text) answer.textContent = "error: stream failed (is Ollama running?)";
  };
});
const cursorNode = () => {
  const s = document.createElement("span");
  s.className = "cursor";
  s.textContent = "▍";
  return s;
};

/* ---------- build ---------- */
const PITCH_GROUPS = [
  { key: "loadout", title: "Loadout — equipment & weapons", cls: "loadout",
    match: (c) => c.loadout },
  { key: "p1", title: "Pitch 1 — red", cls: "p1", match: (c) => !c.loadout && c.pitch === 1 },
  { key: "p2", title: "Pitch 2 — yellow", cls: "p2", match: (c) => !c.loadout && c.pitch === 2 },
  { key: "p3", title: "Pitch 3 — blue", cls: "p3", match: (c) => !c.loadout && c.pitch === 3 },
  { key: "other", title: "Other", cls: "", match: (c) => !c.loadout && ![1, 2, 3].includes(c.pitch) },
];

$("build-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const status = $("build-status");
  status.textContent =
    "building… the LLM proposes, the validator pushes back, repeat — expect 1–3 minutes";
  status.classList.add("spin");
  $("build-result").classList.add("hidden");
  try {
    const resp = await fetch("/api/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        hero: $("build-hero").value,
        strategy: $("build-strategy").value || "a balanced, efficient deck",
        format: $("build-format").value,
      }),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    const d = await resp.json();
    status.textContent = "";
    status.classList.remove("spin");

    const main = d.cards.filter((c) => !c.loadout)
                        .reduce((n, c) => n + (c.copies || 1), 0);
    $("build-meta").textContent =
      `${d.hero.name} — ${d.format} · ${main} main-deck cards · legal: ${d.legal} · ` +
      `${d.rounds} LLM round(s)` +
      (d.finisher_notes.length ? `\nfinisher: ${d.finisher_notes.join("; ")}` : "") +
      (d.warnings.length ? `\n${d.warnings.map((w) => "note: " + w).join("\n")}` : "");

    const deckEl = $("build-deck");
    deckEl.replaceChildren();
    for (const g of PITCH_GROUPS) {
      const cards = d.cards.filter(g.match);
      if (!cards.length) continue;
      const section = document.createElement("section");
      section.className = "pitch-group " + g.cls;
      const h = document.createElement("h3");
      const n = cards.reduce((acc, c) => acc + (c.copies || 1), 0);
      h.textContent = `${g.title} (${n})`;
      const grid = document.createElement("div");
      grid.className = "card-grid small";
      cards.forEach((c) => grid.appendChild(renderCardCell(c)));
      section.append(h, grid);
      deckEl.appendChild(section);
    }
    $("build-explanation").textContent = d.explanation || "";
    $("build-result").classList.remove("hidden");
  } catch (err) {
    status.classList.remove("spin");
    status.textContent = "error: " + err.message;
  }
});
