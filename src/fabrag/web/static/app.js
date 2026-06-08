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

function renderResults(results, gridEl, rulesEl, { append = false } = {}) {
  if (!append) {
    gridEl.replaceChildren();
    rulesEl.replaceChildren();
  }
  for (const r of results) {
    if (r.kind === "card") gridEl.appendChild(renderCardCell(r));
    else rulesEl.appendChild(renderRuleHit(r));
  }
}

/* ---------- search (numbered pages + sorting) ----------
 * The server sorts the FULL result set then slices the requested page, so
 * page 1 of "cost ascending" is the cheapest matches overall. The pager
 * renders a window of page numbers around the current one. */
async function runSearch(pageNum = 1) {
  const k = parseInt($("search-k").value, 10);
  const params = new URLSearchParams({
    q: $("search-q").value,
    source: $("search-source").value,
    sort: $("search-sort").value,
    k: String(k),
    page: String(pageNum),
  });
  const hero = $("search-hero").value.trim();
  if (hero) params.set("hero", hero);

  const status = $("search-status");
  status.textContent = "searching…";
  try {
    const resp = await fetch("/api/search?" + params);
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    const data = await resp.json();
    renderResults([...data.rules, ...data.cards], $("search-grid"), $("search-rules"));
    const totalLabel = data.total === 500 ? "500+" : data.total; // server caps at 500
    status.textContent = data.total
      ? `${totalLabel} result${data.total === 1 ? "" : "s"} · page ${data.page} of ${data.pages}`
      : "nothing matched";
    renderPager(data.page, data.pages);
    scrollTo({ top: 0, behavior: "smooth" });
  } catch (err) {
    status.textContent = "error: " + err.message;
    $("search-pager").replaceChildren();
  }
}

function renderPager(current, pages) {
  const pager = $("search-pager");
  pager.replaceChildren();
  if (pages <= 1) return;

  // windowed page list: 1 … (c-2 c-1 c c+1 c+2) … N
  const want = new Set([1, pages]);
  for (let p = current - 2; p <= current + 2; p++) {
    if (p >= 1 && p <= pages) want.add(p);
  }
  const list = [...want].sort((a, b) => a - b);

  const add = (p) => {
    const b = document.createElement("button");
    b.textContent = p;
    if (p === current) b.classList.add("current");
    else b.addEventListener("click", () => runSearch(p));
    pager.appendChild(b);
  };
  let prev = 0;
  for (const p of list) {
    if (p - prev > 1) {
      const gap = document.createElement("span");
      gap.className = "gap";
      gap.textContent = "…";
      pager.appendChild(gap);
    }
    add(p);
    prev = p;
  }
}

$("search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  runSearch(1);
});
$("search-sort").addEventListener("change", () => {
  if ($("search-q").value.trim()) runSearch(1);
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

/* ---------- build ----------
 * A build returns a full card-pool REGISTRATION: equipped inventory, the
 * starting deck (grouped by pitch), and a matchup sideboard. Each is its own
 * section. */
const PITCH_GROUPS = [
  { title: "Pitch 1 — red", cls: "p1", match: (c) => c.pitch === 1 },
  { title: "Pitch 2 — yellow", cls: "p2", match: (c) => c.pitch === 2 },
  { title: "Pitch 3 — blue", cls: "p3", match: (c) => c.pitch === 3 },
  { title: "Other", cls: "", match: (c) => ![1, 2, 3].includes(c.pitch) },
];

/* A titled card grid; `tag` optionally adds a per-card caption (slot, reason). */
function buildSection(title, cls, cards, tag) {
  const section = document.createElement("section");
  section.className = "pitch-group " + cls;
  const h = document.createElement("h3");
  const n = cards.reduce((acc, c) => acc + (c.copies || 1), 0);
  h.textContent = `${title} (${n})`;
  const grid = document.createElement("div");
  grid.className = "card-grid small";
  cards.forEach((c) => {
    const cell = renderCardCell(c);
    const text = tag && tag(c);
    if (text) {
      const d = document.createElement("div");
      d.className = "reg-tag";
      d.textContent = text;
      cell.appendChild(d);
    }
    grid.appendChild(cell);
  });
  section.append(h, grid);
  return section;
}

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

    const deckN = d.deck.reduce((n, c) => n + (c.copies || 1), 0);
    const cap = d.pool_max ? `/${d.pool_max}` : "";
    $("build-meta").textContent =
      `${d.hero.name} — ${d.format} · ${deckN} deck cards · ` +
      `pool ${d.pool_total}${cap} · legal: ${d.legal} · ${d.rounds} LLM round(s)` +
      (d.finisher_notes.length ? `\nfinisher: ${d.finisher_notes.join("; ")}` : "") +
      (d.warnings.length ? `\n${d.warnings.map((w) => "note: " + w).join("\n")}` : "");

    const deckEl = $("build-deck");
    deckEl.replaceChildren();
    if (d.inventory.length) {
      deckEl.appendChild(buildSection("Inventory — equipped", "loadout", d.inventory,
        (c) => c.slot));
    }
    for (const g of PITCH_GROUPS) {
      const cards = d.deck.filter(g.match);
      if (cards.length) deckEl.appendChild(buildSection(g.title, g.cls, cards));
    }
    if (d.sideboard.length) {
      deckEl.appendChild(buildSection("Sideboard — matchup tech", "sideboard", d.sideboard,
        (c) => c.reason));
    }
    $("build-explanation").textContent = d.explanation || "";
    $("build-result").classList.remove("hidden");
  } catch (err) {
    status.classList.remove("spin");
    status.textContent = "error: " + err.message;
  }
});
