"use strict";

/* ==========================================================================
   Team Builder - Oberfläche
   Zustand: state.rows (Spielerliste) ist die einzige Wahrheit. Die Tabelle wird
   nur bei Strukturänderungen neu aufgebaut; beim Tippen werden lediglich
   Validierung, Vorschläge und Zusammenfassung aktualisiert (Fokus bleibt).
   ========================================================================== */

const ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"];
const ROLE_LABEL = { TOP: "Top", JUNGLE: "Jungle", MIDDLE: "Mid", BOTTOM: "ADC", UTILITY: "Support" };
const ROLE_SHORT = { TOP: "Top", JUNGLE: "Jgl", MIDDLE: "Mid", BOTTOM: "ADC", UTILITY: "Sup" };
const APEX = new Set(["MASTER", "GRANDMASTER", "CHALLENGER"]);
const STORE_KEY = "teambuilder.v1";
const MODE_TEXT = {
  balanced: "Möglichst gleich starke Teams, gut für faire Matches.",
  strongest: "Stärke gebündelt: Team 1 so stark wie möglich, dann Team 2 usw.",
};
const STRENGTH_TIP = "Stärke = gewichtete Kombination aus Rang, LP und Winrate (Gewichte aus der Regression). 0 entspricht dem Durchschnitt im Studiendatensatz.";

const ICON = {
  warn: '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 1.5c.5 0 .9.3 1.2.7l5.6 9.9c.5.9-.1 2-1.2 2H2.4c-1 0-1.7-1.1-1.2-2l5.6-9.9c.3-.4.7-.7 1.2-.7Zm0 9.3a.9.9 0 1 0 0 1.8.9.9 0 0 0 0-1.8Zm0-5.3c-.5 0-.8.4-.8.9l.2 3.1c0 .3.3.6.6.6s.6-.3.6-.6l.2-3.1c0-.5-.3-.9-.8-.9Z"/></svg>',
  crit: '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 1a7 7 0 1 1 0 14A7 7 0 0 1 8 1Zm2.5 3.7L8 7.2 5.5 4.7l-.8.8L7.2 8l-2.5 2.5.8.8L8 8.8l2.5 2.5.8-.8L8.8 8l2.5-2.5-.8-.8Z"/></svg>',
  info: '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 1a7 7 0 1 1 0 14A7 7 0 0 1 8 1Zm0 5.6c-.4 0-.7.3-.7.7v3.9a.7.7 0 1 0 1.4 0V7.3c0-.4-.3-.7-.7-.7Zm0-2.6a.9.9 0 1 0 0 1.8A.9.9 0 0 0 8 4Z"/></svg>',
  x: '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M2 2l6 6M8 2 2 8"/></svg>',
  trash: '<svg class="i" viewBox="0 0 20 20" aria-hidden="true"><path d="M4.5 6h11M8 6V4.5h4V6m-6 0 .7 9.2c.1.7.6 1.3 1.3 1.3h4c.7 0 1.2-.6 1.3-1.3L14 6"/></svg>',
  download: '<svg class="i" viewBox="0 0 20 20" aria-hidden="true"><path d="M10 3v10m0 0-3.5-3.5M10 13l3.5-3.5M4 13.5V15a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2v-1.5"/></svg>',
  copy: '<svg class="i" viewBox="0 0 20 20" aria-hidden="true"><rect x="7" y="7" width="9.5" height="9.5" rx="2"/><path d="M13 7V5.5A2 2 0 0 0 11 3.5H5.5a2 2 0 0 0-2 2V11a2 2 0 0 0 2 2H7"/></svg>',
};

/* ---------- Helfer -------------------------------------------------------- */
const $ = (sel, root = document) => root.querySelector(sel);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

const idKey = (s) => String(s || "").replace(/\s+/g, "").toLowerCase();
const fmt = (v, d = 2) =>
  v == null ? "–" : v.toLocaleString("de-DE", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtSigned = (v) => (v == null ? "–" : (v > 0.0005 ? "+" : v < -0.0005 ? "−" : "±") + fmt(Math.abs(v)));
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

function splitId(id) {
  const i = id.lastIndexOf("#");
  return i > 0 ? [id.slice(0, i), id.slice(i)] : [id, ""];
}

function rankText(m) {
  if (!m.ranked) return "Unranked";
  const tier = m.tier.charAt(0) + m.tier.slice(1).toLowerCase();
  return APEX.has(m.tier) ? tier : `${tier} ${m.division}`;
}

/* ---------- Zustand ------------------------------------------------------- */
let uid = 0;
const state = {
  rows: [],
  status: null,
  result: null,        // letzte Antwort von /api/build
  builtFrom: null,     // Eingabe-Signatur der letzten Berechnung
  view: { mode: "balanced", solver: null },
  building: false,
  timeLimit: 60,
};
const rowEls = new Map();

const newRow = (d = {}) => ({
  id: ++uid,
  riot_id: d.riot_id || "",
  main: d.main || "",
  seconds: [...(d.seconds || [])],
  with: [...(d.with || [])],
  not_with: [...(d.not_with || [])],
});

const isEmptyRow = (r) =>
  !r.riot_id.trim() && !r.main && !r.seconds.length && !r.with.length && !r.not_with.length;

function save() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify({
      rows: state.rows.filter((r) => !isEmptyRow(r)).map(({ id, ...r }) => r),
      timeLimit: state.timeLimit,
    }));
  } catch { /* Speicher nicht verfügbar - egal */ }
}

function restore() {
  try {
    const d = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
    if (!d) return;
    state.rows = (d.rows || []).map(newRow);
    if (d.timeLimit) state.timeLimit = d.timeLimit;
  } catch { /* ignorieren */ }
}

function payloadRows() {
  return state.rows.filter((r) => !isEmptyRow(r)).map((r) => ({
    riot_id: r.riot_id.trim(), main: r.main, seconds: r.seconds, with: r.with, not_with: r.not_with,
  }));
}

/* ---------- Mehrere Riot IDs aus Text (OP.GG, Lobby-Chat, Listen) ---------- */
const LOBBY_SUFFIX = /\s+(?:joined the lobby|left the lobby|hat die lobby betreten|hat die lobby verlassen)\.?\s*$/i;

function parseRiotIds(text) {
  let src = String(text || "");
  // OP.GG-Multisearch-Link: ...?summoners=Name%23TAG%2CName2%23TAG2
  const url = src.match(/[?&]summoners=([^&\s]+)/i);
  if (url) {
    try { src = decodeURIComponent(url[1].replace(/\+/g, " ")); } catch { /* Rohtext nutzen */ }
  }
  const seen = new Set();
  const ids = [];
  const invalid = [];
  for (let part of src.split(/[\r\n,;]+/)) {
    part = part.replace(LOBBY_SUFFIX, "").replace(/\s*#\s*/g, "#").replace(/\s+/g, " ").trim();
    if (!part) continue;
    if (!/^[^#]+#[^#\s]+$/.test(part)) { invalid.push(part); continue; }
    const k = idKey(part);
    if (seen.has(k)) continue;
    seen.add(k);
    ids.push(part);
  }
  return { ids, invalid };
}

function addPlayers(ids, focusRowId = null) {
  const have = new Set(state.rows.map((r) => idKey(r.riot_id)).filter(Boolean));
  const fresh = ids.filter((id) => !have.has(idKey(id)));
  if (!fresh.length) { toast("Diese Spieler stehen schon in der Liste."); return; }
  const before = state.rows.slice();
  const added = fresh.map((riot_id) => newRow({ riot_id }));
  state.rows = state.rows.filter((r) => !isEmptyRow(r)).concat(added);
  renderRows(null, new Set(added.map((r) => r.id)));
  save();
  const first = rowEls.get(focusRowId ?? added[0].id);
  first?.scrollIntoView({ block: "center", behavior: "smooth" });
  first?.querySelector("select.main")?.focus({ preventScroll: true });
  const skipped = ids.length - fresh.length;
  toast(`${plural(added.length, "Spieler", "Spieler")} hinzugefügt${skipped ? ` (${skipped} schon vorhanden)` : ""} · jetzt Rollen wählen`,
    "Rückgängig", () => { state.rows = before; renderRows(); save(); });
}

function openPasteDialog() {
  $("#paste-input").value = "";
  renderPastePreview();
  $("#paste-dialog").showModal();
  $("#paste-input").focus();
}

function pasteCandidates() {
  const { ids, invalid } = parseRiotIds($("#paste-input").value);
  const have = new Set(state.rows.map((r) => idKey(r.riot_id)).filter(Boolean));
  return { ids, invalid, have, fresh: ids.filter((id) => !have.has(idKey(id))) };
}

function renderPastePreview() {
  const { ids, invalid, have, fresh } = pasteCandidates();
  const btn = $("#paste-add");
  btn.disabled = !fresh.length;
  btn.textContent = fresh.length ? `${plural(fresh.length, "Spieler", "Spieler")} hinzufügen` : "Hinzufügen";
  const box = $("#paste-preview");
  if (!ids.length && !invalid.length) return box.replaceChildren();
  const dupes = ids.length - fresh.length;
  box.replaceChildren(
    el("div", { class: "pp-summary" },
      el("span", {}, el("strong", {}, fresh.length), fresh.length === 1 ? " neuer Spieler erkannt" : " neue Spieler erkannt"),
      dupes ? el("span", { class: "pp-muted" }, `${dupes} schon in der Liste`) : null,
      invalid.length ? el("span", {
        class: "pp-warn", "data-tip": `Ohne #TAG, wird übersprungen: ${invalid.join(", ")}`,
        html: `${ICON.warn}${plural(invalid.length, "Eintrag", "Einträge")} ohne #TAG`,
      }) : null),
    el("div", { class: "pp-list" }, ids.map((id) => {
      const [name, tag] = splitId(id);
      const dupe = have.has(idKey(id));
      return el("span", { class: `token${dupe ? " is-dupe" : ""}`, "data-tip": dupe ? `${id} steht schon in der Liste` : id },
        el("span", { class: "token-name" }, name, el("span", { class: "token-tag" }, tag)));
    })));
}

function submitPaste(e) {
  e.preventDefault();
  const { fresh, ids } = pasteCandidates();
  if (!fresh.length) return;
  $("#paste-dialog").close();
  addPlayers(ids);
}

/* ---------- Validierung --------------------------------------------------- */
function analyze() {
  const counts = new Map();
  for (const r of state.rows) {
    const k = idKey(r.riot_id);
    if (k) counts.set(k, (counts.get(k) || 0) + 1);
  }
  const errors = new Map();
  const ids = new Map(); // idKey -> kanonische Schreibweise
  let valid = 0;
  for (const r of state.rows) {
    if (isEmptyRow(r)) continue;
    const e = {};
    const id = r.riot_id.trim();
    if (!id) e.riot = "Riot ID fehlt";
    else if (!/^[^#]+#[^#\s]+$/.test(id)) e.riot = "Format: Name#TAG";
    else if (counts.get(idKey(id)) > 1) e.riot = "Steht doppelt in der Liste";
    if (!r.main) e.main = "Main-Rolle wählen";
    if (Object.keys(e).length) errors.set(r.id, e);
    else valid++;
    if (id && !e.riot) ids.set(idKey(id), id);
  }
  // Präferenzen, die andere über einen Spieler angegeben haben (gelten beidseitig)
  const ghosts = new Map();
  for (const a of state.rows) {
    const aKey = idKey(a.riot_id);
    if (!ids.has(aKey)) continue;
    for (const key of ["with", "not_with"]) {
      for (const t of a[key]) {
        const b = state.rows.find((r) => idKey(r.riot_id) === idKey(t));
        if (!b || b === a || b[key].some((x) => idKey(x) === aKey)) continue;
        if (!ghosts.has(b.id)) ghosts.set(b.id, { with: [], not_with: [] });
        ghosts.get(b.id)[key].push(a.riot_id.trim());
      }
    }
  }
  const k = Math.floor(valid / 5);
  const named = state.rows.filter((r) => !isEmptyRow(r)).length;
  // Zeilen, denen nur die Main-Rolle fehlt (offener Schritt statt Fehler)
  const onlyMain = [...errors.values()].filter((e) => e.main && !e.riot).length;
  return { errors, valid, ids, ghosts, k, reserves: valid - 5 * k, named, onlyMain };
}

/* ---------- Tabelle ------------------------------------------------------- */
function renderRows(focusId = null, animate = null) {
  const box = $("#rows");
  const anim = animate instanceof Set ? animate : new Set(animate != null ? [animate] : []);
  rowEls.clear();
  box.replaceChildren(...state.rows.map((r) => {
    const node = rowEl(r);
    if (anim.has(r.id)) node.classList.add("is-new");
    rowEls.set(r.id, node);
    return node;
  }));
  refresh();
  if (focusId != null) {
    const input = rowEls.get(focusId)?.querySelector(".riot");
    if (input) { input.focus(); input.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
  }
}

function rowEl(r) {
  const riot = el("input", {
    class: "field riot", value: r.riot_id, placeholder: "Name#TAG", autocomplete: "off",
    spellcheck: "false", "aria-label": "Riot ID",
  });
  riot.addEventListener("input", () => { r.riot_id = riot.value; changed(); });
  riot.addEventListener("blur", () => {
    const t = riot.value.replace(/\s*#\s*/, "#").trim();
    if (t !== riot.value) { riot.value = t; r.riot_id = t; save(); }
    refresh(); // Fehler erst nach dem Verlassen des Feldes zeigen
  });
  riot.addEventListener("paste", (e) => {
    // Mehrere IDs (oder eine Lobby-Zeile) -> automatisch auf Zeilen verteilen
    const text = e.clipboardData?.getData("text") || "";
    const { ids } = parseRiotIds(text);
    if (!ids.length || (ids.length === 1 && ids[0] === text.trim())) return; // normales Einfügen
    if (ids.length === 1 && riot.value.trim()) return;
    e.preventDefault();
    if (!riot.value.trim()) {
      r.riot_id = ids.shift();
      riot.value = r.riot_id;
    }
    if (ids.length) addPlayers(ids, r.id);
    else changed();
  });
  riot.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    e.preventDefault();
    const i = state.rows.indexOf(r);
    const next = state.rows[i + 1];
    if (next) rowEls.get(next.id)?.querySelector(".riot")?.focus();
    else addRow();
  });

  const main = el("select", { class: "field main", "aria-label": "Main-Rolle" },
    el("option", { value: "" }, "Wählen …"),
    ROLES.map((role) => el("option", { value: role }, ROLE_LABEL[role])));
  main.value = r.main;
  main.addEventListener("change", () => {
    r.main = main.value;
    r.seconds = r.seconds.filter((x) => x !== r.main);
    updateChips(node, r);
    changed();
  });

  const chips = el("div", { class: "chips", role: "group", "aria-label": "Zweitrollen" },
    ROLES.map((role) => el("button", {
      type: "button", class: "chip", "data-role": role,
      onclick: () => {
        if (role === r.main) return;
        r.seconds = r.seconds.includes(role) ? r.seconds.filter((x) => x !== role)
          : ROLES.filter((x) => x === role || r.seconds.includes(x));
        updateChips(node, r);
        changed();
      },
    }, ROLE_SHORT[role])));

  const del = el("button", {
    type: "button", class: "icon-btn", "aria-label": "Spieler entfernen", html: ICON.trash,
    onclick: () => removeRow(r),
  });

  const node = el("div", { class: "prow", role: "row", "data-id": r.id },
    el("div", { class: "cell cell-idx", role: "cell" }),
    el("div", { class: "cell c-riot", role: "cell" }, el("span", { class: "cell-label" }, "Riot ID"), riot, el("p", { class: "cell-msg" })),
    el("div", { class: "cell c-main", role: "cell" }, el("span", { class: "cell-label" }, "Main-Rolle"), main, el("p", { class: "cell-msg" })),
    el("div", { class: "cell c-roles", role: "cell" }, el("span", { class: "cell-label" }, "Zweitrollen"), chips),
    el("div", { class: "cell c-with", role: "cell" }, el("span", { class: "cell-label" }, "Mit"), tokenField(r, "with")),
    el("div", { class: "cell c-not", role: "cell" }, el("span", { class: "cell-label" }, "Nicht mit"), tokenField(r, "not_with")),
    el("div", { class: "cell c-del", role: "cell" }, del));
  updateChips(node, r);
  return node;
}

function updateChips(node, r) {
  node.querySelectorAll(".chip").forEach((chip) => {
    const role = chip.dataset.role;
    const isMain = role === r.main;
    chip.classList.toggle("is-main", isMain);
    chip.setAttribute("aria-pressed", String(!isMain && r.seconds.includes(role)));
    chip.title = isMain ? "Main-Rolle" : r.seconds.includes(role) ? "Zweitrolle – klicken zum Entfernen" : "Als Zweitrolle hinzufügen";
  });
  const sel = node.querySelector("select.main");
  sel.classList.toggle("is-empty", !r.main);
}

function tokenField(r, key) {
  const input = el("input", {
    class: "token-input", list: "riot-ids", autocomplete: "off", spellcheck: "false",
    "aria-label": key === "with" ? "Mit Spieler" : "Nicht mit Spieler",
  });
  const pills = el("span", { class: "pills" });
  const wrap = el("div", { class: "tokens", "data-key": key }, pills, input);
  wrap.addEventListener("mousedown", (e) => {
    if (e.target === wrap) { e.preventDefault(); input.focus(); }
  });

  const commit = () => {
    const parts = input.value.split(/[,;]/).map((s) => s.trim()).filter(Boolean);
    input.value = "";
    if (parts.length) { parts.forEach((p) => addToken(r, key, p)); changed(); }
  };
  input.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === "," || e.key === ";" || e.key === "Tab") && input.value.trim()) {
      if (e.key !== "Tab") e.preventDefault();
      commit();
    } else if (e.key === "Backspace" && !input.value && r[key].length) {
      r[key].pop();
      changed();
    }
  });
  input.addEventListener("input", (e) => {
    // Auswahl aus der Vorschlagsliste übernimmt sofort
    if (!e.inputType || e.inputType === "insertReplacementText") {
      const known = analyze().ids;
      if (known.has(idKey(input.value))) commit();
    }
  });
  input.addEventListener("blur", commit);
  return wrap;
}

function addToken(r, key, raw) {
  const a = analyze();
  const k = idKey(raw);
  const value = a.ids.get(k) || raw;
  const other = key === "with" ? "not_with" : "with";
  if (k === idKey(r.riot_id)) return toast("Ein Spieler kann sich nicht selbst angeben.");
  if (r[key].some((x) => idKey(x) === k)) return;
  if (r[other].some((x) => idKey(x) === k)) {
    return toast(`${splitId(value)[0]} steht schon bei „${other === "with" ? "Mit" : "Nicht mit"}“.`);
  }
  r[key].push(value);
}

function renderTokens(node, r, a) {
  for (const key of ["with", "not_with"]) {
    const wrap = node.querySelector(`.tokens[data-key="${key}"]`);
    const pills = wrap.querySelector(".pills");
    const ghosts = a.ghosts.get(r.id)?.[key] || [];
    const items = r[key].map((t, i) => {
      const known = a.ids.has(idKey(t));
      const [name, tag] = splitId(t);
      return el("span", {
        class: `token${known ? "" : " is-unknown"}`,
        "data-tip": known ? t : `${t} ist nicht in der Spielerliste und wird ignoriert.`,
      },
      el("span", { class: "token-name" }, name, el("span", { class: "token-tag" }, tag)),
      el("button", {
        type: "button", class: "token-x", "aria-label": `${t} entfernen`, html: ICON.x,
        onclick: (e) => { e.stopPropagation(); r[key].splice(i, 1); changed(); },
      }));
    });
    const ghostEls = ghosts.map((g) => el("span", {
      class: "token is-ghost",
      "data-tip": `${g} hat dich angegeben. Präferenzen gelten automatisch für beide.`,
    }, el("span", { class: "token-name" }, splitId(g)[0])));
    pills.replaceChildren(...items, ...ghostEls);
    wrap.querySelector(".token-input").placeholder = items.length || ghostEls.length ? "" : "Riot ID …";
  }
}

/* Alles Abgeleitete aktualisieren, ohne die Eingabefelder neu zu bauen */
function refresh() {
  const a = analyze();
  state.rows.forEach((r, i) => {
    const node = rowEls.get(r.id);
    if (!node) return;
    node.querySelector(".cell-idx").textContent = i + 1;
    const e = a.errors.get(r.id) || {};
    const showErr = (cell, msg, touched) => {
      const c = node.querySelector(cell);
      const show = Boolean(msg && touched);
      c.classList.toggle("has-error", show);
      c.querySelector(".cell-msg").textContent = show ? msg : "";
    };
    // Fehler nicht schon beim Tippen zeigen: Riot ID erst nach Verlassen des
    // Feldes, Main-Rolle erst wenn eine Riot ID steht und man weitergegangen ist
    const typingRiot = document.activeElement === node.querySelector(".riot");
    const hasId = r.riot_id.trim().length > 0;
    showErr(".c-riot", e.riot, !typingRiot && (hasId || r.main));
    // Fehlende Main-Rolle ist ein offener Schritt, kein Fehler: dezent hervorheben,
    // rot erst nach "Teams bilden" (focusFirstError)
    if (r.main) showErr(".c-main", null, false);
    const sel = node.querySelector("select.main");
    sel.classList.toggle("needs", hasId && !r.main);
    renderTokens(node, r, a);
  });

  $("#riot-ids").replaceChildren(...[...a.ids.values()].map((id) => el("option", { value: id })));

  const empty = state.rows.length === 0;
  $("#empty").hidden = !empty;
  $("#ptable").classList.toggle("is-empty", empty);
  $("#add-row").hidden = empty;

  $("#player-summary").textContent = empty
    ? "Noch keine Spieler"
    : a.errors.size
      ? `${plural(a.named, "Spieler", "Spieler")} · ${a.errors.size === a.onlyMain
        ? `${a.onlyMain} noch ohne Main-Rolle` : `${a.errors.size} noch unvollständig`}`
    : a.valid < 5
      ? `${plural(a.valid, "Spieler", "Spieler")} · mindestens 5 für ein Team`
      : `${a.valid} Spieler · ${plural(a.k, "Team", "Teams")} à 5${a.reserves ? ` · ${a.reserves} auf der Ersatzbank` : ""}`;

  updateActionbar(a);
  updateStale();
  return a;
}

function changed() {
  refresh();
  save();
}

function addRow(data) {
  const r = newRow(data);
  state.rows.push(r);
  renderRows(r.id, r.id);
  save();
}

function removeRow(r) {
  const index = state.rows.indexOf(r);
  state.rows.splice(index, 1);
  renderRows();
  save();
  if (isEmptyRow(r)) return;
  toast(`${r.riot_id.trim() || "Spieler"} entfernt`, "Rückgängig", () => {
    state.rows.splice(Math.min(index, state.rows.length), 0, r);
    renderRows(null, r.id);
    save();
  });
}

/* ---------- Aktionsleiste ------------------------------------------------- */
function keyState() {
  const k = state.status?.key;
  if (!k) return "unknown";
  if (!k.present) return "missing";
  if (k.valid === false) return "invalid";
  return k.valid ? "ok" : "unknown";
}

function updateActionbar(a = analyze()) {
  const btn = $("#build-btn");
  const msg = $("#ab-msg");
  msg.className = "ab-msg";
  msg.replaceChildren();
  let enabled = true;

  if (state.building) {
    enabled = false;
    msg.append("Berechnung läuft …");
  } else if (a.errors.size > 0 && a.errors.size === a.onlyMain) {
    // nur fehlende Main-Rollen: neutraler Hinweis auf den nächsten Schritt
    enabled = false;
    msg.append(`${plural(a.onlyMain, "Spieler", "Spieler")} ohne Main-Rolle · `,
      el("button", { type: "button", onclick: focusFirstError }, "Zur ersten"));
  } else if (a.errors.size) {
    enabled = false;
    msg.classList.add("warn");
    msg.insertAdjacentHTML("beforeend", ICON.warn.replace("<svg", '<svg class="i"'));
    msg.append(`${plural(a.errors.size, "Zeile braucht", "Zeilen brauchen")} noch Angaben · `,
      el("button", { type: "button", onclick: focusFirstError }, "Anzeigen"));
  } else if (a.valid < 5) {
    enabled = false;
    msg.append(a.valid === 0 ? "Füge mindestens 5 Spieler hinzu." : `Noch ${5 - a.valid} Spieler bis zum ersten Team.`);
  } else {
    const ks = keyState();
    if (ks === "missing" || ks === "invalid") {
      msg.classList.add("warn");
      msg.insertAdjacentHTML("beforeend", ICON.warn.replace("<svg", '<svg class="i"'));
      msg.append(ks === "missing" ? "Riot API Key fehlt · " : "Riot API Key ungültig oder abgelaufen · ",
        el("button", { type: "button", onclick: openKeyDialog }, "Key hinterlegen"));
    } else {
      msg.append(el("strong", {}, `${a.valid} Spieler`),
        ` → ${plural(a.k, "Team", "Teams")}${a.reserves ? ` + ${a.reserves} Ersatz` : ""}`);
    }
  }
  btn.setAttribute("aria-disabled", String(!enabled));
  btn.textContent = state.building ? "Wird berechnet …" : state.result ? "Neu berechnen" : "Teams bilden";
}

function focusFirstError() {
  const a = analyze();
  const r = state.rows.find((x) => a.errors.has(x.id));
  if (!r) return;
  const e = a.errors.get(r.id);
  const node = rowEls.get(r.id);
  // Fehler sichtbar machen, auch wenn die Zeile noch "frisch" ist
  node.querySelector(".c-riot").classList.toggle("has-error", Boolean(e.riot));
  node.querySelector(".c-riot .cell-msg").textContent = e.riot || "";
  node.querySelector(".c-main").classList.toggle("has-error", Boolean(e.main));
  node.querySelector(".c-main .cell-msg").textContent = e.main || "";
  node.scrollIntoView({ block: "center", behavior: "smooth" });
  (e.riot ? node.querySelector(".riot") : node.querySelector("select.main")).focus({ preventScroll: true });
}

/* ---------- Status (API-Key, Gurobi) -------------------------------------- */
async function loadStatus() {
  try {
    const res = await fetch("/api/status");
    state.status = await res.json();
  } catch {
    state.status = null;
  }
  renderStatus();
}

function renderStatus() {
  const s = state.status;
  const keyPill = $("#key-pill");
  const ks = keyState();
  keyPill.className = `pill ${{ ok: "ok", missing: "warn", invalid: "bad" }[ks] || ""}`;
  keyPill.querySelector(".pill-text").textContent =
    { ok: "Riot API", missing: "API Key fehlt", invalid: "API Key ungültig" }[ks] || "Riot API";
  keyPill.dataset.tip = {
    ok: `API Key gültig (${s?.key?.source}). Klicken zum Ändern.`,
    missing: "Kein Riot API Key gefunden. Klicken zum Hinterlegen.",
    invalid: "Die Riot API lehnt den Key ab. Development Keys laufen nach 24 h ab.",
  }[ks] || "Key konnte nicht geprüft werden (offline?). Klicken zum Ändern.";

  const gp = $("#gurobi-pill");
  gp.className = `pill ${s?.gurobi ? "ok" : ""}`;
  gp.querySelector(".pill-text").textContent = s?.gurobi ? "Gurobi" : "Nur Heuristik";
  gp.dataset.tip = s?.gurobi
    ? "Gurobi ist installiert: zusätzlich zur Heuristik wird die exakt optimale Einteilung berechnet."
    : "Gurobi ist nicht installiert: Teams werden mit der Heuristik (Simulated Annealing) gebildet. Für exakte Lösungen: pip install gurobipy + Lizenz.";
  $("#region-pill").textContent = s?.region || "EUW";
  $("#region-pill").dataset.tip = "Region der Riot API";
  $("#tl-wrap").hidden = !s?.gurobi;
  $("#tl-select").value = String(state.timeLimit);
  updateActionbar();
}

function openKeyDialog() {
  const d = $("#key-dialog");
  const ks = keyState();
  $("#key-state").textContent = {
    ok: `Aktuell ist ein gültiger Key hinterlegt (${state.status.key.source}).`,
    missing: "Für die Ränge der Spieler wird ein Riot API Key benötigt.",
    invalid: `Der hinterlegte Key (${state.status?.key?.source}) wird von der Riot API abgelehnt, ist also ungültig oder abgelaufen.`,
  }[ks] || "Der Key konnte gerade nicht geprüft werden.";
  $("#key-input").value = "";
  $("#key-error").hidden = true;
  d.showModal();
  $("#key-input").focus();
}

async function saveKey(e) {
  e.preventDefault();
  const key = $("#key-input").value.trim();
  const err = $("#key-error");
  const btn = $("#key-save");
  if (!key) { err.textContent = "Bitte einen Key eingeben."; err.hidden = false; return; }
  btn.disabled = true;
  btn.textContent = "Wird geprüft …";
  try {
    const res = await fetch("/api/key", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    state.status = data;
    renderStatus();
    $("#key-dialog").close();
    toast(data.key.valid ? "API Key gespeichert und gültig." : "API Key gespeichert (konnte nicht geprüft werden).");
  } catch (ex) {
    err.textContent = ex.message || "Speichern fehlgeschlagen.";
    err.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = "Prüfen & speichern";
  }
}

/* ---------- CSV-Import / -Export ------------------------------------------ */
async function importFile(file) {
  if (!file) return;
  if (!/\.(csv|txt)$/i.test(file.name) && file.type && !file.type.includes("csv") && !file.type.startsWith("text/")) {
    return toast("Bitte eine CSV-Datei wählen.");
  }
  try {
    const res = await fetch("/api/parse-csv", { method: "POST", body: await file.arrayBuffer() });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    await applyImport(data, file.name);
  } catch (ex) {
    toast(ex.message || "Import fehlgeschlagen.");
  }
}

async function loadExample() {
  try {
    const res = await fetch("/api/example");
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    await applyImport(data, "Beispiel");
  } catch (ex) {
    toast(ex.message || "Beispiel konnte nicht geladen werden.");
  }
}

function askImportMode(count, label) {
  const existing = state.rows.filter((r) => !isEmptyRow(r)).length;
  if (!existing) return Promise.resolve("replace");
  const d = $("#import-dialog");
  $("#import-text").textContent =
    `„${label}“ enthält ${plural(count, "Spieler", "Spieler")}. Die aktuelle Liste ersetzen (${existing} Spieler) oder die neuen anhängen?`;
  d.returnValue = "cancel";
  d.showModal();
  return new Promise((resolve) => d.addEventListener("close", () => resolve(d.returnValue), { once: true }));
}

async function applyImport(data, label) {
  const mode = await askImportMode(data.players.length, label);
  if (mode === "cancel") return;
  let incoming = data.players.map(newRow);
  let skipped = 0;
  if (mode === "append") {
    const have = new Set(state.rows.map((r) => idKey(r.riot_id)));
    const before = incoming.length;
    incoming = incoming.filter((r) => !have.has(idKey(r.riot_id)));
    skipped = before - incoming.length;
    state.rows = state.rows.filter((r) => !isEmptyRow(r)).concat(incoming);
  } else {
    state.rows = incoming;
  }
  renderRows();
  save();
  const notes = data.notes?.length ? ` · ${plural(data.notes.length, "Hinweis", "Hinweise")} – bitte markierte Zeilen prüfen` : "";
  toast(`${plural(incoming.length, "Spieler", "Spieler")} importiert${skipped ? ` (${skipped} schon vorhanden)` : ""}${notes}`);
  if (data.notes?.length) console.info("Import-Hinweise:", data.notes);
}

function csvCell(v) {
  const s = String(v ?? "");
  return /[;"\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function download(name, text) {
  const blob = new Blob(["﻿" + text], { type: "text/csv;charset=utf-8" });
  const a = el("a", { href: URL.createObjectURL(blob), download: name });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
}

function exportPlayers() {
  const rows = payloadRows();
  if (!rows.length) return toast("Keine Spieler zum Exportieren.");
  const lines = [["RIOT ID", "Main Position", "Second Positions", "With", "Not With"].join(";")];
  for (const r of rows) {
    lines.push([r.riot_id, r.main ? ROLE_LABEL[r.main] : "", r.seconds.map((x) => ROLE_LABEL[x]).join(","),
      r.with.join(","), r.not_with.join(",")].map(csvCell).join(";"));
  }
  download("spieler.csv", lines.join("\r\n") + "\r\n");
}

/* ---------- Berechnung ---------------------------------------------------- */
async function build() {
  const btn = $("#build-btn");
  if (btn.getAttribute("aria-disabled") === "true") {
    const a = analyze();
    if (a.errors.size) focusFirstError();
    return;
  }
  if (["missing", "invalid"].includes(keyState())) return openKeyDialog();

  const players = payloadRows();
  const signature = JSON.stringify(players);
  state.building = true;
  updateActionbar();
  const progress = showProgress(players.length);
  let finished = false;

  try {
    const res = await fetch("/api/build", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ players, time_limit: state.timeLimit }),
    });
    if (!res.ok || !res.body) throw new Error((await res.json().catch(() => ({}))).error || `Serverfehler ${res.status}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === "result") {
          finished = true;
          state.result = ev;
          state.builtFrom = signature;
          const hasGurobi = ev.results.some((x) => x.solver === "gurobi");
          state.view.solver = hasGurobi ? "gurobi" : "heuristik";
          renderResults(true);
        } else if (ev.type === "error") {
          finished = true;
          if (/API Key/.test(ev.message)) loadStatus();
          showError(ev.message);
        } else {
          progress.update(ev);
        }
      }
    }
    if (!finished) throw new Error("Die Verbindung zum Server wurde unterbrochen.");
  } catch (ex) {
    showError(ex.message === "Failed to fetch"
      ? "Der lokale Server ist nicht erreichbar. Läuft webapp.py noch?"
      : ex.message);
  } finally {
    progress.stop();
    state.building = false;
    updateActionbar();
  }
}

function showProgress(total) {
  const section = $("#results");
  const gurobi = Boolean(state.status?.gurobi);
  const steps = [
    { id: "ranks", text: "Ränge von der Riot API laden" },
    { id: "teams", text: "Rollen verteilen und Teams optimieren" },
    ...(gurobi ? [{ id: "gurobi", text: "Exakte Lösung mit Gurobi" }] : []),
  ];
  const stepEls = new Map();
  const list = el("ol", { class: "steps" }, steps.map((s) => {
    const meta = el("span", { class: "s-meta" });
    const li = el("li", { class: "step" }, el("span", { class: "s-icon" }), el("span", {}, s.text), meta);
    stepEls.set(s.id, { li, meta });
    return li;
  }));
  const fill = el("div", { class: "pbar-fill" });
  const bar = el("div", { class: "pbar" }, fill);
  const caption = el("p", { class: "p-caption" }, "Verbinde …");
  section.replaceChildren(el("div", { class: "card progress-card" },
    el("h3", {}, "Teams werden gebildet"),
    el("p", { class: "p-sub" }, `${plural(total, "Spieler", "Spieler")} · bereits geladene Ränge werden 15 Minuten zwischengespeichert`),
    list, bar, caption));
  section.hidden = false;
  section.scrollIntoView({ behavior: "smooth", block: "start" });

  let waitTimer = null;
  const activate = (id) => {
    let reached = false;
    for (const s of steps) {
      const { li } = stepEls.get(s.id);
      if (s.id === id) { reached = true; li.className = "step active"; }
      else li.className = reached ? "step" : "step done";
    }
  };
  activate("ranks");

  return {
    update(ev) {
      if (ev.type === "stage") {
        activate(ev.stage);
        if (ev.stage === "teams") {
          stepEls.get("ranks").meta.textContent = `${total} / ${total}`;
          bar.classList.add("indeterminate");
          caption.textContent = "Rollen und Präferenzen werden abgeglichen …";
        }
        if (ev.stage === "gurobi") {
          caption.textContent = `Gurobi rechnet – je Schritt bis zu ${ev.time_limit} s.`;
          stepEls.get("gurobi").meta.textContent = `max. ${Math.round(3 * ev.time_limit / 60 * 10) / 10} min`;
        }
      } else if (ev.type === "progress") {
        clearInterval(waitTimer);
        stepEls.get("ranks").meta.textContent = `${ev.done} / ${ev.total}`;
        fill.style.width = `${(ev.done / ev.total) * 100}%`;
        caption.textContent = `${ev.player}${ev.ranked ? "" : " – kein Solo/Duo-Rang"}`;
      } else if (ev.type === "wait") {
        let left = ev.seconds;
        clearInterval(waitTimer);
        const tick = () => { caption.textContent = `Rate-Limit der Riot API erreicht – weiter in ${left} s …`; left--; if (left < 0) clearInterval(waitTimer); };
        tick();
        waitTimer = setInterval(tick, 1000);
      } else if (ev.type === "log" && ev.message) {
        caption.textContent = ev.message.replace(/^\[.*?\]\s*/, "");
      }
    },
    stop() { clearInterval(waitTimer); },
  };
}

function showError(message) {
  const section = $("#results");
  const keyIssue = /API Key/.test(message);
  section.replaceChildren(el("div", { class: "card error-card", role: "alert" },
    el("div", { class: "e-icon", html: ICON.warn }),
    el("div", {},
      el("h3", {}, "Teams konnten nicht gebildet werden"),
      el("p", {}, message),
      keyIssue
        ? el("button", { type: "button", class: "btn btn-primary", onclick: openKeyDialog }, "Neuen Key hinterlegen")
        : el("button", { type: "button", class: "btn btn-secondary", onclick: () => $("#players-card").scrollIntoView({ behavior: "smooth" }) }, "Zur Spielerliste"))));
  section.hidden = false;
  state.result = null;
  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------- Ergebnis ------------------------------------------------------ */
function currentResult() {
  const all = state.result?.results || [];
  const { mode, solver } = state.view;
  return all.find((r) => r.mode === mode && r.solver === solver)
    || all.find((r) => r.mode === mode) || all[0];
}

function segmented(options, value, onChange, label) {
  return el("div", { class: "seg", role: "group", "aria-label": label },
    options.map((o) => el("button", {
      type: "button", "aria-pressed": String(o.value === value), "data-tip": o.tip,
      onclick: () => onChange(o.value),
    }, o.label, o.badge ? el("span", { class: "seg-badge" }, o.badge) : null)));
}

function renderResults(scroll = false) {
  const section = $("#results");
  const data = state.result;
  const res = currentResult();
  if (!data || !res) return;
  const all = data.results;
  const hasGurobi = all.some((r) => r.solver === "gurobi" && r.mode === state.view.mode);

  const head = el("div", { class: "results-head" },
    el("h2", {}, "Ergebnis"),
    el("div", { class: "res-controls" },
      segmented([
        { value: "balanced", label: "Ausgeglichen", tip: MODE_TEXT.balanced },
        { value: "strongest", label: "Stärkste", tip: MODE_TEXT.strongest },
      ], state.view.mode, (v) => { state.view.mode = v; renderResults(); }, "Modus"),
      hasGurobi ? segmented([
        { value: "heuristik", label: "Heuristik", tip: "Simulated Annealing, schnell und meist sehr nah am Optimum." },
        { value: "gurobi", label: "Gurobi", tip: "Exakter Solver: beweisbar optimal, sofern im Zeitlimit gelöst." },
      ], res.solver, (v) => { state.view.solver = v; renderResults(); }, "Verfahren") : null,
      el("button", { type: "button", class: "btn btn-icon", "aria-label": "Als Text kopieren", "data-tip": "Als Text kopieren (z. B. für Discord)", html: ICON.copy, onclick: () => copyResult(res) }),
      el("button", {
        type: "button", class: "btn btn-secondary", html: `${ICON.download}CSV`, "data-tip": "Diese Einteilung als CSV herunterladen",
        onclick: () => download(`teams_${res.id}.csv`, res.csv),
      })));

  const solverText = res.solver === "gurobi"
    ? `Gurobi · ${res.status === "optimal" ? "beweisbar optimal" : res.status}${res.gap ? ` · Gap ${fmt(res.gap * 100)} %` : ""}`
    : "Heuristik · Simulated Annealing";
  const meta = el("p", { class: "res-meta" },
    MODE_TEXT[res.mode], el("span", { class: "sep" }, "·"), solverText);

  // replaceChildren würde null als Text "null" einfügen -> herausfiltern
  section.replaceChildren(...[
    el("div", { id: "stale-slot" }),
    head, meta,
    kpiRow(res, data.summary),
    strengthChart(res),
    el("div", { class: "teams-grid" }, res.teams.map((t, i) => teamCard(t, i))),
    res.reserves.length ? benchCard(res.reserves) : null,
    notes(res, data.summary),
  ].filter(Boolean));
  section.hidden = false;
  updateStale();
  if (scroll) section.scrollIntoView({ behavior: "smooth", block: "start" });
}

function kpiRow(res, summary) {
  const m = res.metrics;
  const main = m.active - m.second - m.foreign;
  const tile = (label, value, note, tip) => el("div", { class: "card kpi", "data-tip": tip },
    el("div", { class: "kpi-label" }, label), el("div", { class: "kpi-value" }, value), el("div", { class: "kpi-note" }, note));

  const roleNote = m.foreign
    ? el("span", { class: "badge crit", html: `${ICON.crit}${m.foreign} auf nicht angegebener Rolle` })
    : "Spieler auf einer Zweitrolle";
  return el("div", { class: "kpis" },
    tile("Spanne Ø-Stärke", fmt(m.spread), res.mode === "balanced" ? "kleiner = ausgeglichener" : "stärkstes vs. schwächstes Team",
      "Differenz zwischen der höchsten und der niedrigsten durchschnittlichen Teamstärke."),
    tile("Auf Main-Rolle", el("span", {}, main, el("small", {}, ` / ${m.active}`)), "Spieler",
      "Wie viele Spieler ihre Main-Rolle spielen. Die Main-Rolle hat immer Vorrang vor der Teamstärke."),
    tile("Zweitrolle", String(m.second), roleNote,
      "Spieler auf einer Zweitrolle. Das passiert nur, wenn es sonst keine gültige Verteilung gibt."),
    tile("Präferenzen", m.pref_total ? el("span", {}, m.pref_ok, el("small", {}, ` / ${m.pref_total}`)) : "–",
      m.pref_total ? "„Mit“ / „Nicht mit“ erfüllt" : "keine angegeben",
      "„Mit“- und „Nicht mit“-Wünsche werden immer vollständig erfüllt, sofern das logisch möglich ist."));
}

function strengthChart(res) {
  const means = res.teams.map((t) => t.mean);
  const lo = Math.min(0, ...means);
  const hi = Math.max(0, ...means);
  const pad = (hi - lo) * 0.04 || 0.5;
  const min = lo < 0 ? lo - pad : 0;
  const max = hi > 0 ? hi + pad : 0;
  const span = max - min || 1;
  const x = (v) => ((v - min) / span) * 100;
  const zero = x(0);

  const rows = res.teams.map((t, i) => {
    const v = t.mean;
    const left = Math.min(x(v), zero);
    const width = Math.abs(x(v) - zero);
    return el("div", {
      class: "bar-row",
      "data-tip": `Team ${i + 1} · Ø ${fmtSigned(v)} · Summe ${fmtSigned(t.sum)}`,
    },
    el("span", { class: "bar-label" }, `Team ${i + 1}`),
    el("span", { class: "bar-track" },
      el("span", { class: "bar-zero", style: `left:${zero}%` }),
      el("span", { class: `bar-fill ${v >= 0 ? "pos" : "neg"}`, style: `left:${left}%;width:${Math.max(width, 0.4)}%` })),
    el("span", { class: "bar-value" }, fmtSigned(v)));
  });

  return el("div", { class: "card chart-card" },
    el("div", { class: "chart-head" },
      el("h3", {}, "Ø Stärke je Team"),
      el("p", { "data-tip": STRENGTH_TIP }, "0 = Durchschnitt im Datensatz ⓘ")),
    el("div", { class: "bars", role: "list" }, rows));
}

function memberRow(m, roleLabel) {
  const badges = [];
  if (m.kind === "second") badges.push(el("span", { class: "badge warn", html: `${ICON.warn}Zweitrolle` }));
  if (m.kind === "foreign") badges.push(el("span", { class: "badge crit", html: `${ICON.crit}Nicht angegebene Rolle` }));
  if (!m.ranked) badges.push(el("span", { class: "badge info", "data-tip": "Kein Solo/Duo-Rang gefunden. Die Stärke wird mit dem Median des Modells geschätzt." }, "geschätzt"));

  const sub = [rankText(m)];
  if (m.ranked && m.lp != null) sub.push(`${Math.round(m.lp)} LP`);
  if (m.win_rate != null) sub.push(`${Math.round(m.win_rate * 100)} % WR`);
  const subEls = [];
  sub.forEach((s, i) => { if (i) subEls.push(el("span", { class: "dotsep" }, "·")); subEls.push(s); });

  const [name, tag] = splitId(m.riot_id);
  return el("li", { class: "member" },
    el("span", { class: "m-role" }, roleLabel),
    el("span", { class: "m-main" },
      el("span", { class: "m-name", title: m.riot_id }, name, el("span", { class: "m-tag" }, tag)),
      el("span", { class: "m-sub" }, subEls, badges)),
    el("span", { class: "m-str", "data-tip": STRENGTH_TIP }, fmtSigned(m.strength)));
}

function teamCard(t, i) {
  return el("article", { class: "card team" },
    el("div", { class: "team-head" },
      el("h3", {}, `Team ${i + 1}`),
      el("span", { class: "team-avg" }, "Ø", el("strong", {}, fmtSigned(t.mean)))),
    el("ul", { class: "members" }, t.members.map((m) => memberRow(m, ROLE_LABEL[m.role]))));
}

function benchCard(reserves) {
  return el("section", { class: "card bench" },
    el("div", { class: "team-head" },
      el("h3", {}, "Ersatzbank"),
      el("span", { class: "team-avg" }, plural(reserves.length, "Spieler", "Spieler"))),
    el("ul", { class: "members" }, reserves.map((m) => memberRow(m, ROLE_LABEL[m.role]))));
}

function notes(res, summary) {
  const items = [];
  const note = (kind, title, body) => el("div", { class: `note ${kind}` },
    el("span", { html: ICON[kind] }), el("div", {}, el("strong", {}, title), " ", body));
  if (summary.unranked.length) {
    items.push(note("info", "Ohne Solo/Duo-Rang:",
      `${summary.unranked.join(", ")}. Die Stärke wird mit dem Median des Modells geschätzt.`));
  }
  if (summary.problems.length) {
    items.push(el("div", { class: "note warn" }, el("span", { html: ICON.warn }),
      el("div", {}, el("strong", {}, "Hinweise zur Eingabe"), el("ul", {}, summary.problems.map((p) => el("li", {}, p))))));
  }
  if (res.metrics.violations.length) {
    items.push(note("crit", "Nicht erfüllte Präferenzen:", res.metrics.violations.join(", ")));
  }
  if (!summary.gurobi) {
    items.push(note("info", "Gurobi ist nicht installiert.",
      "Die Teams wurden mit der Heuristik gebildet. Für beweisbar optimale Einteilungen: pip install gurobipy + Lizenz."));
  } else if (res.solver === "gurobi" && res.status !== "optimal") {
    items.push(note("warn", "Zeitlimit erreicht.",
      "Gurobi hat die Optimalität in der vorgegebenen Zeit nicht bewiesen. Für ein besseres Ergebnis das Zeitlimit in der Leiste unten erhöhen."));
  }
  return items.length ? el("div", { class: "notes" }, items) : null;
}

function copyResult(res) {
  const lines = [];
  res.teams.forEach((t, i) => {
    lines.push(`Team ${i + 1} (Ø ${fmtSigned(t.mean)})`);
    t.members.forEach((m) => lines.push(`  ${ROLE_LABEL[m.role].padEnd(8)} ${m.riot_id}  ${rankText(m)}${m.kind === "second" ? "  (Zweitrolle)" : ""}`));
    lines.push("");
  });
  if (res.reserves.length) lines.push(`Ersatz: ${res.reserves.map((m) => m.riot_id).join(", ")}`);
  const text = lines.join("\n").trim();
  (navigator.clipboard?.writeText(text) || Promise.reject())
    .then(() => toast("Teams in die Zwischenablage kopiert."))
    .catch(() => toast("Kopieren nicht möglich."));
}

function updateStale() {
  const slot = $("#stale-slot");
  if (!slot) return;
  const stale = state.result && !state.building && state.builtFrom !== JSON.stringify(payloadRows());
  slot.replaceChildren(stale ? el("div", { class: "stale" },
    el("span", {}, "Die Spielerliste wurde seit der Berechnung geändert."),
    el("button", { type: "button", class: "btn btn-primary", onclick: build }, "Neu berechnen")) : "");
}

/* ---------- Toast & Tooltip ----------------------------------------------- */
let toastTimer = null;
function toast(text, actionLabel, action) {
  const t = $("#toast");
  clearTimeout(toastTimer);
  t.replaceChildren(el("span", {}, text));
  if (actionLabel) {
    t.append(el("button", { type: "button", onclick: () => { t.hidden = true; action(); } }, actionLabel));
  }
  t.hidden = false;
  t.style.animation = "none";
  void t.offsetWidth;
  t.style.animation = "";
  toastTimer = setTimeout(() => { t.hidden = true; }, actionLabel ? 6000 : 3500);
}

function setupTooltips() {
  const tip = $("#tooltip");
  let current = null;
  const show = (target) => {
    current = target;
    tip.textContent = target.dataset.tip;
    tip.hidden = false;
    const r = target.getBoundingClientRect();
    const w = tip.offsetWidth;
    const hgt = tip.offsetHeight;
    let left = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), window.innerWidth - w - 8);
    let top = r.top - hgt - 8;
    if (top < 60) top = r.bottom + 8;
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  };
  const hide = () => { current = null; tip.hidden = true; };
  document.addEventListener("mouseover", (e) => {
    const t = e.target.closest("[data-tip]");
    if (t && t.dataset.tip) { if (t !== current) show(t); } else if (current) hide();
  });
  document.addEventListener("focusin", (e) => {
    const t = e.target.closest?.("[data-tip]");
    if (t && t.dataset.tip && e.target.matches(":focus-visible")) show(t); else hide();
  });
  document.addEventListener("scroll", hide, true);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hide(); });
}

/* ---------- Drag & Drop --------------------------------------------------- */
function setupDrop() {
  const overlay = $("#drop-overlay");
  let depth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  window.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; depth++; overlay.classList.add("show"); });
  window.addEventListener("dragleave", (e) => { if (!hasFiles(e)) return; depth = Math.max(0, depth - 1); if (!depth) overlay.classList.remove("show"); });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    overlay.classList.remove("show");
    importFile(e.dataTransfer.files[0]);
  });
}

/* ---------- Hell / Dunkel ------------------------------------------------- */
const THEME_KEY = "teambuilder.theme";

function setupTheme() {
  const root = document.documentElement;
  const btn = $("#theme-btn");
  const label = () => {
    const dark = root.dataset.theme === "dark";
    btn.dataset.tip = dark ? "Zum hellen Design wechseln" : "Zum dunklen Design wechseln";
    btn.setAttribute("aria-pressed", String(dark));
  };
  label();
  btn.addEventListener("click", () => {
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem(THEME_KEY, next); } catch { /* nur für diese Sitzung */ }
    label();
    $("#tooltip").hidden = true;
  });
  // Solange man nichts gewählt hat, folgt die Seite der Systemeinstellung
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
    let saved = null;
    try { saved = localStorage.getItem(THEME_KEY); } catch { /* ignorieren */ }
    if (!saved) { root.dataset.theme = e.matches ? "dark" : "light"; label(); }
  });
}

/* ---------- Start --------------------------------------------------------- */
function setup() {
  restore();
  renderRows();

  $("#add-row").addEventListener("click", () => addRow());
  $("#import-btn").addEventListener("click", () => $("#file-input").click());
  $("#paste-btn").addEventListener("click", openPasteDialog);
  $("#paste-form").addEventListener("submit", submitPaste);
  $("#paste-cancel").addEventListener("click", () => $("#paste-dialog").close());
  $("#paste-input").addEventListener("input", renderPastePreview);
  $("#paste-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submitPaste(e);
  });
  // Strg+V außerhalb von Eingabefeldern fügt Riot IDs direkt als Spieler hinzu
  document.addEventListener("paste", (e) => {
    if (e.target.closest?.("input, textarea, select, [contenteditable], dialog")) return;
    const { ids } = parseRiotIds(e.clipboardData?.getData("text") || "");
    if (!ids.length) return;
    e.preventDefault();
    addPlayers(ids);
  });
  $("#file-input").addEventListener("change", (e) => { importFile(e.target.files[0]); e.target.value = ""; });
  $("#build-btn").addEventListener("click", build);
  $("#key-pill").addEventListener("click", openKeyDialog);
  $("#key-form").addEventListener("submit", saveKey);
  $("#key-cancel").addEventListener("click", () => $("#key-dialog").close());
  $("#tl-select").addEventListener("change", (e) => { state.timeLimit = Number(e.target.value); save(); });

  $("#empty").addEventListener("click", (e) => {
    const action = e.target.closest("[data-action]")?.dataset.action;
    if (action === "add") addRow();
    if (action === "paste") openPasteDialog();
    if (action === "import") $("#file-input").click();
    if (action === "example") loadExample();
  });

  const moreBtn = $("#more-btn");
  const menu = $("#more-menu");
  const closeMenu = () => { menu.hidden = true; moreBtn.setAttribute("aria-expanded", "false"); };
  moreBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    menu.hidden = !menu.hidden;
    moreBtn.setAttribute("aria-expanded", String(!menu.hidden));
    if (!menu.hidden) menu.querySelector("button").focus();
  });
  menu.addEventListener("click", (e) => {
    const action = e.target.closest("[data-action]")?.dataset.action;
    closeMenu();
    if (action === "export") exportPlayers();
    if (action === "example") loadExample();
    if (action === "clear") {
      const before = state.rows;
      if (!before.length) return;
      state.rows = [];
      renderRows();
      save();
      toast(`${plural(before.filter((r) => !isEmptyRow(r)).length, "Spieler", "Spieler")} entfernt`, "Rückgängig", () => {
        state.rows = before;
        renderRows();
        save();
      });
    }
  });
  document.addEventListener("click", (e) => { if (!menu.hidden && !e.target.closest(".menu-wrap")) closeMenu(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !menu.hidden) { closeMenu(); moreBtn.focus(); } });

  setupTooltips();
  setupDrop();
  setupTheme();
  loadStatus();
}

setup();
