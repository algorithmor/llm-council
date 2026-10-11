const $ = (id) => document.getElementById(id);
const state = { cluster: null, offset: 0, limit: 48 };

const money = (n) => {
  if (n == null) return "—";
  if (n >= 1e9) return "$" + (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + "B";
  if (n >= 1e6) return "$" + Math.round(n / 1e6) + "M";
  return "$" + Math.round(n / 1e3) + "K";
};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const api = (p) => fetch(p).then((r) => r.json());
// Only allow http(s) links from scraped data.
const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? u : null);

function logo(s) {
  const u = safeUrl(s.logo);
  return u ? `<img class="logo" src="${esc(u)}" alt="" loading="lazy">`
           : `<span class="logo">${esc(s.name[0])}</span>`;
}

function card(s) {
  return `<div class="card" data-id="${esc(s.id)}">
    <h3>${logo(s)}${esc(s.name)}</h3>
    <p>${esc(s.one_liner)}</p>
    <div class="fin"><span>Raised <b>${money(s.capital_raised_usd)}</b></span>
      <span>Founded <b>${s.founded_year ?? "—"}</b></span>
      <span>${esc(s.last_round || "")}</span></div>
    <div class="meta"><span class="pill acc">${esc(s.accelerator)} ${esc(s.batch || "")}</span>
      ${s.country ? `<span class="pill">${esc(s.country)}</span>` : ""}
      ${(s.tags || []).slice(0, 2).map((t) => `<span class="pill">${esc(t)}</span>`).join("")}</div>
  </div>`;
}

function params() {
  const p = new URLSearchParams({ sort: $("sort").value, limit: state.limit, offset: state.offset });
  for (const k of ["q", "accelerator", "country"]) if ($(k).value) p.set(k, $(k).value);
  if (state.cluster != null) p.set("cluster_id", state.cluster);
  if ($("min_raised").value) p.set("min_raised", $("min_raised").value * 1e6);
  return p;
}

async function load(reset = true) {
  if (reset) { state.offset = 0; $("grid").innerHTML = ""; }
  const { total, items } = await api("/api/startups?" + params());
  $("grid").insertAdjacentHTML("beforeend", items.map(card).join(""));
  state.offset += items.length;
  $("count").textContent = `${total.toLocaleString()} compan${total === 1 ? "y" : "ies"}`;
  $("more").hidden = state.offset >= total;
}

async function loadClusters() {
  const cs = await api("/api/clusters");
  $("clusters").innerHTML = cs.map((c) => `<li data-id="${c.id}" class="${state.cluster === c.id ? "on" : ""}">
    <div class="l">${esc(c.label)}</div>
    <div class="m">${c.size} companies · ${money(c.total_raised || null)} raised<br>${esc(c.keywords.slice(0, 5).join(", "))}</div></li>`).join("");
}

async function showDetail(id) {
  const s = await api("/api/startups/" + encodeURIComponent(id));
  const site = safeUrl(s.website);
  $("detail").innerHTML = `
    <h2>${logo(s)} ${esc(s.name)}</h2>
    <div>${site ? `<a href="${esc(site)}" target="_blank" rel="noopener noreferrer">${esc(s.website)}</a>` : ""}</div>
    <p>${esc(s.description || s.one_liner)}</p>
    <div class="kv">
      <div><span>Accelerator</span><b>${esc(s.accelerator)} ${esc(s.batch || "")}</b></div>
      <div><span>Founded</span><b>${s.founded_year ?? "—"}</b></div>
      <div><span>Capital raised</span><b>${money(s.capital_raised_usd)}</b></div>
      <div><span>Last round</span><b>${esc(s.last_round || "—")}</b></div>
      <div><span>Valuation</span><b>${money(s.valuation_usd)}</b></div>
      <div><span>Revenue</span><b>${money(s.revenue_usd)}</b></div>
      <div><span>Team size</span><b>${s.team_size ?? "—"}</b></div>
      <div><span>Status</span><b>${esc(s.status || "—")}</b></div>
      <div><span>HQ</span><b>${esc([s.city, s.country].filter(Boolean).join(", ") || "—")}</b></div>
      <div><span>Industry</span><b>${esc(s.industry || "—")}</b></div>
    </div>
    <div class="meta">${(s.tags || []).map((t) => `<span class="pill">${esc(t)}</span>`).join("")}</div>
    <h3>Similar companies</h3>
    ${s.similar.map((x) => `<div class="sim" data-id="${esc(x.id)}"><span>${esc(x.name)} <small>${esc(x.one_liner || "")}</small></span><span>${Math.round(x.score * 100)}%</span></div>`).join("") || "<p class='src'>None found.</p>"}
    <h3>Sources</h3>
    <div class="src">${(s.sources || []).map((x) => safeUrl(x.url) ? `<a href="${esc(x.url)}" target="_blank" rel="noopener noreferrer">${esc(x.name)}</a>` : esc(x.name)).join(" · ")}</div>`;
  $("drawer").hidden = false;
}

async function init() {
  const [st, f] = await Promise.all([api("/api/stats"), api("/api/facets")]);
  $("stats").innerHTML = [
    [st.companies.toLocaleString(), "companies"], [st.accelerators, "accelerators"],
    [st.countries, "countries"], [st.clusters, "domain clusters"],
    [money(st.total_raised), `raised (${st.with_funding} reporting)`],
  ].map(([b, s]) => `<div class="stat"><b>${b}</b><span>${s}</span></div>`).join("");
  $("sample-banner").hidden = !st.has_sample;
  for (const [id, key] of [["accelerator", "accelerators"], ["country", "countries"]])
    $(id).insertAdjacentHTML("beforeend", f[key].map((v) => `<option>${esc(v)}</option>`).join(""));
  await loadClusters();
  load();
}

let t;
for (const id of ["q", "min_raised"]) $(id).addEventListener("input", () => { clearTimeout(t); t = setTimeout(load, 250); });
for (const id of ["accelerator", "country", "sort"]) $(id).addEventListener("change", () => load());
$("more").onclick = () => load(false);
$("clear").onclick = () => { state.cluster = null; loadClusters(); load(); };
$("clusters").onclick = (e) => {
  const li = e.target.closest("li"); if (!li) return;
  const id = +li.dataset.id;
  state.cluster = state.cluster === id ? null : id;
  loadClusters(); load();
};
$("grid").onclick = (e) => { const c = e.target.closest(".card"); if (c) showDetail(c.dataset.id); };
$("detail").onclick = (e) => { const c = e.target.closest(".sim"); if (c) showDetail(c.dataset.id); };
$("close").onclick = () => ($("drawer").hidden = true);
document.addEventListener("keydown", (e) => e.key === "Escape" && ($("drawer").hidden = true));
init();
