// DailyBee bulletin & setup page behaviour. Plain JS, no build step.
(function () {
  "use strict";

  var base = document.body.dataset.base || "";

  function api(method, path, body) {
    var init = { method: method, credentials: "same-origin", headers: { "x-requested-by": "dailybee" } };
    if (body !== undefined) {
      init.headers["content-type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    return fetch(base + path, init).then(function (res) {
      if (!res.ok) throw new Error(method + " " + path + ": HTTP " + res.status);
      return res;
    });
  }

  function storageGet(key) {
    try { return localStorage.getItem(key); } catch (e) { return null; }
  }
  function storageSet(key, val) {
    try { localStorage.setItem(key, val); } catch (e) { /* private mode */ }
  }

  // ---- dates, greeting --------------------------------------------------

  var rtf = window.Intl && Intl.RelativeTimeFormat ? new Intl.RelativeTimeFormat(undefined, { numeric: "auto" }) : null;

  function relative(date) {
    var secs = (date.getTime() - Date.now()) / 1000;
    var abs = Math.abs(secs);
    if (!rtf) return date.toLocaleString();
    if (abs < 60) return rtf.format(Math.round(secs), "second");
    if (abs < 3600) return rtf.format(Math.round(secs / 60), "minute");
    if (abs < 86400) return rtf.format(Math.round(secs / 3600), "hour");
    return rtf.format(Math.round(secs / 86400), "day");
  }

  document.querySelectorAll("time[datetime]").forEach(function (el) {
    var d = new Date(el.getAttribute("datetime"));
    if (isNaN(d)) return;
    el.title = d.toLocaleString(undefined, { dateStyle: "full", timeStyle: "short" });
    el.textContent = el.closest(".meta") ? relative(d) : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  });

  var today = document.querySelector("[data-today]");
  if (today) today.textContent = new Date().toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" });

  var greeting = document.querySelector("[data-greeting]");
  if (greeting) {
    var h = new Date().getHours();
    greeting.textContent = (h < 5 ? "Up late? " : h < 12 ? "Good morning! " : h < 18 ? "Good afternoon! " : "Good evening! ") +
      "Here's your bulletin.";
  }

  // ---- watched / saved state -------------------------------------------

  function statusOf(card) {
    if (card.classList.contains("is-saved")) return "starred";
    if (card.classList.contains("is-watched")) return "read";
    return "unread";
  }

  function applyStatus(card, status) {
    card.classList.toggle("is-watched", status !== "unread");
    card.classList.toggle("is-saved", status === "starred");
  }

  function setStatus(card, status) {
    var previous = statusOf(card);
    if (previous === status) return Promise.resolve();
    applyStatus(card, status);
    updateCounts();
    return api("PUT", "/api/items/" + card.dataset.item, { status: status }).catch(function (err) {
      console.error(err);
      applyStatus(card, previous);
      updateCounts();
    });
  }

  function updateCounts() {
    var unwatched = document.querySelectorAll(".video:not(.is-watched)").length;
    document.querySelectorAll("[data-unwatched]").forEach(function (el) { el.textContent = unwatched; });
    document.querySelectorAll(".channel").forEach(function (section) {
      section.classList.toggle("all-watched", !section.querySelector(".video:not(.is-watched)"));
    });
    var markAll = document.getElementById("mark-all");
    if (markAll) markAll.hidden = unwatched === 0;
  }

  // ---- player -----------------------------------------------------------

  var player = document.getElementById("player");
  var frame = player && player.querySelector("iframe");

  function play(card) {
    var id = card.dataset.video;
    if (!id || !player || typeof player.showModal !== "function") {
      window.open(card.dataset.url, "_blank", "noopener");
    } else {
      frame.src = "https://www.youtube-nocookie.com/embed/" + encodeURIComponent(id) + "?autoplay=1&rel=0&modestbranding=1";
      document.getElementById("player-title").textContent = card.dataset.title;
      document.getElementById("player-open").href = card.dataset.url;
      player.showModal();
    }
    if (statusOf(card) === "unread") setStatus(card, "read");
  }

  if (player) {
    var stop = function () { frame.src = "about:blank"; };
    player.addEventListener("close", stop);
    document.getElementById("player-close").addEventListener("click", function () { player.close(); });
    player.addEventListener("click", function (e) { if (e.target === player) player.close(); });
    document.getElementById("player-open").addEventListener("click", function () { player.close(); });
  }

  // ---- card actions -----------------------------------------------------

  document.addEventListener("click", function (e) {
    var target = e.target.closest("[data-action]");
    if (!target) return;
    var card = target.closest(".video");
    if (!card) return;
    switch (target.dataset.action) {
      case "play":
        e.preventDefault();
        play(card);
        break;
      case "open":
        if (statusOf(card) === "unread") setStatus(card, "read");
        break;
      case "toggle-watched":
        setStatus(card, statusOf(card) === "unread" ? "read" : "unread");
        break;
      case "toggle-saved":
        setStatus(card, statusOf(card) === "starred" ? "read" : "starred");
        break;
    }
  });

  var markAll = document.getElementById("mark-all");
  if (markAll) {
    markAll.addEventListener("click", function () {
      var cards = Array.prototype.slice.call(document.querySelectorAll(".video:not(.is-watched)"));
      markAll.disabled = true;
      Promise.all(cards.map(function (c) { return setStatus(c, "read"); })).then(function () { markAll.disabled = false; });
    });
  }

  var hideWatched = document.getElementById("hide-watched");
  if (hideWatched) {
    hideWatched.checked = storageGet("dailybee.hideWatched") === "1";
    var applyHide = function () {
      document.body.classList.toggle("hide-watched", hideWatched.checked);
      storageSet("dailybee.hideWatched", hideWatched.checked ? "1" : "0");
    };
    hideWatched.addEventListener("change", applyHide);
    applyHide();
  }

  // ---- refresh ----------------------------------------------------------

  var banner = document.getElementById("refresh-banner");
  var bannerText = document.getElementById("refresh-text");
  var refreshBtn = document.getElementById("refresh");

  function waitForRefresh() {
    if (banner) banner.hidden = false;
    if (refreshBtn) refreshBtn.disabled = true;
    var seenRunning = false;
    var started = Date.now();
    var poll = function () {
      api("GET", "/api/status").then(function (res) { return res.json(); }).then(function (st) {
        if (st.running > 0) {
          seenRunning = true;
          if (bannerText) bannerText.textContent = "Checking your channels for new videos… " + st.running + " to go.";
          setTimeout(poll, 1500);
        } else if (!seenRunning && Date.now() - started < 4000) {
          setTimeout(poll, 700); // the refresh may not have started yet
        } else {
          window.location.reload();
        }
      }).catch(function () { setTimeout(poll, 3000); });
    };
    poll();
  }

  if (refreshBtn) {
    refreshBtn.addEventListener("click", function () {
      api("POST", "/api/feeds/refresh").then(waitForRefresh).catch(function (err) {
        console.error(err);
        alert("Could not start a refresh: " + err.message);
      });
    });
  }
  if (document.body.hasAttribute("data-refreshing")) waitForRefresh();

  // ---- setup & auth -----------------------------------------------------

  var setupForm = document.getElementById("setup-form");
  if (setupForm) {
    setupForm.addEventListener("submit", function () {
      var btn = document.getElementById("save");
      btn.disabled = true;
      btn.textContent = "Syncing subscriptions…";
    });
  }

  var logout = document.getElementById("logout");
  if (logout) {
    logout.addEventListener("click", function () {
      api("POST", "/logout").finally(function () { window.location.href = base + "/reader"; });
    });
  }

  updateCounts();
})();
