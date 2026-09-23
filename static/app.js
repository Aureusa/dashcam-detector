/* Dashcam Detector UI - plain vanilla JS, no dependencies. */
(function () {
  "use strict";

  var STATS_MS = 500;
  var DETS_MS = 200;
  var SETTINGS_DEBOUNCE_MS = 250;

  function $(id) { return document.getElementById(id); }
  function fmt(v, digits) {
    if (v === null || v === undefined || isNaN(v)) return "–";
    return Number(v).toFixed(digits);
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function fmtUptime(s) {
    if (s === null || s === undefined) return "–";
    s = Math.floor(s);
    var h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    return (h ? h + "h " : "") + (h || m ? m + "m " : "") + sec + "s";
  }

  function fetchJson(url, opts) {
    return fetch(url, Object.assign({ cache: "no-store" }, opts || {})).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (body) {
        if (!r.ok) {
          var err = new Error(body && body.error ? body.error : "HTTP " + r.status);
          err.status = r.status;
          throw err;
        }
        return body;
      });
    });
  }

  /* Poll `fn` every `ms`, never overlapping requests. */
  function poll(fn, ms) {
    function tick() {
      Promise.resolve().then(fn).catch(function () {}).then(function () { setTimeout(tick, ms); });
    }
    tick();
  }

  /* ------------------------------------------------------------ stream */
  var stream = $("stream");
  var overlay = $("overlay");
  var overlayText = $("overlay-text");
  var liveDot = $("live-dot");
  var retryDelay = 500;
  var retryTimer = null;
  var lastFrameId = null;
  var staleSince = null;
  var serverDown = false;

  function showOverlay(text) {
    overlayText.textContent = text;
    overlay.classList.remove("hidden");
    liveDot.classList.remove("live");
  }
  function hideOverlay() {
    overlay.classList.add("hidden");
    liveDot.classList.add("live");
  }
  function reconnectStream() {
    if (retryTimer) return;
    showOverlay("Stream disconnected — reconnecting in " + (retryDelay / 1000).toFixed(1) + " s…");
    retryTimer = setTimeout(function () {
      retryTimer = null;
      stream.src = "/video_feed?t=" + Date.now();
      retryDelay = Math.min(retryDelay * 2, 10000);
    }, retryDelay);
  }
  stream.addEventListener("load", function () { retryDelay = 500; hideOverlay(); });
  stream.addEventListener("error", reconnectStream);
  // MJPEG <img> fires "load" per frame in some browsers only; stats polling also clears the overlay.

  /* ------------------------------------------------------------- stats */
  function updateStats() {
    return fetchJson("/api/stats").then(function (s) {
      if (serverDown) {
        // Server came back (e.g. restarted): the old MJPEG connection is dead, reopen it.
        serverDown = false;
        stream.src = "/video_feed?t=" + Date.now();
      }
      $("s-cap").textContent = fmt(s.capture_fps, 1);
      $("s-inf").textContent = fmt(s.inference_fps, 1);
      $("s-infms").textContent = fmt(s.inference_ms && s.inference_ms.mean, 1);
      $("s-infms-p95").textContent = "ms · p95 " + fmt(s.inference_ms && s.inference_ms.p95, 1);
      $("s-e2e").textContent = fmt(s.e2e_latency_ms && s.e2e_latency_ms.mean, 0);
      $("s-e2e-p95").textContent = "ms · p95 " + fmt(s.e2e_latency_ms && s.e2e_latency_ms.p95, 0);
      $("s-device").textContent = s.device || "–";
      $("s-model").textContent = s.model || "–";
      var cap = s.capture || {};
      $("s-res").textContent = (s.capture_resolution || "–") + (cap.fps ? " @ " + fmt(cap.fps, 0) + " fps" : "") +
        (cap.fourcc ? " " + cap.fourcc : "");
      $("s-frame").textContent = s.frame_id === null ? "–" : s.frame_id;
      $("s-clients").textContent = s.stream_clients;
      $("s-uptime").textContent = fmtUptime(s.uptime_s);
      $("meta").textContent = (s.model || "") + " · " + (s.device || "");
      // Detect a frozen pipeline (frame id not advancing) even if the HTTP stream stays open.
      if (s.frame_id !== lastFrameId) { lastFrameId = s.frame_id; staleSince = null; if (!retryTimer) hideOverlay(); }
      else if (staleSince === null) { staleSince = Date.now(); }
      else if (Date.now() - staleSince > 3000) { showOverlay("No new frames from the pipeline…"); }
      if (!sliderDirty && typeof s.threshold === "number" && document.activeElement !== slider) {
        setSliderValue(s.threshold);
      }
    }).catch(function (e) {
      serverDown = true;
      showOverlay("Server unreachable — retrying…");
      throw e;
    });
  }

  /* -------------------------------------------------------- detections */
  function updateDetections() {
    return fetchJson("/api/detections").then(function (d) {
      var dets = d.detections || [];
      $("det-count").textContent = dets.length;
      var counts = {};
      dets.forEach(function (x) { counts[x.label] = (counts[x.label] || 0) + 1; });
      var names = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a] || a.localeCompare(b); });
      $("class-counts").innerHTML = names.map(function (n) {
        return '<span class="chip"><b>' + counts[n] + "</b> " + escapeHtml(n) + "</span>";
      }).join("");
      var body = $("det-body");
      if (!dets.length) {
        body.innerHTML = '<tr class="empty"><td colspan="3">No detections</td></tr>';
        return;
      }
      body.innerHTML = dets.slice(0, 50).map(function (x) {
        var w = Math.round(x.x2 - x.x1), h = Math.round(x.y2 - x.y1);
        return "<tr><td>" + escapeHtml(x.label) + "</td><td>" + fmt(x.score, 2) +
          '</td><td class="num">' + w + "×" + h + "</td></tr>";
      }).join("");
    });
  }

  /* ---------------------------------------------------------- settings */
  var slider = $("threshold");
  var sliderOut = $("threshold-value");
  var statusEl = $("settings-status");
  var sliderDirty = false;
  var allLabels = [];
  var preset = [];
  var debounceTimer = null;
  var statusTimer = null;

  function setStatus(text, kind) {
    statusEl.textContent = text;
    statusEl.className = "settings-status " + (kind || "");
    if (statusTimer) clearTimeout(statusTimer);
    if (kind !== "error") statusTimer = setTimeout(function () { statusEl.textContent = ""; }, 2500);
  }
  function setSliderValue(v) {
    slider.value = v;
    sliderOut.textContent = Number(v).toFixed(2);
  }

  function postSettings(payload) {
    return fetchJson("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (s) {
      setStatus("Saved: threshold " + Number(s.score_threshold).toFixed(2) + ", " +
        (s.classes.length ? s.classes.length + " classes" : "all classes"), "ok");
      return s;
    }).catch(function (e) {
      setStatus("Settings error: " + e.message, "error");
      throw e;
    });
  }

  function debounced(fn) {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(function () { debounceTimer = null; fn(); }, SETTINGS_DEBOUNCE_MS);
  }

  slider.addEventListener("input", function () {
    sliderDirty = true;
    sliderOut.textContent = Number(slider.value).toFixed(2);
    debounced(function () {
      postSettings({ score_threshold: Number(slider.value) })
        .catch(function () {})
        .then(function () { sliderDirty = false; });
    });
  });

  function checkedClasses() {
    return Array.prototype.slice.call(document.querySelectorAll("#class-list input:checked"))
      .map(function (el) { return el.value; });
  }
  function sendClasses() {
    var sel = checkedClasses();
    if (!sel.length) {
      setStatus("Select at least one class (an empty selection means all classes).", "warn");
      return;
    }
    // All selected == no filter.
    postSettings({ classes: sel.length === allLabels.length ? [] : sel }).catch(function () {});
  }
  function setChecked(names) {
    var set = {};
    names.forEach(function (n) { set[n.toLowerCase()] = true; });
    document.querySelectorAll("#class-list input").forEach(function (el) {
      el.checked = !!set[el.value.toLowerCase()];
    });
  }

  function buildClassList(labels, active) {
    var list = $("class-list");
    var activeSet = {};
    active.forEach(function (n) { activeSet[n.toLowerCase()] = true; });
    var presetSet = {};
    preset.forEach(function (n) { presetSet[n] = true; });
    // Driving-relevant classes first, then the rest alphabetically.
    var ordered = labels.slice().sort(function (a, b) {
      var pa = presetSet[a] ? 0 : 1, pb = presetSet[b] ? 0 : 1;
      return pa - pb || a.localeCompare(b);
    });
    list.innerHTML = ordered.map(function (n) {
      var on = active.length === 0 || activeSet[n.toLowerCase()];
      return '<label class="class-item' + (presetSet[n] ? " preset" : "") + '" data-name="' + escapeHtml(n.toLowerCase()) +
        '"><input type="checkbox" value="' + escapeHtml(n) + '"' + (on ? " checked" : "") + "> " + escapeHtml(n) + "</label>";
    }).join("");
    list.addEventListener("change", function () { debounced(sendClasses); });
  }

  $("btn-preset").addEventListener("click", function () { setChecked(preset); sendClasses(); });
  $("btn-all").addEventListener("click", function () { setChecked(allLabels); sendClasses(); });
  $("btn-none").addEventListener("click", function () {
    setChecked([]);
    setStatus("Nothing selected — tick the classes to show (filter unchanged until then).", "warn");
  });
  $("class-search").addEventListener("input", function (e) {
    var q = e.target.value.trim().toLowerCase();
    document.querySelectorAll("#class-list .class-item").forEach(function (el) {
      el.style.display = !q || el.getAttribute("data-name").indexOf(q) !== -1 ? "" : "none";
    });
  });

  function initSettings() {
    return Promise.all([fetchJson("/api/labels"), fetchJson("/api/settings")]).then(function (r) {
      allLabels = r[0].labels || [];
      preset = r[0].driving_preset || [];
      setSliderValue(r[1].score_threshold);
      buildClassList(allLabels, r[1].classes || []);
    }).catch(function (e) {
      setStatus("Could not load settings: " + e.message + " (retrying)", "error");
      setTimeout(initSettings, 2000);
    });
  }

  initSettings();
  poll(updateStats, STATS_MS);
  poll(updateDetections, DETS_MS);
})();
