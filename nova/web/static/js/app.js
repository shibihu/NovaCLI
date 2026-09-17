/* ==========================================================================
 * NovaCLI web IDE — client
 *
 * No framework, no build step: this file is served as-is, which keeps the
 * whole IDE installable on a phone. Its only job is to render the event
 * stream produced by the shared Nova Core agent.
 * ========================================================================== */
(function () {
  "use strict";

  // --- Small helpers ----------------------------------------------------

  const $ = (id) => document.getElementById(id);

  const HTML_ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => HTML_ESCAPES[c]);

  /** Minimal, safe markdown: code fences, inline code, bold, lists. */
  function renderMarkdown(text) {
    const chunks = esc(text).split(/```/);
    return chunks
      .map((chunk, index) => {
        if (index % 2 === 1) {
          return "<pre><code>" + chunk.replace(/^[a-zA-Z0-9]*\n/, "") + "</code></pre>";
        }
        const html = chunk
          .replace(/`([^`\n]+)`/g, "<code>$1</code>")
          .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
          .replace(/^[-*]\s+(.+)$/gm, "<li>$1</li>")
          .replace(/(?:<li>[\s\S]*?<\/li>\s*)+/g, (list) => "<ul>" + list + "</ul>")
          .replace(/\n{2,}/g, "</p><p>")
          .replace(/\n/g, "<br>");
        return "<p>" + html + "</p>";
      })
      .join("");
  }

  function truncate(text, limit) {
    const value = String(text ?? "");
    return value.length > limit ? value.slice(0, limit) + "\n… [" + (value.length - limit) + " more chars]" : value;
  }

  let toastTimer = null;
  function toast(message, kind) {
    const el = $("toast");
    el.textContent = message;
    el.className = "toast" + (kind ? " " + kind : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.add("hidden"), 4000);
  }

  async function api(path, options) {
    const response = await fetch(path, Object.assign({ headers: {} }, options || {}));
    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json") ? await response.json() : await response.text();
    if (!response.ok) {
      const detail = payload && payload.detail ? payload.detail : payload;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return payload;
  }

  function setStatus(state, label) {
    const chip = $("chip-status");
    chip.dataset.state = state;
    chip.textContent = label;
  }

  // --- Tabs -------------------------------------------------------------

  function showView(name) {
    document.querySelectorAll(".view").forEach((view) => {
      view.classList.toggle("is-active", view.id === "view-" + name);
    });
    document.querySelectorAll(".tab").forEach((tab) => {
      const active = tab.dataset.view === name;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    if (name === "files") loadFiles(filesPath);
    if (name === "term") initTerminal();
    if (name === "project") loadProject();
  }

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => showView(tab.dataset.view));
  });

  // --- Chat -------------------------------------------------------------

  const state = { sessionId: null, source: null, busy: false, approvalId: null };

  function addMessage(html, className) {
    $("chat-empty")?.remove();
    const node = document.createElement("div");
    node.className = "msg " + className;
    node.innerHTML = html;
    $("messages").appendChild(node);
    $("messages").scrollTop = $("messages").scrollHeight;
    return node;
  }

  function agentMessage(title, bodyHtml) {
    return addMessage('<div class="msg-head"><span>' + esc(title) + "</span></div>" + bodyHtml, "agent");
  }

  function setProgress(percent) {
    const bar = $("progress-bar");
    if (percent >= 100) {
      bar.style.width = "100%";
      setTimeout(() => $("progress").classList.add("hidden"), 400);
    } else {
      $("progress").classList.remove("hidden");
      bar.style.width = Math.max(2, percent) + "%";
    }
  }

  function setBusy(busy) {
    state.busy = busy;
    $("send").disabled = busy;
    $("stop").classList.toggle("hidden", !busy);
    setStatus(busy ? "busy" : "ok", busy ? "working" : "ready");
  }

  function handleEvent(event) {
    const data = event.data || {};
    switch (event.type) {
      case "agent_start":
        addMessage(
          '<div class="msg-head"><span>Nova</span><span class="badge">' +
            esc(data.model || "") +
            "</span></div><p class=\"muted small\">Working in " +
            esc(data.project || "") +
            " · " +
            esc(data.safety_mode || "smart") +
            " mode</p>",
          "step"
        );
        break;

      case "step_start":
        setProgress(Math.min(95, (event.step / (data.of || 8)) * 100));
        break;

      case "thought":
        if (data.text) addMessage("<p>" + esc(data.text) + "</p>", "thought");
        break;

      case "tool_call":
        addMessage(
          '<div class="msg-head"><span>🔧 tool</span><span class="badge">' +
            esc(data.tool) +
            "</span></div><p>" +
            esc(truncate(JSON.stringify(data.input || {}), 260)) +
            "</p>",
          "step"
        );
        break;

      case "tool_result": {
        const ok = Boolean(data.ok);
        const cls = data.blocked ? "blocked" : ok ? "ok" : "fail";
        const output = truncate(data.output || data.error || "(no output)", 4000);
        addMessage(
          '<div class="msg-head"><span class="status-dot ' +
            cls +
            '"></span><span>' +
            esc(data.name) +
            "</span><span class=\"badge\">" +
            (data.duration_ms || 0) +
            "ms</span></div>" +
            "<details class=\"tool-out\"><summary>output</summary><pre>" +
            esc(output) +
            "</pre></details>",
          "step"
        );
        break;
      }

      case "blocked":
        addMessage(
          '<div class="msg-head"><span>⛔ blocked</span><span class="badge">' +
            esc(data.tool || "") +
            "</span></div><p>" +
            esc(data.reason || "blocked by safety policy") +
            "</p>",
          "error"
        );
        break;

      case "approval_request":
        state.approvalId = data.id;
        $("approval-summary").textContent = data.summary || data.tool || "";
        $("approval-reason").textContent = data.reason ? "Reason: " + data.reason : "";
        $("approval-detail").textContent = truncate(data.detail || "", 1500);
        $("approval-modal").classList.remove("hidden");
        setStatus("busy", "waiting for you");
        break;

      case "approval_resolved":
        $("approval-modal").classList.add("hidden");
        state.approvalId = null;
        break;

      case "progress":
        if (typeof data.percent === "number") setProgress(data.percent);
        break;

      case "final":
        $("approval-modal").classList.add("hidden");
        let finalHtml = renderMarkdown(data.answer || "(no answer)");
        if (data.checkpoint_id) {
          const changed = data.changed_files || [];
          finalHtml += '<div class="checkpoint-box" style="margin-top: 12px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 6px;">' +
                       '<div><strong>Checkpoint:</strong> <code>' + esc(data.checkpoint_id) + '</code> <span class="muted">(' + changed.length + ' changed files)</span></div>' +
                       '<div style="margin-top: 8px; display: flex; gap: 8px;">' +
                       '<button type="button" class="btn btn-ghost btn-view-diff" data-cp="' + esc(data.checkpoint_id) + '">View Diff</button>' +
                       '<button type="button" class="btn btn-danger btn-undo-cp" data-cp="' + esc(data.checkpoint_id) + '">Undo Changes</button>' +
                       '</div></div>';
        }
        agentMessage("Nova", finalHtml);
        setProgress(100);
        setBusy(false);
        break;

      case "error":
        $("approval-modal").classList.add("hidden");
        addMessage(
          '<div class="msg-head"><span>error</span></div><p>' + esc(data.message || "unknown error") + "</p>",
          "error"
        );
        setBusy(false);
        break;

      case "cancelled":
        $("approval-modal").classList.add("hidden");
        addMessage("<p>Cancelled.</p>", "thought");
        setBusy(false);
        break;
    }
  }

  function closeStream() {
    if (state.source) {
      state.source.close();
      state.source = null;
    }
  }

  function openStream(sessionId) {
    closeStream();
    const source = new EventSource("/api/agent/stream?session_id=" + encodeURIComponent(sessionId));
    state.source = source;

    source.onmessage = (message) => {
      let parsed;
      try {
        parsed = JSON.parse(message.data);
      } catch (error) {
        return;
      }
      handleEvent(parsed);
      if (parsed.type === "final" || parsed.type === "error" || parsed.type === "cancelled") {
        closeStream();
      }
    };

    source.onerror = () => {
      // The browser retries automatically; only surface a hint when idle.
      if (!state.busy) closeStream();
    };
  }

  async function submitTask(task) {
    if (!task.trim() || state.busy) return;
    addMessage("<p>" + esc(task) + "</p>", "user");
    setBusy(true);
    setProgress(3);

    try {
      const created = await api("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: task }),
      });
      state.sessionId = created.session_id;
      openStream(created.session_id);
    } catch (error) {
      setBusy(false);
      addMessage('<div class="msg-head"><span>error</span></div><p>' + esc(error.message) + "</p>", "error");
      setProgress(100);
    }
  }

  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const input = $("task");
    const task = input.value;
    input.value = "";
    input.style.height = "auto";
    submitTask(task);
  });

  const taskInput = $("task");
  taskInput.addEventListener("input", () => {
    taskInput.style.height = "auto";
    taskInput.style.height = Math.min(120, taskInput.scrollHeight) + "px";
  });
  taskInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });

  document.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => submitTask(button.textContent.trim()));
  });

  $("stop").addEventListener("click", async () => {
    if (!state.sessionId) return;
    try {
      await api("/api/agent/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId }),
      });
      toast("Stopping…");
    } catch (error) {
      toast(error.message, "error");
    }
  });

  // --- Approvals --------------------------------------------------------

  async function decide(decision) {
    const requestId = state.approvalId;
    $("approval-modal").classList.add("hidden");
    if (!requestId || !state.sessionId) return;
    try {
      await api("/api/agent/approve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: state.sessionId, request_id: requestId, decision: decision }),
      });
      state.approvalId = null;
      if (decision === "deny") addMessage("<p>Denied.</p>", "thought");
    } catch (error) {
      toast(error.message, "error");
    }
  }

  $("approve-once").addEventListener("click", () => decide("approve"));
  $("approve-always").addEventListener("click", () => decide("always"));
  $("approve-deny").addEventListener("click", () => decide("deny"));

  // --- Files ------------------------------------------------------------

  let filesPath = ".";
  let openFilePath = null;

  function humanSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1024 / 1024).toFixed(1) + " MB";
  }

  async function loadFiles(path) {
    filesPath = path || ".";
    $("files-path").textContent = filesPath;
    const list = $("file-list");
    list.innerHTML = '<li class="muted">Loading…</li>';
    try {
      const data = await api("/api/files?path=" + encodeURIComponent(filesPath));
      list.innerHTML = "";
      const entries = data.entries || [];
      if (!entries.length) {
        list.innerHTML = '<li class="muted">Empty directory</li>';
        return;
      }
      entries.forEach((entry) => {
        const li = document.createElement("li");
        const icon = entry.is_dir ? "📁" : "📄";
        li.innerHTML =
          '<span class="entry-icon">' +
          icon +
          '</span><span class="entry-name">' +
          esc(entry.path.split("/").pop()) +
          '</span><span class="entry-size">' +
          (entry.is_dir ? "" : humanSize(entry.size)) +
          "</span>";
        li.addEventListener("click", () => {
          if (entry.is_dir) loadFiles(entry.path);
          else openFile(entry.path);
        });
        list.appendChild(li);
      });
    } catch (error) {
      list.innerHTML = '<li class="muted">' + esc(error.message) + "</li>";
    }
  }

  async function openFile(path) {
    try {
      const data = await api("/api/file?path=" + encodeURIComponent(path));
      openFilePath = data.path;
      $("editor").value = data.content;
      $("editor-path").textContent = data.path;
      $("editor-meta").textContent = data.lines + " lines · " + humanSize(data.size);
      $("editor-pane").classList.remove("hidden");
    } catch (error) {
      toast(error.message, "error");
    }
  }

  $("files-up").addEventListener("click", () => {
    const parts = filesPath.split("/").filter(Boolean);
    parts.pop();
    loadFiles(parts.join("/") || ".");
  });
  $("files-refresh").addEventListener("click", () => loadFiles(filesPath));
  $("editor-close").addEventListener("click", () => $("editor-pane").classList.add("hidden"));
  $("editor-save").addEventListener("click", async () => {
    if (!openFilePath) return;
    try {
      await api("/api/file", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: openFilePath, content: $("editor").value }),
      });
      toast("Saved " + openFilePath, "ok");
    } catch (error) {
      toast(error.message, "error");
    }
  });

  // --- Interactive PTY Terminal (xterm.js + WebSocket) -------------------

  let termInstance = null;
  let fitAddon = null;
  let termWs = null;
  let ctrlActive = false;

  function initTerminal() {
    const container = $("terminal-container");
    if (!container) return;

    if (!termInstance && typeof Terminal !== "undefined") {
      termInstance = new Terminal({
        cursorBlink: true,
        fontSize: 14,
        fontFamily: 'Consolas, Monaco, "Courier New", monospace',
        theme: {
          background: "#0d1117",
          foreground: "#c9d1d9",
          cursor: "#58a6ff",
        },
        // A real PTY/ConPTY already emits CRLF and cursor positioning. Rewriting
        // line endings here would corrupt full-screen and interactive programs.
        convertEol: false,
      });

      if (typeof FitAddon !== "undefined" && FitAddon.FitAddon) {
        fitAddon = new FitAddon.FitAddon();
        termInstance.loadAddon(fitAddon);
      }

      termInstance.open(container);
      if (fitAddon) fitAddon.fit();

      // Focus the real xterm surface (an off-screen textarea owned by xterm.js).
      // Without this the terminal renders but keystrokes have nowhere to go.
      termInstance.focus();

      // The container only reaches its final size after the tab is laid out, so
      // re-fit and re-focus once the browser has settled.
      setTimeout(() => {
        if (fitAddon) fitAddon.fit();
        if (termInstance) termInstance.focus();
      }, 50);

      // Tapping the terminal focuses xterm directly (desktop and mobile).
      container.addEventListener("pointerdown", () => {
        if (termInstance) termInstance.focus();
      });

      connectTerminalWs();

      termInstance.onData((data) => {
        if (ctrlActive && data.length === 1) {
          ctrlActive = false;
          const ctrlBtn = $("term-ctrl-btn");
          if (ctrlBtn) ctrlBtn.classList.remove("is-active");
          const code = data.toUpperCase().charCodeAt(0);
          if (code >= 64 && code <= 95) {
            data = String.fromCharCode(code - 64);
          }
        }
        sendTermMsg({ type: "input", data: data });
      });

      termInstance.onResize((size) => {
        sendTermMsg({ type: "resize", cols: size.cols, rows: size.rows });
      });

      window.addEventListener("resize", () => {
        if (fitAddon && termInstance) {
          fitAddon.fit();
        }
      });
    } else if (termInstance) {
      setTimeout(() => {
        if (fitAddon) fitAddon.fit();
        termInstance.focus();
      }, 50);
    }
  }

  function connectTerminalWs() {
    if (termWs && (termWs.readyState === WebSocket.OPEN || termWs.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const cols = termInstance ? termInstance.cols : 80;
    const rows = termInstance ? termInstance.rows : 24;
    const wsUrl = protocol + "//" + location.host + "/ws/terminal?cols=" + cols + "&rows=" + rows;

    try {
      termWs = new WebSocket(wsUrl);

      termWs.onopen = () => {
        if (fitAddon && termInstance) fitAddon.fit();
        if (termInstance) termInstance.focus();
      };

      termWs.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "output" && termInstance) {
            termInstance.write(msg.data);
          } else if (msg.type === "exit" && termInstance) {
            termInstance.write("\r\n\x1b[33m[Process exited with code " + msg.code + "]\x1b[0m\r\n");
          } else if (msg.type === "error" && termInstance) {
            termInstance.write("\r\n\x1b[31m[Error: " + esc(msg.message) + "]\x1b[0m\r\n");
          }
        } catch (e) {
          if (termInstance) termInstance.write(event.data);
        }
      };

      termWs.onclose = () => {
        if (termInstance) {
          termInstance.write("\r\n\x1b[33m[Terminal disconnected. Switch tab or refresh to reconnect.]\x1b[0m\r\n");
        }
        termWs = null;
      };
    } catch (err) {
      if (termInstance) termInstance.write("\r\n\x1b[31m[WebSocket connection failed: " + err.message + "]\x1b[0m\r\n");
    }
  }

  function sendTermMsg(msg) {
    if (termWs && termWs.readyState === WebSocket.OPEN) {
      termWs.send(JSON.stringify(msg));
    }
  }

  // Mobile quick control buttons
  document.querySelectorAll(".mobile-term-controls .btn-term").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const key = btn.dataset.key;
      if (!key) return;

      if (key === "Ctrl") {
        ctrlActive = !ctrlActive;
        btn.classList.toggle("is-active", ctrlActive);
        if (termInstance) termInstance.focus();
        return;
      }

      let data = "";
      if (key === "Esc") data = "\x1b";
      else if (key === "Tab") data = "\t";
      else if (key === "Up") data = "\x1b[A";
      else if (key === "Down") data = "\x1b[B";
      else if (key === "Right") data = "\x1b[C";
      else if (key === "Left") data = "\x1b[D";

      if (data) {
        sendTermMsg({ type: "input", data: data });
        if (termInstance) termInstance.focus();
      }
    });
  });

  // Preserve non-interactive command execution API helper for external/CLI callers
  async function runCommand(command, approve) {
    try {
      const data = await api("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: command, approve: Boolean(approve) }),
      });
      return data;
    } catch (error) {
      throw error;
    }
  }

  // --- Project ----------------------------------------------------------

  function card(label, value, sub) {
    return (
      '<div class="card"><div class="card-label">' +
      esc(label) +
      '</div><div class="card-value">' +
      esc(value) +
      "</div>" +
      (sub ? '<div class="card-sub">' + esc(sub) + "</div>" : "") +
      "</div>"
    );
  }

  async function loadProject() {
    const cards = $("project-cards");
    cards.innerHTML = '<div class="card"><div class="card-label">Loading</div></div>';
    try {
      const data = await api("/api/project");
      const summary = data.summary || {};
      const languages = Object.keys(summary.languages || {});
      cards.innerHTML = [
        card("Project", summary.name || "—", summary.root || ""),
        card("Files", summary.total_files ?? 0, humanSize(summary.total_bytes || 0)),
        card("Languages", languages.slice(0, 3).join(", ") || "unknown", languages.length > 3 ? "+" + (languages.length - 3) + " more" : ""),
        card("Tests", (summary.tests || []).length, (summary.tests || []).slice(0, 2).join(", ")),
        card("Git", summary.has_git ? "yes" : "no", (summary.entry_points || []).slice(0, 2).join(", ")),
        card("Commands", Object.keys(summary.commands || {}).length, Object.values(summary.commands || {})[0] || ""),
      ].join("");
      $("project-tree").textContent = data.tree || "(no files)";
    } catch (error) {
      cards.innerHTML = '<div class="card"><div class="card-label">Error</div><div class="card-value">' + esc(error.message) + "</div></div>";
    }
  }

  $("project-refresh").addEventListener("click", loadProject);

  // --- Boot -------------------------------------------------------------

  async function boot() {
    try {
      const health = await api("/api/health");
      $("chip-model").textContent = health.model;
      $("chip-safety").textContent = health.safety_mode;
      if (!health.has_api_key) {
        $("setup-banner").classList.remove("hidden");
        setStatus("error", "no api key");
      } else {
        setStatus("ok", "ready");
      }
    } catch (error) {
      setStatus("error", "offline");
    }
  }

  boot();



  // Checkpoint & Diff Modal Event Listeners
  const msgContainer = $("messages");
  if (msgContainer) {
    msgContainer.addEventListener("click", async (evt) => {
      const diffBtn = evt.target.closest(".btn-view-diff");
      if (diffBtn) {
        const cpId = diffBtn.getAttribute("data-cp");
        try {
          const res = await api("/api/git/diff");
          $("diff-meta").textContent = "Diff for checkpoint " + cpId;
          $("diff-content").textContent = res.diff || "(no diff)";
          $("diff-modal").classList.remove("hidden");
        } catch (err) {
          toast("Error fetching diff: " + err.message);
        }
        return;
      }

      const undoBtn = evt.target.closest(".btn-undo-cp");
      if (undoBtn) {
        const cpId = undoBtn.getAttribute("data-cp");
        if (!confirm("Are you sure you want to undo all changes for checkpoint " + cpId + "?")) {
          return;
        }
        try {
          const res = await api("/api/agent/checkpoint/" + encodeURIComponent(cpId) + "/rollback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirm: true }) });
          let msg = "Rollback complete: " + (res.restored ? res.restored.length : 0) + " restored, " + (res.removed ? res.removed.length : 0) + " removed.";
          if (res.preserved && res.preserved.length > 0) {
            msg += " (" + res.preserved.length + " files preserved due to user conflicts)";
          }
          toast(msg);
        } catch (err) {
          toast("Undo failed: " + err.message);
        }
      }
    });
  }

  const diffCloseBtn = $("diff-close");
  if (diffCloseBtn) {
    diffCloseBtn.addEventListener("click", () => {
      $("diff-modal").classList.add("hidden");
    });
  }

})();
