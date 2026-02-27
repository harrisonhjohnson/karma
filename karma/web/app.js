/**
 * Karma Web UI — app.js (KARMA-014)
 *
 * Three-panel SPA:
 *   - Graph panel: vis.js network, click node -> open seed in Editor
 *   - Editor panel: title + textarea with live markdown preview, Save / New
 *   - Chat panel: POST /api/chat, renders answer + context seeds
 *
 * All data comes from the FastAPI backend at /api/*.
 */

const API = "";  // Same origin — served by FastAPI on localhost:8765

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let network = null;           // vis.js Network instance
let nodesDataset = null;      // vis.DataSet for nodes (allows live updates)
let edgesDataset = null;      // vis.DataSet for edges
let currentSlug = null;       // slug of the seed open in the editor (null = new)

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  await refreshStatus();
  await loadGraph();
  setupEditor();
  setupChat();
});

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

async function refreshStatus() {
  try {
    const resp = await fetch(`${API}/api/status`);
    if (!resp.ok) return;
    const data = await resp.json();
    document.getElementById("karma-score").textContent = `K ${data.points} pts`;
  } catch (_) {
    // silently ignore — status bar is cosmetic
  }
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

async function loadGraph() {
  let data;
  try {
    const resp = await fetch(`${API}/api/graph`);
    if (!resp.ok) return;
    data = await resp.json();
  } catch (e) {
    document.getElementById("graph-empty").classList.remove("hidden");
    return;
  }

  const container = document.getElementById("graph-container");
  const empty = document.getElementById("graph-empty");
  const statsEl = document.getElementById("graph-stats");

  if (!data.nodes || data.nodes.length === 0) {
    container.style.display = "none";
    empty.classList.remove("hidden");
    statsEl.textContent = "";
    return;
  }

  container.style.display = "";
  empty.classList.add("hidden");

  const nodeCount = data.nodes.length;
  const edgeCount = data.edges.length;
  statsEl.textContent = `${nodeCount} seeds  ${edgeCount} roots`;

  // Build vis.js datasets
  const visNodes = data.nodes.map(n => ({
    id: n.id,
    label: n.label,
    color: {
      background: n.color,
      border: n.color,
      highlight: { background: n.color, border: "#ffffff" },
    },
    font: { color: "#ffffff", size: 13 },
    value: Math.max(n.value, 1),   // vis sizes nodes by `value`
    title: `${n.label}\n${n.value} root(s)`,
  }));

  const visEdges = data.edges.map((e, i) => ({
    id: i,
    from: e.from,
    to: e.to,
    color: { color: "#30363d", highlight: "#4a9eff" },
    width: 1.5,
  }));

  if (network) {
    // Update existing network instead of re-creating (preserves positions)
    nodesDataset.clear();
    edgesDataset.clear();
    nodesDataset.add(visNodes);
    edgesDataset.add(visEdges);
    return;
  }

  // Create new network
  nodesDataset = new vis.DataSet(visNodes);
  edgesDataset = new vis.DataSet(visEdges);

  const options = {
    nodes: {
      shape: "dot",
      scaling: { min: 14, max: 42 },
      borderWidth: 1.5,
    },
    edges: {
      smooth: { type: "continuous" },
    },
    physics: {
      solver: "barnesHut",
      barnesHut: {
        gravitationalConstant: -8000,
        centralGravity: 0.3,
        springLength: 120,
      },
    },
    interaction: {
      hover: true,
      tooltipDelay: 200,
    },
    background: { color: "#0e1117" },
  };

  network = new vis.Network(container, { nodes: nodesDataset, edges: edgesDataset }, options);

  // Click node -> open seed in editor
  network.on("click", params => {
    if (params.nodes && params.nodes.length > 0) {
      openSeed(params.nodes[0]);
    }
  });
}

async function openSeed(slug) {
  try {
    const resp = await fetch(`${API}/api/seeds/${encodeURIComponent(slug)}`);
    if (!resp.ok) return;
    const seed = await resp.json();

    currentSlug = seed.slug;
    document.getElementById("editor-title").value = seed.title;
    document.getElementById("editor-input").value = seed.body;
    updatePreview();
    setEditorStatus(`Editing: ${seed.title}`, "");
  } catch (e) {
    setEditorStatus(`Error loading seed: ${e.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Editor
// ---------------------------------------------------------------------------

function setupEditor() {
  const textarea = document.getElementById("editor-input");
  const titleInput = document.getElementById("editor-title");

  textarea.addEventListener("input", updatePreview);

  document.getElementById("btn-save").addEventListener("click", saveSeed);
  document.getElementById("btn-new").addEventListener("click", newSeed);

  // Ctrl+S / Cmd+S to save
  document.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "s") {
      e.preventDefault();
      saveSeed();
    }
  });

  // Enter in title field jumps to editor
  titleInput.addEventListener("keydown", e => {
    if (e.key === "Enter") {
      e.preventDefault();
      textarea.focus();
    }
  });
}

function updatePreview() {
  const input = document.getElementById("editor-input").value;
  const preview = document.getElementById("editor-preview");
  if (typeof marked !== "undefined") {
    preview.innerHTML = marked.parse(input || "_Start typing to see preview..._");
  } else {
    preview.textContent = input;
  }
}

async function saveSeed() {
  const title = document.getElementById("editor-title").value.trim();
  const body = document.getElementById("editor-input").value;

  if (!title) {
    setEditorStatus("Title is required.", "error");
    return;
  }

  setEditorStatus("Saving...", "");

  try {
    let resp;
    if (currentSlug) {
      // Update existing seed
      resp = await fetch(`${API}/api/seeds/${encodeURIComponent(currentSlug)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ body }),
      });
    } else {
      // Create new seed
      resp = await fetch(`${API}/api/seeds`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, body }),
      });
    }

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      setEditorStatus(`Error: ${err.detail || resp.statusText}`, "error");
      return;
    }

    const result = await resp.json();
    currentSlug = result.slug;

    const rootsMsg = result.new_roots.length > 0
      ? ` | +${result.new_roots.length} root(s) discovered`
      : "";
    setEditorStatus(`Saved. +${result.pts_awarded} pts${rootsMsg}`, "success");

    // Refresh graph and score
    await loadGraph();
    await refreshStatus();

  } catch (e) {
    setEditorStatus(`Save failed: ${e.message}`, "error");
  }
}

function newSeed() {
  currentSlug = null;
  document.getElementById("editor-title").value = "";
  document.getElementById("editor-input").value = "";
  document.getElementById("editor-preview").innerHTML =
    "<em>Start typing to see preview...</em>";
  setEditorStatus("", "");
  document.getElementById("editor-title").focus();
}

function setEditorStatus(msg, type) {
  const bar = document.getElementById("editor-status");
  bar.textContent = msg;
  bar.className = "status-bar";
  if (type) bar.classList.add(type);
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

function setupChat() {
  const input = document.getElementById("chat-input");
  document.getElementById("btn-send").addEventListener("click", sendChatMessage);
  input.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });
}

async function sendChatMessage() {
  const input = document.getElementById("chat-input");
  const query = input.value.trim();
  if (!query) return;

  input.value = "";
  input.disabled = true;
  document.getElementById("btn-send").disabled = true;

  // Append user message
  appendChatMessage("user", query, []);

  // Thinking indicator
  const thinkingEl = appendChatMessage("thinking", "Thinking...", []);

  try {
    const resp = await fetch(`${API}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    thinkingEl.remove();

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      appendChatMessage("assistant", `Error: ${err.detail || resp.statusText}`, []);
    } else {
      const data = await resp.json();
      appendChatMessage("assistant", data.answer, data.sources || []);
    }
  } catch (e) {
    thinkingEl.remove();
    appendChatMessage("assistant", `Network error: ${e.message}`, []);
  }

  input.disabled = false;
  document.getElementById("btn-send").disabled = false;
  input.focus();
}

function appendChatMessage(role, content, sources) {
  const messages = document.getElementById("chat-messages");

  const el = document.createElement("div");
  el.className = `chat-msg ${role}`;

  const text = document.createElement("div");
  text.textContent = content;
  el.appendChild(text);

  if (sources && sources.length > 0) {
    const srcEl = document.createElement("div");
    srcEl.className = "context-seeds";
    srcEl.textContent = "Context seeds: " + sources.join(", ");
    el.appendChild(srcEl);
  }

  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}
