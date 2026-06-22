// ==========================================
// GLOBALS & STATE MANAGEMENT
// ==========================================
const state = {
    activePanel: "chat-panel",
    chunks: {
        list: [],
        total: 0,
        page: 1,
        limit: 12,
        searchQuery: ""
    },
    statusPoller: null,
    isProcessing: false,
    uploadedFiles: []
};

// ==========================================
// TOAST NOTIFICATIONS HELPER
// ==========================================
function showToast(message, type = "info") {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    
    // Icon mappings
    let icon = "";
    if (type === "success") icon = "✓";
    else if (type === "error") icon = "✗";
    else if (type === "warning") icon = "⚠";
    else icon = "ℹ";
    
    toast.innerHTML = `<span style="font-weight: 700; font-size: 1.1rem;">${icon}</span> <span>${message}</span>`;
    container.appendChild(toast);
    
    // Auto-remove toast after 4s
    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(10px) scale(0.9)";
        toast.style.transition = "all 0.3s ease-out";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ==========================================
// MARKDOWN PARSER FOR LLM CHAT
// ==========================================
function parseMarkdown(text) {
    if (!text) return "";
    let html = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    
    // Triple backtick code blocks
    html = html.replace(/```([\s\S]*?)```/g, '<pre class="prompt-pre"><code style="font-family: \'JetBrains Mono\', monospace;">$1</code></pre>');
    
    // Single backtick inline code
    html = html.replace(/`([^`]+)`/g, '<code style="font-family: \'JetBrains Mono\', monospace; background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px;">$1</code>');
    
    // Bold text (**word**)
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Bullet lists
    html = html.replace(/^\s*-\s+(.*?)$/gm, '• $1');
    
    // Line breaks
    html = html.split("\n").join("<br>");
    
    return html;
}

// ==========================================
// PANEL NAVIGATION SYSTEM
// ==========================================
const navItems = document.querySelectorAll(".nav-item");
const panels = document.querySelectorAll(".panel");
const panelTitle = document.getElementById("panel-title");

navItems.forEach(item => {
    item.addEventListener("click", () => {
        const target = item.getAttribute("data-target");
        
        // Update navigation active state
        navItems.forEach(i => i.classList.remove("active"));
        item.classList.add("active");
        
        // Update visible panel
        panels.forEach(p => p.classList.remove("active"));
        const targetPanel = document.getElementById(target);
        if (targetPanel) {
            targetPanel.classList.add("active");
        }
        
        // Update header title
        panelTitle.textContent = item.textContent.trim();
        state.activePanel = target;
        
        // Trigger specific panel setups
        if (target === "settings-panel") {
            fetchSettings();
        } else if (target === "explorer-panel") {
            fetchChunks();
        }
    });
});

// ==========================================
// BACKEND STATUS POLLING
// ==========================================
async function pollServerStatus() {
    try {
        const response = await fetch("/api/status");
        if (!response.ok) throw new Error("Status failed");
        
        const data = await response.json();
        
        // Update server indicator
        const dot = document.getElementById("server-status-dot");
        const txt = document.getElementById("server-status-text");
        dot.className = "status-dot online";
        txt.textContent = "Server Online";
        
        // Update Stats Bar
        document.getElementById("stat-transcripts").textContent = data.transcripts_count;
        document.getElementById("stat-chunks").textContent = data.chunks_count;
        
        const embBadge = document.getElementById("stat-embeddings");
        if (data.needs_embeddings) {
            embBadge.textContent = `${data.embeddings_count}/${data.chunks_count} (Partial)`;
            embBadge.className = "stat-val pending-badge";
            embBadge.style.color = "var(--warning)";
        } else if (data.embeddings_loaded) {
            embBadge.textContent = "Sync Ready";
            embBadge.className = "stat-val";
            embBadge.style.color = "var(--accent-teal)";
        } else {
            embBadge.textContent = "None";
            embBadge.className = "stat-val";
            embBadge.style.color = "var(--error)";
        }
        
        // Update Background Job Controller
        handleJobStatus(data.job);
        
    } catch (error) {
        const dot = document.getElementById("server-status-dot");
        const txt = document.getElementById("server-status-text");
        dot.className = "status-dot offline";
        txt.textContent = "Connection Lost";
    }
}

function handleJobStatus(job) {
    const runAllBtn = document.getElementById("run-all-pipeline-btn");
    const stepBtns = document.querySelectorAll(".step-card .btn");
    const logsPre = document.getElementById("console-logs");
    
    if (job.status === "processing") {
        state.isProcessing = true;
        
        // Disable pipeline actions
        runAllBtn.disabled = true;
        runAllBtn.textContent = `⏳ Running: ${job.current_task} (${job.progress}%)`;
        stepBtns.forEach(btn => btn.disabled = true);
        
        // Fill console box
        if (job.logs && job.logs.length > 0) {
            logsPre.textContent = job.logs.join("\n");
            // Auto scroll console
            const box = document.getElementById("console-box");
            box.scrollTop = box.scrollHeight;
        }
        
        // Speed up polling to 1.5s when active
        clearInterval(state.statusPoller);
        state.statusPoller = setInterval(pollServerStatus, 1500);
        
    } else {
        if (state.isProcessing) {
            // State just transitioned from processing to idle
            state.isProcessing = false;
            showToast("Pipeline task finished!", "success");
            
            // Re-enable pipeline actions
            runAllBtn.disabled = false;
            runAllBtn.textContent = "⚡ Run Full Pipeline (End-to-End)";
            stepBtns.forEach(btn => btn.disabled = false);
            
            // Slow down polling back to 5s
            clearInterval(state.statusPoller);
            state.statusPoller = setInterval(pollServerStatus, 5000);
            
            // Re-fetch status parameters
            pollServerStatus();
            fetchChunks();
        }
    }
}

// Start polling
pollServerStatus();
state.statusPoller = setInterval(pollServerStatus, 5000);

// ==========================================
// CHAT ASSISTANT LOGIC
// ==========================================
const chatForm = document.getElementById("chat-form");
const queryInput = document.getElementById("query-input");
const chatHistory = document.getElementById("chat-history");
const inspectNoData = document.getElementById("no-inspect-data");
const inspectData = document.getElementById("inspect-data");
const inspectSources = document.getElementById("inspect-sources");
const inspectPrompt = document.getElementById("inspect-prompt");

chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const queryText = queryInput.value.trim();
    if (!queryText) return;
    
    // 1. Add User message
    appendMessage(queryText, "user");
    queryInput.value = "";
    
    // 2. Add loading spinner bubble
    const loadingId = appendMessage("Thinking...", "bot typing");
    
    try {
        const response = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: queryText })
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "RAG retrieval failed");
        }
        
        const data = await response.json();
        
        // 3. Remove typing bubble and append actual answer
        removeMessage(loadingId);
        appendMessage(data.answer, "bot");
        
        // 4. Update Inspector panel
        renderRAGInspector(data);
        
    } catch (err) {
        removeMessage(loadingId);
        appendMessage(`Error: ${err.message}`, "bot", true);
        showToast(err.message, "error");
    }
});

function appendMessage(text, sender, isError = false) {
    const msgId = "msg-" + Date.now();
    const wrapper = document.createElement("div");
    wrapper.id = msgId;
    
    if (sender === "user") {
        wrapper.className = "message user-message";
        wrapper.innerHTML = `<div class="message-content">${text}</div>`;
    } else if (sender === "bot typing") {
        wrapper.className = "message bot-message";
        wrapper.innerHTML = `<div class="message-content" style="color: var(--text-muted); italic; font-size: 0.85rem;">
            <span class="pulse-text">Searching embeddings database & querying Ollama...</span>
        </div>`;
    } else {
        wrapper.className = "message bot-message";
        const contentClass = isError ? "message-content error-msg" : "message-content";
        const parsedText = parseMarkdown(text);
        wrapper.innerHTML = `<div class="${contentClass}">${parsedText}</div>`;
    }
    
    chatHistory.appendChild(wrapper);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return msgId;
}

function removeMessage(id) {
    const element = document.getElementById(id);
    if (element) element.remove();
}

function renderRAGInspector(data) {
    inspectNoData.classList.add("hidden");
    inspectData.classList.remove("hidden");
    
    // Render source chunk list
    inspectSources.innerHTML = "";
    if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
        data.retrieved_chunks.forEach(c => {
            const card = document.createElement("div");
            card.className = "source-card";
            card.innerHTML = `
                <div class="source-header">
                    <span>[${c.tutorial_number}] ${c.tutorial_name} (ch ${c.chunk_id})</span>
                    <span class="source-score">${c.similarity.toFixed(4)}</span>
                </div>
                <div class="source-text">${c.text}</div>
            `;
            inspectSources.appendChild(card);
        });
    } else {
        inspectSources.innerHTML = '<p class="empty-state">No source chunks retrieved.</p>';
    }
    
    // Render compiled prompt text
    inspectPrompt.textContent = data.prompt;
}

// ==========================================
// CHUNK EXPLORER LOGIC
// ==========================================
const chunkSearch = document.getElementById("chunk-search");
const searchBtn = document.getElementById("search-btn");
const chunksTbody = document.getElementById("chunks-tbody");
const prevPageBtn = document.getElementById("prev-page");
const nextPageBtn = document.getElementById("next-page");
const paginationInfo = document.getElementById("pagination-info");

async function fetchChunks() {
    chunksTbody.innerHTML = '<tr><td colspan="5" class="empty-table">Fetching chunks...</td></tr>';
    
    try {
        const queryParams = new URLSearchParams({
            q: state.chunks.searchQuery,
            page: state.chunks.page,
            limit: state.chunks.limit
        });
        
        const response = await fetch(`/api/chunks?${queryParams}`);
        if (!response.ok) throw new Error("Could not fetch chunks");
        
        const data = await response.json();
        state.chunks.list = data.chunks;
        state.chunks.total = data.total;
        
        renderChunksTable();
    } catch (err) {
        chunksTbody.innerHTML = `<tr><td colspan="5" class="empty-table" style="color: var(--error);">Error loading data: ${err.message}</td></tr>`;
    }
}

function renderChunksTable() {
    chunksTbody.innerHTML = "";
    
    if (state.chunks.list.length === 0) {
        chunksTbody.innerHTML = '<tr><td colspan="5" class="empty-table">No matching chunks found in database.</td></tr>';
        prevPageBtn.disabled = true;
        nextPageBtn.disabled = true;
        paginationInfo.textContent = "Showing 0 of 0 chunks";
        return;
    }
    
    state.chunks.list.forEach(c => {
        const tr = document.createElement("tr");
        
        const tagClass = c.has_embedding ? "tag success" : "tag pending";
        const tagText = c.has_embedding ? "Generated" : "Pending";
        
        tr.innerHTML = `
            <td style="font-family: monospace; font-size: 0.8rem; color: var(--text-muted);">${c.chunk_id}</td>
            <td style="font-family: monospace; font-size: 0.8rem;">${c.tutorial_number || "-"}</td>
            <td style="font-weight: 500;">${c.tutorial_name}</td>
            <td><div class="text-excerpt-cell" title="${c.text}">${c.text}</div></td>
            <td style="text-align: center;"><span class="${tagClass}">${tagText}</span></td>
        `;
        chunksTbody.appendChild(tr);
    });
    
    // Update pagination controls
    const totalPages = Math.ceil(state.chunks.total / state.chunks.limit);
    prevPageBtn.disabled = state.chunks.page <= 1;
    nextPageBtn.disabled = state.chunks.page >= totalPages;
    
    const startIdx = (state.chunks.page - 1) * state.chunks.limit + 1;
    const endIdx = Math.min(startIdx + state.chunks.list.length - 1, state.chunks.total);
    paginationInfo.textContent = `Showing ${startIdx}-${endIdx} of ${state.chunks.total} chunks`;
}

searchBtn.addEventListener("click", () => {
    state.chunks.searchQuery = chunkSearch.value.trim();
    state.chunks.page = 1;
    fetchChunks();
});

chunkSearch.addEventListener("keyup", (e) => {
    if (e.key === "Enter") {
        state.chunks.searchQuery = chunkSearch.value.trim();
        state.chunks.page = 1;
        fetchChunks();
    }
});

prevPageBtn.addEventListener("click", () => {
    if (state.chunks.page > 1) {
        state.chunks.page--;
        fetchChunks();
    }
});

nextPageBtn.addEventListener("click", () => {
    const totalPages = Math.ceil(state.chunks.total / state.chunks.limit);
    if (state.chunks.page < totalPages) {
        state.chunks.page++;
        fetchChunks();
    }
});

// ==========================================
// UPLOADS & FILE INGESTION LOGIC
// ==========================================
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const uploadList = document.getElementById("upload-list");

// Trigger browse file dialog
dropzone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
    const files = Array.from(e.target.files);
    handleFileUploads(files);
});

// Drag Over
dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
});

// Drag Leave
dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
});

// Drop
dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const files = Array.from(e.dataTransfer.files);
    handleFileUploads(files);
});

async function handleFileUploads(files) {
    if (files.length === 0) return;
    
    // Clear initial empty state
    if (uploadList.querySelector(".empty-item")) {
        uploadList.innerHTML = "";
    }
    
    for (const file of files) {
        const item = document.createElement("li");
        const fileId = "upload-" + Date.now() + Math.random().toString(36).substr(2, 5);
        item.id = fileId;
        item.innerHTML = `
            <span>${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)</span>
            <div class="upload-progress">
                <div class="upload-progress-bar" id="${fileId}-bar"></div>
            </div>
        `;
        uploadList.appendChild(item);
        
        try {
            const formData = new FormData();
            formData.append("file", file);
            
            const xhr = new XMLHttpRequest();
            xhr.open("POST", "/api/upload", true);
            
            // Track upload progress
            xhr.upload.addEventListener("progress", (e) => {
                if (e.lengthComputable) {
                    const pct = (e.loaded / e.total) * 100;
                    document.getElementById(`${fileId}-bar`).style.width = pct + "%";
                }
            });
            
            xhr.onreadystatechange = () => {
                if (xhr.readyState === 4) {
                    if (xhr.status === 200) {
                        item.innerHTML = `
                            <span>${file.name}</span>
                            <span style="color: var(--success); font-weight: 600;">Success</span>
                        `;
                        showToast(`Uploaded: ${file.name}`, "success");
                        // Refresh status parameters after upload
                        pollServerStatus();
                    } else {
                        let errMsg = "Upload failed";
                        try {
                            const resp = JSON.parse(xhr.responseText);
                            errMsg = resp.detail || errMsg;
                        } catch(e) {}
                        
                        item.innerHTML = `
                            <span>${file.name}</span>
                            <span style="color: var(--error); font-weight: 600;" title="${errMsg}">Failed</span>
                        `;
                        showToast(`${file.name}: ${errMsg}`, "error");
                    }
                }
            };
            
            xhr.send(formData);
        } catch (err) {
            item.innerHTML = `
                <span>${file.name}</span>
                <span style="color: var(--error); font-weight: 600;">Error</span>
            `;
            showToast(`Upload failed for: ${file.name}`, "error");
        }
    }
}

// ==========================================
// PIPELINE RUNNER TRIGGER LOGIC
// ==========================================
const runAllBtn = document.getElementById("run-all-pipeline-btn");
const clearLogsBtn = document.getElementById("clear-logs-btn");
const consoleLogs = document.getElementById("console-logs");

async function startPipelineAction(action) {
    try {
        const response = await fetch("/api/process/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action })
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Failed to start pipeline");
        
        showToast(data.message, "info");
        consoleLogs.textContent = `Awaiting output stream from: ${action}...`;
        
        // Trigger status poll immediately to lock UI controls
        pollServerStatus();
        
    } catch (err) {
        showToast(err.message, "error");
    }
}

// Attach action event handlers to step card buttons
document.querySelectorAll(".step-card button").forEach(btn => {
    btn.addEventListener("click", () => {
        const action = btn.getAttribute("data-action");
        startPipelineAction(action);
    });
});

runAllBtn.addEventListener("click", () => {
    startPipelineAction("all");
});

clearLogsBtn.addEventListener("click", async () => {
    // We can just clear locally if idle
    if (!state.isProcessing) {
        consoleLogs.textContent = "Console idle. Awaiting command execution...";
    }
});

// ==========================================
// CONFIGURATION SETTINGS PANEL
// ==========================================
const settingsForm = document.getElementById("settings-form");

async function fetchSettings() {
    try {
        const response = await fetch("/api/config");
        if (!response.ok) throw new Error("Could not load config");
        const data = await response.json();
        
        document.getElementById("ollama-url").value = data.ollama_url;
        document.getElementById("embeddings-model").value = data.embeddings_model;
        document.getElementById("completions-url").value = data.completions_url;
        document.getElementById("llm-model").value = data.llm_model;
        document.getElementById("max-context-chunks").value = data.max_context_chunks;
        document.getElementById("max-chunk-characters").value = data.max_chunk_characters;
        document.getElementById("chunk-size").value = data.chunk_size;
        document.getElementById("overlap").value = data.overlap;
        
    } catch (err) {
        showToast("Error loading config: " + err.message, "error");
    }
}

settingsForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    
    const payload = {
        ollama_url: document.getElementById("ollama-url").value.trim(),
        embeddings_model: document.getElementById("embeddings-model").value.trim(),
        completions_url: document.getElementById("completions-url").value.trim(),
        llm_model: document.getElementById("llm-model").value.trim(),
        max_context_chunks: parseInt(document.getElementById("max-context-chunks").value),
        max_chunk_characters: parseInt(document.getElementById("max-chunk-characters").value),
        chunk_size: parseInt(document.getElementById("chunk-size").value),
        overlap: parseInt(document.getElementById("overlap").value)
    };
    
    try {
        const response = await fetch("/api/config", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) throw new Error("Save config failed");
        
        showToast("Configuration saved successfully", "success");
        pollServerStatus();
    } catch (err) {
        showToast("Error saving config: " + err.message, "error");
    }
});
