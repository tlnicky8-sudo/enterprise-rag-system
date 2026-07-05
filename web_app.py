"""劳动法智能问答系统 - Web 版"""
import json
import uuid
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, Response, request, session, render_template_string
from openai import OpenAI

from base import Config, logger
from core.qa_pipeline import QAPipeline
from core.rag_system import RAGSystem
from core.vector_store import VectorStore

conf = Config()
app = Flask(__name__)
app.secret_key = str(uuid.uuid4())

print("正在初始化系统组件...")
try:
    print("  [1/4] 连接 LLM API...")
    client = OpenAI(api_key=conf.DASHSCOPE_API_KEY, base_url=conf.DASHSCOPE_BASE_URL)
    print("  [2/4] 加载向量数据库 (Milvus + BGE-M3)...")
    vector_store = VectorStore()
    print("  [3/4] 初始化 FAQ + RAG 流水线...")
    pipeline = None

    def call_dashscope(prompt):
        completion = client.chat.completions.create(
            model=conf.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的劳动法助手，请基于提供的法律条文上下文回答用户问题。",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content if completion.choices else ""

    def call_llm_stream(prompt):
        try:
            completion = client.chat.completions.create(
                model=conf.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的劳动法助手，请基于提供的法律条文上下文回答用户问题。",
                    },
                    {"role": "user", "content": prompt},
                ],
                timeout=60,
                stream=True,
            )
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            logger.error(f"LLM 调用失败: {exc}")
            yield f"\n\n[错误：服务暂时不可用，请稍后重试 - {exc}]"

    rag_system = RAGSystem(vector_store, call_dashscope, stream_llm=call_llm_stream)
    pipeline = QAPipeline(rag_system=rag_system)
    print("  [4/4] 流水线就绪")
    logger.info("所有组件初始化成功")
except Exception as exc:
    print(f"  错误：组件初始化失败 - {exc}")
    logger.error(f"组件初始化失败: {exc}")
    raise


# ── 路由 ──
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, sources=conf.VALID_SOURCES)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    source_filter = data.get("source_filter") or None

    if not query:
        return {"error": "问题不能为空"}, 400
    if len(query) > 1000:
        return {"error": "问题过长，请控制在 1000 字以内"}, 400
    if source_filter and source_filter not in conf.VALID_SOURCES:
        return {"error": "无效的文档来源"}, 400

    sid = session.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        session["sid"] = sid

    def generate():
        full_answer = ""
        try:
            result = pipeline.answer(query, sid, source_filter=source_filter, stream=True)
            citations = list(result.citations)
            response_source = result.source
            trust_payload = {
                "source": response_source,
                "grounded": result.grounded,
                "refusal_reason": result.refusal_reason,
                "citation_count": len(citations),
            }
            if not result.stream:
                full_answer = result.answer
                yield f"data: {json.dumps({'token': full_answer}, ensure_ascii=False)}\n\n"
            else:
                for token in result.token_iterator:
                    full_answer += token
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                pipeline.save_streamed_answer(
                    sid,
                    query,
                    full_answer,
                    source=result.source,
                    generation=result.generation,
                )
            yield f"data: {json.dumps({'done': True, 'citations': citations, **trust_payload}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error(f"处理查询失败: {exc}")
            yield f"data: {json.dumps({'error': '服务暂时不可用，请稍后重试'}, ensure_ascii=False)}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/history", methods=["GET"])
def history():
    sid = session.get("sid", "")
    return {"history": pipeline.get_history(sid) if sid else []}


@app.route("/api/clear", methods=["POST"])
def clear():
    sid = session.get("sid")
    if sid:
        pipeline.clear_session(sid)
    return {"ok": True}


# ── HTML 模板 ──
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>劳动法智能问答</title>
<style>
  *,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
  :root{
    --bg:#0a0e17;--surface:#111827;--border:rgba(255,255,255,.06);
    --text:#e2e8f0;--text-muted:#94a3b8;--accent:#22d3ee;
    --user-bg:rgba(34,211,238,.12);--ai-bg:#1a1f2e;
  }
  body{
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
    background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column;
  }
  .bg-grid{
    position:fixed;inset:0;pointer-events:none;z-index:0;
    background-image:linear-gradient(rgba(34,211,238,.03) 1px,transparent 1px),
      linear-gradient(90deg,rgba(34,211,238,.03) 1px,transparent 1px);
    background-size:60px 60px;
    mask-image:radial-gradient(ellipse 70% 70% at 50% 50%,black 30%,transparent 70%);
  }
  header{
    position:relative;z-index:1;padding:16px 24px;border-bottom:1px solid var(--border);
    display:flex;align-items:center;justify-content:space-between;flex-shrink:0;
  }
  header h1{font-size:18px;font-weight:600;color:var(--accent)}
  .header-actions{display:flex;gap:10px;align-items:center}
  .btn{
    padding:6px 14px;border-radius:8px;border:1px solid var(--border);
    background:rgba(255,255,255,.04);color:var(--text-muted);cursor:pointer;
    font-size:13px;transition:all .2s;
  }
  .btn:hover{background:rgba(34,211,238,.1);border-color:rgba(34,211,238,.2);color:var(--accent)}
  .source-select{
    padding:6px 10px;border-radius:8px;border:1px solid var(--border);
    background:var(--surface);color:var(--text);font-size:13px;outline:none;
  }
  .source-select:focus{border-color:var(--accent)}
  .chat-area{
    position:relative;z-index:1;flex:1;overflow-y:auto;padding:20px 24px;
    display:flex;flex-direction:column;gap:16px;
  }
  .msg{display:flex;gap:10px;max-width:80%;animation:fadeIn .3s ease}
  .msg.user{align-self:flex-end;flex-direction:row-reverse}
  .msg.ai{align-self:flex-start}
  @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
  .avatar{
    width:36px;height:36px;border-radius:50%;flex-shrink:0;
    display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:700;
  }
  .msg.user .avatar{background:linear-gradient(135deg,#22d3ee,#6366f1);color:#fff}
  .msg.ai .avatar{background:linear-gradient(135deg,#6366f1,#a855f7);color:#fff}
  .bubble{
    padding:12px 16px;border-radius:14px;font-size:14px;line-height:1.65;
    word-break:break-word;
  }
  .msg.user .bubble{background:var(--user-bg);border:1px solid rgba(34,211,238,.15)}
  .msg.ai .bubble{background:var(--ai-bg);border:1px solid var(--border)}
  .answer-text{white-space:pre-wrap}
  .trust-bar{
    margin-top:10px;padding-top:10px;border-top:1px solid var(--border);
    font-size:12px;color:var(--text-muted);display:flex;align-items:center;gap:8px;flex-wrap:wrap;
  }
  .trust-pill{
    display:inline-flex;align-items:center;padding:3px 8px;border-radius:999px;
    border:1px solid var(--border);background:rgba(255,255,255,.04);font-size:12px;
  }
  .trust-pill.grounded{color:#86efac;border-color:rgba(134,239,172,.25);background:rgba(134,239,172,.08)}
  .trust-pill.refused{color:#fca5a5;border-color:rgba(252,165,165,.25);background:rgba(252,165,165,.08)}
  .trust-pill.faq{color:var(--accent);border-color:rgba(34,211,238,.25);background:rgba(34,211,238,.08)}
  .evidence-bar{margin-top:10px;padding-top:10px;border-top:1px solid var(--border)}
  .evidence-btn{
    display:inline-flex;align-items:center;gap:6px;padding:5px 10px;border-radius:8px;
    border:1px solid rgba(34,211,238,.25);background:rgba(34,211,238,.08);
    color:var(--accent);font-size:12px;cursor:pointer;transition:all .2s;
  }
  .evidence-btn:hover{background:rgba(34,211,238,.16)}
  .evidence-modal{
    position:fixed;inset:0;z-index:20;display:none;align-items:center;justify-content:center;
    background:rgba(2,6,23,.72);padding:20px;
  }
  .evidence-modal.open{display:flex}
  .evidence-panel{
    width:min(720px,100%);max-height:80vh;overflow:hidden;
    background:var(--surface);border:1px solid var(--border);border-radius:16px;
    box-shadow:0 20px 60px rgba(0,0,0,.35);display:flex;flex-direction:column;
  }
  .evidence-header{
    display:flex;align-items:center;justify-content:space-between;
    padding:16px 18px;border-bottom:1px solid var(--border);
  }
  .evidence-header h3{font-size:15px;font-weight:600}
  .evidence-close{
    border:none;background:transparent;color:var(--text-muted);font-size:20px;cursor:pointer;
  }
  .evidence-list{padding:14px 18px 18px;overflow-y:auto;display:flex;flex-direction:column;gap:12px}
  .citation-card{
    padding:12px 14px;border-radius:12px;border:1px solid var(--border);
    background:rgba(255,255,255,.02);
  }
  .citation-title{font-size:14px;font-weight:600;color:var(--text);margin-bottom:6px}
  .citation-meta{font-size:12px;color:var(--text-muted);margin-bottom:8px;line-height:1.5}
  .citation-excerpt{
    font-size:13px;line-height:1.7;color:var(--text);white-space:pre-wrap;
    background:rgba(0,0,0,.18);padding:10px 12px;border-radius:10px;
  }
  .input-area{
    position:relative;z-index:1;padding:16px 24px;border-top:1px solid var(--border);
    display:flex;gap:10px;flex-shrink:0;
  }
  .input-area input{
    flex:1;padding:12px 16px;border-radius:12px;border:1px solid var(--border);
    background:var(--surface);color:var(--text);font-size:14px;outline:none;
  }
  .input-area input:focus{border-color:var(--accent)}
  .input-area input::placeholder{color:var(--text-muted)}
  .send-btn{
    padding:12px 20px;border-radius:12px;border:none;
    background:linear-gradient(135deg,#22d3ee,#6366f1);color:#fff;
    font-size:14px;font-weight:600;cursor:pointer;transition:all .2s;
  }
  .send-btn:hover{opacity:.9;transform:translateY(-1px)}
  .send-btn:disabled{opacity:.4;cursor:not-allowed;transform:none}
  .typing-indicator{display:flex;gap:6px;padding:8px 0}
  .typing-indicator span{
    width:8px;height:8px;border-radius:50%;background:var(--accent);
    animation:bounce 1.4s infinite ease-in-out both;
  }
  .typing-indicator span:nth-child(1){animation-delay:-.32s}
  .typing-indicator span:nth-child(2){animation-delay:-.16s}
  @keyframes bounce{0%,80%,100%{transform:scale(0)}40%{transform:scale(1)}}
  .empty-state{
    flex:1;display:flex;align-items:center;justify-content:center;color:var(--text-muted);
    font-size:15px;text-align:center;line-height:1.8;
  }
  .empty-state strong{color:var(--accent)}
  .welcome-hint{font-size:13px;margin-top:8px;opacity:.6}
</style>
</head>
<body>

<div class="bg-grid"></div>

<header>
  <h1>劳动法智能问答</h1>
  <div class="header-actions">
    <select class="source-select" id="sourceFilter">
      <option value="">全部来源</option>
      {% for s in sources %}
      <option value="{{ s }}">{{ s }}</option>
      {% endfor %}
    </select>
    <button class="btn" onclick="clearHistory()">清空对话</button>
  </div>
</header>

<div class="chat-area" id="chatArea">
  <div class="empty-state">
    <div>
      <div>欢迎使用 <strong>劳动法智能问答系统</strong></div>
      <div class="welcome-hint">基于《劳动合同法》和《劳动法》条文，为您提供专业解答</div>
    </div>
  </div>
</div>

<div class="input-area">
  <input id="queryInput" type="text" placeholder="输入您的劳动法问题…" onkeydown="if(event.key==='Enter')send()">
  <button class="send-btn" id="sendBtn" onclick="send()">发送</button>
</div>

<div class="evidence-modal" id="evidenceModal" onclick="if(event.target===this)closeEvidenceModal()">
  <div class="evidence-panel">
    <div class="evidence-header">
      <h3 id="evidenceTitle">检索证据</h3>
      <button class="evidence-close" type="button" onclick="closeEvidenceModal()">×</button>
    </div>
    <div class="evidence-list" id="evidenceList"></div>
  </div>
</div>

<script>
const chatArea = document.getElementById("chatArea");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const sourceFilter = document.getElementById("sourceFilter");
const evidenceModal = document.getElementById("evidenceModal");
const evidenceList = document.getElementById("evidenceList");
const evidenceTitle = document.getElementById("evidenceTitle");
let isStreaming = false;
let emptyState = chatArea.querySelector(".empty-state");

function send() {
  const query = queryInput.value.trim();
  if (!query || isStreaming) return;
  isStreaming = true;
  sendBtn.disabled = true;
  queryInput.value = "";

  if (emptyState) { emptyState.remove(); emptyState = null; }

  addMessage("user", query);
  const aiMsg = addMessage("ai", "");
  const typing = showTyping(aiMsg);

  const decoder = new TextDecoder();
  let fullText = "";

  fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, source_filter: sourceFilter.value || null }),
  }).then(async resp => {
    if (!resp.ok) {
      let message = "请求失败，请稍后重试";
      try {
        const payload = await resp.json();
        message = payload.error || message;
      } catch (e) {
        message = await resp.text() || message;
      }
      getAnswerTextEl(aiMsg).textContent = "[错误] " + message;
      typing.remove();
      isStreaming = false;
      sendBtn.disabled = false;
      queryInput.focus();
      return;
    }

    const reader = resp.body.getReader();
    let buffer = "";
    let typingVisible = true;
    const handleStreamEvent = (data) => {
      if (data.token) {
        if (typingVisible) {
          typing.remove();
          typingVisible = false;
        }
        fullText += data.token;
        getAnswerTextEl(aiMsg).textContent = fullText;
        chatArea.scrollTop = chatArea.scrollHeight;
      }
      if (data.done) {
        if (typingVisible) {
          typing.remove();
          typingVisible = false;
        }
        attachTrustBar(aiMsg, data);
        if (Array.isArray(data.citations) && data.citations.length > 0) {
          attachEvidenceButton(aiMsg, data.citations);
        }
      }
      if (data.error) {
        getAnswerTextEl(aiMsg).textContent = "[错误] " + data.error;
        if (typingVisible) {
          typing.remove();
          typingVisible = false;
        }
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      for (const eventText of events) {
        const dataLines = eventText
          .split("\n")
          .filter(line => line.startsWith("data: "))
          .map(line => line.slice(6));
        if (dataLines.length === 0) continue;
        try {
          handleStreamEvent(JSON.parse(dataLines.join("\n")));
        } catch (e) {
          console.warn("忽略无法解析的 SSE 消息", e);
        }
      }
    }
    buffer += decoder.decode();
    const trailing = buffer.trim();
    if (trailing.startsWith("data: ")) {
      try {
        handleStreamEvent(JSON.parse(trailing.slice(6)));
      } catch (e) {
        console.warn("忽略无法解析的 SSE 尾包", e);
      }
    }
    isStreaming = false;
    sendBtn.disabled = false;
    queryInput.focus();
  }).catch(() => {
    getAnswerTextEl(aiMsg).textContent = fullText || "[网络错误，请重试]";
    typing.remove();
    isStreaming = false;
    sendBtn.disabled = false;
  });
}

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (role === "ai") {
    div.innerHTML = `
      <div class="avatar">法</div>
      <div class="bubble">
        <div class="answer-text">${escapeHtml(text)}</div>
      </div>
    `;
  } else {
    div.innerHTML = `
      <div class="avatar">我</div>
      <div class="bubble">${escapeHtml(text)}</div>
    `;
  }
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
  return div;
}

function getAnswerTextEl(parentMsg) {
  return parentMsg.querySelector(".answer-text") || parentMsg.querySelector(".bubble");
}

function attachEvidenceButton(parentMsg, citations) {
  const bubble = parentMsg.querySelector(".bubble");
  if (!bubble || bubble.querySelector(".evidence-btn")) return;

  const bar = document.createElement("div");
  bar.className = "evidence-bar";
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "evidence-btn";
  btn.textContent = `查看检索证据 (${citations.length})`;
  btn.onclick = () => openEvidenceModal(citations);
  bar.appendChild(btn);
  bubble.appendChild(bar);
}

function attachTrustBar(parentMsg, data) {
  const bubble = parentMsg.querySelector(".bubble");
  if (!bubble || bubble.querySelector(".trust-bar")) return;

  const bar = document.createElement("div");
  bar.className = "trust-bar";

  const pill = document.createElement("span");
  if (data.refusal_reason) {
    pill.className = "trust-pill refused";
    pill.textContent = "证据不足，已拒答";
  } else if (data.grounded) {
    pill.className = "trust-pill grounded";
    pill.textContent = `基于 ${data.citation_count || 0} 条检索证据`;
  } else if (data.source === "faq") {
    pill.className = "trust-pill faq";
    pill.textContent = data.citation_count > 0 ? "FAQ 缓存回答，含原始证据" : "FAQ 固定问答";
  } else if (data.source === "direct_llm") {
    pill.className = "trust-pill";
    pill.textContent = "通用问题，未检索法条";
  } else {
    pill.className = "trust-pill refused";
    pill.textContent = "未检索到可引用证据";
  }
  bar.appendChild(pill);

  if (data.refusal_reason) {
    const reason = document.createElement("span");
    reason.textContent = data.refusal_reason;
    bar.appendChild(reason);
  }
  bubble.appendChild(bar);
}

function openEvidenceModal(citations) {
  evidenceTitle.textContent = `检索证据 (${citations.length})`;
  evidenceList.innerHTML = citations.map(item => `
    <div class="citation-card">
      <div class="citation-title">[${item.id}] ${escapeHtml(item.title || "未命名片段")}</div>
      <div class="citation-meta">
        ${item.source ? `来源：${escapeHtml(item.source)}` : ""}
        ${item.source_file ? `<br>文件：${escapeHtml(item.source_file)}` : ""}
        ${item.section_path && item.section_path !== item.title ? `<br>章节：${escapeHtml(item.section_path)}` : ""}
      </div>
      <div class="citation-excerpt">${escapeHtml(item.excerpt || "")}</div>
    </div>
  `).join("");
  evidenceModal.classList.add("open");
}

function closeEvidenceModal() {
  evidenceModal.classList.remove("open");
}

function showTyping(parentMsg) {
  const div = document.createElement("div");
  div.className = "typing-indicator";
  div.innerHTML = "<span></span><span></span><span></span>";
  getAnswerTextEl(parentMsg).appendChild(div);
  return div;
}

function escapeHtml(s) {
  const el = document.createElement("span");
  el.textContent = s;
  return el.innerHTML;
}

async function clearHistory() {
  await fetch("/api/clear", { method: "POST" });
  chatArea.innerHTML = "";
  emptyState = document.createElement("div");
  emptyState.className = "empty-state";
  emptyState.innerHTML = `
    <div>
      <div>对话已清空</div>
      <div class="welcome-hint">请继续提问</div>
    </div>
  `;
  chatArea.appendChild(emptyState);
}
</script>

</body>
</html>
"""


def main():
    print("\n" + "=" * 50)
    print("  劳动法智能问答系统 - Web 版")
    print("=" * 50)
    print(f"\n  浏览器访问: http://127.0.0.1:5000")
    print(f"  按 Ctrl+C 停止服务\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
