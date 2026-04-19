import { useState, useRef, useEffect } from "react"
import NodeGraph from "./NodeGraph"
import "./TaskRunner.css"

const API = "http://localhost:8000"

const SAMPLE_TASKS = [
  "What is the time complexity of Dijkstra's algorithm and why?",
  "Compare the transformer architecture, RWKV, and Mamba — explain how each handles long-range dependencies differently",
  "Is RAG still relevant now that context windows are so large?",
  "Find the latest benchmark scores for Llama 4, Qwen3, and Gemini 2.5 Pro and rank them across reasoning, coding, and math",
]

export default function TaskRunner() {
  const [task,       setTask]       = useState("")
  const [status,     setStatus]     = useState("idle")   // idle | running | done | error
  const [tokens,     setTokens]     = useState("")
  const [activeNode, setActiveNode] = useState(null)
  const [metadata,   setMetadata]   = useState(null)
  const [runId,      setRunId]      = useState(null)

  const outputRef  = useRef(null)
  const esRef      = useRef(null)

  // auto-scroll output
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight
    }
  }, [tokens])

  function reset() {
    setTokens("")
    setActiveNode(null)
    setMetadata(null)
    setRunId(null)
    setStatus("idle")
    if (esRef.current) { esRef.current.close(); esRef.current = null }
  }

  async function handleRun() {
    if (!task.trim() || status === "running") return
    reset()
    setStatus("running")
    setActiveNode("classifier")          // highlight classifier immediately on click

    try {
      const resp = await fetch(`${API}/run`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ task }),
      })

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)

      const reader = resp.body.getReader()
      const dec    = new TextDecoder()
      let   buf    = ""

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buf += dec.decode(value, { stream: true })

        // SSE spec: events are separated by double newline
        const parts = buf.split("\n\n")
        buf = parts.pop()               // Added cause last part may be potentially incomplete

        for (const part of parts) {
          // each part may have multiple "data:" lines — join them
          const dataLine = part
            .split("\n")
            .filter(l => l.startsWith("data: "))
            .map(l => l.slice(6))
            .join("")

          if (!dataLine.trim()) continue

          try {
            handleSSEEvent(JSON.parse(dataLine))
          } catch { /* malformed */ }
        }
      }
    } catch (err) {
      setStatus("error")
      console.error(err)
    }
  }

  function handleSSEEvent(event) {
    switch (event.type) {
      case "token": {
        setTokens(prev => prev + event.content)
        break
      }
      case "node_change": {
        if (event.node) setActiveNode(event.node)
        break
      }
      case "metadata": {
        setMetadata(event)
        // after routing is known, highlight the correct branch node
        if (event.routing === "single") setActiveNode("single_agent_node")
        if (event.routing === "multi")  setActiveNode("synthesizer")
        break
      }
      case "done": {
        setRunId(event.run_id)
        setActiveNode("optimal_decision")
        setTimeout(() => {
          setActiveNode(null)
          setStatus("done")
        }, 1200)
        break
      }
      case "error": {
        setStatus("error")
        setActiveNode(null)
        break
      }
    }
  }

  const canRun = task.trim().length > 0 && status !== "running"

  return (
    <div className="runner-layout">

      {/* left: pipeline */}
      <aside className="runner-sidebar">
        <NodeGraph activeNode={activeNode} routing={metadata?.routing} />

        {metadata && (
          <div className="meta-card animate-fade-in">
            <div className="card-title">Routing</div>
            <div className="meta-row">
              <span className="meta-key">architecture</span>
              <span className={`meta-val arch-${metadata.routing}`}>
                {metadata.routing ?? "—"}
              </span>
            </div>
            <div className="meta-row">
              <span className="meta-key">task type</span>
              <span className="meta-val">{metadata.task_type ?? "—"}</span>
            </div>
            <div className="meta-row">
              <span className="meta-key">confidence</span>
              <ConfidenceBar value={metadata.confidence} />
            </div>
            {runId && (
              <div className="meta-row">
                <span className="meta-key">run id</span>
                <span className="meta-val run-id">{runId.slice(0, 8)}…</span>
              </div>
            )}
          </div>
        )}
      </aside>

      {/* right: input + output */}
      <section className="runner-main">

        {/* input */}
        <div className="input-block">
          <textarea
            className="task-input"
            placeholder="enter a task…"
            value={task}
            onChange={e => setTask(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && e.metaKey) handleRun() }}
            rows={3}
            disabled={status === "running"}
          />
          <div className="input-actions">
            <div className="sample-pills">
              {SAMPLE_TASKS.map((t, i) => (
                <button
                  key={i}
                  className="sample-pill"
                  onClick={() => setTask(t)}
                  disabled={status === "running"}
                >
                  {t.length > 42 ? t.slice(0, 42) + "…" : t}
                </button>
              ))}
            </div>
            <div className="run-actions">
              {status !== "idle" && (
                <button className="btn-ghost" onClick={reset}>clear</button>
              )}
              <button
                className={`btn-run ${status === "running" ? "is-running" : ""}`}
                onClick={handleRun}
                disabled={!canRun}
              >
                {status === "running" ? <RunningIndicator /> : "Run  ⌘↵"}
              </button>
            </div>
          </div>
        </div>

        {/* output */}
        {(tokens || status === "running") && (
          <div className="output-block animate-fade-in">
            <div className="output-header">
              <span className="card-title">Output</span>
              <StatusBadge status={status} node={activeNode} />
            </div>
            <div className="output-body" ref={outputRef}>
              <span className="output-text">{tokens}</span>
              {status === "running" && <span className="cursor-blink">▋</span>}
            </div>
          </div>
        )}

        {status === "idle" && !tokens && (
          <div className="empty-state">
            <div className="empty-hex">⬡</div>
            <p>submit a task to watch the pipeline run</p>
          </div>
        )}
      </section>
    </div>
  )
}

function ConfidenceBar({ value }) {
  if (value == null) return <span className="meta-val">—</span>
  const pct = Math.round(value * 100)
  const color = pct >= 75 ? "var(--green)" : pct >= 50 ? "var(--amber)" : "var(--red)"
  return (
    <div className="conf-bar-wrap">
      <div className="conf-bar">
        <div className="conf-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="meta-val" style={{ color }}>{pct}%</span>
    </div>
  )
}

function StatusBadge({ status, node }) {
  if (status === "running") return (
    <span className="status-badge running">
      <span className="badge-dot" /> {node ?? "processing"}
    </span>
  )
  if (status === "done")  return <span className="status-badge done">complete</span>
  if (status === "error") return <span className="status-badge error">error</span>
  return null
}

function RunningIndicator() {
  return (
    <span className="running-ind">
      <span /><span /><span />
    </span>
  )
}
