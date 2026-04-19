import { useState, useEffect } from "react"
import "./Dashboard.css"

const API = "http://localhost:8000"

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      fetch(`${API}/metrics`).then(r => r.json()),
      fetch(`${API}/history`).then(r => r.json()),
    ]).then(([m, h]) => {
      setMetrics(m)
      setHistory(h.slice().reverse()) // newest first
    }).catch(console.error)
    .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="dash-loading">
      <span className="loading-hex">⬡</span>
      <span>loading metrics…</span>
    </div>
  )

  return (
    <div className="dashboard">

      {/*Stats Row*/}
      <div className="stat-row">
        <StatCard label="Total Runs"      value={metrics?.total_runs ?? 0} />
        <StatCard label="Avg Latency (single)" value={metrics?.avg_latency_ms?.single ? `${metrics.avg_latency_ms.single}ms` : "—"} color="var(--node-single)" />
        <StatCard label="Avg Latency (multi)"  value={metrics?.avg_latency_ms?.multi  ? `${metrics.avg_latency_ms.multi}ms`  : "—"} color="var(--node-multi)"  />
        <StatCard label="Avg Judge (single)"   value={metrics?.avg_judge_score?.single ?? "—"} color="var(--node-single)" />
        <StatCard label="Avg Judge (multi)"    value={metrics?.avg_judge_score?.multi  ?? "—"} color="var(--node-multi)"  />
      </div>

      {/*Distribution of routing*/}
      {metrics?.routing_distribution && (
        <div className="card animate-fade-in">
          <div className="card-title">Routing Distribution</div>
          <RoutingBars dist={metrics.routing_distribution} total={metrics.total_runs} />
        </div>
      )}

      {/*Run History*/}
      <div className="card animate-fade-in">
        <div className="card-title">Run History</div>
        {history.length === 0 ? (
          <p className="no-data">no runs yet</p>
        ) : (
          <div className="history-table">
            <div className="ht-header">
              <span>task</span>
              <span>arch</span>
              <span>type</span>
              <span>conf</span>
              <span>judge</span>
              <span>latency</span>
            </div>
            {history.map((run, i) => (
              <HistoryRow key={i} run={run} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function StatCard({ label, value, color }) {
  return (
    <div className="stat-card animate-fade-in">
      <span className="stat-label">{label}</span>
      <span className="stat-value" style={color ? { color } : {}}>{value}</span>
    </div>
  )
}

function RoutingBars({ dist, total }) {
  const entries = Object.entries(dist)
  const colors = { single: "var(--node-single)", multi: "var(--node-multi)", ambiguous: "var(--node-judge)" }
  return (
    <div className="routing-bars">
      {entries.map(([key, val]) => {
        const pct = total > 0 ? Math.round((val / total) * 100) : 0
        return (
          <div key={key} className="rbar-row">
            <span className="rbar-label">{key}</span>
            <div className="rbar-track">
              <div
                className="rbar-fill"
                style={{ width: `${pct}%`, background: colors[key] ?? "var(--accent)" }}
              />
            </div>
            <span className="rbar-count">{val} <span className="rbar-pct">({pct}%)</span></span>
          </div>
        )
      })}
    </div>
  )
}

function HistoryRow({ run }) {
  const arch  = run.architectures_run?.[0]
  const conf  = run.confidence != null ? `${Math.round(run.confidence * 100)}%` : "—"
  const judge = run.judge_score ?? "—"
  const lat   = run.latency_ms  ? `${run.latency_ms}ms` : "—"
  const task  = run.task ?? "—"

  return (
    <div className="ht-row animate-slide-in">
      <span className="ht-task" title={task}>{task.length > 55 ? task.slice(0, 55) + "…" : task}</span>
      <span className={`ht-arch arch-${arch}`}>{arch ?? "—"}</span>
      <span className="ht-cell">{run.task_type ?? "—"}</span>
      <span className="ht-cell">{conf}</span>
      <span className="ht-cell">{judge}</span>
      <span className="ht-cell">{lat}</span>
    </div>
  )
}
