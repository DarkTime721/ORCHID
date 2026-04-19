import { useState } from "react"
import TaskRunner from "./components/TaskRunner"
import Dashboard from "./components/Dashboard"
import "./index.css"

export default function App() {
  const [view, setView] = useState("runner")

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <span className="logo-mark">⬡</span>
          <span className="logo-text">ORCHID</span>
          <span className="logo-sub">orchestration research</span>
        </div>
        <nav className="header-nav">
          <button
            className={`nav-btn ${view === "runner" ? "active" : ""}`}
            onClick={() => setView("runner")}
          >
            Run
          </button>
          <button
            className={`nav-btn ${view === "dashboard" ? "active" : ""}`}
            onClick={() => setView("dashboard")}
          >
            Dashboard
          </button>
        </nav>
      </header>

      <main className="app-main">
        {view === "runner" ? <TaskRunner /> : <Dashboard />}
      </main>
    </div>
  )
}
