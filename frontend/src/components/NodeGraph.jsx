import { useEffect, useRef } from "react"
import "./NodeGraph.css"

const NODES = [
  { id: "classifier",       label: "Classifier",      color: "var(--node-router)",  x: 50,  y: 50  },
  { id: "single_agent_node",label: "Single Agent",    color: "var(--node-single)", x: 20,  y: 50  },
  { id: "multi_agent",      label: "Multi Agent",     color: "var(--node-multi)",  x: 80,  y: 50  },
  { id: "synthesizer",      label: "Synthesizer",     color: "var(--node-multi)",  x: 80,  y: 50  },
  { id: "judge",            label: "Judge",           color: "var(--node-judge)",  x: 50,  y: 80  },
  { id: "optimal_decision", label: "Decision",        color: "var(--node-router)", x: 50,  y: 95  },
]

// which node label maps to which display color category
const NODE_COLOR_MAP = {
  classifier:        "router",
  single_agent_node: "single",
  synthesizer:       "multi",
  judge:             "judge",
  optimal_decision:  "router",
}

export default function NodeGraph({ activeNode, routing }) {
  return (
    <div className="node-graph">
      <div className="graph-title">PIPELINE</div>

      <div className="pipeline-track">
        {/* Step 1: Classifier */}
        <PipelineNode
          id="classifier"
          label="Classifier"
          sublabel="task taxonomy"
          type="router"
          active={activeNode === "classifier"}
          done={activeNode && activeNode !== "classifier"}
        />

        <Connector active={!!activeNode} />

        {/* Step 2: Branch */}
        <div className="branch-row">
          <PipelineNode
            id="single_agent_node"
            label="Single Agent"
            sublabel="direct LLM"
            type="single"
            active={activeNode === "single_agent_node"}
            done={isDone(activeNode, "single_agent_node")}
            dimmed={routing === "multi"}
          />
          <div className="branch-divider">OR</div>
          <PipelineNode
            id="synthesizer"
            label="Multi Agent"
            sublabel="parallel → synthesize"
            type="multi"
            active={activeNode === "synthesizer"}
            done={isDone(activeNode, "synthesizer")}
            dimmed={routing === "single"}
          />
        </div>

        <Connector active={activeNode === "judge" || activeNode === "optimal_decision"} />

        {/* Step 3: Judge */}
        <PipelineNode
          id="judge"
          label="Judge"
          sublabel="quality eval"
          type="judge"
          active={activeNode === "judge"}
          done={activeNode === "optimal_decision"}
        />

        <Connector active={activeNode === "optimal_decision"} />

        {/* Step 4: Decision */}
        <PipelineNode
          id="optimal_decision"
          label="Decision"
          sublabel="optimal output"
          type="router"
          active={activeNode === "optimal_decision"}
          done={false}
        />
      </div>
    </div>
  )
}

function PipelineNode({ id, label, sublabel, type, active, done, dimmed }) {
  return (
    <div
      className={[
        "pipeline-node",
        `type-${type}`,
        active  ? "is-active" : "",
        done    ? "is-done"   : "",
        dimmed  ? "is-dimmed" : "",
      ].join(" ")}
    >
      <div className="pnode-indicator">
        {done   ? <DoneIcon /> : null}
        {active ? <ActiveDot /> : null}
        {!active && !done ? <IdleDot /> : null}
      </div>
      <div className="pnode-body">
        <span className="pnode-label">{label}</span>
        <span className="pnode-sub">{sublabel}</span>
      </div>
      {active && <div className="pnode-glow" />}
    </div>
  )
}

function Connector({ active }) {
  return (
    <div className={`pipe-connector ${active ? "is-lit" : ""}`}>
      <div className="connector-line" />
      {active && <div className="connector-dot" />}
    </div>
  )
}

function ActiveDot() {
  return <span className="active-dot" />
}

function IdleDot() {
  return <span className="idle-dot" />
}

function DoneIcon() {
  return <span className="done-icon">✓</span>
}

function isDone(activeNode, nodeId) {
  const order = ["classifier", "single_agent_node", "synthesizer", "judge", "optimal_decision"]
  const activeIdx = order.indexOf(activeNode)
  const nodeIdx   = order.indexOf(nodeId)
  return activeIdx > nodeIdx
}
