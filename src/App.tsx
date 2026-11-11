import React from "react";
import { CHAPTERS, Atlas, href, useRoute } from "./lib/data";
import Overview from "./pages/Overview";
import Architecture from "./pages/Architecture";
import Lens from "./pages/Lens";
import Anatomy from "./pages/Anatomy";
import AttentionPage from "./pages/Attention";
import Circuit from "./pages/Circuit";
import Features from "./pages/Features";
import Emergence from "./pages/Emergence";
import GeometryPage from "./pages/Geometry";
import AblationPage from "./pages/Ablation";
import Landscape from "./pages/Landscape";
import Flow from "./pages/Flow";
import Waterfall from "./pages/Waterfall";
import Commitment from "./pages/Commitment";
import FunctionMap from "./pages/FunctionMap";

class Boundary extends React.Component<
  { children: React.ReactNode; k: string },
  { err: Error | null }
> {
  state = { err: null as Error | null };
  static getDerivedStateFromError(err: Error) {
    return { err };
  }
  componentDidUpdate(prev: { k: string }) {
    if (prev.k !== this.props.k && this.state.err) this.setState({ err: null });
  }
  componentDidCatch(err: Error, info: React.ErrorInfo) {
    console.error("chapter render error", err, info.componentStack);
  }
  render() {
    if (this.state.err) {
      return (
        <section className="panel" style={{ marginTop: 40 }}>
          <div className="panel-head">
            <div className="panel-title">This chapter failed to render</div>
          </div>
          <div className="panel-body">
            <pre style={{ whiteSpace: "pre-wrap", color: "var(--rust)", fontFamily: "var(--mono)", fontSize: 11, margin: 0 }}>
              {String(this.state.err?.stack || this.state.err)}
            </pre>
          </div>
        </section>
      );
    }
    return this.props.children;
  }
}

export default function App({ atlas }: { atlas: Atlas }) {
  const route = useRoute();
  const idx = Math.max(0, CHAPTERS.findIndex((c) => c.slug === route));
  const chapter = CHAPTERS[idx] ?? CHAPTERS[0];

  const pages: Record<string, React.ReactNode> = {
    "": <Overview atlas={atlas} />,
    architecture: <Architecture atlas={atlas} />,
    lens: <Lens atlas={atlas} />,
    anatomy: <Anatomy atlas={atlas} />,
    attention: <AttentionPage atlas={atlas} />,
    circuit: <Circuit atlas={atlas} />,
    features: <Features atlas={atlas} />,
    emergence: <Emergence atlas={atlas} />,
    geometry: <GeometryPage atlas={atlas} />,
    ablation: <AblationPage atlas={atlas} />,
    landscape: <Landscape atlas={atlas} />,
    flow: <Flow atlas={atlas} />,
    waterfall: <Waterfall atlas={atlas} />,
    commitment: <Commitment atlas={atlas} />,
    function: <FunctionMap atlas={atlas} />,
  };

  return (
    <div className="shell">
      <aside className="rail">
        <a className="brand" href={href("")}>
          <div className="mark">
            Depth <em>Atlas</em>
          </div>
          <div className="sub">a layer-by-layer reading of Qwen</div>
        </a>

        <div className="rail-meta">
          <div>
            <b>{atlas.meta.modelName}</b>
          </div>
          <div>
            {atlas.meta.layers} layers · d={atlas.meta.hidden} · {atlas.meta.heads} heads
          </div>
          <div>
            vocab {atlas.meta.vocab.toLocaleString()} · {atlas.meta.corpus} prompts
          </div>
          <div className="ok">● measured, not simulated</div>
        </div>

        <nav className="rail-nav">
          <div className="rail-group">Read the atlas</div>
          {CHAPTERS.map((c, i) => (
            <a key={c.slug || "overview"} href={href(c.slug)} className={"rail-item" + (i === idx ? " active" : "")}>
              <span className="idx">{c.n}</span>
              <span>{c.short}</span>
            </a>
          ))}
        </nav>

        <div className="rail-foot">
          Every number on these pages is computed from forward and backward passes
          of this checkpoint. Data generated{" "}
          {atlas.meta.generatedAt.replace("T", " ").replace("Z", " UTC")}.
          <br />
          <br />
          {atlas.meta.tokens.toLocaleString()} tokens · {atlas.meta.evidence}
        </div>
      </aside>

      <main className="main">
        <div className="topbar">
          <span className="crumb">
            {chapter.n} / {chapter.short}
          </span>
          <span className="spacer" />
          <span className="chip">{atlas.meta.modelName}</span>
          <span className="chip amber">{atlas.meta.layers} layers</span>
          <span className="chip cyan">{atlas.meta.vocab.toLocaleString()} vocab</span>
        </div>
        <div className="page">
          <Boundary k={route}>{pages[route] ?? pages[""]}</Boundary>
        </div>
      </main>
    </div>
  );
}
