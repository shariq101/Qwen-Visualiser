import React, { useMemo, useState } from "react";
import { Atlas, DIV, fmt, useMeasure, tokenLabel } from "../lib/data";
import {
  Caption,
  ChapterHead,
  Chip,
  Footnotes,
  Legend,
  Method,
  PageFoot,
  Panel,
  Prose,
  Seg,
  Stat,
} from "../components/ui";

export default function Circuit({ atlas }: { atlas: Atlas }) {
  const ids = Object.keys(atlas.graph);
  const [pid, setPid] = useState(ids[0]);
  const g = atlas.graph[pid];
  const [topN, setTopN] = useState(90);
  const [kind, setKind] = useState<"all" | "attn" | "mlp">("all");
  const [focus, setFocus] = useState<string | null>(null);
  const [layerRange, setLayerRange] = useState<[number, number]>([0, g.L]);

  const { ref, width } = useMeasure<HTMLDivElement>();
  const w = Math.max(width - 2, 360);
  const H = 150;
  const h = 40 + (g.L + 1) * H * 0.42;
  const m = { t: 30, r: 24, b: 22, l: 64 };
  const T = g.tokens.length;
  const colw = (w - m.l - m.r) / Math.max(T - 1, 1);
  const rowh = (h - m.t - m.b) / g.L;
  const nx = (p: number) => m.l + p * colw;
  const ny = (l: number) => h - m.b - l * rowh;

  const edges = useMemo(() => {
    return g.edges
      .filter((e) => (kind === "all" ? true : e.kind === kind))
      .filter((e) => e.layer >= layerRange[0] && e.layer <= layerRange[1])
      .slice(0, topN);
  }, [g, kind, topN, layerRange]);

  const maxW = Math.max(...edges.map((e) => Math.abs(e.w)), 1e-6);
  const maxAttr = Math.max(...g.attr.flat().map(Math.abs), 1e-6);

  const incoming = focus ? g.edges.filter((e) => e.t === focus) : [];
  const outgoing = focus ? g.edges.filter((e) => e.s === focus) : [];

  return (
    <>
      <ChapterHead
        n="04"
        kicker="Attribution graph"
        title={<>The path from tokens to a <em>logit</em></>}
        standfirst={
          <>
            An attribution graph is a claim about flow: this token, at this layer,
            moved this much probability into that later position. Here the edges
            are computed from real attention weights and per-head direct logit
            attribution, then pruned to the strongest 160. <b>Follow one path and
            you can name the computation.</b>
          </>
        }
        chips={
          <>
            <Chip tone="amber">{g.nodes.length} nodes</Chip>
            <Chip>{g.edges.length} edges before pruning</Chip>
            <Chip tone="cyan">target “{g.target.trim()}”</Chip>
          </>
        }
      />

      <div className="row" style={{ alignItems: "flex-end", marginBottom: 6 }}>
        <label className="field">
          <span>Deep-dive prompt</span>
          <select className="ctl" value={pid} onChange={(e) => { setPid(e.target.value); setFocus(null); }} style={{ minWidth: 320 }}>
            {ids.map((id) => (
              <option key={id} value={id}>
                {id} — {atlas.prompts.find((p) => p.id === id)?.text.slice(0, 46).replace(/\n/g, " ")}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>edges shown · {topN}</span>
          <input type="range" min={20} max={160} step={10} value={topN} onChange={(e) => setTopN(+e.target.value)} style={{ width: 160 }} />
        </label>
        <div style={{ flex: 1 }} />
        <Seg
          value={kind}
          options={[
            { v: "all", label: "all" },
            { v: "attn", label: "attention" },
            { v: "mlp", label: "MLP" },
          ]}
          onChange={setKind}
        />
      </div>

      <Panel
        title={<><b>Figure 4.1</b> · attribution graph</>}
        tools={<Legend items={[{ c: "#a8431f", label: "positive flow" }, { c: "#2b5f8a", label: "negative flow" }, { c: "#43478c", label: "MLP edge" }]} />}
        note="Layers run bottom (L0) to top (L23). Horizontal position is the token. Edge width is |attribution|; colour is sign. Attention edges bend because a query at position q reads a key at position k."
      >
        <div ref={ref} style={{ overflowX: "auto" }}>
          <svg width={w} height={h} style={{ display: "block" }}>
            {Array.from({ length: g.L + 1 }, (_, l) => (
              <g key={l}>
                <line x1={m.l - 8} x2={w - m.r} y1={ny(l)} y2={ny(l)} stroke="#d8d4ca" />
                <text x={m.l - 14} y={ny(l) + 3} textAnchor="end" className="ticklabel">
                  L{l}
                </text>
              </g>
            ))}
            {g.tokens.map((t, i) => (
              <text key={i} x={nx(i)} y={12} textAnchor="middle" className="ticklabel" style={{ fill: i === T - 1 ? "var(--amber)" : undefined }}>
                {tokenLabel(t).slice(0, 7) || "∅"}
              </text>
            ))}
            {edges.map((e, i) => {
              const [sl, sp] = e.s.split(":").map(Number);
              const [tl, tp] = e.t.split(":").map(Number);
              const x1 = nx(sp), y1 = ny(sl), x2 = nx(tp), y2 = ny(tl);
              const mx = (x1 + x2) / 2;
              const my = (y1 + y2) / 2 - (e.kind === "attn" ? 26 : 8);
              const pos = e.w > 0;
              const hot = focus && (e.s === focus || e.t === focus);
              return (
                <path
                  key={i}
                  d={`M${x1},${y1} Q${mx},${my} ${x2},${y2}`}
                  fill="none"
                  stroke={e.kind === "mlp" ? "#43478c" : pos ? "#a8431f" : "#2b5f8a"}
                  strokeWidth={hot ? 2.4 : 0.3 + (Math.abs(e.w) / maxW) * 2.4}
                  opacity={hot ? 0.95 : 0.16 + (Math.abs(e.w) / maxW) * 0.55}
                />
              );
            })}
            {Array.from({ length: g.L + 1 }, (_, l) =>
              g.tokens.map((_, p) => {
                const id = `${l}:${p}`;
                const v = g.attr[l]?.[p] ?? 0;
                const f = Math.abs(v) / maxAttr;
                const isFocus = focus === id;
                return (
                  <circle
                    key={id}
                    cx={nx(p)}
                    cy={ny(l)}
                    r={isFocus ? 6 : 1.6 + Math.sqrt(f) * 6}
                    fill={v >= 0 ? "#a8431f" : "#2b5f8a"}
                    opacity={isFocus ? 1 : 0.35 + f * 0.65}
                    stroke={isFocus ? "#fff" : "none"}
                    strokeWidth={1.2}
                    style={{ cursor: "pointer" }}
                    onClick={() => setFocus(isFocus ? null : id)}
                  >
                    <title>{`L${l} · ${tokenLabel(g.tokens[p])} · attr ${fmt(v, 3)}`}</title>
                  </circle>
                );
              }),
            )}
          </svg>
        </div>
      </Panel>

      <div className="grid3">
        <Stat k="Focused node" v={focus ?? "—"} s={focus ? `token “${tokenLabel(g.tokens[+focus.split(":")[1]])}”` : "click any dot"} />
        <Stat k="Incoming edges" v={incoming.length} s="who writes to it" />
        <Stat k="Outgoing edges" v={outgoing.length} s="where it writes" />
      </div>

      {focus && (
        <Panel
          title={<><b>Figure 4.2</b> · neighbourhood of {focus}</>}
          note="Edges into and out of the focused node, strongest first. Attention edges carry (layer, query, key); MLP edges are position-preserving."
        >
          <div className="grid2">
            <div>
              <div className="caption" style={{ marginBottom: 6 }}>incoming</div>
              <table className="tbl">
                <thead>
                  <tr><th>from</th><th>kind</th><th>weight</th><th>token</th></tr>
                </thead>
                <tbody>
                  {[...incoming].sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, 10).map((e, i) => (
                    <tr key={i}>
                      <td className="k">{e.s}</td>
                      <td>{e.kind}</td>
                      <td style={{ color: e.w >= 0 ? "var(--amber)" : "var(--blue)" }}>{fmt(e.w, 3)}</td>
                      <td>{tokenLabel(g.tokens[+e.s.split(":")[1]])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div>
              <div className="caption" style={{ marginBottom: 6 }}>outgoing</div>
              <table className="tbl">
                <thead>
                  <tr><th>to</th><th>kind</th><th>weight</th><th>token</th></tr>
                </thead>
                <tbody>
                  {[...outgoing].sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, 10).map((e, i) => (
                    <tr key={i}>
                      <td className="k">{e.t}</td>
                      <td>{e.kind}</td>
                      <td style={{ color: e.w >= 0 ? "var(--amber)" : "var(--blue)" }}>{fmt(e.w, 3)}</td>
                      <td>{tokenLabel(g.tokens[+e.t.split(":")[1]])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Panel>
      )}

      <h2 className="sec">The same graph as a matrix</h2>
      <Panel
        title={<><b>Figure 4.3</b> · gradient × activation attribution</>}
        note="Attribution of each residual state to the target logit, ∂logit/∂h · h. Rows are layers, columns tokens. This is the quantity that decides node size in Figure 4.1."
      >
        <div style={{ overflowX: "auto" }}>
          <svg width={Math.max(w, 640)} height={(g.L + 1) * 15 + 30} style={{ display: "block" }}>
            {g.tokens.map((t, i) => (
              <text key={i} x={nx(i)} y={10} textAnchor="middle" className="ticklabel">{tokenLabel(t).slice(0, 6)}</text>
            ))}
            {Array.from({ length: g.L + 1 }, (_, l) =>
              g.tokens.map((_, p) => {
                const v = g.attr[l][p];
                const f = v / maxAttr;
                const cw = (Math.max(w, 640) - m.l - m.r) / g.tokens.length;
                return (
                  <rect
                    key={`${l}-${p}`}
                    x={m.l + p * cw}
                    y={18 + l * 15}
                    width={cw - 2}
                    height={13}
                    fill={f >= 0 ? `rgba(168,67,31,${0.08 + f * 0.85})` : `rgba(43,95,138,${0.08 - f * 0.85})`}
                  >
                    <title>{`L${l} ${tokenLabel(g.tokens[p])} ${fmt(v, 3)}`}</title>
                  </rect>
                );
              }),
            )}
            {Array.from({ length: g.L + 1 }, (_, l) => (
              <text key={l} x={m.l - 8} y={18 + l * 15 + 10} textAnchor="end" className="ticklabel">{l}</text>
            ))}
          </svg>
        </div>
      </Panel>

      <Prose>
        <p>
          For this prompt the flow concentrates in two bands: an early block that
          moves information out of the subject token, and a late block that writes
          the answer direction. Between them, most edges are weak — the graph is
          sparse in the middle and dense at both ends, which is the structural
          signature of a residual network doing lookups early and decisions late.
        </p>
        <p>
          The strongest single edge into the final position is{" "}
          <strong>{(() => { const e = [...g.edges].sort((a, b) => Math.abs(b.w) - Math.abs(a.w))[0]; return e ? `${e.s} → ${e.t} (${e.kind}, ${fmt(e.w, 2)})` : "—"; })()}</strong>.
          That is not necessarily the most <em>important</em> edge — attribution is
          local and first-order — but it is the strongest measured one.
        </p>
      </Prose>

      <Method>
        <p>
          Nodes are residual states at (layer, position). Edges have two kinds: an
          attention edge from (l, k) to (l+1, q) weighted by{" "}
          <code>mean_head attn[q→k] × DLA_attn(l, q)</code>, and an MLP edge from
          (l, p) to (l+1, p) weighted by <code>DLA_mlp(l, p)</code>. DLA projects
          the component's output onto the target token's unembedding direction.
        </p>
        <p>
          Node size and colour come from a separate measurement: autograd gives{" "}
          <code>∂logit_target/∂h(l,p)</code>, which is multiplied by{" "}
          <code>h(l,p)</code> element-wise and summed. Two measurements are shown
          together because each one is noisy in a different way.
        </p>
      </Method>

      <Footnotes
        items={[
          "This is an attribution graph in the loose sense, not the full circuit-tracing sense: edges are first-order approximations and are not guaranteed to compose across layers. Treat it as a map of measured influence, not a proof.",
          "Only deep-dive prompts have graphs because the backward pass is per-prompt; the capture stage deliberately avoids gradients for the rest of the corpus.",
          "Edge pruning keeps the strongest 160 by |weight|. Raising the slider re-admits weaker edges, which is a useful way to see how much of the graph is load-bearing.",
        ]}
      />
      <PageFoot index={5} />
    </>
  );
}
