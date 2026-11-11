import React, { useMemo, useState } from "react";
import { Atlas, fmt, pct, tokenLabel, useMeasure } from "../lib/data";
import {
  Caption,
  ChapterHead,
  Chip,
  Footnotes,
  Method,
  PageFoot,
  Panel,
  Prose,
  Seg,
  Stat,
} from "../components/ui";

export default function Features({ atlas }: { atlas: Atlas }) {
  const L = atlas.meta.layers;
  const [layer, setLayer] = useState(12);
  const [sort, setSort] = useState<"max" | "density" | "sparse">("max");
  const [query, setQuery] = useState("");

  const rows = useMemo(() => {
    let r = atlas.neurons.filter((n) => n.layer === layer);
    if (sort === "max") r = [...r].sort((a, b) => b.max - a.max);
    if (sort === "density") r = [...r].sort((a, b) => b.density - a.density);
    if (sort === "sparse") r = [...r].sort((a, b) => a.density - b.density);
    if (query) {
      const q = query.toLowerCase();
      r = r.filter(
        (n) =>
          n.contexts.some((c) => (c.before + c.token).toLowerCase().includes(q)) ||
          String(n.neuron) === q,
      );
    }
    return r;
  }, [atlas.neurons, layer, sort, query]);

  const peaks = useMemo(
    () =>
      Array.from({ length: L }, (_, l) =>
        Math.max(...atlas.neurons.filter((n) => n.layer === l).map((n) => n.max), 1),
      ),
    [atlas.neurons, L],
  );
  const maxPeak = Math.max(...peaks, 1);
  const { ref: chartRef, width: chartW } = useMeasure<HTMLDivElement>();

  const meanDensity = atlas.neurons.reduce((a, n) => a + n.density, 0) / atlas.neurons.length;
  const selective = atlas.neurons.filter((n) => n.density < 0.08).length;

  return (
    <>
      <ChapterHead
        n="05"
        kicker="Feature catalogue"
        title={<>Neurons, caught in the <em>act</em></>}
        standfirst={
          <>
            An MLP neuron is a direction in the residual stream and a gate. The
            most direct way to understand one is to ask: <b>what was the model
            reading when this neuron fired hardest?</b> Every context below is a
            real passage from the corpus, with the measured activation attached.
          </>
        }
        chips={
          <>
            <Chip tone="amber">{atlas.neurons.length} neurons catalogued</Chip>
            <Chip>top 14 per layer by peak activation</Chip>
            <Chip tone="cyan">{selective} selective (density &lt; 0.08)</Chip>
          </>
        }
      />

      <div className="row" style={{ alignItems: "flex-end", marginBottom: 6, flexWrap: "wrap", rowGap: 10 }}>
        <label className="field">
          <span>Layer · {layer}</span>
          <input type="range" min={0} max={L - 1} value={layer} onChange={(e) => setLayer(+e.target.value)} style={{ width: "min(220px, 60vw)" }} />
        </label>
        <div style={{ flex: 1 }} />
        <label className="field">
          <span>search contexts</span>
          <input className="ctl" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. Paris, def, 7" style={{ width: 180 }} />
        </label>
        <Seg
          value={sort}
          options={[
            { v: "max", label: "peak" },
            { v: "density", label: "dense" },
            { v: "sparse", label: "selective" },
          ]}
          onChange={setSort}
        />
      </div>

      <Panel
        title={<><b>Figure 5.1</b> · peak activation by layer</>}
        note="The largest activation reached by any of the 14 catalogued neurons in each layer, on a logarithmic vertical axis, because magnitudes grow steeply with depth as residual norms grow (chapter 02). A linear axis would flatten everything before layer 30. Click a bar to load that layer."
      >
        <div ref={chartRef}>
          <svg width={Math.max(chartW, 320)} height={134} style={{ display: "block" }}>
            {[1, 10, 100, 1000, 10000, maxPeak]
              .filter((v, i, a) => v <= maxPeak && a.indexOf(v) === i)
              .map((v) => {
                const y = 108 - (Math.log10(1 + v) / Math.log10(1 + maxPeak)) * 94;
                return (
                  <g key={v}>
                    <line x1={44} x2={Math.max(chartW, 320) - 6} y1={y} y2={y} className="gridline" />
                    <text x={40} y={y + 3} textAnchor="end" className="ticklabel">
                      {v >= 1000 ? `${v / 1000}k` : v}
                    </text>
                  </g>
                );
              })}
            {peaks.map((peak, l) => {
              const bw = (Math.max(chartW, 320) - 50) / L;
              const y = 108 - (Math.log10(1 + peak) / Math.log10(1 + maxPeak)) * 94;
              return (
                <g key={l} onClick={() => setLayer(l)} style={{ cursor: "pointer" }}>
                  <rect
                    x={44 + l * bw + 0.5}
                    y={y}
                    width={Math.max(bw - 1.2, 1)}
                    height={Math.max(108 - y, 1.5)}
                    fill={l === layer ? "var(--rust)" : "var(--rule-2)"}
                  >
                    <title>{`L${l}: peak activation ${fmt(peak, 2)}`}</title>
                  </rect>
                </g>
              );
            })}
            <line x1={44} x2={Math.max(chartW, 320) - 6} y1={108} y2={108} stroke="var(--ink)" />
            <text x={44} y={122} className="ticklabel">L0</text>
            <text x={Math.max(chartW, 320) - 6} y={122} textAnchor="end" className="ticklabel">
              L{L - 1}
            </text>
            <text x={44} y={132} className="ticklabel" style={{ fill: "var(--ink-3)" }}>
              peak activation (log scale) · selected layer L{layer}
            </text>
          </svg>
        </div>
      </Panel>

      <div className="grid2" style={{ alignItems: "start" }}>
        {rows.map((n) => (
          <Panel
            key={`${n.layer}-${n.neuron}`}
            title={<><b>L{n.layer}·{n.neuron}</b></>}
            tools={
              <>
                <span className="chip amber">peak {fmt(n.max, 1)}</span>
                <span className="chip">density {pct(n.density, 0)}</span>
              </>
            }
            note={`Mean activation ${fmt(n.mean, 2)} across prompts. Density = fraction of prompts where this neuron reaches 10% of its peak — a crude selectivity measure.`}
          >
            <div className="stack" style={{ gap: 6 }}>
              {n.contexts.slice(0, 5).map((c, i) => (
                <div
                  key={i}
                  style={{
                    fontFamily: "var(--mono)",
                    fontSize: 11,
                    lineHeight: 1.6,
                    color: "var(--ink2)",
                    borderLeft: `2px solid rgba(168,67,31,${0.2 + (c.act / n.max) * 0.8})`,
                    paddingLeft: 9,
                    background: "rgba(23,24,28,0.012)",
                  }}
                >
                  <div style={{ color: "var(--ink3)", fontSize: 9.5, marginBottom: 2 }}>
                    {c.prompt} · pos {c.pos} · act {fmt(c.act, 1)}
                  </div>
                  <span>{c.before.replace(/\s+/g, " ").slice(-90)}</span>
                  <span
                    style={{
                      background: `rgba(168,67,31,${0.15 + (c.act / n.max) * 0.55})`,
                      color: "#f7f5f0",
                      padding: "1px 4px",
                      borderRadius: 2,
                      fontWeight: 500,
                    }}
                  >
                    {tokenLabel(c.token)}
                  </span>
                </div>
              ))}
            </div>
          </Panel>
        ))}
      </div>

      {rows.length === 0 && (
        <Panel title={<><b>No match</b></>}>
          <div className="caption">No neuron contexts on this layer match “{query}”. Try clearing the search or moving layers.</div>
        </Panel>
      )}

      <Prose>
        <p>
          Read a few contexts and a pattern usually emerges — not always a clean
          one. Some neurons fire on a specific token in any context; others fire on
          a <em>role</em> (a word that is being defined, a number that is being
          incremented). Density separates the two: a token-specific neuron fires
          whenever that token appears, so it can look dense in a small corpus by
          accident, while a role neuron fires in many different prompts.
        </p>
        <p>
          The catalogue is deliberately small. These are the <strong>top 14
          neurons per layer by peak activation</strong> over 38 prompts — a
          biased sample of the most excitable units, not a random one. A full
          feature atlas would need sparse autoencoders or a much larger corpus;
          chapter 13 is explicit about that limit.
        </p>
      </Prose>

      <Method>
        <p>
          Qwen's MLP is SwiGLU: <code>down_proj(act(gate_proj(x)) ⊙ up_proj(x))</code>.
          The "neuron" is an index in the {atlas.meta.hidden ? "" : ""}intermediate
          dimension, and its activation is the gated value captured at the input to{" "}
          <code>down_proj</code> — i.e. after the nonlinearity and the
          element-wise product, which is the quantity that actually reaches the
          residual stream.
        </p>
        <p>
          For each neuron, the six strongest contexts across the corpus are kept,
          with the exact token position of the maximum. Contexts are truncated for
          display but the activation is exact.
        </p>
      </Method>

      <Footnotes
        items={[
          "Selectivity measured over 38 prompts is weak evidence. A neuron with density 0.05 may simply be rare in this corpus rather than genuinely selective.",
          "No auto-interpretation labels are applied here: the catalogue shows raw contexts so the reader can judge. Any label in the prose is the author's reading of those contexts, not a model output.",
          "Activation magnitudes are not comparable across layers: later layers have larger residual norms and correspondingly larger MLP activations.",
        ]}
      />
      <PageFoot index={6} />
    </>
  );
}
