import React, { useMemo, useState } from "react";
import { Atlas, SEQ, fmt, pct, ramp, useMeasure, usePrompt, tokenLabel } from "../lib/data";
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

const LAYER_RAMP: [number, string][] = [
  [0, "#2b5f8a"],
  [0.35, "#1f5c6b"],
  [0.7, "#6b3f6e"],
  [1, "#a8431f"],
];

export default function AttentionPage({ atlas }: { atlas: Atlas }) {
  const [prompt, setPrompt] = usePrompt(atlas);
  const L = atlas.meta.layers;
  const H = atlas.meta.heads;
  const entry = atlas.attention[prompt.id];
  const [head, setHead] = useState<{ l: number; h: number }>({ l: Math.floor(L * 0.6), h: 0 });
  const [hover, setHover] = useState<string | null>(null);
  const [scatterMode, setScatterMode] = useState<"prev" | "sink" | "ent">("prev");

  const { ref, width } = useMeasure<HTMLDivElement>();
  const sc = { w: Math.max(width - 2, 340), h: 320, m: { t: 18, r: 22, b: 40, l: 52 } };
  const heads = atlas.heads;
  const xKey = scatterMode === "prev" ? "prev" : scatterMode === "sink" ? "sink" : "ent";
  const xMax = Math.max(...heads.map((d) => d[xKey]), 0.05);
  const yMax = Math.max(...heads.map((d) => d.ind), 0.05);
  const sx = (v: number) => sc.m.l + (v / xMax) * (sc.w - sc.m.l - sc.m.r);
  const sy = (v: number) => sc.h - sc.m.b - (v / yMax) * (sc.h - sc.m.t - sc.m.b);

  const row = entry.heads[head.l]?.[head.h] ?? [];
  const sorted = useMemo(
    () => row.map((v, i) => [v, i] as [number, number]).sort((a, b) => b[0] - a[0]).slice(0, 6),
    [row],
  );
  const indBest = [...heads].sort((a, b) => b.ind - a.ind)[0];
  const prevBest = [...heads].sort((a, b) => b.prev - a.prev)[0];

  const layerAgg = useMemo(() => {
    return Array.from({ length: L }, (_, l) => {
      const vals = entry.heads[l] ?? [];
      const maxByHead = vals.map((r) => Math.max(...r));
      return {
        l,
        max: Math.max(...maxByHead, 0),
        meanEnt: atlas.heads.filter((d) => d.l === l).reduce((a, d) => a + d.ent, 0) / H,
      };
    });
  }, [entry, atlas.heads, H, L]);

  return (
    <>
      <ChapterHead
        n="03"
        kicker="Attention atlas"
        title={<>Where every head <em>looks</em></>}
        standfirst={
          <>
            Attention is the only part of the model with an interpretable
            intermediate: a row of weights saying which earlier tokens a head read.
            Across the corpus, those rows form recognisable types —{" "}
            <b>previous-token heads, sink heads, induction heads</b> — and the type
            is stable even when the weights are not.
          </>
        }
        chips={
          <>
            <Chip tone="amber">{L * H} heads</Chip>
            <Chip>{L} layers × {H} heads</Chip>
            <Chip tone="cyan">{prompt.id}</Chip>
          </>
        }
      />

      <div className="row" style={{ alignItems: "flex-end", marginBottom: 6 }}>
        <label className="field">
          <span>Probe prompt</span>
          <select className="ctl" value={prompt.id} onChange={(e) => setPrompt(e.target.value)} style={{ minWidth: 320 }}>
            {atlas.prompts.map((p) => (
              <option key={p.id} value={p.id}>
                {p.id} — {p.text.slice(0, 50).replace(/\n/g, " ")}
              </option>
            ))}
          </select>
        </label>
        <div style={{ flex: 1 }} />
        <label className="field">
          <span>taxonomy x-axis</span>
          <Seg
            value={scatterMode}
            options={[
              { v: "prev", label: "prev-token" },
              { v: "sink", label: "sink" },
              { v: "ent", label: "entropy" },
            ]}
            onChange={setScatterMode}
          />
        </label>
      </div>

      <Panel
        title={<><b>Figure 3.1</b> · head taxonomy across the corpus</>}
        tools={<Legend items={[{ c: "#2b5f8a", label: "L0" }, { c: "#1f5c6b", label: "L8" }, { c: "#6b3f6e", label: "L16" }, { c: "#a8431f", label: "L23" }]} />}
        note="Every dot is one head, averaged over all 38 prompts and all query positions. Up = induction/copying score (attention to the token after a previous occurrence). Right = the chosen behaviour. Heads in the top-right corner are copy machinery; heads pinned to the left are positional."
      >
        <div ref={ref} style={{ overflowX: "auto" }}>
          <svg width={sc.w} height={sc.h} style={{ display: "block" }}>
            {[0, 0.25, 0.5, 0.75, 1].map((f) => (
              <g key={f}>
                <line x1={sc.m.l} x2={sc.w - sc.m.r} y1={sy(yMax * f)} y2={sy(yMax * f)} className="gridline" />
                <text x={sc.m.l - 6} y={sy(yMax * f) + 3} textAnchor="end" className="ticklabel">
                  {(yMax * f).toFixed(2)}
                </text>
              </g>
            ))}
            {[0, 0.25, 0.5, 0.75, 1].map((f) => (
              <g key={f}>
                <line x1={sx(xMax * f)} x2={sx(xMax * f)} y1={sc.m.t} y2={sc.h - sc.m.b} className="gridline" />
                <text x={sx(xMax * f)} y={sc.h - sc.m.b + 14} textAnchor="middle" className="ticklabel">
                  {(xMax * f).toFixed(2)}
                </text>
              </g>
            ))}
            <text x={sc.m.l} y={12} className="ticklabel">induction / copy score ↑</text>
            <text x={sc.w - sc.m.r} y={sc.h - 6} textAnchor="end" className="ticklabel">
              {scatterMode === "prev" ? "attention to previous token" : scatterMode === "sink" ? "attention to position 0" : "mean attention entropy"} →
            </text>
            {heads.map((d) => {
              const active = d.l === head.l && d.h === head.h;
              return (
                <circle
                  key={`${d.l}-${d.h}`}
                  cx={sx(d[xKey])}
                  cy={sy(d.ind)}
                  r={active ? 5.5 : 3.4}
                  fill={ramp(d.l / L, LAYER_RAMP)}
                  opacity={active ? 1 : 0.72}
                  stroke={active ? "#fff" : "none"}
                  strokeWidth={active ? 1.4 : 0}
                  style={{ cursor: "pointer" }}
                  onMouseEnter={() => setHover(`L${d.l}·H${d.h}  prev ${fmt(d.prev, 2)}  ind ${fmt(d.ind, 2)}  sink ${fmt(d.sink, 2)}  H ${fmt(d.ent, 2)}`)}
                  onMouseLeave={() => setHover(null)}
                  onClick={() => setHead({ l: d.l, h: d.h })}
                >
                  <title>{`L${d.l} H${d.h} — prev ${d.prev}, ind ${d.ind}, sink ${d.sink}`}</title>
                </circle>
              );
            })}
            <g>
              <rect x={sx(0.55 * xMax)} y={sy(0.55 * yMax)} width={sx(xMax) - sx(0.55 * xMax)} height={sy(0) - sy(0.55 * yMax)} fill="rgba(168,67,31,0.05)" stroke="rgba(168,67,31,0.25)" strokeDasharray="3 3" />
              <text x={sx(0.57 * xMax)} y={sy(0.55 * yMax) + 14} className="ticklabel" style={{ fill: "var(--amber)" }}>
                copy / induction corner
              </text>
            </g>
          </svg>
          <div className="hoverbox" style={{ marginTop: 8 }}>
            {hover ?? "Hover a head for its measured scores; click to open its attention row below."}
          </div>
        </div>
      </Panel>

      <h2 className="sec">The attention row, unrolled</h2>
      <Panel
        title={<><b>Figure 3.2</b> · all {L} layers × {H} heads for “{prompt.text.slice(0, 34).replace(/\n/g, " ")}…”</>}
        tools={
          <span className="mono" style={{ fontSize: 10, color: "var(--ink3)" }}>
            last query position → {prompt.n} keys
          </span>
        }
        note="Each mini-strip is one head's attention from the final token to every earlier token; brighter means more attention. Rows are layers. Click any strip."
      >
        <div className="scroll" style={{ maxHeight: 620, padding: 4 }}>
          {Array.from({ length: L }, (_, l) => (
            <div key={l} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
              <span className="ticklabel" style={{ width: 24, textAlign: "right" }}>{l}</span>
              <div style={{ display: "flex", gap: 2, flex: 1 }}>
                {(entry.heads[l] ?? []).map((rowv, hi) => (
                  <div
                    key={hi}
                    onClick={() => setHead({ l, h: hi })}
                    style={{
                      flex: 1,
                      height: 15,
                      display: "grid",
                      gridTemplateColumns: `repeat(${rowv.length}, 1fr)`,
                      gap: 0.6,
                      outline: head.l === l && head.h === hi ? "1px solid var(--amber)" : "1px solid transparent",
                      cursor: "pointer",
                      padding: 1,
                    }}
                    title={`L${l}·H${hi} · max ${fmt(Math.max(...rowv), 2)}`}
                  >
                    {rowv.map((v, k) => (
                      <div
                        key={k}
                        style={{
                          background: ramp(Math.sqrt(v / 0.9), SEQ),
                          height: "100%",
                          borderRadius: 0.5,
                        }}
                      />
                    ))}
                  </div>
                ))}
              </div>
              <span className="ticklabel" style={{ width: 40 }}>
                {layerAgg[l] ? fmt(layerAgg[l].meanEnt, 1) : ""}
              </span>
            </div>
          ))}
          <div className="spread" style={{ marginTop: 8 }}>
            <span className="ticklabel">← layers · heads across · right column: mean entropy</span>
            <span className="ticklabel">keys →</span>
          </div>
        </div>
      </Panel>

      <div className="grid2" style={{ alignItems: "start" }}>
        <Panel
          title={<><b>Figure 3.3</b> · L{head.l}·H{head.h}</>}
          tools={<span className="chip">{fmt(Math.max(...row), 2)} max</span>}
          note="Attention from the final token to each key position, for the selected head only."
        >
          <div className="stack" style={{ gap: 4 }}>
            {prompt.tokens.map((t, i) => {
              const v = row[i] ?? 0;
              return (
                <div key={i} style={{ display: "grid", gridTemplateColumns: "86px 1fr 44px", gap: 8, alignItems: "center" }}>
                  <span className="token" style={{ textAlign: "right", border: 0 }}>{tokenLabel(t).slice(0, 10) || "∅"}</span>
                  <div style={{ height: 10, background: "rgba(23,24,28,0.03)" }}>
                    <div style={{ width: `${(v / (Math.max(...row) || 1)) * 100}%`, height: "100%", background: "linear-gradient(90deg,#1f5c6b,#1f5c6b)" }} />
                  </div>
                  <span className="ticklabel">{pct(v, 0)}</span>
                </div>
              );
            })}
            <div className="caption">
              Top keys:{" "}
              {sorted.map(([v, i], k) => (
                <span key={k} className="token" title={`${pct(v, 1)}`}>
                  {tokenLabel(prompt.tokens[i]) || "∅"}
                </span>
              ))}
            </div>
          </div>
        </Panel>

        <div className="stack">
          <div className="grid2">
            <Stat k="Heads with ind > 0.25" v={heads.filter((d) => d.ind > 0.25).length} s="induction-like" />
            <Stat k="Heads with sink > 0.4" v={heads.filter((d) => d.sink > 0.4).length} s="attend to token 0" />
            <Stat k="Most previous-token-ish" v={(() => { const b = [...heads].sort((a, c) => c.prev - a.prev)[0]; return `L${b.l}·H${b.h}`; })()} s={fmt([...heads].sort((a, c) => c.prev - a.prev)[0].prev, 2)} />
            <Stat k="Lowest entropy" v={(() => { const b = [...heads].sort((a, c) => a.ent - c.ent)[0]; return `L${b.l}·H${b.h}`; })()} s={fmt([...heads].sort((a, c) => a.ent - c.ent)[0].ent, 2)} />
          </div>
          <Panel title={<><b>Figure 3.4</b> · per-layer character</>} note="Mean over heads: attention to token 0 (sink), to the previous token, and entropy. The sink tendency is a positional artefact with a real function.">
            <div className="stack" style={{ gap: 3 }}>
              {layerAgg.map((d) => {
                const st = atlas.heads.filter((x) => x.l === d.l);
                const prev = st.reduce((a, x) => a + x.prev, 0) / st.length;
                const sink = st.reduce((a, x) => a + x.sink, 0) / st.length;
                return (
                  <div key={d.l} style={{ display: "grid", gridTemplateColumns: "26px 1fr 1fr 38px", gap: 5, alignItems: "center" }}>
                    <span className="ticklabel">{d.l}</span>
                    <div style={{ height: 7, background: "rgba(23,24,28,0.03)" }}>
                      <div style={{ width: `${Math.min(prev, 1) * 100}%`, height: "100%", background: "#1f5c6b" }} title={`prev ${fmt(prev, 2)}`} />
                    </div>
                    <div style={{ height: 7, background: "rgba(23,24,28,0.03)" }}>
                      <div style={{ width: `${Math.min(sink, 1) * 100}%`, height: "100%", background: "#43478c" }} title={`sink ${fmt(sink, 2)}`} />
                    </div>
                    <span className="ticklabel">{fmt(d.meanEnt, 1)}</span>
                  </div>
                );
              })}
            </div>
          </Panel>
        </div>
      </div>

      <Prose>
        <p>
          The taxonomy is not decoration. The strongest <strong>induction
          score</strong> in this model is {fmt(indBest.ind, 2)} at{" "}
          <strong>L{indBest.l}·H{indBest.h}</strong> — attention to the token that
          followed the previous occurrence of the current token, averaged over every
          position where that condition held. The most positional head is{" "}
          L{prevBest.l}·H{prevBest.h}, at {fmt(prevBest.prev, 2)} attention to the
          previous token. <strong>Sink heads</strong> dump attention on position 0;
          they look like a bug, but they act as a "no information needed" channel
          that lets a head abstain without contributing zero.
        </p>
        <p>
          The count above ({heads.filter((d) => d.ind > 0.25).length} heads above an
          induction score of 0.25) is measured, not annotated by hand. Averaging over
          the whole corpus — not only repeated-token prompts — dilutes the signal, so
          the copy corner of Figure 3.1 is sparse and the threshold matters. An
          induction score is a behavioural fingerprint, not proof that a head
          implements copying.
        </p>
      </Prose>

      <Method>
        <p>
          For every prompt, every layer and every head, attention is captured for
          all query positions, then reduced to five scores: mean attention to
          <code>t−1</code> (previous token), to <code>t</code> (self), to position 0
          (sink), mean entropy over keys, and an induction score — for each query
          position whose token occurred earlier, the attention paid to the token
          immediately <em>after</em> that earlier occurrence, averaged over all such
          positions.
        </p>
        <p>
          With grouped-query attention ({atlas.meta.kvHeads} KV heads shared across{" "}
          {atlas.meta.heads} query heads) the value/key projections are shared, so
          "head" here means a query head with its own attention row and its own
          slice of the output projection.
        </p>
      </Method>

      <Footnotes
        items={[
          "Attention weights are softmax outputs, summed to 1 per query row. They are not causal explanations; a head can attend somewhere and write nothing useful.",
          "Induction score requires a repeated token with a following token, so it is only defined on positions where the current token has appeared before. Prompts without repetition contribute no samples to that head's induction score.",
          "The sink score is attention to position 0 only. Qwen uses no explicit BOS in this tokenisation path, so position 0 is the first real token — a stable anchor nonetheless.",
        ]}
      />
      <PageFoot index={4} />
    </>
  );
}
