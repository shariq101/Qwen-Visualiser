import React, { useMemo, useState } from "react";
import { Atlas, bare, fmt, pct, useMeasure, usePrompt, tokenLabel } from "../lib/data";
import {
  Axis,
  Caption,
  ChapterHead,
  Chip,
  Footnotes,
  Legend,
  Method,
  PageFoot,
  Panel,
  Prose,
  Stat,
} from "../components/ui";

function RankRiver({
  entry,
  L,
  topK,
  onLayer,
  layer,
}: {
  entry: Atlas["lens"][string];
  L: number;
  topK: number;
  onLayer: (l: number) => void;
  layer: number;
}) {
  const colW = 64;
  const rowH = 26;
  const left = 30;
  const top = 16;
  const H = top + topK * rowH + 10;
  const lines: React.ReactNode[] = [];
  const cells: React.ReactNode[] = [];

  for (let l = 0; l < L; l++) {
    const a = entry.top[l].slice(0, topK).map((d) => d[0]);
    const b = entry.top[l + 1].slice(0, topK).map((d) => d[0]);
    a.forEach((tok, i) => {
      const j = b.indexOf(tok);
      if (j >= 0) {
        lines.push(
          <line
            key={`${l}-${i}`}
            x1={left + l * colW + 18}
            y1={top + i * rowH + 10}
            x2={left + (l + 1) * colW + 18}
            y2={top + j * rowH + 10}
            stroke={i === 0 ? "rgba(168,67,31,0.5)" : "rgba(130,133,140,0.22)"}
            strokeWidth={i === 0 ? 1.4 : 1}
          />,
        );
      }
    });
  }

  for (let l = 0; l <= L; l++) {
    const items = entry.top[l].slice(0, topK);
    items.forEach(([tok, p], i) => {
      const isT = bare(tok) === bare(entry.targetStr);
      cells.push(
        <g key={`${l}-${i}`} onClick={() => onLayer(l)} style={{ cursor: "pointer" }}>
          <rect
            x={left + l * colW}
            y={top + i * rowH}
            width={colW - 12}
            height={rowH - 6}
            fill={isT ? "rgba(168,67,31,0.22)" : i === 0 ? "rgba(31,92,107,0.14)" : "rgba(23,24,28,0.025)"}
            stroke={l === layer ? "rgba(168,67,31,0.8)" : "transparent"}
          />
          <text
            x={left + l * colW + 4}
            y={top + i * rowH + 13}
            style={{ fontFamily: "var(--mono)", fontSize: 8.5, fill: isT ? "#a8431f" : i === 0 ? "#2b2d33" : "#8d9199" }}
          >
            {(tokenLabel(tok) || "∅").slice(0, 7)}
          </text>
          <text x={left + l * colW + colW - 14} y={top + i * rowH + 13} textAnchor="end" style={{ fontFamily: "var(--mono)", fontSize: 7.5, fill: "#8a8d93" }}>
            {(p * 100).toFixed(0)}
          </text>
        </g>,
      );
    });
    cells.push(
      <text key={`lab-${l}`} x={left + l * colW + 12} y={H - 2} textAnchor="middle" className="ticklabel">
        {l}
      </text>,
    );
  }

  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={left + (L + 1) * colW} height={H} style={{ display: "block" }}>
        {lines}
        {cells}
      </svg>
    </div>
  );
}

export default function Landscape({ atlas }: { atlas: Atlas }) {
  const [prompt, setPrompt] = usePrompt(atlas);
  const entry = atlas.lens[prompt.id];
  const L = atlas.meta.layers;
  const [layer, setLayer] = useState(Math.floor(L * 0.5));
  const [topK, setTopK] = useState(5);
  const [compareA, setCompareA] = useState(4);
  const [compareB, setCompareB] = useState(L);

  const churn = useMemo(() => {
    return Array.from({ length: L + 1 }, (_, l) => {
      if (l === 0) return { l, overlap: 1, kl: 0, newTop: 0 };
      const a = new Set(entry.top[l - 1].slice(0, 8).map((d) => d[0]));
      const b = entry.top[l].slice(0, 8).map((d) => d[0]);
      const overlap = b.filter((t) => a.has(t)).length / 8;
      return { l, overlap, kl: entry.kl[l] - entry.kl[l - 1], newTop: 8 - Math.round(overlap * 8) };
    });
  }, [entry, L]);

  const { ref, width } = useMeasure<HTMLDivElement>();
  const w = Math.max(width - 2, 340);
  const h = 180;
  const m = { t: 16, r: 16, b: 30, l: 46 };
  const xs = (l: number) => m.l + (l / L) * (w - m.l - m.r);
  const maxKl = Math.max(...churn.map((c) => Math.abs(c.kl)), 1e-3);
  const y0 = m.t + (h - m.t - m.b) / 2;
  const kls = (v: number) => y0 - (v / maxKl) * ((h - m.t - m.b) / 2);
  const ovs = (v: number) => m.t + (1 - v) * (h - m.t - m.b);

  const union = useMemo(() => {
    const s = new Map<string, { a: number; b: number }>();
    entry.top[compareA].forEach(([t, p]) => s.set(t, { a: p, b: 0 }));
    entry.top[compareB].forEach(([t, p]) => {
      const cur = s.get(t) ?? { a: 0, b: 0 };
      s.set(t, { ...cur, b: p });
    });
    return [...s.entries()].sort((x, y) => y[1].a + y[1].b - (x[1].a + x[1].b)).slice(0, 10);
  }, [entry, compareA, compareB]);

  const maxU = Math.max(...union.map(([, v]) => Math.max(v.a, v.b)), 0.01);

  return (
    <>
      <ChapterHead
        n="09"
        kicker="Probability landscape"
        title={<>The distribution is the <em>terrain</em></>}
        standfirst={
          <>
            The argmax is a single point on a surface with {atlas.meta.vocab.toLocaleString()} coordinates.
            Two layers can agree on the top token while their distributions are
            nearly disjoint. This chapter looks at the surface itself —{" "}
            <b>rank churn, KL between layers, and the shape of the tail</b>.
          </>
        }
        chips={
          <>
            <Chip tone="amber">vocab {atlas.meta.vocab.toLocaleString()}</Chip>
            <Chip>top-12 per layer</Chip>
            <Chip tone="cyan">KL L{compareA} → L{compareB}</Chip>
          </>
        }
      />

      <div className="row" style={{ alignItems: "flex-end", marginBottom: 6 }}>
        <label className="field">
          <span>Probe prompt</span>
          <select className="ctl" value={prompt.id} onChange={(e) => setPrompt(e.target.value)} style={{ minWidth: 300 }}>
            {atlas.prompts.map((p) => (
              <option key={p.id} value={p.id}>
                {p.id} — {p.text.slice(0, 48).replace(/\n/g, " ")}
              </option>
            ))}
          </select>
        </label>
        <div style={{ flex: 1 }} />
        <label className="field">
          <span>rows per layer · {topK}</span>
          <input type="range" min={2} max={8} value={topK} onChange={(e) => setTopK(+e.target.value)} style={{ width: 120 }} />
        </label>
      </div>

      <Panel
        title={<><b>Figure 9.1</b> · the rank river</>}
        tools={<Legend items={[{ c: "#a8431f", label: "lens top-1 path" }, { c: "#1f5c6b", label: "top-1 cell" }]} />}
        note="Each column is a layer's top-k distribution; rows are ranks. Lines connect a token to the same token in the next layer. When lines cross, the ordering is being rewritten."
      >
        <RankRiver entry={entry} L={L} topK={topK} layer={layer} onLayer={setLayer} />
      </Panel>

      <h2 className="sec">Two layers, side by side</h2>
      <Panel
        title={<><b>Figure 9.2</b> · distribution distance</>}
        tools={
          <div className="row" style={{ gap: 10 }}>
            <label className="field">
              <span>layer A · {compareA}</span>
              <input type="range" min={0} max={L} value={compareA} onChange={(e) => setCompareA(+e.target.value)} style={{ width: 110 }} />
            </label>
            <label className="field">
              <span>layer B · {compareB}</span>
              <input type="range" min={0} max={L} value={compareB} onChange={(e) => setCompareB(+e.target.value)} style={{ width: 110 }} />
            </label>
          </div>
        }
        note="Union of the two layers' top-10 tokens. Left bar = layer A, right bar = layer B. Tokens that appear in one and not the other are the ones the transformation invented or discarded."
      >
        <div className="stack" style={{ gap: 6 }}>
          {union.map(([tok, v]) => (
            <div key={tok} style={{ display: "grid", gridTemplateColumns: "52px 1fr 1px 1fr 52px", gap: 8, alignItems: "center" }}>
              <span style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--ink3)", textAlign: "right" }}>{pct(v.a, 1)}</span>
              <div style={{ display: "flex", justifyContent: "flex-end", height: 12, background: "rgba(23,24,28,0.02)" }}>
                <div style={{ width: `${(v.a / maxU) * 100}%`, background: "#1f5c6b", height: "100%" }} />
              </div>
              <div style={{ width: 1, background: "#b9b5aa", height: 16 }} />
              <div style={{ height: 12, background: "rgba(23,24,28,0.02)" }}>
                <div style={{ width: `${(v.b / maxU) * 100}%`, background: "#a8431f", height: "100%" }} />
              </div>
              <span style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--ink3)" }}>{pct(v.b, 1)}</span>
              <span />
              <span className="token" style={{ gridColumn: "2 / span 3", justifySelf: "center", fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--ink)" }}>
                {tokenLabel(tok) || "∅"}
              </span>
            </div>
          ))}
        </div>
      </Panel>

      <h2 className="sec">Where the surface changes</h2>
      <Panel
        title={<><b>Figure 9.3</b> · rank churn and KL between consecutive layers</>}
        note="Teal line: fraction of the top-8 that survives from the previous layer. Amber bars: KL divergence between consecutive lens distributions. A spike means a layer rewrote the ranking."
      >
        <div ref={ref}>
          <svg width={w} height={h} style={{ display: "block" }}>
            <line x1={m.l} x2={w - m.r} y1={y0} y2={y0} stroke="#b9b5aa" />
            {churn.map((c) => (
              <rect
                key={c.l}
                x={xs(c.l) - 5}
                y={c.kl >= 0 ? kls(c.kl) : y0}
                width={10}
                height={Math.abs(kls(c.kl) - y0)}
                fill="rgba(168,67,31,0.65)"
              >
                <title>{`L${c.l}: KL ${c.kl.toFixed(3)}, overlap ${(c.overlap * 100).toFixed(0)}%`}</title>
              </rect>
            ))}
            <path d={churn.map((c, i) => `${i === 0 ? "M" : "L"}${xs(c.l)},${ovs(c.overlap)}`).join(" ")} fill="none" stroke="#1f5c6b" strokeWidth={1.8} />
            {[0, 6, 12, 18, L].map((l) => (
              <text key={l} x={xs(l)} y={h - 12} textAnchor="middle" className="ticklabel">{l}</text>
            ))}
            <text x={m.l - 6} y={12} className="ticklabel">KL</text>
            <text x={w - m.r} y={12} textAnchor="end" className="ticklabel" style={{ fill: "#1f5c6b" }}>top-8 overlap</text>
          </svg>
        </div>
        <div className="grid4" style={{ marginTop: 8 }}>
          <Stat k="Largest KL step" v={`L${churn.reduce((a, b) => (Math.abs(b.kl) > Math.abs(a.kl) ? b : a)).l}`} s={fmt(Math.max(...churn.map((c) => Math.abs(c.kl))), 2)} />
          <Stat k="Most stable step" v={`L${churn.slice(1).reduce((a, b) => (b.overlap > a.overlap ? b : a)).l}`} s={`${pct(Math.max(...churn.map((c) => c.overlap)), 0)} overlap`} />
          <Stat k="Total flips" v={churn.reduce((a, b) => a + b.newTop, 0)} s="tokens entering top-8" />
          <Stat k="Final entropy" v={fmt(entry.entropy[L], 2)} s={`nats (max ${Math.log(atlas.meta.vocab).toFixed(1)})`} />
        </div>
      </Panel>

      <Prose>
        <p>
          The lens distributions are not a smooth sequence of improvements. They are
          a sequence of <strong>regimes</strong>: a few layers preserve the top-8
          almost exactly (overlap 1.0, KL near zero), and the rest replace most of
          it. That is what a residual network looks like when it routes information
          conditionally — nothing happens for several layers, then a block reads the
          accumulated state and rewrites the hypothesis.
        </p>
        <p>
          Note how large the final entropy is: {fmt(entry.entropy[L], 2)} nats over{" "}
          {atlas.meta.vocab.toLocaleString()} vocabulary items. Even at the last
          layer the model is not choosing from a shortlist — it is choosing from a
          distribution with enormous support, and the apparent confidence in the
          top-1 token is a property of the tail being spread very thin.
        </p>
      </Prose>

      <Method>
        <p>
          KL between consecutive layers is{" "}
          <code>KL(P_l ‖ P_(l−1))</code>, computed on the full vocabulary
          distribution, not on the top-12 — the stored top-12 is only for display.
          A positive bar means layer l moved mass; the direction of movement is not
          shown by a scalar.
        </p>
        <p>
          Rank churn is computed on the top-8 sets. It is a set overlap, not a rank
          correlation: a token can stay in the top-8 while moving from rank 1 to
          rank 7.
        </p>
      </Method>

      <Footnotes
        items={[
          "The lens distribution is not the model's final distribution at intermediate layers; it is a projection. KL between two projections is a measure of change in the projection, not in the model's beliefs.",
          "Top-12 is a display window. Everything in Figure 9.3 is computed on all 151,936 logits.",
          `For this prompt, entropy falls from ${fmt(entry.entropy[0], 1)} nats at L0 to ${fmt(entry.entropy[L], 1)} at L${L}.`,
        ]}
      />
      <PageFoot index={10} />
    </>
  );
}
