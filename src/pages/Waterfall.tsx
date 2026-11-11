import React, { useMemo, useState } from "react";
import { Atlas, fmt, pct, useMeasure, usePrompt } from "../lib/data";
import {
  Axis,
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

type Step = { label: string; v: number; kind: "base" | "attn" | "mlp"; layer: number };

export default function Waterfall({ atlas }: { atlas: Atlas }) {
  const [prompt, setPrompt] = usePrompt(atlas);
  const L = atlas.meta.layers;
  const d = atlas.dla[prompt.id];
  const [mode, setMode] = useState<"all" | "attn" | "mlp">("all");
  const { ref, width } = useMeasure<HTMLDivElement>();

  const steps = useMemo(() => {
    const s: Step[] = [{ label: "embed", v: d.resid[0], kind: "base", layer: -1 }];
    for (let l = 0; l < L; l++) {
      if (mode !== "mlp") s.push({ label: `L${l}`, v: d.attn[l], kind: "attn", layer: l });
      if (mode !== "attn") s.push({ label: `L${l}`, v: d.mlp[l], kind: "mlp", layer: l });
    }
    return s;
  }, [d, L, mode]);

  const cum = useMemo(() => {
    let run = 0;
    return steps.map((s) => (run += s.v));
  }, [steps]);

  const W = Math.max(width - 2, 300);
  const h = 300;
  const m = { t: 18, r: 14, b: 40, l: 56 };
  const lo = Math.min(0, ...cum);
  const hi = Math.max(0, ...cum);
  const pad = (hi - lo) * 0.08 || 1;
  const y = (v: number) => m.t + (1 - (v - (lo - pad)) / (hi - lo + 2 * pad)) * (h - m.t - m.b);
  const bw = (W - m.l - m.r) / steps.length;
  const zero = y(0);
  const finalVal = d.resid[L];

  const totals = {
    attn: d.attn.reduce((a, b) => a + b, 0),
    mlp: d.mlp.reduce((a, b) => a + b, 0),
  };
  const ranked = [...steps.map((s, i) => ({ ...s, v: s.v, i }))].filter((s) => s.kind !== "base").sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const best = ranked[0];

  return (
    <>
      <ChapterHead
        n="11"
        kicker="Logit waterfall"
        title={<>Every block, <em>signed and summed</em></>}
        standfirst={
          <>
            Direct logit attribution is additive: each block writes a vector into the
            residual stream, and its projection onto the target token's unembedding
            direction is a signed number of logits. Stack those numbers in order and
            you get a waterfall from the embedding to the final logit —{" "}
            <b>the accounting of how the answer was paid for</b>.
          </>
        }
        chips={
          <>
            <Chip tone="amber">target “{d.targetStr.trim()}”</Chip>
            <Chip>{L} blocks × 2 components</Chip>
            <Chip tone="cyan">final {fmt(finalVal, 2)} logits</Chip>
          </>
        }
      />

      <div className="row" style={{ alignItems: "flex-end", marginBottom: 4 }}>
        <label className="field">
          <span>Probe prompt</span>
          <select className="ctl" value={prompt.id} onChange={(e) => setPrompt(e.target.value)}>
            {atlas.prompts.map((p) => (
              <option key={p.id} value={p.id}>
                {p.id} — {p.text.slice(0, 50).replace(/\n/g, " ")}
              </option>
            ))}
          </select>
        </label>
        <div style={{ flex: 1 }} />
        <Seg
          value={mode}
          options={[
            { v: "all", label: "both" },
            { v: "attn", label: "attention" },
            { v: "mlp", label: "MLP" },
          ]}
          onChange={setMode}
        />
      </div>

      <Panel
        title={<><b>Figure 11.1</b> · signed contribution, accumulated</>}
        tools={<Legend items={[{ c: "var(--ink-3)", label: "embedding" }, { c: "var(--teal)", label: "attention" }, { c: "var(--rust)", label: "MLP" }]} />}
        note="Each bar starts where the previous one ended. Upward bars push the target token's logit higher; downward bars push it lower. The dashed line is the final residual attribution, which should equal the sum of all steps."
      >
        <div ref={ref} style={{ overflowX: "auto" }}>
          <svg width={W} height={h} style={{ display: "block" }}>
            <Axis
              x={(v) => m.l + (v / Math.max(steps.length - 1, 1)) * (W - m.l - m.r)}
              y={y}
              w={W - m.r}
              h={h - m.b}
              xTicks={[0, Math.round(steps.length * 0.25), Math.round(steps.length * 0.5), Math.round(steps.length * 0.75), steps.length - 1]}
              yTicks={[lo, lo / 2, 0, hi / 2, hi].filter((v, i, a) => a.indexOf(v) === i)}
              yFormat={(v) => v.toFixed(1)}
              xLabel="step (layers left to right)"
              yLabel="cumulative logit attribution"
            />
            <line x1={m.l} x2={W - m.r} y1={zero} y2={zero} stroke="var(--ink)" strokeWidth={0.8} />
            <line x1={m.l} x2={W - m.r} y1={y(finalVal)} y2={y(finalVal)} stroke="var(--ink-3)" strokeDasharray="3 3" />
            <text x={W - m.r} y={y(finalVal) - 4} textAnchor="end" className="ticklabel">final {fmt(finalVal, 2)}</text>
            {steps.map((s, i) => {
              const start = i === 0 ? 0 : cum[i - 1];
              const end = cum[i];
              const top = Math.min(y(start), y(end));
              const height = Math.max(1, Math.abs(y(end) - y(start)));
              const fill = s.kind === "base" ? "var(--ink-3)" : s.kind === "attn" ? "var(--teal)" : "var(--rust)";
              return (
                <rect
                  key={i}
                  x={m.l + i * bw + bw * 0.15}
                  y={top}
                  width={Math.max(bw * 0.7, 1)}
                  height={height}
                  fill={fill}
                  opacity={Math.abs(s.v) < 1e-4 ? 0.25 : 1}
                >
                  <title>{`${s.kind} ${s.layer >= 0 ? "L" + s.layer : "embed"} · ${fmt(s.v, 3)} logits · running ${fmt(end, 3)}`}</title>
                </rect>
              );
            })}
            {[0, 8, 16, 24, 32].filter((l) => l < L).map((l) => {
              const idx = steps.findIndex((s) => s.layer === l);
              if (idx < 0) return null;
              return (
                <text key={l} x={m.l + idx * bw} y={h - m.b + 12} className="ticklabel" textAnchor="middle">
                  L{l}
                </text>
              );
            })}
          </svg>
        </div>
      </Panel>

      <div className="grid4">
        <Stat k="Embedding" v={fmt(d.resid[0], 2)} s="position 0 of the waterfall" />
        <Stat k="Attention total" v={(totals.attn >= 0 ? "+" : "") + fmt(totals.attn, 2)} s={`${d.attn.filter((v) => v > 0).length} of ${L} positive`} />
        <Stat k="MLP total" v={(totals.mlp >= 0 ? "+" : "") + fmt(totals.mlp, 2)} s={`${d.mlp.filter((v) => v > 0).length} of ${L} positive`} />
        <Stat k="Largest single step" v={best ? `${best.kind === "attn" ? "attn" : "mlp"} L${best.layer}` : "—"} s={best ? `${fmt(best.v, 2)} logits` : ""} />
      </div>

      <h2 className="sec">The ten largest contributions</h2>
      <Panel
        title={<><b>Figure 11.2</b> · ranked contributions</>}
        note="Signed, in logit units. A block can be a large contributor to this particular answer without being important to the model in general; chapter 08 is the causal counterpart."
        tight
      >
        <div className="scroll">
          <table className="tbl">
            <thead>
              <tr>
                <th>rank</th>
                <th>component</th>
                <th>layer</th>
                <th>contribution</th>
                <th>running total</th>
                <th>share of |total|</th>
              </tr>
            </thead>
            <tbody>
              {ranked.slice(0, 10).map((s, i) => {
                const denom = Math.abs(totals.attn) + Math.abs(totals.mlp) || 1;
                return (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td className="k">{s.kind}</td>
                    <td>{s.layer}</td>
                    <td style={{ color: s.v >= 0 ? "var(--rust)" : "var(--teal)" }}>{fmt(s.v, 3)}</td>
                    <td>{fmt(cum[s.i], 2)}</td>
                    <td>{pct(Math.abs(s.v) / denom, 1)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>

      <Prose>
        <p>
          The waterfall makes a point that averages hide: the final logit is a{" "}
          <strong>small difference between large numbers</strong>. Attention and MLP
          contributions at this prompt sum to {fmt(totals.attn, 1)} and{" "}
          {fmt(totals.mlp, 1)} logits respectively, while the answer itself sits at{" "}
          {fmt(finalVal, 1)}. The model is not accumulating evidence monotonically;
          it is balancing opposing pushes.
        </p>
        <p>
          It also shows why "which layer decides" has no single answer. The largest
          single step is {best ? <><strong>{best.kind}</strong> at <strong>L{best.layer}</strong> ({fmt(best.v, 2)} logits)</> : "—"}, but
          the final decision depends on where the running total crosses its
          neighbours, not on any one block.
        </p>
      </Prose>

      <Method>
        <p>
          Each component's contribution is{" "}
          <code>⟨output, γ ⊙ W_U[target]⟩ / rms(h_final)</code>: the component's
          output projected onto the target token's unembedding direction, scaled by
          the final RMSNorm. Adding the embedding term and every block's two
          components reproduces the final residual attribution up to the
          first-order norm approximation.
        </p>
        <p>
          The running total is therefore an accounting identity, not a model of
          causation. Chapter 08 perturbs the model to test which of these terms the
          network actually depends on.
        </p>
      </Method>

      <Footnotes
        items={[
          "DLA ignores the derivative of the final RMSNorm with respect to each component, which is small but non-zero; the sum of steps can differ from the measured final residual attribution by a few percent.",
          "Contributions are to one token's logit. The same block may contribute in the opposite direction to the runner-up token, which is often what decides the argmax.",
          "Switching to 'attention' or 'MLP' only hides the other series; the running total is recomputed for the visible steps, so the final value changes.",
        ]}
      />
      <PageFoot index={12} />
    </>
  );
}
