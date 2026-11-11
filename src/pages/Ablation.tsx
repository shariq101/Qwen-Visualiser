import React, { useMemo, useState } from "react";
import { Atlas, fmt, useMeasure } from "../lib/data";
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

const KIND_COLOR: Record<string, string> = { attn: "#1f5c6b", mlp: "#6b3f6e", block: "#a8431f" };

export default function AblationPage({ atlas }: { atlas: Atlas }) {
  const { ablation } = atlas;
  const L = atlas.meta.layers;
  const [kind, setKind] = useState<"attn" | "mlp" | "block">("block");
  const [layer, setLayer] = useState(12);
  const { ref, width } = useMeasure<HTMLDivElement>();

  const series = ablation[kind];
  const vals = Array.from({ length: L }, (_, l) => series[String(l)]?.delta ?? 0);
  const maxV = Math.max(...vals, 0.1);
  const w = Math.max(width - 2, 340);
  const h = 230;
  const m = { t: 18, r: 14, b: 34, l: 54 };
  const xs = (l: number) => m.l + (l / (L - 1 || 1)) * (w - m.l - m.r);
  const ys = (v: number) => h - m.b - (v / maxV) * (h - m.t - m.b);

  const perPrompt = series[String(layer)]?.perPrompt ?? [];
  const promMax = Math.max(...perPrompt.map(Math.abs), 0.01);

  const attnMean = Array.from({ length: L }, (_, l) => ablation.attn[String(l)].delta).reduce((a, b) => a + b, 0) / L;
  const mlpMean = Array.from({ length: L }, (_, l) => ablation.mlp[String(l)].delta).reduce((a, b) => a + b, 0) / L;
  const blockMean = Array.from({ length: L }, (_, l) => ablation.block[String(l)].delta).reduce((a, b) => a + b, 0) / L;
  const worst = vals.indexOf(Math.max(...vals));
  const mildest = vals.indexOf(Math.min(...vals));

  const redundancy = useMemo(() => {
    return Array.from({ length: L }, (_, l) => {
      const a = ablation.attn[String(l)].delta;
      const m = ablation.mlp[String(l)].delta;
      const b = ablation.block[String(l)].delta;
      return { l, a, m, b, alone: a + m, ratio: (a + m) / Math.max(b, 1e-6) };
    });
  }, [ablation, L]);

  return (
    <>
      <ChapterHead
        n="08"
        kicker="Ablation lab"
        title={<>Take a layer out. <em>Measure the damage.</em></>}
        standfirst={
          <>
            Everything else in this atlas is a measurement of what a layer writes.
            This chapter is the only one that measures what the rest of the network{" "}
            <b>needs</b>. Zero a module's output at every position, re-run the same
            prompts, and record how much the next-token loss moves. No surrogate, no
            probe — just the loss.
          </>
        }
        chips={
          <>
            <Chip tone="amber">baseline NLL {fmt(ablation.baseline, 3)}</Chip>
            <Chip>{L * 3} knockout conditions</Chip>
            <Chip tone="cyan">{ablation.prompts.length} prompts</Chip>
          </>
        }
      />

      <div className="row" style={{ alignItems: "flex-end", marginBottom: 6 }}>
        <label className="field">
          <span>condition</span>
          <Seg
            value={kind}
            options={[
              { v: "attn", label: "attention" },
              { v: "mlp", label: "MLP" },
              { v: "block", label: "whole block" },
            ]}
            onChange={setKind}
          />
        </label>
        <div style={{ flex: 1 }} />
        <span className="chip">ΔNLL, nats · positive = damage</span>
      </div>

      <Panel
        title={<><b>Figure 8.1</b> · damage by layer</>}
        tools={<Legend items={[{ c: KIND_COLOR[kind], label: `${kind} knockout` }]} />}
        note="Each bar is the mean increase in next-token negative log-likelihood when that module is zeroed across all positions. Click a bar to see which prompts suffer."
      >
        <div ref={ref} style={{ overflowX: "auto" }}>
          <svg width={w} height={h} style={{ display: "block" }}>
            <line x1={m.l} x2={w - m.r} y1={ys(0)} y2={ys(0)} stroke="#b9b5aa" />
            {[0.25, 0.5, 0.75, 1].map((f) => (
              <g key={f}>
                <line x1={m.l} x2={w - m.r} y1={ys(maxV * f)} y2={ys(maxV * f)} className="gridline" />
                <text x={m.l - 6} y={ys(maxV * f) + 3} textAnchor="end" className="ticklabel">
                  {(maxV * f).toFixed(1)}
                </text>
              </g>
            ))}
            {vals.map((v, l) => (
              <g key={l} onClick={() => setLayer(l)} style={{ cursor: "pointer" }}>
                <rect
                  x={xs(l) - (w - m.l - m.r) / L / 2 + 1.5}
                  y={ys(Math.max(v, 0))}
                  width={(w - m.l - m.r) / L - 3}
                  height={Math.abs(ys(0) - ys(Math.max(v, 0)))}
                  fill={l === layer ? "#c9704c" : KIND_COLOR[kind]}
                  opacity={l === layer ? 1 : 0.72}
                />
                <text x={xs(l)} y={h - 14} textAnchor="middle" className="ticklabel" style={{ fill: l === layer ? "var(--amber)" : undefined }}>
                  {l}
                </text>
              </g>
            ))}
            <text x={m.l - 6} y={12} className="ticklabel">ΔNLL</text>
          </svg>
        </div>
      </Panel>

      <div className="grid4">
        <Stat k="Worst layer" v={`L${worst}`} s={`+${fmt(vals[worst], 2)} nats`} />
        <Stat k="Mildest layer" v={`L${mildest}`} s={`+${fmt(vals[mildest], 2)} nats`} />
        <Stat k="Selected" v={`L${layer}`} s={`ΔNLL ${fmt(series[String(layer)]?.delta ?? 0, 2)}`} />
        <Stat k="Mean damage" v={fmt(kind === "attn" ? attnMean : kind === "mlp" ? mlpMean : blockMean, 2)} s={`${kind}, all layers`} />
      </div>

      <h2 className="sec">Per-prompt breakdown</h2>
      <Panel
        title={<><b>Figure 8.2</b> · who feels L{layer} disappear</>}
        note="ΔNLL for each prompt in the ablation subset. A layer can be essential for arithmetic and irrelevant for idiom; the mean hides that."
      >
        <div className="stack" style={{ gap: 3 }}>
          {ablation.prompts.map((p, i) => {
            const v = perPrompt[i] ?? 0;
            return (
              <div key={p} style={{ display: "grid", gridTemplateColumns: "170px 1fr 56px", gap: 8, alignItems: "center" }}>
                <span style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--ink2)" }}>{p}</span>
                <div style={{ height: 11, background: "rgba(23,24,28,0.03)", position: "relative" }}>
                  <div
                    style={{
                      position: "absolute",
                      left: v >= 0 ? "50%" : undefined,
                      right: v < 0 ? "50%" : undefined,
                      width: `${(Math.abs(v) / promMax) * 50}%`,
                      height: "100%",
                      background: v >= 0 ? KIND_COLOR[kind] : "#2b5f8a",
                    }}
                  />
                  <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "#b9b5aa" }} />
                </div>
                <span style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: v > 0 ? "var(--ink2)" : "var(--blue)", textAlign: "right" }}>
                  {v > 0 ? "+" : ""}{fmt(v, 2)}
                </span>
              </div>
            );
          })}
        </div>
      </Panel>

      <h2 className="sec">The redundancy result</h2>
      <Panel
        title={<><b>Figure 8.3</b> · knocking out attention + MLP ≠ knocking out the block</>}
        note="Sum of the two component knockouts (teal) against the whole-block knockout (amber). If the network were a chain of independent modules, the two would be equal."
      >
        <div className="stack" style={{ gap: 3 }}>
          {redundancy.map((r) => (
            <div key={r.l} style={{ display: "grid", gridTemplateColumns: "26px 1fr 1fr 52px", gap: 6, alignItems: "center" }}>
              <span className="ticklabel">{r.l}</span>
              <div style={{ height: 9, background: "rgba(23,24,28,0.03)" }}>
                <div style={{ width: `${Math.min(1, r.alone / Math.max(...redundancy.map((x) => x.b))) * 100}%`, height: "100%", background: "#1f5c6b" }} title={`attn+mlp ${fmt(r.alone, 2)}`} />
              </div>
              <div style={{ height: 9, background: "rgba(23,24,28,0.03)" }}>
                <div style={{ width: `${Math.min(1, r.b / Math.max(...redundancy.map((x) => x.b))) * 100}%`, height: "100%", background: "#a8431f" }} title={`block ${fmt(r.b, 2)}`} />
              </div>
              <span className="ticklabel">{fmt(r.ratio, 2)}×</span>
            </div>
          ))}
        </div>
        <div className="spread" style={{ marginTop: 8 }}>
          <span className="ticklabel">teal = attn + mlp · amber = whole block · right = ratio</span>
          <span className="ticklabel">tokens of bars → nats</span>
        </div>
      </Panel>

      <Prose>
        <p>
          This is the most important negative result in the atlas. Zeroing a whole
          block costs on the order of <strong>{fmt(blockMean, 1)} nats</strong>,
          but zeroing its attention and MLP separately costs about{" "}
          <strong>{fmt((attnMean + mlpMean) / 2, 2)} nats</strong> each. The two
          components are not additive: removing either one alone barely matters,
          because the residual stream carries a copy of the information around the
          missing module. The block knockout also removes the residual bypass — it
          deletes the layer's contribution to the stream entirely, which is a much
          harsher intervention.
        </p>
        <p>
          So "layer L is important" is not a single fact. It depends on whether you
          mean <em>the computation this layer performs</em> (small effect when
          removed) or <em>the vector this layer writes into the stream</em> (large
          effect when removed). Both are true simultaneously, and a report that
          only measures one of them is describing a different model.
        </p>
      </Prose>

      <Method>
        <p>
          A forward hook replaces a module's output with zeros. For attention, the
          output tuple's first element is zeroed; for MLP and for the whole block,
          the returned tensor is zeroed. The residual connection is <em>not</em>{" "}
          touched in the component conditions — <code>h + attn(ln h)</code> becomes{" "}
          <code>h + 0</code> — but the block condition zeroes the block's total
          output, which includes the residual add. That asymmetry is deliberate and
          is the reason Figure 8.3 exists.
        </p>
        <p>
          The prompts are tokenised as one right-padded batch. Loss is read at each
          row's last real token, using the row's own attention mask, so padding
          cannot leak into the measurement.
        </p>
      </Method>

      <Footnotes
        items={[
          `Baseline NLL on this 14-prompt subset is ${fmt(ablation.baseline, 4)} nats. Delta values are per-prompt means, so a single catastrophic prompt can dominate a layer.`,
          "Ablation is a blunt instrument: it removes a module everywhere, not just where it matters. Position-specific ablation would be a stronger test and is not measured here.",
          "The block knockout also changes the residual stream's scale, which interacts with the final RMSNorm. Part of the large block damage is therefore a scale effect, not lost computation.",
        ]}
      />
      <PageFoot index={9} />
    </>
  );
}
