import React, { useMemo, useState } from "react";
import { Atlas, INK, SEQ, pct, ramp, useMeasure, usePrompt, tokenLabel } from "../lib/data";
import {
  Axis,
  ChapterHead,
  Chip,
  Colorbar,
  Footnotes,
  Legend,
  Method,
  PageFoot,
  Panel,
  Prose,
  Stat,
} from "../components/ui";

const SERIES = [INK.rust, INK.teal, INK.indigo, INK.olive, INK.plum];

export default function Flow({ atlas }: { atlas: Atlas }) {
  const [prompt, setPrompt] = usePrompt(atlas);
  const L = atlas.meta.layers;
  const entry = atlas.attention[prompt.id];
  const T = entry.tokens.length;
  const [layer, setLayer] = useState(L - 2);
  const { ref, width } = useMeasure<HTMLDivElement>();

  const mean = entry.mean; // [layer][key position], head-averaged
  const maxAttn = Math.max(...mean.flat());
  const W = Math.max(width - 2, 300);
  const left = 44;
  const cw = (W - left - 8) / T;
  const rh = 11;

  const profileLayers = useMemo(
    () => [1, Math.round(L * 0.35), Math.round(L * 0.6), Math.round(L * 0.85), L - 1],
    [L],
  );

  const budget = useMemo(() => {
    return Array.from({ length: L }, (_, l) => {
      const row = mean[l];
      const self = row[T - 1] ?? 0;
      const first = row[0] ?? 0;
      const prev = T > 2 ? row[T - 2] ?? 0 : 0;
      return { l, self, first, prev, rest: Math.max(0, 1 - self - first - prev) };
    });
  }, [mean, L, T]);

  const sinkLayer = budget.reduce((a, b) => (b.first > a.first ? b : a));
  const selfLayer = budget.reduce((a, b) => (b.self > a.self ? b : a));
  const selected = budget[layer] ?? budget[0];

  const h = profileLayers.length ? 220 : 0;
  const prof = { m: { t: 14, r: 12, b: 34, l: 46 } };
  const maxProfile = Math.max(
    ...profileLayers.map((l) =>
      Math.max(...Array.from({ length: T }, (_, k) => mean[l][k] ?? 0)),
    ),
  );
  const px = (d: number) => prof.m.l + (d / Math.max(T - 1, 1)) * (W - prof.m.l - prof.m.r);
  const py = (v: number) => prof.m.t + (1 - v / (maxProfile || 1)) * (h - prof.m.t - prof.m.b);

  return (
    <>
      <ChapterHead
        n="10"
        kicker="Attention budget"
        title={<>Where the <em>mass</em> goes</>}
        standfirst={
          <>
            A softmax row is a budget: every query distributes exactly one unit of
            attention across the keys it can see. This chapter measures that budget
            directly — per layer, per key position, and against{" "}
            <b>distance</b> — for the final query position, which is the one that
            produces the next-token prediction.
          </>
        }
        chips={
          <>
            <Chip tone="amber">{prompt.id}</Chip>
            <Chip>{T} keys · {L} layers</Chip>
            <Chip tone="cyan">head-averaged</Chip>
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
      </div>

      <Panel
        title={<><b>Figure 10.1</b> · attention mass, layer × key position</>}
        tools={<Colorbar stops={SEQ} min={0} max={maxAttn} label="mean head attention" width={120} format={(v) => v.toFixed(2)} />}
        note="Rows are layers, columns are key positions. Each row is a distribution over the same T keys and sums to 1; the dark band on the right is attention to the final position itself, and the left edge shows the position-0 sink."
      >
        <div ref={ref} style={{ overflowX: "auto" }}>
          <svg width={W} height={22 + L * rh} style={{ display: "block" }}>
            {entry.tokens.map((t, k) => (
              <text key={k} x={left + k * cw + cw / 2} y={10} textAnchor="middle" className="ticklabel">
                {k % Math.ceil(T / 14) === 0 ? tokenLabel(t).slice(0, 5) || "∅" : ""}
              </text>
            ))}
            {Array.from({ length: L }, (_, l) => (
              <g key={l} onClick={() => setLayer(l)} style={{ cursor: "pointer" }}>
                <text x={left - 6} y={22 + l * rh + 8} textAnchor="end" className="ticklabel">
                  {l % 4 === 0 ? l : ""}
                </text>
                {Array.from({ length: T }, (_, k) => (
                  <rect
                    key={k}
                    x={left + k * cw}
                    y={22 + l * rh}
                    width={Math.max(cw - 0.6, 0.6)}
                    height={rh - 0.6}
                    fill={ramp((mean[l][k] ?? 0) / (maxAttn || 1), SEQ)}
                  >
                    <title>{`L${l} · key ${k} “${tokenLabel(entry.tokens[k])}” · ${pct(mean[l][k] ?? 0, 2)}`}</title>
                  </rect>
                ))}
                {l === layer && (
                  <rect x={left - 1} y={22 + l * rh - 1} width={W - left} height={rh} fill="none" stroke="var(--rust)" />
                )}
              </g>
            ))}
            <text x={left - 6} y={16} textAnchor="end" className="ticklabel">layer</text>
          </svg>
        </div>
      </Panel>

      <h2 className="sec">Against distance</h2>
      <Panel
        title={<><b>Figure 10.2</b> · attention versus distance from the query</>}
        tools={<Legend items={profileLayers.map((l, i) => ({ c: SERIES[i % SERIES.length], label: `L${l}`, line: true }))} />}
        note="Distance 0 is the query itself; distance 1 is the previous token. Plotted for five layers spanning the depth. A curve that stays high at small distance is positional; a curve that decays slowly is content-based retrieval over a long range."
      >
        <svg width={W} height={h} style={{ display: "block" }}>
          <Axis
            x={px}
            y={py}
            w={W - prof.m.r}
            h={h - prof.m.b}
            xTicks={[0, Math.round((T - 1) / 4), Math.round((T - 1) / 2), Math.round((3 * (T - 1)) / 4), T - 1]}
            yTicks={[0, maxProfile / 2, maxProfile]}
            yFormat={(v) => v.toFixed(2)}
            xLabel="distance from final query (tokens)"
            yLabel="mean attention"
          />
          {profileLayers.map((l, i) => (
            <path
              key={l}
              d={Array.from({ length: T }, (_, d) => {
                const k = T - 1 - d;
                return `${d === 0 ? "M" : "L"}${px(d)},${py(mean[l][k] ?? 0)}`;
              }).join(" ")}
              fill="none"
              stroke={SERIES[i % SERIES.length]}
              strokeWidth={1.4}
            />
          ))}
        </svg>
      </Panel>

      <div className="grid3">
        <Stat k="Selected layer" v={`L${layer}`} s={`self ${pct(selected.self, 1)} · prev ${pct(selected.prev, 1)}`} />
        <Stat k="Largest position-0 sink" v={`L${sinkLayer.l}`} s={`${pct(sinkLayer.first, 1)} of the budget`} />
        <Stat k="Largest self-attention" v={`L${selfLayer.l}`} s={`${pct(selfLayer.self, 1)} of the budget`} />
      </div>

      <h2 className="sec">The budget, decomposed</h2>
      <Panel
        title={<><b>Figure 10.3</b> · how each layer spends its attention</>}
        tools={<Legend items={[{ c: "var(--ink)", label: "self" }, { c: "var(--sand)", label: "previous" }, { c: "var(--teal)", label: "first" }, { c: "var(--rule-2)", label: "rest" }]} />}
        note="Head-averaged share of the final query's attention. Categories are mutually exclusive; 'rest' is everything else. Hover any bar for the underlying percentages."
      >
        <div className="stack" style={{ gap: 2 }}>
          {budget.map((b) => (
            <div key={b.l} style={{ display: "grid", gridTemplateColumns: "24px 1fr 74px", gap: 8, alignItems: "center" }}>
              <span className="ticklabel" style={{ textAlign: "right" }}>{b.l}</span>
              <div
                style={{ display: "flex", height: 9, border: b.l === layer ? "1px solid var(--rust)" : "none" }}
                onClick={() => setLayer(b.l)}
                title={`L${b.l} — self ${pct(b.self, 1)}, prev ${pct(b.prev, 1)}, first ${pct(b.first, 1)}, rest ${pct(b.rest, 1)}`}
              >
                <div style={{ width: `${b.self * 100}%`, background: "var(--ink)" }} />
                <div style={{ width: `${b.prev * 100}%`, background: "var(--sand)" }} />
                <div style={{ width: `${b.first * 100}%`, background: "var(--teal)" }} />
                <div style={{ width: `${b.rest * 100}%`, background: "var(--rule-2)" }} />
              </div>
              <span className="ticklabel">
                {pct(b.self + b.prev, 0)} local
              </span>
            </div>
          ))}
        </div>
      </Panel>

      <Prose>
        <p>
          The budget is not stationary with depth. Early layers spend most of it
          locally — self and the immediately previous token — which is what you would
          expect from a stage that is still resolving token identity. Later layers
          spread mass further back, and the position-0 sink becomes prominent in a
          minority of layers, where it functions as a way to write almost nothing.
        </p>
        <p>
          At the selected layer <strong>L{layer}</strong>, {pct(selected.self, 1)} of
          the final query's attention goes to itself, {pct(selected.prev, 1)} to the
          previous token, {pct(selected.first, 1)} to position 0, and{" "}
          {pct(selected.rest, 1)} to everything else. The layer with the largest sink
          is <strong>L{sinkLayer.l}</strong>; the largest self-attention is{" "}
          <strong>L{selfLayer.l}</strong>.
        </p>
      </Prose>

      <Method>
        <p>
          For each layer and head, the softmax row of the final query token is
          captured during the forward pass. Figure 10.1 and 10.3 average those rows
          over the {atlas.meta.heads} heads; Figure 10.2 plots the same quantity as a
          function of <code>distance = (T − 1) − key_index</code>, so distance 0 is
          the query itself.
        </p>
        <p>
          Averaging over heads first is a deliberate simplification: it hides head
          specialisation (chapter 03) in exchange for a stable picture of where the
          layer as a whole is reading from.
        </p>
      </Method>

      <Footnotes
        items={[
          "Only the final query position is captured. Attention rows for earlier positions exist in the attention tensors but are not exported, to keep the artefact small.",
          "Attention weights are not contribution weights. A head can attend strongly to a position whose value vector is nearly orthogonal to the output direction.",
          "Sink, self, previous and rest are computed from mutually exclusive key positions, except when the prompt is shorter than three tokens.",
        ]}
      />
      <PageFoot index={11} />
    </>
  );
}
