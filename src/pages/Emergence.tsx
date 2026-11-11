import React, { useMemo, useState } from "react";
import { Atlas, fmt, pct, useMeasure } from "../lib/data";
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

const COLORS = ["#a8431f", "#1f5c6b", "#6b3f6e", "#43478c", "#2f6b3f", "#2b5f8a", "#a8431f"];

export default function Emergence({ atlas }: { atlas: Atlas }) {
  const { probes } = atlas;
  const L = atlas.meta.layers;
  const { ref, width } = useMeasure<HTMLDivElement>();
  const [focus, setFocus] = useState<string>(probes.properties[0].key);
  const [showBands, setShowBands] = useState(true);

  const w = Math.max(width - 2, 340);
  const h = 330;
  const m = { t: 18, r: 130, b: 36, l: 46 };
  const xs = (l: number) => m.l + (l / L) * (w - m.l - m.r);
  const ys = (v: number) => h - m.b - v * (h - m.t - m.b);

  const curves = probes.properties.map((p, i) => {
    const acc = probes.byProp[p.key] ?? [];
    const std = probes.std[p.key] ?? [];
    return { ...p, acc, std, color: COLORS[i % COLORS.length] };
  });

  const crossings = useMemo(() => {
    return curves.map((c) => {
      let first = -1;
      for (let l = 0; l < c.acc.length; l++) {
        if (c.acc[l] > c.chance + 0.08) {
          first = l;
          break;
        }
      }
      return { key: c.key, label: c.label, first, peak: Math.max(...c.acc), final: c.acc[c.acc.length - 1] };
    });
  }, [curves]);

  const focused = curves.find((c) => c.key === focus) ?? curves[0];
  const rel = curves.find((c) => c.key === "next_repeat") ?? curves[0];
  const relPeak = Math.max(...rel.acc);
  const relPeakLayer = rel.acc.indexOf(relPeak);

  return (
    <>
      <ChapterHead
        n="06"
        kicker="Capability emergence"
        title={<>When does a property become <em>readable</em>?</>}
        standfirst={
          <>
            A linear probe is a deliberately dumb instrument: one hyperplane, fit by
            logistic regression, asked to separate a property from the residual
            stream at each layer. If the probe can read a property at layer{" "}
            <em>k</em>, that property is <b>linearly present</b> there — and the
            layer at which it appears is a fact about the model, not about the probe.
            Accuracy is <b>balanced</b>, so a rare property cannot be won by
            predicting the majority class.
          </>
        }
        chips={
          <>
            <Chip tone="amber">{probes.properties.length} properties</Chip>
            <Chip>{L + 1} layers probed</Chip>
            <Chip tone="cyan">GroupKFold(5) by prompt</Chip>
          </>
        }
      />

      <Panel
        title={<><b>Figure 6.1</b> · probe accuracy by depth</>}
        tools={
          <div className="legend">
            {curves.map((c) => (
              <button
                key={c.key}
                onClick={() => setFocus(c.key)}
                style={{
                  background: "none",
                  border: 0,
                  cursor: "pointer",
                  padding: 0,
                  opacity: focus === c.key ? 1 : 0.55,
                }}
              >
                <span className="legend-item">
                  <span className="swatch line" style={{ background: c.color }} />
                  {c.label}
                </span>
              </button>
            ))}
          </div>
        }
        note="Dashed horizontal line is chance for balanced accuracy (0.50). Shaded band is ±1 s.d. across the five held-out folds. Click a legend entry to isolate a property."
      >
        <div ref={ref} style={{ overflowX: "auto" }}>
          <svg width={w} height={h} style={{ display: "block" }}>
            <Axis
              x={xs}
              y={ys}
              w={w - m.r}
              h={h - m.b}
              xTicks={[0, 4, 8, 12, 16, 20, L]}
              yTicks={[0.5, 0.6, 0.7, 0.8, 0.9, 1]}
              yFormat={(v) => `${Math.round(v * 100)}%`}
            />
            {curves.map((c) => {
              const path = c.acc.map((v, l) => `${l === 0 ? "M" : "L"}${xs(l)},${ys(v)}`).join(" ");
              const isFocus = c.key === focus;
              return (
                <g key={c.key} opacity={isFocus ? 1 : 0.3}>
                  {showBands && (
                    <path
                      d={
                        c.acc.map((v, l) => `${l === 0 ? "M" : "L"}${xs(l)},${ys(Math.min(1, v + c.std[l]))}`).join(" ") +
                        " " +
                        c.acc
                          .slice()
                          .reverse()
                          .map((v, li) => {
                            const l = c.acc.length - 1 - li;
                            return `L${xs(l)},${ys(Math.max(0, v - c.std[l]))}`;
                          })
                          .join(" ") +
                        " Z"
                      }
                      fill={c.color}
                      opacity={0.08}
                    />
                  )}
                  <path d={path} fill="none" stroke={c.color} strokeWidth={isFocus ? 2.2 : 1} />
                  <line x1={xs(0)} x2={xs(L)} y1={ys(c.chance)} y2={ys(c.chance)} stroke={c.color} strokeDasharray="2 4" opacity={0.35} />
                  {c.acc.map((v, l) => (
                    <circle key={l} cx={xs(l)} cy={ys(v)} r={isFocus ? 2 : 0} fill={c.color} />
                  ))}
                </g>
              );
            })}
            {focused && (
              <g>
                <text x={xs(L) + 8} y={ys(focused.acc[L]) + 3} className="ticklabel" style={{ fill: focused.color }}>
                  {pct(focused.acc[L], 0)} at L{L}
                </text>
                <text x={xs(L) + 8} y={ys(focused.chance) + 3} className="ticklabel">
                  chance {pct(focused.chance, 0)}
                </text>
              </g>
            )}
          </svg>
        </div>
      </Panel>

      <div className="grid4">
        {crossings.slice(0, 4).map((c) => (
          <Stat key={c.key} k={c.label} v={c.first < 0 ? "not linearly" : `L${c.first}`} s={c.first < 0 ? "no layer crosses chance+8" : `peak ${pct(c.peak, 0)}`} />
        ))}
      </div>

      <h2 className="sec">The full table</h2>
      <Panel
        title={<><b>Figure 6.2</b> · every property, every layer</>}
        note="Accuracy of a linear probe trained on that layer's residual states, averaged over 5 folds. The colour ramp is relative to chance+1, so 1.0 means 'the probe has learned nothing above the majority class'."
        tight
      >
        <div className="scroll">
          <table className="tbl">
            <thead>
              <tr>
                <th>property</th>
                <th>chance</th>
                {probes.layers.map((l) => (
                  <th key={l}>{l}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {curves.map((c) => (
                <tr key={c.key}>
                  <td className="k">{c.label}</td>
                  <td>{pct(c.chance, 0)}</td>
                  {c.acc.map((v, l) => {
                    const above = Math.max(0, v - c.chance) / Math.max(0.01, 1 - c.chance);
                    return (
                      <td
                        key={l}
                        style={{
                          background: `rgba(168,67,31,${above * 0.55})`,
                          color: above > 0.5 ? "#f7f5f0" : "var(--ink2)",
                          padding: "4px 5px",
                          textAlign: "center",
                        }}
                        title={`${pct(v, 1)} ± ${fmt(c.std[l], 3)}`}
                      >
                        {Math.round(v * 100)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Prose>
        <p>
          The curves disagree in an informative way. Surface properties that are
          essentially token identity — <strong>punctuation</strong>,{" "}
          <strong>starts a word</strong>, <strong>capitalised</strong> — are
          readable from the earliest layers, because the embedding already encodes
          them. The <em>relational</em> property,{" "}
          <strong>next token seen earlier</strong>, behaves differently: it starts at{" "}
          {pct(rel.acc[0], 0)} in the embedding, peaks at{" "}
          <strong>{pct(relPeak, 0)}</strong> by <strong>L{relPeakLayer}</strong>, then
          decays to {pct(rel.acc[rel.acc.length - 1], 0)} at the output. The
          information is assembled early and then either used or overwritten — and
          the probe cannot tell which, which is exactly why chapter 08 exists.
        </p>
        <p>
          Nothing here proves the model <em>uses</em> these features. A probe can
          succeed on information that is present but discarded. That is why chapter
          08 exists: probe curves tell you what is readable, ablation tells you what
          is load-bearing, and the two do not always agree.
        </p>
      </Prose>

      <Method>
        <p>
          Features are the residual state at every token of every prompt —{" "}
          {atlas.meta.corpus} prompts, {atlas.meta.tokens} tokens — giving a matrix
          of roughly {atlas.meta.tokens} × {atlas.meta.hidden} per layer. Labels are
          computed mechanically from the token string and its context (punctuation
          class, whether the token contains a digit, whether the decoded word
          appeared earlier in the prompt, and so on), so there is no human
          annotation noise and no label leakage from the future.
        </p>
        <p>
          Splits are <code>GroupKFold(5)</code> grouped by <em>prompt</em>: a probe
          never evaluates on a prompt whose tokens it trained on. Without grouping,
          repeated tokens across folds would inflate accuracy substantially.
        </p>
      </Method>

      <Footnotes
        items={[
          "Balanced accuracy is the mean per-class recall, so chance is 0.50 for every binary property and rare properties are not swallowed by the majority class. Raw accuracy would be easier to read but systematically flattering.",
          "Probes are linear and standardised. A nonlinear probe would find more properties, earlier; the point here is a lower bound on what is explicitly represented.",
          `The probe uses C=0.1 regularisation, chosen before looking at the results. Accuracy is averaged over 5 folds, and the reported band is the fold-to-fold standard deviation.`,
        ]}
      />
      <PageFoot index={7} />
    </>
  );
}
