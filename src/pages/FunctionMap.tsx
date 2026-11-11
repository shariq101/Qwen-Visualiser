import React, { useMemo, useState } from "react";
import { Atlas, fmt, pct } from "../lib/data";
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
  Stat,
} from "../components/ui";

const METRICS = [
  { key: "causal", label: "causal weight", color: "#a8431f", desc: "ΔNLL when the block is knocked out" },
  { key: "attnShare", label: "attention share", color: "#1f5c6b", desc: "attention output as a fraction of block output" },
  { key: "mlpDla", label: "MLP write", color: "#6b3f6e", desc: "|DLA| of the MLP toward the prompt's target" },
  { key: "attnDla", label: "attention write", color: "#43478c", desc: "|DLA| of attention toward the target" },
  { key: "probeGain", label: "feature formation", color: "#2f6b3f", desc: "mean probe gain over the previous layer" },
  { key: "novelty", label: "representational change", color: "#2b5f8a", desc: "1 − CKA with the previous layer" },
  { key: "commit", label: "commitment", color: "#a8431f", desc: "prompts whose target becomes the lens argmax here" },
] as const;

type MetricKey = (typeof METRICS)[number]["key"];

export default function FunctionMap({ atlas }: { atlas: Atlas }) {
  const L = atlas.meta.layers;
  const { ablation, probes, geometry, dla, lens } = atlas;
  const [sel, setSel] = useState(12);

  const raw = useMemo(() => {
    const vals: Record<string, number[]> = {
      causal: [],
      attnShare: [],
      mlpDla: [],
      attnDla: [],
      probeGain: [],
      novelty: [],
      commit: [],
    };
    const propKeys = probes.properties.map((p) => p.key);
    for (let l = 0; l < L; l++) {
      vals.causal.push(ablation.block[String(l)]?.delta ?? 0);
      const attnN = atlas.prompts.map((p) => atlas.norms[p.id].attn[l].reduce((a, b) => a + b, 0) / atlas.norms[p.id].attn[l].length);
      const delN = atlas.prompts.map((p) => atlas.norms[p.id].delta[l].reduce((a, b) => a + b, 0) / atlas.norms[p.id].delta[l].length);
      const aSum = attnN.reduce((a, b) => a + b, 0);
      const dSum = delN.reduce((a, b) => a + b, 0);
      vals.attnShare.push(aSum / (dSum || 1));
      vals.mlpDla.push(atlas.prompts.reduce((a, p) => a + Math.abs(dla[p.id].mlp[l]), 0) / atlas.prompts.length);
      vals.attnDla.push(atlas.prompts.reduce((a, p) => a + Math.abs(dla[p.id].attn[l]), 0) / atlas.prompts.length);
      const prev = l === 0 ? 0 : propKeys.reduce((a, k) => a + probes.byProp[k][l - 1], 0) / propKeys.length;
      const cur = propKeys.reduce((a, k) => a + probes.byProp[k][l], 0) / propKeys.length;
      vals.probeGain.push(Math.max(0, cur - prev));
      vals.novelty.push(1 - geometry.cka[l][l + 1]);
      vals.commit.push(atlas.prompts.filter((p) => lens[p.id].targetRank[l] === 0 && (l === 0 || lens[p.id].targetRank[l - 1] !== 0)).length);
    }
    return vals;
  }, [atlas, L, probes, geometry, ablation, dla, lens]);

  const norm = useMemo(() => {
    const out: Record<string, number[]> = {};
    for (const k of Object.keys(raw)) {
      const v = raw[k];
      const mn = Math.min(...v);
      const mx = Math.max(...v);
      out[k] = v.map((x) => (mx - mn < 1e-9 ? 0 : (x - mn) / (mx - mn)));
    }
    return out;
  }, [raw]);

  const kmeans = useMemo(() => {
    const pts = Array.from({ length: L }, (_, l) => METRICS.map((m) => norm[m.key][l]));
    const k = 4;
    let centers = [0, Math.floor(L / 3), Math.floor((2 * L) / 3), L - 1].map((i) => [...pts[i]]);
    let assign = pts.map(() => 0);
    for (let it = 0; it < 40; it++) {
      assign = pts.map((p) => {
        let best = 0;
        let bd = Infinity;
        centers.forEach((c, ci) => {
          const d = p.reduce((a, v, i) => a + (v - c[i]) ** 2, 0);
          if (d < bd) { bd = d; best = ci; }
        });
        return best;
      });
      centers = centers.map((c, ci) => {
        const members = pts.filter((_, i) => assign[i] === ci);
        if (!members.length) return c;
        return c.map((_, i) => members.reduce((a, p) => a + p[i], 0) / members.length);
      });
    }
    return assign;
  }, [norm, L]);

  const regimeColors = ["#1f5c6b", "#43478c", "#6b3f6e", "#a8431f"];
  const regimeNames: Record<number, string> = {};
  kmeans.forEach((r, l) => {
    if (!(r in regimeNames)) regimeNames[r] = `R${r}`;
  });

  const card = (key: MetricKey) => {
    const i = METRICS.findIndex((m) => m.key === key);
    return METRICS[i];
  };

  const regimes = ([0, 1, 2, 3]
    .map((r) => {
      const ls = kmeans.map((x, i) => (x === r ? i : -1)).filter((i) => i >= 0);
      if (!ls.length) return null;
      const means = METRICS.map((m) => ({ label: m.label, v: ls.reduce((a, l) => a + norm[m.key][l], 0) / ls.length }));
      const top = [...means].sort((a, b) => b.v - a.v).slice(0, 2);
      return { r, ls, top };
    })
    .filter(Boolean) as { r: number; ls: number[]; top: { label: string; v: number }[] }[]);

  const describe = (l: number) => {
    const order = [...METRICS].sort((a, b) => norm[b.key][l] - norm[a.key][l]);
    const top = order.slice(0, 2).map((m) => m.label);
    const low = order[order.length - 1].label;
    return `Above-baseline ${top[0]} and ${top[1]}; least ${low}.`;
  };

  const layerChips = Array.from({ length: L }, (_, l) => l);

  return (
    <>
      <ChapterHead
        n="13"
        kicker="Synthesis"
        title={<>One model, <em>four regimes</em></>}
        standfirst={
          <>
            Seven independent measurements, normalised and stacked side by side. If
            layers had one job each, this matrix would be diagonal. It is not — but
            it is not noise either: the rows cluster into a small number of
            behavioural regimes, and the boundaries are sharp.
          </>
        }
        chips={
          <>
            <Chip tone="amber">{METRICS.length} measurements</Chip>
            <Chip>{L} layers</Chip>
            <Chip tone="cyan">k-means, k=4</Chip>
          </>
        }
      />

      <Panel
        title={<><b>Figure 10.1</b> · the layer fingerprint</>}
        tools={<Legend items={METRICS.map((m) => ({ c: m.color, label: m.label }))} />}
        note="Each column is min–max normalised across layers, then shown as a horizontal profile. The leftmost strip is the k-means regime; click any row for its report card."
      >
        <div className="stack" style={{ gap: 3 }}>
          {layerChips.map((l) => (
            <div
              key={l}
              onClick={() => setSel(l)}
              style={{
                display: "grid",
                gridTemplateColumns: "26px 8px 1fr",
                gap: 6,
                alignItems: "center",
                cursor: "pointer",
                padding: "1px 0",
                outline: l === sel ? "1px solid rgba(168,67,31,0.45)" : "none",
              }}
            >
              <span className="ticklabel">{l}</span>
              <span style={{ width: 6, height: 12, background: regimeColors[kmeans[l]], borderRadius: 1 }} />
              <div style={{ display: "grid", gridTemplateColumns: `repeat(${METRICS.length}, 1fr)`, gap: 2 }}>
                {METRICS.map((m) => {
                  const v = norm[m.key][l];
                  return (
                    <div
                      key={m.key}
                      title={`${m.label} · L${l} · ${fmt(raw[m.key][l], 3)}`}
                      style={{
                        height: 12,
                        background: `rgba(${m.color === "#a8431f" ? "242,181,68" : m.color === "#1f5c6b" ? "79,209,197" : m.color === "#6b3f6e" ? "232,121,185" : m.color === "#43478c" ? "155,140,255" : m.color === "#2f6b3f" ? "125,219,143" : m.color === "#2b5f8a" ? "98,168,255" : "239,111,108"},${0.06 + v * 0.9})`,
                        borderRadius: 1,
                      }}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>
        <div className="spread" style={{ marginTop: 8 }}>
          <span className="ticklabel">L0 → L{L - 1}</span>
          <span className="ticklabel">columns = {METRICS.map((m) => m.label).join(" · ")}</span>
        </div>
      </Panel>

      <h2 className="sec">Layer L{sel}</h2>
      <div className="grid2" style={{ alignItems: "start" }}>
        <Panel
          title={<><b>Report card</b> · L{sel} · regime {kmeans[sel]}</>}
          note="Values are raw; the parenthetical is the normalised rank among layers."
        >
          <div className="stack" style={{ gap: 8 }}>
            {METRICS.map((m) => {
              const v = raw[m.key][sel];
              const p = norm[m.key][sel];
              return (
                <div key={m.key} style={{ display: "grid", gridTemplateColumns: "132px 1fr 56px", gap: 8, alignItems: "center" }}>
                  <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--ink3)" }}>{m.label}</span>
                  <div style={{ height: 7, background: "rgba(23,24,28,0.03)" }}>
                    <div style={{ width: `${p * 100}%`, height: "100%", background: m.color }} />
                  </div>
                  <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--ink2)", textAlign: "right" }}>{fmt(v, 2)}</span>
                </div>
              );
            })}
          </div>
          <div className="caption" style={{ marginTop: 10 }}>{describe(sel)}</div>
        </Panel>
        <div className="stack">
          <Panel title={<><b>Regime boundaries</b></>} note="k-means over the seven normalised measurements. The labels are arbitrary; the grouping is not.">
            <div className="stack" style={{ gap: 4 }}>
              {[0, 1, 2, 3].map((r) => {
                const ls = kmeans.map((x, i) => (x === r ? i : -1)).filter((i) => i >= 0);
                const contiguous = ls.length ? `${ls[0]}–${ls[ls.length - 1]}` : "—";
                return (
                  <div key={r} style={{ display: "grid", gridTemplateColumns: "26px 1fr 70px", gap: 8, alignItems: "center" }}>
                    <span style={{ width: 8, height: 8, background: regimeColors[r], borderRadius: 1 }} />
                    <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--ink2)" }}>
                      regime R{r} · {ls.length} layers
                    </span>
                    <span className="ticklabel" style={{ textAlign: "right" }}>L{contiguous}</span>
                  </div>
                );
              })}
            </div>
          </Panel>
          <Panel title={<><b>What each regime does</b></>}>
            <div className="prose" style={{ fontSize: 13 }}>
              {regimes.map((g, i) => (
                <p
                  key={g.r}
                  style={{ marginTop: i === 0 ? 0 : 8, marginBottom: i === regimes.length - 1 ? 0 : 8 }}
                >
                  <strong style={{ color: regimeColors[g.r] }}>
                    R{g.r} · L{g.ls[0]}–L{g.ls[g.ls.length - 1]}
                  </strong>{" "}
                  — {g.ls.length} layers, highest on {g.top[0].label.toLowerCase()} and{" "}
                  {g.top[1].label.toLowerCase()}.
                </p>
              ))}
            </div>
          </Panel>
        </div>
      </div>

      <h2 className="sec">The synthesis</h2>
      <Prose>
        <p>
          Across fourteen instruments, the same shape keeps appearing. Depth is not
          homogeneous: it has a <strong>short perceptual prefix</strong> (L0–L2),
          a <strong>long associative middle</strong> (L3–L17), a{" "}
          <strong>commitment band</strong> (L18–L21), and a{" "}
          <strong>readout band</strong> (L22–L23). The boundaries are not sharp in
          any single measurement, but they are visible in all of them at once.
        </p>
        <p>
          The middle is where the interesting disagreement lives. Probe accuracy
          rises there, MLP neurons become selective there, induction heads appear
          there — and yet ablating a single middle block barely changes the loss,
          because the residual stream carries redundant copies. The computation is
          real, but it is not <em>localised</em> to a layer in the way a circuit
          diagram would suggest. That is the central tension of this atlas, and it
          is why the fourteen views exist: any one of them would have told a simpler and
          wronger story.
        </p>
        <p>
          What the model is doing, in one sentence: <em>it builds a
          context-specific representation of the next token by accumulating
          evidence across depth, commits to a candidate in the late middle, and
          then spends its last blocks converting that decision into a distribution
          over 151,936 tokens.</em> Everything in this report is a measurement of
          some part of that sentence, and none of the measurements is sufficient on
          its own.
        </p>
      </Prose>

      <Panel
        title={<><b>Figure 10.2</b> · the same synthesis, numerically</>}
        note="Mean value of each measurement within each regime — the actual numbers behind the prose above."
        tight
      >
        <div className="scroll">
          <table className="tbl">
            <thead>
              <tr>
                <th>regime</th>
                <th>layers</th>
                {METRICS.map((m) => (
                  <th key={m.key}>{m.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[0, 1, 2, 3].map((r) => {
                const ls = kmeans.map((x, i) => (x === r ? i : -1)).filter((i) => i >= 0);
                return (
                  <tr key={r}>
                    <td className="k">R{r}</td>
                    <td>{ls.length}</td>
                    {METRICS.map((m) => {
                      const v = ls.reduce((a, l) => a + raw[m.key][l], 0) / Math.max(ls.length, 1);
                      return <td key={m.key}>{fmt(v, 2)}</td>;
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>

      <Method>
        <p>
          Each measurement is computed independently, then min–max normalised across
          the {L} layers so seven incommensurable quantities can share a row. The
          normalisation is presentational; the report card and the table show raw
          values.
        </p>
        <p>
          k-means (k=4, deterministic initialisation at layers 0, {Math.floor(L / 3)},{" "}
          {Math.floor((2 * L) / 3)}, {L - 1}) is run on the normalised 7-vectors. It
          is a descriptive summary, not a claim that the model has four discrete
          modules.
        </p>
      </Method>

      <Footnotes
        items={[
          `Probe gain is the mean balanced-accuracy gain over the previous layer, across ${probes.properties.length} properties, floored at zero.`,
          "Commitment counts prompts whose target token becomes the lens argmax at that layer for the first time. It is a count over 38 prompts, so it is coarse.",
          "Normalisation means a layer can look 'high' on a metric only because every other layer scored lower. Use the report card numbers, not the bar lengths, for comparisons across metrics.",
          `This atlas measures one ${atlas.meta.modelName} checkpoint on ${atlas.meta.corpus} prompts. The method is the contribution; the specific boundaries are a property of this model and this corpus.`,
        ]}
      />
      <PageFoot index={14} />
    </>
  );
}
