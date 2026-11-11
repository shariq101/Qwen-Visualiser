import React, { useMemo, useState } from "react";
import { Atlas, SEQ, fmt, ramp, useMeasure, tokenLabel } from "../lib/data";
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
  Seg,
  Stat,
} from "../components/ui";

const LAYER_RAMP: [number, string][] = [
  [0, "#1f5c6b"],
  [0.3, "#2f7f8f"],
  [0.6, "#6b3f6e"],
  [1, "#a8431f"],
];

export default function GeometryPage({ atlas }: { atlas: Atlas }) {
  const geo = atlas.geometry;
  const L = atlas.meta.layers;
  const [pid, setPid] = useState(Object.keys(geo.pca)[0]);
  const [mode, setMode] = useState<"trajectory" | "layers">("trajectory");
  const { ref, width } = useMeasure<HTMLDivElement>();

  const w = 520;
  const h = 460;
  const m = { t: 20, r: 20, b: 26, l: 20 };
  const pca = geo.pca[pid];
  const allPts = pca ? pca.points.flat() : [[0, 0]];
  const xmin = Math.min(...allPts.map((p) => p[0]));
  const xmax = Math.max(...allPts.map((p) => p[0]));
  const ymin = Math.min(...allPts.map((p) => p[1]));
  const ymax = Math.max(...allPts.map((p) => p[1]));
  const sx = (v: number) => m.l + ((v - xmin) / (xmax - xmin || 1)) * (w - m.l - m.r);
  const sy = (v: number) => h - m.b - ((v - ymin) / (ymax - ymin || 1)) * (h - m.t - m.b);

  const eff = geo.effectiveRank;
  const maxEff = Math.max(...eff);
  const minEff = Math.min(...eff);
  const minEffLayer = eff.indexOf(minEff);
  const pair = (() => {
    let best = { i: 0, j: 1, v: -1 };
    let weakest = { i: 0, j: 1, v: 2 };
    for (let i = 0; i < geo.cka.length; i++)
      for (let j = i + 1; j < geo.cka.length; j++) {
        const v = geo.cka[i][j];
        if (v > best.v) best = { i, j, v };
        if (v < weakest.v) weakest = { i, j, v };
      }
    return { best, weakest };
  })();

  const { ref: ref2, width: w2 } = useMeasure<HTMLDivElement>();
  const h2 = 150;
  const xs2 = (l: number) => 40 + (l / L) * (Math.max(w2, 400) - 60);
  const ys2 = (v: number) => h2 - 24 - (v / maxEff) * (h2 - 40);

  return (
    <>
      <ChapterHead
        n="07"
        kicker="Geometry"
        title={<>Representations have a <em>shape</em></>}
        standfirst={
          <>
            A residual stream is a point in {atlas.meta.hidden} dimensions. Two
            layers can carry the same information in different coordinate systems,
            and the geometry of that change — how much the basis rotates, how many
            dimensions are actually used — is a different kind of evidence from
            probabilities. <b>CKA measures shared structure; effective rank
            measures how much room the model is using.</b>
          </>
        }
        chips={
          <>
            <Chip tone="amber">CKA {L + 1}×{L + 1}</Chip>
            <Chip>peak rank {Math.round(maxEff)}</Chip>
            <Chip tone="cyan">PCA shared across layers</Chip>
          </>
        }
      />

      <h2 className="sec">Layer-to-layer similarity</h2>
      <div className="grid2" style={{ alignItems: "start" }}>
        <Panel
          title={<><b>Figure 7.1</b> · linear CKA</>}
          note="Centred kernel alignment between every pair of layers, on all tokens of the corpus. Bright blocks are layers that represent the data in the same way; off-diagonal brightness means a layer is doing something the layers around it undo."
        >
          <div style={{ overflowX: "auto" }}>
            <svg width={Math.max(width / 2 - 20, 380)} height={Math.max(width / 2 - 20, 380)} style={{ display: "block" }}>
              {Array.from({ length: L + 1 }, (_, i) =>
                Array.from({ length: L + 1 }, (_, j) => {
                  const cw = Math.max(width / 2 - 20, 380) / (L + 1);
                  const v = geo.cka[i][j];
                  return (
                    <rect
                      key={`${i}-${j}`}
                      x={j * cw}
                      y={i * cw}
                      width={cw - 1}
                      height={cw - 1}
                      fill={ramp(Math.pow(v, 0.8), [
                        [0, "#0d1016"],
                        [0.4, "#1d3a4a"],
                        [0.7, "#2f8f8f"],
                        [1, "#a8431f"],
                      ])}
                    >
                      <title>{`L${i} ↔ L${j}: CKA ${v.toFixed(3)}`}</title>
                    </rect>
                  );
                }),
              )}
              {[0, 6, 12, 18, L].map((l) => {
                const cw = Math.max(width / 2 - 20, 380) / (L + 1);
                return (
                  <text key={l} x={l * cw + cw / 2} y={Math.max(width / 2 - 20, 380) - 4} textAnchor="middle" className="ticklabel">
                    {l}
                  </text>
                );
              })}
            </svg>
          </div>
          <div className="caption" style={{ marginTop: 6 }}>
            rows and columns are layers 0…{L}. The diagonal is 1 by definition.
          </div>
        </Panel>

        <div className="stack">
          <Panel
            title={<><b>Figure 7.2</b> · effective rank</>}
            note="exp(entropy of normalised singular values) of the centred hidden-state matrix. A low value means the layer's variance lives in a few directions."
          >
            <div ref={ref2} style={{ overflowX: "auto" }}>
              <svg width={Math.max(w2, 400)} height={h2} style={{ display: "block" }}>
                <Axis x={xs2} y={ys2} w={Math.max(w2, 400) - 40} h={h2 - 24} xTicks={[0, 6, 12, 18, L]} yTicks={[0, 50, 100, 150, 200]} />
                <path
                  d={eff.map((v, l) => `${l === 0 ? "M" : "L"}${xs2(l)},${ys2(v)}`).join(" ")}
                  fill="none"
                  stroke="#1f5c6b"
                  strokeWidth={2}
                />
                {eff.map((v, l) => (
                  <circle key={l} cx={xs2(l)} cy={ys2(v)} r={2.2} fill="#1f5c6b">
                    <title>{`L${l}: ${v.toFixed(1)}`}</title>
                  </circle>
                ))}
              </svg>
            </div>
            <div className="grid2" style={{ marginTop: 8 }}>
              <Stat k="Minimum" v={fmt(Math.min(...eff), 1)} s={`at L${eff.indexOf(Math.min(...eff))}`} />
              <Stat k="Final" v={fmt(eff[L], 1)} s="output layer" />
            </div>
          </Panel>
          <Panel
            title={<><b>Figure 7.3</b> · direction drift</>}
            note="Mean cosine between each layer's final-token state and the last layer's state. A jump means the answer direction rotated."
          >
            <svg width={Math.max(w2, 400)} height={118} style={{ display: "block" }}>
              {geo.drift.map((v, l) => {
                const bw = (Math.max(w2, 400) - 40) / (L + 1);
                return (
                  <g key={l}>
                    <rect
                      x={40 + l * bw}
                      y={96 - v * 72}
                      width={bw - 1.5}
                      height={v * 72}
                      fill={l === L ? "#a8431f" : "#2f7f8f"}
                    >
                      <title>{`L${l}: cos ${v.toFixed(3)}`}</title>
                    </rect>
                    <text x={40 + l * bw + bw / 2} y={110} textAnchor="middle" className="ticklabel">
                      {l % 4 === 0 ? l : ""}
                    </text>
                  </g>
                );
              })}
              <line x1={40} x2={Math.max(w2, 400) - 20} y1={96} y2={96} stroke="#b9b5aa" />
            </svg>
          </Panel>
        </div>
      </div>

      <h2 className="sec">Token trajectories</h2>
      <div className="row" style={{ alignItems: "flex-end", marginBottom: 6 }}>
        <label className="field">
          <span>Prompt</span>
          <select className="ctl" value={pid} onChange={(e) => setPid(e.target.value)} style={{ minWidth: 280 }}>
            {Object.keys(geo.pca).map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </label>
        <div style={{ flex: 1 }} />
        <Seg
          value={mode}
          options={[
            { v: "trajectory", label: "trajectories" },
            { v: "layers", label: "layer clouds" },
          ]}
          onChange={setMode}
        />
      </div>

      <Panel
        title={<><b>Figure 7.4</b> · {mode === "trajectory" ? "each token's path through depth" : "each layer's cloud of token states"}</>}
        tools={
          <Legend
            items={[
              { c: "#1f5c6b", label: "L0" },
              { c: "#2f7f8f", label: "L8" },
              { c: "#6b3f6e", label: "L16" },
              { c: "#a8431f", label: `L${L}` },
            ]}
          />
        }
        note="A single PCA basis is fit to the hidden states of all layers at once, so a token's position is comparable across depth. Movement is real movement under one shared projection."
      >
        {pca && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))", gap: 18, alignItems: "start" }}>
            <div style={{ overflowX: "auto" }}>
            <svg width={w} height={h} style={{ display: "block", maxWidth: "100%" }}>
              <rect x={0} y={0} width={w} height={h} fill="rgba(23,24,28,0.012)" stroke="#d8d4ca" />
              {mode === "trajectory"
                ? pca.points.map((pts, t) => (
                    <g key={t}>
                      <path
                        d={pts.map((p, l) => `${l === 0 ? "M" : "L"}${sx(p[0])},${sy(p[1])}`).join(" ")}
                        fill="none"
                        stroke={ramp(1, LAYER_RAMP)}
                        strokeWidth={0.9}
                        opacity={0.22}
                      />
                      {pts.map((p, l) => (
                        <circle
                          key={l}
                          cx={sx(p[0])}
                          cy={sy(p[1])}
                          r={l === pts.length - 1 ? 3.6 : 1.9}
                          fill={ramp(l / L, LAYER_RAMP)}
                          opacity={l === pts.length - 1 ? 1 : 0.75}
                        >
                          <title>{`${tokenLabel(pca.tokens[t])} · L${l}`}</title>
                        </circle>
                      ))}
                      <text
                        x={sx(pts[pts.length - 1][0]) + 6}
                        y={sy(pts[pts.length - 1][1]) + 3}
                        className="ticklabel"
                        style={{ fill: t === pca.tokens.length - 1 ? "var(--amber)" : "var(--ink3)" }}
                      >
                        {tokenLabel(pca.tokens[t]).slice(0, 9) || "∅"}
                      </text>
                    </g>
                  ))
                : Array.from({ length: L + 1 }, (_, l) => (
                    <g key={l}>
                      {pca.points.map((pts, t) => (
                        <circle key={t} cx={sx(pts[l][0])} cy={sy(pts[l][1])} r={2.4} fill={ramp(l / L, LAYER_RAMP)} opacity={0.55}>
                          <title>{`L${l} · ${tokenLabel(pca.tokens[t])}`}</title>
                        </circle>
                      ))}
                      <text x={sx(geo.centroids[l][0]) + 5} y={sy(geo.centroids[l][1])} className="ticklabel" style={{ fill: "var(--amber)" }}>
                        L{l}
                      </text>
                    </g>
                  ))}
            </svg>
            </div>
            <div className="stack" style={{ gap: 5 }}>
              <div className="caption">Tokens at the final layer, ordered top→bottom by their position on the page:</div>
              {pca.tokens.map((t, i) => (
                <div key={i} style={{ display: "grid", gridTemplateColumns: "1fr 44px", gap: 6, alignItems: "center" }}>
                  <span className="token" style={{ borderColor: i === pca.tokens.length - 1 ? "var(--amber)" : "transparent" }}>
                    {tokenLabel(t).slice(0, 14) || "∅"}
                  </span>
                  <span className="ticklabel">L{L}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </Panel>

      <Prose>
        <p>
          The CKA matrix is close to diagonal-dominant, but not perfectly. The
          strongest off-diagonal pair is <strong>L{pair.best.i} ↔ L{pair.best.j}</strong>{" "}
          (CKA {pair.best.v.toFixed(2)}); the weakest is{" "}
          <strong>L{pair.weakest.i} ↔ L{pair.weakest.j}</strong>{" "}
          ({pair.weakest.v.toFixed(2)}). Effective rank falls to{" "}
          <strong>{minEff.toFixed(0)}</strong> at L{minEffLayer}, then returns to{" "}
          {eff[L].toFixed(0)} at the output. The model compresses into a
          task-relevant subspace and re-expands into vocabulary-sized coordinates;
          the layers in between are where that subspace is actually built.
        </p>
        <p>
          The final-layer token trajectories show something the probabilities hide:
          tokens that end up in the same semantic region often <em>start</em> near
          each other and only separate late. The last token (amber) usually takes
          the longest path, because it is the one that has to become a prediction.
        </p>
      </Prose>

      <Method>
        <p>
          Linear CKA between layer activation matrices X and Y is{" "}
          <code>‖XᵀY‖²_F / (‖XᵀX‖_F · ‖YᵀY‖_F)</code> after centring. It is
          invariant to orthogonal rotation and isotropic scaling, so it measures
          shared structure rather than shared coordinates.
        </p>
        <p>
          Effective rank is <code>exp(H(σ/Σσ))</code> where σ are singular values
          of the centred activation matrix. It is a soft count of dimensions
          carrying variance.
        </p>
      </Method>

      <Footnotes
        items={[
          "PCA is fit once on the concatenation of all layers' hidden states, not separately per layer — otherwise the axes would rotate between panels and trajectories would be meaningless.",
          "CKA on 38 prompts is a small sample; the block structure is robust across prompts but individual off-diagonal values are noisy.",
          "The drift curve uses the final token only, because that is the position that determines the next-token prediction.",
        ]}
      />
      <PageFoot index={8} />
    </>
  );
}
