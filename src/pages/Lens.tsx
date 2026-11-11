import React, { useMemo, useState } from "react";
import {
  Atlas,
  Prompt,
  SEQ,
  fmt,
  href,
  bare,
  mix,
  pct,
  ramp,
  useMeasure,
  usePrompt,
  tokenLabel,
} from "../lib/data";
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

const W = 1000;

function Spectrum({ entry, layer, final }: { entry: Atlas["lens"][string]; layer: number; final: boolean }) {
  const items = entry.top[layer].slice(0, 12);
  const max = Math.max(...items.map((d) => d[1]), 0.001);
  return (
    <div className="stack" style={{ gap: 5 }}>
      {items.map(([tok, p], i) => {
        const isTarget = bare(tok) === bare(entry.targetStr);
        return (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "120px 1fr 62px", gap: 10, alignItems: "center" }}>
            <span
              style={{
                fontFamily: "var(--mono)",
                fontSize: 12,
                color: isTarget ? "var(--amber)" : i === 0 ? "var(--ink)" : "var(--ink2)",
                textAlign: "right",
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
              title={tok}
            >
              {tokenLabel(tok) === "" ? "∅" : tokenLabel(tok)}
            </span>
            <div style={{ height: 13, background: "rgba(23,24,28,0.03)", borderRadius: 2, position: "relative" }}>
              <div
                style={{
                  width: `${(p / max) * 100}%`,
                  height: "100%",
                  background: isTarget
                    ? "linear-gradient(90deg,#8a6420,#a8431f)"
                    : i === 0
                      ? "linear-gradient(90deg,#c08f4a,#1f5c6b)"
                      : "linear-gradient(90deg,#c9c4b8,#8a8d93)",
                  borderRadius: 2,
                  transition: "width 220ms cubic-bezier(.2,.8,.2,1)",
                }}
              />
            </div>
            <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: isTarget ? "var(--amber)" : "var(--ink3)" }}>
              {pct(p, p > 0.1 ? 1 : 2)}
            </span>
          </div>
        );
      })}
      <div className="caption" style={{ marginTop: 4 }}>
        {final ? "Final layer: this is the model's actual output distribution." : "Intermediate layer: a projection through the final norm and unembedding."}
      </div>
    </div>
  );
}

function Trajectory({
  entry,
  layer,
  onLayer,
  L,
}: {
  entry: Atlas["lens"][string];
  layer: number;
  onLayer: (l: number) => void;
  L: number;
}) {
  const { ref, width } = useMeasure<HTMLDivElement>();
  const h = 210;
  const m = { t: 14, r: 42, b: 26, l: 46 };
  const xs = (l: number) => m.l + (l / L) * (width - m.l - m.r);
  const maxP = Math.max(...entry.targetProb, ...entry.top1Prob, 0.02);
  const maxEnt = Math.max(...entry.entropy);
  const ys = (v: number) => m.t + (1 - v / maxP) * (h - m.t - m.b);
  const ye = (v: number) => m.t + (1 - v / maxEnt) * (h - m.t - m.b);

  const path = (vals: number[], yfn: (v: number) => number) =>
    vals.map((v, i) => `${i === 0 ? "M" : "L"}${xs(i).toFixed(1)},${yfn(v).toFixed(1)}`).join(" ");

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg width={width} height={h} style={{ display: "block" }}>
        <Axis
          x={xs}
          y={ys}
          w={width - m.r}
          h={h - m.b}
          xTicks={[0, 4, 8, 12, 16, 20, L]}
          yTicks={[0, 0.25, 0.5, 0.75, 1].map((v) => v * maxP)}
          yFormat={(v) => pct(v, 0)}
        />
        <path d={path(entry.entropy, ye)} fill="none" stroke="#43478c" strokeWidth={1.2} strokeDasharray="2 3" opacity={0.75} />
        <path d={path(entry.top1Prob, ys)} fill="none" stroke="#1f5c6b" strokeWidth={1.4} opacity={0.9} />
        <path d={path(entry.targetProb, ys)} fill="none" stroke="#a8431f" strokeWidth={2} />
        {entry.targetProb.map((v, i) => (
          <circle key={i} cx={xs(i)} cy={ys(v)} r={2.1} fill="#a8431f" />
        ))}
        <line x1={xs(layer)} x2={xs(layer)} y1={m.t - 6} y2={h - m.b} stroke="#a8431f" strokeWidth={1} opacity={0.55} />
        <rect
          x={xs(layer) - 9}
          y={h - m.b}
          width={18}
          height={6}
          fill="#a8431f"
          style={{ cursor: "ew-resize" }}
        />
        {Array.from({ length: L + 1 }, (_, l) => (
          <rect
            key={l}
            x={xs(l) - (width - m.l - m.r) / (2 * (L + 1))}
            y={0}
            width={(width - m.l - m.r) / (L + 1)}
            height={h}
            fill="transparent"
            style={{ cursor: "pointer" }}
            onClick={() => onLayer(l)}
          />
        ))}
      </svg>
      <Legend
        items={[
          { c: "#a8431f", label: "P(target token)", line: true },
          { c: "#1f5c6b", label: "P(lens top-1)", line: true },
          { c: "#43478c", label: "entropy (dashed, right-scaled)", line: true },
        ]}
      />
    </div>
  );
}

function River({ entry, onPick, picked }: { entry: Atlas["lens"][string]; onPick: (t: number) => void; picked: number }) {
  if (!entry.river) {
    return (
      <div className="caption" style={{ padding: 16 }}>
        This prompt was not captured with per-position lens rows. Select one of the deep-dive prompts
        (<code>fact-multihop</code>, <code>syn-attract</code>, <code>ind-1</code>, <code>idiom-1</code>) to see the full river.
      </div>
    );
  }
  const rivers = entry.river;
  const T = rivers[0].p.length;
  const L = rivers.length - 1;
  const cw = 34;
  const ch = 20;
  const left = 118;
  const top = 24;
  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={left + (L + 1) * cw + 20} height={top + T * ch + 16} style={{ display: "block" }}>
        {Array.from({ length: L + 1 }, (_, l) => (
          <text key={l} x={left + l * cw + cw / 2} y={14} textAnchor="middle" className="ticklabel">
            {l}
          </text>
        ))}
        <text x={left - 8} y={14} textAnchor="end" className="ticklabel">
          layer →
        </text>
        {Array.from({ length: T }, (_, t) => (
          <g key={t}>
            <text x={left - 8} y={top + t * ch + 13} textAnchor="end" className="ticklabel">
              {tokenLabel(entry.tokens[t] ?? "").slice(0, 9) || "∅"}
            </text>
            {Array.from({ length: L + 1 }, (_, l) => {
              const r = rivers[l];
              const p = r.p[t];
              const tok = r.top1[t] ?? "";
              const isTarget = bare(tok) === bare(entry.targetStr);
              return (
                <g key={l}>
                  <rect
                    x={left + l * cw}
                    y={top + t * ch}
                    width={cw - 1.5}
                    height={ch - 1.5}
                    fill={ramp(Math.sqrt(p), SEQ)}
                    stroke={isTarget ? "#a8431f" : "rgba(0,0,0,0.35)"}
                    strokeWidth={isTarget ? 1.2 : 0.5}
                    style={{ cursor: "pointer" }}
                    onClick={() => onPick(t)}
                  >
                    <title>{`L${l} · position ${t} · ${tokenLabel(tok) || "∅"} ${(p * 100).toFixed(1)}%`}</title>
                  </rect>
                  <text
                    x={left + l * cw + cw / 2}
                    y={top + t * ch + 13}
                    textAnchor="middle"
                    style={{ fontFamily: "var(--mono)", fontSize: 8, fill: p > 0.55 ? "#f7f5f0" : "#3b3e45", pointerEvents: "none" }}
                  >
                    {(tokenLabel(tok) || "∅").slice(0, 5)}
                  </text>
                </g>
              );
            })}
            {t === picked && (
              <rect x={left - 1} y={top + t * ch - 1} width={left + (L + 1) * cw + 1} height={ch} fill="none" stroke="#a8431f" strokeWidth={1} opacity={0.6} />
            )}
          </g>
        ))}
      </svg>
    </div>
  );
}

export default function Lens({ atlas }: { atlas: Atlas }) {
  const [prompt, setPrompt] = usePrompt(atlas);
  const entry = atlas.lens[prompt.id];
  const L = atlas.meta.layers;
  const [layer, setLayer] = useState(Math.floor(L * 0.55));
  const [view, setView] = useState<"spectrum" | "river">("spectrum");
  const [tokIdx, setTokIdx] = useState(prompt.n - 1);

  React.useEffect(() => {
    setTokIdx(prompt.n - 1);
  }, [prompt.id, prompt.n]);

  const commit = useMemo(() => {
    for (let l = 0; l <= L; l++) if (entry.targetRank[l] === 0) return l;
    return -1;
  }, [entry, L]);

  const finalProb = entry.targetProb[L];
  const firstNonZero = entry.targetProb.findIndex((v) => v > 0.001);
  const commitText =
    commit >= 0
      ? `the target token is the lens argmax from layer ${commit} onward`
      : `the target token never becomes the lens argmax — the final answer is a different token`;

  return (
    <>
      <ChapterHead
        n="01"
        kicker="Logit lens"
        title={<>The answer, forming in <em>layers</em></>}
        standfirst={
          <>
            Take the residual stream after every block, run it through the final
            norm and the unembedding, and you get an intermediate guess at the next
            token. Read in sequence, those guesses are a time-lapse of the answer
            being decided. <b>{prompt.text.replace(/\n/g, " ⏎ ")}</b>
          </>
        }
        chips={
          <>
            <Chip tone="amber">P(answer) {pct(finalProb, 1)}</Chip>
            <Chip>{commit >= 0 ? `commits at L${commit}` : "never commits"}</Chip>
            <Chip tone="cyan">first signal at L{firstNonZero < 0 ? "—" : firstNonZero}</Chip>
          </>
        }
      />

      <div className="row" style={{ alignItems: "center", gap: 14, marginBottom: 6 }}>
        <label className="field">
          <span>Probe prompt</span>
          <select className="ctl" value={prompt.id} onChange={(e) => setPrompt(e.target.value)} style={{ minWidth: 300 }}>
            {atlas.prompts.map((p) => (
              <option key={p.id} value={p.id}>
                {p.id} — {p.text.slice(0, 52).replace(/\n/g, " ")}
              </option>
            ))}
          </select>
        </label>
        <div style={{ flex: 1 }} />
        <Seg
          value={view}
          options={[
            { v: "spectrum", label: "spectrum" },
            { v: "river", label: "prediction river" },
          ]}
          onChange={setView}
        />
      </div>

      <Panel
        title={<><b>Figure 1.1</b> · {view === "spectrum" ? "the distribution at one layer" : "every position, every layer"}</>}
        tools={
          view === "spectrum" ? (
            <span className="mono" style={{ fontSize: 10, color: "var(--ink3)" }}>
              layer <b style={{ color: "var(--amber)" }}>{layer}</b> / {L}
            </span>
          ) : (
            <span className="mono" style={{ fontSize: 10, color: "var(--ink3)" }}>
              {entry.river ? `${L + 1} layers × ${entry.river[0].p.length} positions` : "not captured"}
            </span>
          )
        }
        note={
          view === "spectrum"
            ? "Bars are the lens distribution top-12. The target token is highlighted in amber. Watch how a hypothesis appears as a low-probability tail, then gets promoted."
            : "Each cell is the lens argmax at that (layer, position). Fill is its probability; amber outline marks a cell whose top-1 is the prompt's target token. Rows are token positions."
        }
      >
        {view === "spectrum" ? (
          <div className="grid2" style={{ gap: 26, alignItems: "start" }}>
            <div>
              <Spectrum entry={entry} layer={layer} final={layer === L} />
            </div>
            <div>
              <div className="spread" style={{ marginBottom: 6 }}>
                <span className="mono" style={{ fontSize: 10, color: "var(--ink3)" }}>
                  layer scrubber
                </span>
                <span className="mono" style={{ fontSize: 10, color: "var(--amber)" }}>
                  L{layer}
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={L}
                value={layer}
                onChange={(e) => setLayer(+e.target.value)}
              />
              <div className="grid2" style={{ gap: 8, marginTop: 10 }}>
                <Stat k="P(target)" v={pct(entry.targetProb[layer], 2)} />
                <Stat k="rank of target" v={entry.targetRank[layer] === 0 ? "#1" : `#${entry.targetRank[layer] + 1}`} />
                <Stat k="entropy" v={fmt(entry.entropy[layer], 2)} s="nats" />
                <Stat k="KL → final" v={fmt(entry.kl[layer], 3)} s="nats" />
              </div>
              <div style={{ marginTop: 16 }}>
                <div className="caption" style={{ marginBottom: 6 }}>
                  Hypotenuse of the layer: the ten most likely tokens at L{layer}.
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
                  {entry.top[layer].slice(0, 10).map(([t, p], i) => (
                    <span
                      key={i}
                      className="token"
                      style={{
                        borderColor: i === 0 ? "var(--cyan)" : "var(--line)",
                        color: i === 0 ? "var(--ink)" : "var(--ink2)",
                        background: `rgba(31,92,107,${0.04 + p * 0.5})`,
                      }}
                      title={`${pct(p, 2)}`}
                    >
                      {tokenLabel(t) || "∅"}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <River entry={entry} picked={tokIdx} onPick={setTokIdx} />
        )}
      </Panel>

      <h2 className="sec">When does the model make up its mind?</h2>
      <Panel
        title={<><b>Figure 1.2</b> · probability of the answer, by depth</>}
        note={`Click anywhere on the plot to move the lens to that layer. Entropy is rescaled to the same box; its true maximum is ${fmt(Math.max(...entry.entropy), 1)} nats.`}
      >
        <Trajectory entry={entry} layer={layer} onLayer={setLayer} L={L} />
      </Panel>

      <Prose>
        <p>
          The curve is rarely a smooth ramp. For this prompt, the target sits in
          the noise floor until <strong>L{Math.max(firstNonZero, 0)}</strong>, then{" "}
          {commit >= 0 ? (
            <>becomes the leading hypothesis at <strong>L{commit}</strong> and stays there.</>
          ) : (
            <>never becomes the leading hypothesis — the model settles on a different
            continuation, and the lens shows the answer losing a competition rather
            than winning one.</>
          )}{" "}
          The final probability is <strong>{pct(finalProb, 1)}</strong>, against a
          lens top-1 probability of <strong>{pct(entry.top1Prob[L], 1)}</strong>.
        </p>
        <p>
          The interesting part is what happens <em>before</em> the commitment.
          Intermediate layers often hold the right answer at ranks 5–50: enough
          signal to be readable, not enough to win. That is the signature of a
          model that has found the relevant fact but has not yet resolved it
          against the alternatives. <strong>{commitText}</strong>.
        </p>
      </Prose>

      <Panel
        title={<><b>Figure 1.3</b> · all {L + 1} layers at once</>}
        note="Each column is one layer's top-1 lens token; height is its probability. This is the whole 'career' of a prediction in one strip."
      >
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${L + 1}, minmax(0,1fr))`, gap: 3 }}>
          {entry.top.map((items, l) => {
            const p = entry.top1Prob[l];
            const tok = items[0]?.[0] ?? "";
            const isT = bare(tok) === bare(entry.targetStr);
            return (
              <button
                key={l}
                onClick={() => setLayer(l)}
                style={{
                  appearance: "none",
                  border: l === layer ? "1px solid var(--amber)" : "1px solid var(--line)",
                  background: "rgba(16,19,25,0.7)",
                  borderRadius: 3,
                  padding: "6px 3px 5px",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 4,
                  cursor: "pointer",
                  minHeight: 92,
                  justifyContent: "flex-end",
                }}
                title={`L${l}: ${tokenLabel(tok)} ${pct(p, 1)}`}
              >
                <div
                  style={{
                    width: "100%",
                    height: Math.max(2, p * 46),
                    background: isT ? "linear-gradient(180deg,#c9704c,#8a5a1f)" : "linear-gradient(180deg,#5fc8c0,#1f5c6b)",
                    borderRadius: 1,
                  }}
                />
                <span style={{ fontFamily: "var(--mono)", fontSize: 8.5, color: isT ? "var(--amber)" : "var(--ink3)", maxWidth: "100%", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {(tokenLabel(tok) || "∅").slice(0, 6)}
                </span>
                <span style={{ fontFamily: "var(--mono)", fontSize: 7.5, color: "var(--ink3)" }}>{l}</span>
              </button>
            );
          })}
        </div>
      </Panel>

      <Method>
        <p>
          The logit lens applies the model's <em>final</em> RMSNorm and unembedding
          matrix to every intermediate residual state. Nothing is trained: this is
          the model's own output head, used early. It is not guaranteed to be
          faithful — intermediate states were never optimised to be decodable — but
          when a token rises monotonically and then stays, the projection is
          tracking something real.
        </p>
        <p>
          KL→final measures how far each layer's distribution is from the model's
          final distribution, in nats. It is a sharper summary than the argmax:
          a layer can keep the same top-1 token while its distribution changes
          radically underneath.
        </p>
      </Method>

      <Prose>
        <p>
          Read against chapter 08, this view becomes a warning. A layer can look
          decisive in the lens and be causally irrelevant under ablation, because
          the lens measures <em>legibility</em>, not <em>use</em>. The two
          measurements are genuinely different instruments, and the gap between
          them is one of the most interesting things in the model.
        </p>
      </Prose>

      <Footnotes
        items={[
          "Layer 0 is the embedding output after the first block's input norm has not yet been applied — i.e. the token embedding plus position, before any computation.",
          `For this prompt the lens argmax first equals the target at L${commit >= 0 ? commit : "—"}; the target's probability first exceeds 0.1% at L${Math.max(firstNonZero, 0)}.`,
          `The per-position river is captured only for the four deep-dive prompts to keep the artefact small; it is a ${L + 1} × T × 4 array per prompt (top-1 token, its probability, target probability, entropy).`,
          "Lens probabilities are not calibrated. A layer reporting 40% early in the network is not saying the model is 40% confident; it is saying the residual state projects onto that token with 40% of the unembedding mass.",
        ]}
      />
      <PageFoot index={2} />
    </>
  );
}
