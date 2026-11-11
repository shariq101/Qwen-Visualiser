import React, { useMemo, useState } from "react";
import { Atlas, DIV, SEQ, fmt, pct, ramp, useMeasure, usePrompt, tokenLabel } from "../lib/data";
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

export default function Anatomy({ atlas }: { atlas: Atlas }) {
  const [prompt, setPrompt] = usePrompt(atlas);
  const L = atlas.meta.layers;
  const d = atlas.dla[prompt.id];
  const n = atlas.norms[prompt.id];
  const [sel, setSel] = useState(Math.floor(L / 2));

  const { ref, width } = useMeasure<HTMLDivElement>();
  const h = 260;
  const m = { t: 18, r: 16, b: 30, l: 54 };
  const maxAbs = Math.max(
    ...d.attn.map(Math.abs),
    ...d.mlp.map(Math.abs),
    0.5,
  );
  const xs = (l: number) => m.l + (l / L) * (width - m.l - m.r);
  const y0 = m.t + (h - m.t - m.b) / 2;
  const scale = (h - m.t - m.b) / 2 / maxAbs;
  const bw = Math.max(2, (width - m.l - m.r) / L / 2 - 2);

  const bestLayer = useMemo(() => {
    let bi = 0;
    d.attn.forEach((v, i) => {
      if (Math.abs(v) + Math.abs(d.mlp[i]) > Math.abs(d.attn[bi]) + Math.abs(d.mlp[bi])) bi = i;
    });
    return bi;
  }, [d]);

  const netsum = d.attn.reduce((a, b) => a + b, 0) + d.mlp.reduce((a, b) => a + b, 0);
  const peak = Math.max(...n.norm.map((row) => Math.max(...row)));

  return (
    <>
      <ChapterHead
        n="02"
        kicker="Layer anatomy"
        title={<>What each block <em>writes</em></>}
        standfirst={
          <>
            Each block reads the residual stream, computes something, and adds its
            result back. Direct logit attribution asks a narrow, answerable
            question: <b>how much did this block move the logit of the token we
            care about?</b> The answer separates <em>layers that compute</em> from{" "}
            <em>layers that pass things along</em>.
          </>
        }
        chips={
          <>
            <Chip tone="amber">target “{prompt.target.trim()}”</Chip>
            <Chip>peak block L{bestLayer}</Chip>
            <Chip tone="cyan">net attributed {fmt(netsum, 2)}</Chip>
          </>
        }
      />

      <div className="row" style={{ alignItems: "center", marginBottom: 6 }}>
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
      </div>

      <Panel
        title={<><b>Figure 2.1</b> · per-block contribution to the target logit</>}
        tools={
          <Legend
            items={[
              { c: "#a8431f", label: "attention output" },
              { c: "#1f5c6b", label: "MLP output" },
            ]}
          />
        }
        note="Amber bars rise above the axis when a block's attention output pushes the target token up; cyan below. Signed values, first-order, final-norm scaled."
      >
        <div ref={ref}>
          <svg width={width} height={h} style={{ display: "block" }}>
            <line x1={m.l} x2={width - m.r} y1={y0} y2={y0} stroke="#b9b5aa" />
            {[-1, -0.5, 0.5, 1].map((f) => (
              <g key={f}>
                <line x1={m.l} x2={width - m.r} y1={y0 + f * maxAbs * scale} y2={y0 + f * maxAbs * scale} className="gridline" />
                <text x={m.l - 6} y={y0 + f * maxAbs * scale + 3} textAnchor="end" className="ticklabel">
                  {(f * maxAbs).toFixed(1)}
                </text>
              </g>
            ))}
            {d.attn.map((v, l) => (
              <g key={l} onClick={() => setSel(l)} style={{ cursor: "pointer" }}>
                <rect
                  x={xs(l) - bw - 1}
                  y={v >= 0 ? y0 - v * scale : y0}
                  width={bw}
                  height={Math.abs(v * scale)}
                  fill={sel === l ? "#c9704c" : "#a8431f"}
                  opacity={sel === l ? 1 : 0.78}
                />
                <rect
                  x={xs(l) + 1}
                  y={d.mlp[l] >= 0 ? y0 - d.mlp[l] * scale : y0}
                  width={bw}
                  height={Math.abs(d.mlp[l] * scale)}
                  fill={sel === l ? "#2f7f8f" : "#1f5c6b"}
                  opacity={sel === l ? 1 : 0.75}
                />
                <text x={xs(l)} y={h - 12} textAnchor="middle" className="ticklabel" style={{ fill: sel === l ? "var(--amber)" : undefined }}>
                  {l}
                </text>
              </g>
            ))}
            <text x={m.l - 6} y={12} className="ticklabel" textAnchor="end">
              logits
            </text>
          </svg>
        </div>
      </Panel>

      <div className="grid3">
        <Stat k="Selected layer" v={`L${sel}`} s="click a bar to change" />
        <Stat k="attention Δ" v={fmt(d.attn[sel], 3)} s="toward the target logit" />
        <Stat k="MLP Δ" v={fmt(d.mlp[sel], 3)} s="toward the target logit" />
      </div>

      <h2 className="sec">The residual stream, as a heat map</h2>
      <Panel
        title={<><b>Figure 2.2</b> · ‖residual‖ at every (layer, position)</>}
        tools={<span className="chip">max {fmt(peak, 0)}</span>}
        note="Normalised magnitude of the residual stream. Early layers grow sharply as token identity is written in; later layers grow more slowly, and the last token often carries a distinct magnitude because it is the only position that also does next-token work."
      >
        <div className="scroll" style={{ maxHeight: 560 }}>
          <svg width={Math.max(width - 2, 340)} height={(L + 1) * 19 + 56} style={{ display: "block" }}>
            {Array.from({ length: prompt.n }, (_, t) => (
              <text key={t} x={96} y={40 + t * 19 + 13} textAnchor="end" className="ticklabel">
                {tokenLabel(prompt.tokens[t]).slice(0, 11) || "∅"}
              </text>
            ))}
            {Array.from({ length: L + 1 }, (_, l) => (
              <text key={l} x={100 + l * ((width - 130) / (L + 1)) + 6} y={16} className="ticklabel">
                {l}
              </text>
            ))}
            {Array.from({ length: L + 1 }, (_, l) =>
              Array.from({ length: prompt.n }, (_, t) => {
                const v = n.norm[l]?.[t] ?? 0;
                const cw = (Math.max(width - 2, 340) - 130) / (L + 1);
                return (
                  <g key={`${l}-${t}`}>
                    <rect
                      x={100 + l * cw}
                      y={26 + t * 19}
                      width={cw - 2}
                      height={17}
                      fill={ramp(v / peak, SEQ)}
                      style={{ cursor: "pointer" }}
                      onClick={() => setSel(l)}
                    >
                      <title>{`L${l} · ${tokenLabel(prompt.tokens[t])} · ‖h‖=${v.toFixed(1)}`}</title>
                    </rect>
                    {l === sel && (
                      <rect x={100 + l * cw} y={26} width={cw - 2} height={prompt.n * 19} fill="none" stroke="#a8431f" strokeWidth={1} opacity={0.55} />
                    )}
                    {t === prompt.n - 1 && (
                      <rect x={100 + l * cw} y={26} width={cw - 2} height={prompt.n * 19} fill="none" stroke="rgba(23,24,28,0.08)" />
                    )}
                  </g>
                );
              }),
            )}
            <text x={100} y={40 + prompt.n * 19 + 14} className="ticklabel">
              ← position (tokens) · layers across →
            </text>
          </svg>
        </div>
      </Panel>

      <h2 className="sec">The shape of a single layer</h2>
      <Panel
        title={<><b>Figure 2.3</b> · layer L{sel}, in detail</>}
        note="How much this block writes (‖Δh‖), how much of that is attention versus MLP, and how much it changes the direction of the residual stream."
      >
        <div className="grid2" style={{ gap: 22 }}>
          <div className="stack">
            <div className="spread">
              <span className="mono" style={{ fontSize: 10, color: "var(--ink3)" }}>block output norm per token</span>
              <span className="chip">L{sel}</span>
            </div>
            <svg width={Math.min(width / 2 - 20, 440)} height={26 + Math.max(...(n.delta[sel] ?? [1])) * 0 + 74}>
              {(n.delta[sel] ?? []).map((v, t) => {
                const mx = Math.max(...(n.delta[sel] ?? [1]));
                const bw2 = Math.min(width / 2 - 20, 440) / prompt.n;
                return (
                  <g key={t}>
                    <rect
                      x={t * bw2 + 2}
                      y={80 - (v / mx) * 66}
                      width={bw2 - 3}
                      height={(v / mx) * 66}
                      fill={t === prompt.n - 1 ? "#a8431f" : "#2f7f8f"}
                      opacity={t === prompt.n - 1 ? 1 : 0.8}
                    />
                    <title>{`${tokenLabel(prompt.tokens[t])} · ‖Δh‖=${v.toFixed(1)}`}</title>
                  </g>
                );
              })}
              <line x1={0} x2={Math.min(width / 2 - 20, 440)} y1={80} y2={80} stroke="#b9b5aa" />
              <text x={0} y={95} className="ticklabel">tokens →</text>
            </svg>
            <div className="caption">
              The last token (amber) is usually written to more strongly than the
              others: it is the one position that has to produce a next-token
              prediction.
            </div>
          </div>
          <div className="stack">
            <Stat k="Attention share of this block's output" v={pct(n.attn[sel]?.reduce((a, b) => a + b, 0) / (n.delta[sel]?.reduce((a, b) => a + b, 0) || 1), 0)} s="L2 mass, all positions" />
            <Stat k="mean cosine(L, L+1)" v={fmt(n.cosine[sel]?.reduce((a, b) => a + b, 0) / prompt.n, 3)} s="direction preserved" />
            <Stat k="residual norm in → out" v={`${fmt(n.norm[sel]?.[prompt.n - 1] ?? 0, 0)} → ${fmt(n.norm[sel + 1]?.[prompt.n - 1] ?? 0, 0)}`} s="at the final position" />
            <Stat k="DLA, attention" v={fmt(d.attn[sel], 3)} s={`${d.attn[sel] > 0 ? "supports" : "opposes"} “${prompt.target.trim()}”`} />
          </div>
        </div>
      </Panel>

      <Prose>
        <p>
          Two facts fall out of this view immediately. First, direct attribution is{" "}
          <strong>sparse and signed</strong>: a handful of blocks do most of the
          work for any given answer, and several blocks push actively against it.
          Second, the largest writers are not always the most useful writers — a
          block can pour a large vector into the residual stream that is orthogonal
          to the answer.
        </p>
        <p>
          The peak contributor for this prompt is <strong>L{bestLayer}</strong>.
          Notice how attention and MLP contributions trade off with depth: early
          blocks are attention-heavy (routing token identity), middle blocks are
          MLP-heavy (retrieval), and the last few blocks are dominated by the
          output direction itself. Chapter 13 turns this into a full layer
          fingerprint.
        </p>
      </Prose>

      <Method>
        <p>
          DLA here is <code>⟨component_output, γ ⊙ W_U[target]⟩ / rms(h_final)</code>.
          It is the first-order effect of that component's output on the target
          logit, holding everything else fixed. It ignores the derivative of the
          final RMSNorm with respect to the component, which is a small correction
          but not zero.
        </p>
        <p>
          Residual norm is the L2 norm of the hidden state, per layer, per position.
          It is not "information content" — a large norm can be a single dominant
          coordinate (often a saliency/bias direction). Chapter 07 examines that
          coordinate directly.
        </p>
      </Method>

      <Footnotes
        items={[
          "Blocks are pre-norm: <code>h ← h + attn(ln(h))</code>, then <code>h ← h + mlp(ln(h))</code>. The heat map shows the state <em>after</em> each block.",
          `For this prompt the net attributed logit mass is ${fmt(netsum, 2)}, against a final target probability of ${pct(prompt.targetProb, 1)}. These are different quantities and should not be expected to match.`,
          "Attention share is computed as Σ‖attn_out‖ / Σ‖attn_out + mlp_out‖ over all positions of this prompt; it is a rough split, not an exact decomposition of information.",
        ]}
      />
      <PageFoot index={3} />
    </>
  );
}
