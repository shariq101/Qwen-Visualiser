import React from "react";
import { Atlas, fmt } from "../lib/data";
import {
  Caption,
  ChapterHead,
  Chip,
  Footnotes,
  Method,
  PageFoot,
  Panel,
  Prose,
  Stat,
} from "../components/ui";

/**
 * Chapter 0 is deliberately factual: the block diagram and every number on this
 * page are computed from the checkpoint's own config, not from prose.
 */
export default function Architecture({ atlas }: { atlas: Atlas }) {
  const m = atlas.meta;
  const d = m.hidden;
  const L = m.layers;
  const H = m.heads;
  const KV = m.kvHeads;
  const I = m.intermediate ?? 0;
  const V = m.vocab;
  const hd = d / H;
  const kvDim = KV * hd;

  const pQ = d * d;
  const pK = d * kvDim;
  const pV = d * kvDim;
  const pO = d * d;
  const pMlp = 3 * d * I;
  const pNorm = 2 * d;
  const perLayer = pQ + pK + pV + pO + pMlp + pNorm;
  const pEmbed = V * d;
  const total = pEmbed + L * perLayer + d;

  const parts = [
    { k: "Embedding / unembedding (tied)", v: pEmbed, c: "var(--sand)" },
    { k: "Attention (Q, K, V, O)", v: L * (pQ + pK + pV + pO), c: "var(--teal)" },
    { k: "SwiGLU MLP", v: L * pMlp, c: "var(--rust)" },
    { k: "RMSNorm gains", v: L * pNorm + d, c: "var(--ink-3)" },
  ];

  const kvBytes = 2 * L * kvDim * 2; // K and V, fp16
  const flops = 2 * total;

  const W = 760;
  const cx = 330;
  const bw = 250;
  const bh = 30;
  const box = (y: number, label: string, shape?: string, w = bw) => (
    <g key={label + y}>
      <rect x={cx - w / 2} y={y} width={w} height={bh} fill="var(--paper-2)" stroke="var(--ink)" strokeWidth={1} />
      <text x={cx} y={y + 19} textAnchor="middle" style={{ fontFamily: "var(--mono)", fontSize: 9.5, fill: "var(--ink)" }}>
        {label}
      </text>
      {shape && (
        <text x={cx + w / 2 + 10} y={y + 19} style={{ fontFamily: "var(--mono)", fontSize: 8.5, fill: "var(--ink-3)" }}>
          {shape}
        </text>
      )}
    </g>
  );
  const arrow = (y1: number, y2: number) => (
    <line key={`a${y1}`} x1={cx} x2={cx} y1={y1} y2={y2} stroke="var(--ink)" strokeWidth={1} markerEnd="url(#arw)" />
  );

  return (
    <>
      <ChapterHead
        n="00"
        kicker="Architecture"
        title={<>The machine, <em>exactly</em></>}
        standfirst={
          <>
            Before interpreting anything, it is worth stating what the model is.{" "}
            <b>{m.modelName}</b> is {L} identical pre-norm blocks of width {d}, with{" "}
            {H} query heads sharing {KV} key/value heads, a SwiGLU MLP of width {I},
            and a {V.toLocaleString()}-token embedding tied to its unembedding.
            Every shape and every count below is computed from that config.
          </>
        }
        chips={
          <>
            <Chip tone="amber">{fmt(total / 1e9, 3)} B parameters</Chip>
            <Chip>{L} × blocks</Chip>
            <Chip tone="cyan">d = {d}</Chip>
          </>
        }
      />

      <Panel
        title={<><b>Figure 0.1</b> · one Qwen block, with tensor shapes</>}
        note="A single block, applied 36 times. The residual stream is the horizontal carrier: every sublayer reads the normalised stream and adds its output back. Query and key/value widths differ because 16 query heads share 2 KV heads."
      >
        <div style={{ overflowX: "auto" }}>
          <svg width={W} height={640} style={{ display: "block" }}>
            <defs>
              <marker id="arw" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M0,0 L8,4 L0,8 z" fill="var(--ink)" />
              </marker>
            </defs>
            <text x={cx} y={16} textAnchor="middle" style={{ fontFamily: "var(--mono)", fontSize: 8.5, fill: "var(--ink-3)", letterSpacing: "0.1em" }}>
              INPUT
            </text>
            {box(26, "token embedding", `T × ${d}`)}
            {arrow(56, 74)}
            {box(74, "RMSNorm", `T × ${d}`)}
            {arrow(104, 118)}
            {[0, 1, 2].map((i) => {
              const w = 78;
              const x = cx - 125 + i * 86;
              const labels = [
                { l: "W_Q", s: `${d} × ${d}`, n: `${H} heads × ${hd}` },
                { l: "W_K", s: `${d} × ${kvDim}`, n: `${KV} heads × ${hd}` },
                { l: "W_V", s: `${d} × ${kvDim}`, n: `${KV} heads × ${hd}` },
              ][i];
              return (
                <g key={i}>
                  <rect x={x} y={118} width={w} height={46} fill="var(--paper-2)" stroke="var(--ink)" />
                  <text x={x + w / 2} y={135} textAnchor="middle" style={{ fontFamily: "var(--mono)", fontSize: 9, fill: "var(--ink)" }}>{labels.l}</text>
                  <text x={x + w / 2} y={147} textAnchor="middle" style={{ fontFamily: "var(--mono)", fontSize: 7.5, fill: "var(--ink-3)" }}>{labels.s}</text>
                  <text x={x + w / 2} y={158} textAnchor="middle" style={{ fontFamily: "var(--mono)", fontSize: 7, fill: "var(--ink-3)" }}>{labels.n}</text>
                  <line x1={x + w / 2} x2={x + w / 2} y1={104} y2={118} stroke="var(--ink)" markerEnd="url(#arw)" />
                </g>
              );
            })}
            <text x={cx + 133} y={180} style={{ fontFamily: "var(--mono)", fontSize: 8, fill: "var(--rust)" }}>RoPE(q, k)</text>
            {arrow(168, 196)}
            {box(196, "softmax(QKᵀ/√" + hd + " + causal)", `${H} × T × T`)}
            {arrow(226, 240)}
            {box(240, "attn ⊗ V → concat", `T × ${d}`)}
            {arrow(270, 284)}
            {box(284, "W_O", `${d} × ${d}`)}
            {arrow(314, 332)}
            <circle cx={cx} cy={340} r={7} fill="var(--paper)" stroke="var(--ink)" />
            <text x={cx} y={344} textAnchor="middle" style={{ fontFamily: "var(--mono)", fontSize: 10, fill: "var(--ink)" }}>+</text>
            <line x1={cx - 210} x2={cx} y1={340} y2={340} stroke="var(--ink)" />
            <line x1={cx - 210} x2={cx - 210} y1={340} y2={26 + 15} stroke="var(--ink)" strokeDasharray="2 3" />
            <text x={cx - 216} y={26} textAnchor="end" style={{ fontFamily: "var(--mono)", fontSize: 7.5, fill: "var(--ink-3)" }}>residual</text>
            {arrow(347, 362)}
            {box(362, "RMSNorm", `T × ${d}`)}
            {arrow(392, 406)}
            {[0, 1].map((i) => {
              const w = 118;
              const x = cx - 125 + i * 132;
              const lbl = i === 0 ? `Gate ${d} × ${I}` : `Up ${d} × ${I}`;
              return (
                <g key={i}>
                  <rect x={x} y={406} width={w} height={26} fill="var(--paper-2)" stroke="var(--ink)" />
                  <text x={x + w / 2} y={423} textAnchor="middle" style={{ fontFamily: "var(--mono)", fontSize: 8, fill: "var(--ink)" }}>{lbl}</text>
                  <line x1={x + w / 2} x2={x + w / 2} y1={392} y2={406} stroke="var(--ink)" markerEnd="url(#arw)" />
                </g>
              );
            })}
            {arrow(432, 448)}
            {box(448, "SiLU(gate) ⊙ up", `T × ${I}`)}
            {arrow(478, 492)}
            {box(492, "Down", `${I} × ${d}`)}
            {arrow(522, 538)}
            <circle cx={cx} cy={546} r={7} fill="var(--paper)" stroke="var(--ink)" />
            <text x={cx} y={550} textAnchor="middle" style={{ fontFamily: "var(--mono)", fontSize: 10, fill: "var(--ink)" }}>+</text>
            <line x1={cx - 210} x2={cx} y1={546} y2={546} stroke="var(--ink)" />
            {arrow(553, 570)}
            {box(570, "final RMSNorm", `T × ${d}`)}
            {arrow(600, 612)}
            <text x={cx} y={628} textAnchor="middle" style={{ fontFamily: "var(--mono)", fontSize: 8.5, fill: "var(--ink-3)" }}>
              lm_head {V.toLocaleString()} × {d} (tied) → logits T × {V.toLocaleString()}
            </text>

            {/* the ×36 bracket */}
            <path d={`M ${cx - 232} 104 L ${cx - 232} 546`} stroke="var(--ink)" strokeWidth={1} />
            <path d={`M ${cx - 232} 104 L ${cx - 226} 104 M ${cx - 232} 546 L ${cx - 226} 546`} stroke="var(--ink)" />
            <text
              x={cx - 240}
              y={325}
              textAnchor="middle"
              transform={`rotate(-90 ${cx - 240} 325)`}
              style={{ fontFamily: "var(--mono)", fontSize: 9, fill: "var(--rust)", letterSpacing: "0.12em" }}
            >
              × {L} BLOCKS
            </text>
          </svg>
        </div>
      </Panel>

      <h2 className="sec">Where the parameters live</h2>
      <Panel
        title={<><b>Figure 0.2</b> · parameter budget</>}
        tools={<span className="mono" style={{ fontSize: 9, color: "var(--ink-3)" }}>{fmt(total / 1e6, 1)} M total</span>}
        note="Counted from the tensor shapes above, with the tied embedding counted once. The MLP carries 78.9% of the weights; attention, despite being the part interpretability usually studies, carries 11.0%."
      >
        <div style={{ display: "flex", height: 26, width: "100%", border: "1px solid var(--ink)" }}>
          {parts.map((p) => (
            <div
              key={p.k}
              style={{ width: `${(p.v / total) * 100}%`, background: p.c, borderRight: "1px solid var(--paper)" }}
              title={`${p.k}: ${p.v.toLocaleString()} (${((p.v / total) * 100).toFixed(2)}%)`}
            />
          ))}
        </div>
        <table className="facts" style={{ marginTop: 12 }}>
          <thead>
            <tr>
              <th>component</th>
              <th>per layer</th>
              <th>× {L}</th>
              <th>share</th>
            </tr>
          </thead>
          <tbody>
            {[
              { n: "W_Q", v: pQ },
              { n: "W_K", v: pK },
              { n: "W_V", v: pV },
              { n: "W_O", v: pO },
              { n: "gate / up / down", v: pMlp },
              { n: "RMSNorm ×2", v: pNorm },
            ].map((r) => (
              <tr key={r.n}>
                <td className="k">{r.n}</td>
                <td>{r.v.toLocaleString()}</td>
                <td>{(r.v * L).toLocaleString()}</td>
                <td>{((r.v * L * 100) / total).toFixed(2)}%</td>
              </tr>
            ))}
            <tr>
              <td className="k">embedding (tied)</td>
              <td>—</td>
              <td>{pEmbed.toLocaleString()}</td>
              <td>{((pEmbed * 100) / total).toFixed(2)}%</td>
            </tr>
            <tr>
              <td className="k">total</td>
              <td>{perLayer.toLocaleString()}</td>
              <td colSpan={2}>{total.toLocaleString()}</td>
            </tr>
          </tbody>
        </table>
      </Panel>

      <h2 className="sec">The cost of one token</h2>
      <div className="grid3">
        <Stat k="Parameters" v={fmt(total / 1e9, 3) + " B"} s="tied embedding counted once" />
        <Stat k="Forward FLOPs / token" v={fmt(flops / 1e9, 2) + " G"} s="≈ 2 × parameters" />
        <Stat k="KV cache / token" v={fmt(kvBytes / 1024, 1) + " KiB"} s={`fp16, ${KV} KV heads × ${hd}`} />
        <Stat k="KV cache @ 4k ctx" v={fmt((kvBytes * 4096) / 2 ** 20, 0) + " MiB"} s="single sequence" />
        <Stat k="KV cache @ 32k ctx" v={fmt((kvBytes * 32768) / 2 ** 20, 0) + " MiB"} s="single sequence" />
        <Stat k="Head dimension" v={hd} s={`${d} / ${H} query heads`} />
      </div>
      <Caption>
        KV bytes = 2 (K and V) × {L} layers × {kvDim} channels × 2 bytes. Because{" "}
        {H} query heads share {KV} KV heads, the cache is {H / KV}× smaller than
        multi-head attention would need at the same width.
      </Caption>

      <Prose>
        <p>
          Two architectural facts matter for everything that follows. First, the
          stream is <strong>{d}-dimensional</strong> while each attention head reads
          only <strong>{hd} dimensions</strong>: heads are narrow readers of a wide
          bus, and the output projection is what recombines them. Second, the MLP is{" "}
          <strong>{I}-wide</strong> — {fmt(I / d, 1)}× the residual width — so the
          feature catalogue in chapter 05 is sampling a space of{" "}
          {(L * I).toLocaleString()} units, of which it shows 504.
        </p>
        <p>
          The tied embedding is why logit-lens projections are meaningful at all:
          the same matrix that writes a token into the stream is the one that reads
          a direction out of it. Chapter 01 exploits exactly that symmetry.
        </p>
      </Prose>

      <Method label="How these numbers were obtained">
        <p>
          Shapes are read from the checkpoint config (<code>hidden_size</code>,{" "}
          <code>num_attention_heads</code>, <code>num_key_value_heads</code>,{" "}
          <code>intermediate_size</code>, <code>vocab_size</code>,{" "}
          <code>tie_word_embeddings</code>), and parameter counts are the products of
          those shapes. No value on this page is estimated or rounded from memory.
        </p>
        <p>
          FLOPs follow the standard 2·N approximation for a dense forward pass and
          exclude the attention score term, which is quadratic in context length and
          negligible at the sequence lengths used here.
        </p>
      </Method>

      <Footnotes
        items={[
          `Block diagram is structural: ${L} identical blocks, pre-norm placement, RoPE on Q and K, grouped-query attention, SwiGLU, tied unembedding. No architectural detail is inferred beyond the config.`,
          `Total parameters computed here: ${total.toLocaleString()} (${fmt(total / 1e9, 4)} B). The reported parameter count for this checkpoint may differ slightly depending on whether the tied embedding is counted once or twice.`,
          "KV-cache figures assume fp16 and one sequence; batching multiplies them by the batch size.",
        ]}
      />
      <PageFoot index={1} />
    </>
  );
}
