import React from "react";
import { Atlas, CHAPTERS, href, pct } from "../lib/data";
import { Caption, Chip, Footnotes, PageFoot, Panel, Prose, Stat } from "../components/ui";

export default function Overview({ atlas }: { atlas: Atlas }) {
  const { meta, prompts, ablation, probes, geometry } = atlas;
  const kindCounts: Record<string, number> = {};
  for (const p of prompts) kindCounts[p.group] = (kindCounts[p.group] ?? 0) + 1;
  const meanTarget = prompts.reduce((a, p) => a + p.targetProb, 0) / prompts.length;

  return (
    <>
      <header className="hero">
        <div>
          <div className="kicker">An interactive atlas of one model's depth</div>
          <h1 className="title">
            Fifteen ways to look <em>inside</em> Qwen
          </h1>
          <p className="lead">
            A {meta.layers}-layer language model is not a black box so much as a very tall
            building with the lights off. This report walks the same {meta.corpus}{" "}
            prompts through fifteen different instruments — lens, microscope, circuit
            tracer, knockout rig — and lets you operate each one yourself.
          </p>
          <div className="byline">
            <Chip tone="amber">{meta.modelName}</Chip>
            <Chip>{meta.layers} layers</Chip>
            <Chip>{meta.tokens.toLocaleString()} measured tokens</Chip>
            <Chip tone="cyan">{meta.generatedAt.slice(0, 10)}</Chip>
          </div>
        </div>
        <div className="hero-side">
          <div className="grid2">
            <Stat k="Corpus" v={meta.corpus} s="hand-built probe prompts" />
            <Stat k="Forward passes" v={meta.corpus + meta.layers * 3} s="capture + ablation sweep" />
            <Stat k="Target mass" v={pct(meanTarget, 0)} s="mean final P(answer)" />
            <Stat k="Effective rank" v={geometry.effectiveRank[meta.layers - 1].toFixed(0)} s={`at layer ${meta.layers - 1}`} />
          </div>
          <Caption>
            All values are measured from this checkpoint, not illustrative. The
            suite exports raw arrays; this page only renders them.
          </Caption>
        </div>
      </header>

      <h2 className="sec">The premise</h2>
      <Prose>
        <p>
          Interpretability usually stops at the activation: <em>this direction
          means "female", this one means "code"</em>. That tells you what the
          model is thinking about, not how the thought was computed. The harder
          and more useful question is <strong>what job each layer is doing</strong> —
          and the honest answer is that the job changes with depth, sometimes
          several times within a single block.
        </p>
        <p>
          So instead of one heroic visualisation, this atlas uses fourteen instruments
          that disagree with each other in useful ways. A layer that looks
          important in the logit lens can be nearly inert under ablation; a layer
          that writes nothing to the output can still be where a probe finds the
          answer. Each chapter is a different measurement of <em>the same{" "}
          {meta.layers} blocks</em>, and each one is interactive, because the interesting
          structure is usually at the boundary between two of them.
        </p>
      </Prose>

      <div className="cards">
        {CHAPTERS.slice(1).map((c) => (
          <a className="card" key={c.slug} href={href(c.slug)}>
            <span className="n">CH {c.n}</span>
            <span className="t">{c.title}</span>
            <span className="d">{c.desc}</span>
          </a>
        ))}
      </div>

      <h2 className="sec">The model, exactly</h2>
      <div className="grid4">
        <Stat k="Checkpoint" v={meta.modelName} s={meta.modelId} />
        <Stat k="Layers" v={meta.layers} s="pre-norm transformer blocks" />
        <Stat k="Residual width" v={meta.hidden} s={`${meta.heads} query heads, ${meta.kvHeads} KV heads`} />
        <Stat k="Vocabulary" v={meta.vocab.toLocaleString()} s="byte-level BPE, tied embeddings" />
      </div>
      <Caption>
        A {meta.modelName} checkpoint. These artefacts were produced by the GPU
        kernel in <code>experiments/kaggle</code>; the identical script also runs
        on CPU against Qwen2.5-0.5B, which is how the pipeline was developed.{" "}
        <b>The pipeline is model-agnostic</b> — every layer count and index on
        this page is read from the measured data, not typed in.
      </Caption>

      <h2 className="sec">Experiment design</h2>
      <Prose>
        <p>
          A layer's function cannot be read off its weights; it has to be provoked.
          The corpus is therefore not random text. It is built as a set of{" "}
          <strong>controlled contrasts</strong> so that different computations are
          forced to happen at measurable moments:
        </p>
        <ul>
          <li>
            <strong>Factual recall</strong> — a single answer token must be
            retrieved, so we can watch the retrieval happen at some layer.
          </li>
          <li>
            <strong>Syntax</strong> — subject–verb agreement with a distracting
            noun between subject and verb: surface statistics and structure
            disagree, so the layer that settles the disagreement is identifiable.
          </li>
          <li>
            <strong>Induction</strong> — repeated random tokens: copying is the
            only usable algorithm, which isolates the induction circuitry.
          </li>
          <li>
            <strong>Arithmetic and code</strong> — algorithmic continuations where
            the correct next token is not a memorised association.
          </li>
          <li>
            <strong>Multilingual, negation, idiom</strong> — three ways a
            surface-level answer is the wrong one.
          </li>
        </ul>
        <p>
          Each prompt is measured under six instruments simultaneously: the{" "}
          <strong>logit lens</strong> at every layer, <strong>direct logit
          attribution</strong> of every block's output, <strong>full attention
          weights</strong> at every head, <strong>MLP activation maxima</strong>,
          and the raw <strong>residual-stream geometry</strong>. Then each block is{" "}
          <strong>knocked out</strong> and the damage to next-token loss recorded —
          a causal check on everything the correlational views suggest.
        </p>
      </Prose>

      <Panel
        title={<><b>Figure 0.1</b> · the corpus, as measured</>}
        tools={<span className="chip">{prompts.length} prompts</span>}
        tight
        note={
          <>
            P(answer) is the model's own probability for the target token after the
            final layer. Prompts where the model is unconfident are the interesting
            ones: the lens has further to travel.
          </>
        }
      >
        <div className="scroll">
          <table className="tbl">
            <thead>
              <tr>
                <th>id</th>
                <th>kind</th>
                <th>prompt</th>
                <th>target</th>
                <th>P(answer)</th>
                <th>model's top-1</th>
                <th>tokens</th>
              </tr>
            </thead>
            <tbody>
              {prompts.map((p) => (
                <tr key={p.id}>
                  <td className="k">{p.id}</td>
                  <td>{p.group}</td>
                  <td style={{ fontFamily: "var(--sans)", fontSize: 12, color: "var(--ink2)", maxWidth: 380, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                    {p.text.replace(/\n/g, " ⏎ ")}
                  </td>
                  <td style={{ color: "var(--amber)" }}>{p.target}</td>
                  <td>{pct(p.targetProb, 1)}</td>
                  <td>{p.finalTop1.trim() || p.finalTop1}</td>
                  <td>{p.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <h2 className="sec">What is actually stored</h2>
      <Prose>
        <p>
          Nothing on these pages is hand-authored data. The pipeline writes
          twelve JSON artefacts, and every chart is a pure function of them:
          per-layer top-12 distributions for all {meta.corpus} prompts, per-head
          attention rows, {meta.layers} × 3 ablation conditions ×{" "}
          {ablation.prompts.length} prompts of loss damage, {probes.properties.length}{" "}
          probe curves on a shared hidden-state matrix, and a {meta.layers + 1} ×{" "}
          {meta.layers + 1} CKA matrix.
        </p>
      </Prose>
      <div className="grid3">
        <Stat k="Probe properties" v={probes.properties.length} s="linear probes per layer" />
        <Stat k="Ablation conditions" v={meta.layers * 3} s={`${meta.layers} layers × {attn, mlp, block}`} />
        <Stat k="Baseline NLL" v={ablation.baseline.toFixed(3)} s="on the ablation subset" />
      </div>

      <Footnotes
        items={[
          `Model: <code>${meta.modelId}</code>, eager attention, measured by <code>experiments/suite.py</code>. Layer indices run 0…${meta.layers - 1}; <code>L${meta.layers}</code> denotes the final normed state before the unembedding.`,
          "The logit lens applies the model's final RMSNorm and unembedding to every intermediate residual state. This is an approximation — intermediate states were not trained to be decodable — but it is a consistent one, and its failures are themselves informative.",
          "Direct logit attribution (DLA) projects a component's output onto the unembedding direction of the target token, scaled by the final norm. It is a first-order measure and ignores the derivative of the norm itself.",
          "Ablation zeroes a module's output at every position and re-measures next-token NLL. Batch padding is right-sided; the loss is read at each row's last real token, so padding cannot contribute.",
          "Probes are <code>StandardScaler → LogisticRegression(C=0.1)</code> with GroupKFold(5) grouped by prompt: a probe never sees a prompt it was trained on.",
        ]}
      />
      <PageFoot index={0} />
    </>
  );
}
