# Depth Atlas

Fifteen interactive, data-backed views of what every layer of a Qwen language
model computes. Everything rendered in the app is measured from the checkpoint —
there are no illustrative or fabricated values anywhere in `public/data/`.

```
npm ci
npm run dev        # http://localhost:3003
npm run build      # static build in dist/
npm run typecheck
```

The measured artefacts are committed in `public/data/`, so the site runs without
a model or a GPU. The build is fully static and uses a relative base, so it can
be served from any path.

## The model

The shipped data is **Qwen2.5-3B-Instruct** — 36 pre-norm blocks, d_model 2048,
16 query heads over 2 KV heads (GQA), SwiGLU at 11008, 151,936 vocabulary, tied
embeddings, 3.086 B parameters. It was produced by the Kaggle GPU kernel in
`experiments/kaggle` (float16, cuda).

The pipeline is model-agnostic and small enough to develop against a CPU
checkpoint. `MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct python experiments/suite.py all`
produced the first complete run, in under a minute per stage; the same script
with `MODEL_ID=Qwen/Qwen2.5-3B-Instruct DTYPE=float16` produced the shipped run.

## The corpus

38 hand-built prompts, grouped into controlled contrasts so that different
computations are forced to happen at measurable moments: factual recall, syntax
(subject–verb agreement with attractors), induction/copying, arithmetic, code,
multilingual lookup, negation, idiom, and long-context retrieval. See
`experiments/corpus.py`. The ablation sweep uses a 14-prompt single-line subset.

## The stages

`experiments/suite.py` has five stages, each cached under `$QWEN_CACHE`
(default `/tmp/qwen_cache`) so they can be re-run independently.

| stage | what it measures | how |
|---|---|---|
| `capture` | residual norms, logit-lens distributions (all positions for deep dives), per-head attention rows, per-head behaviour scores, direct logit attribution, MLP neuron maxima, residual activations | one forward pass per prompt, hooks on `self_attn`, `mlp.down_proj`, and each block |
| `ablate` | causal effect of each layer | 3 × L forward passes with a module output zeroed; Δ next-token NLL |
| `analyse` | linear probes per layer; CKA, effective rank, drift, shared-basis PCA | `LogisticRegression` with `GroupKFold(5)` grouped by prompt |
| `graph` | attribution graph for 4 deep-dive prompts | one forward + backward; `∂logit/∂h · h`, attention-flow edges |
| `build` | compact JSON for the site | rounding and reshaping only |

## The chapters

0. **Architecture** — the block diagram with every tensor shape, and the parameter budget computed from the config.
1. **Logit lens** — the answer forming across depth; per-layer spectra, per-position river.
2. **Layer anatomy** — what each block writes (attention vs MLP), residual norms, cosine drift.
3. **Attention atlas** — every head's row, plus a measured taxonomy (previous-token, sink, induction).
4. **Circuit** — attribution graph with real edge weights and a gradient×activation matrix.
5. **Feature catalogue** — top MLP neurons with their real max-activating contexts.
6. **Capability emergence** — balanced accuracy of linear probes by depth.
7. **Geometry** — CKA between layers, effective rank, token trajectories in a shared PCA basis.
8. **Ablation lab** — knockout damage, per-prompt breakdown, and the attn+MLP ≠ block result.
9. **Probability landscape** — rank churn, layer-to-layer KL, distribution comparison.
10. **Attention budget** — where each layer's attention mass goes, by key position and by distance.
11. **Logit waterfall** — every block's signed, additive contribution to the target logit.
12. **Commitment atlas** — the layer at which each prompt's answer first becomes the argmax.
13. **Function map** — all measurements normalised and clustered into depth regimes.

## Deploying to GitHub Pages

The repository ships a workflow at `.github/workflows/deploy.yml` that builds the
site and publishes `dist/` on every push to `main`.

1. Create a repository and push this folder to `main`.
2. Open **Settings → Pages** and set **Source** to **GitHub Actions**.
3. The workflow runs `npm ci && npm run build`, uploads `dist/` and deploys it.

`vite.config.ts` sets `base: "./"`, so the same build works at a project page
(`user.github.io/repo/`) or at a custom domain without edits. `public/.nojekyll`
stops Pages from ignoring the hashed asset files. There are no API keys,
analytics or server-side components: the site is HTML, CSS, JS and the JSON in
`public/data/`.

To preview the production build locally:

```
npm run build
npm run preview     # http://localhost:3004
```

## Kaggle

`experiments/make_kernel.py` concatenates `corpus.py` and `suite.py` into a
single-file Kaggle script and writes its metadata:

```
python experiments/make_kernel.py
kaggle kernels push -p experiments/kaggle
kaggle kernels status skrik001/depth-atlas-qwen
kaggle kernels output skrik001/depth-atlas-qwen -p /tmp/atlas_out
```

The kernel writes the twelve JSON artefacts into `/kaggle/working/data`; copy
them over `public/data/` to swap the site onto a different checkpoint.

## Limits

The corpus is small (38 prompts, 412 tokens), so probe curves and CKA values are
noisy. Ablation removes a module everywhere rather than where it matters. The
attribution graph is first-order and its edges are not guaranteed to compose.
These are stated on the relevant pages rather than in a footnote here.
