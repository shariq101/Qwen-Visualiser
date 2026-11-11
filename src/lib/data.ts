import { useEffect, useMemo, useRef, useState } from "react";

/* ------------------------------------------------------------------ types */
export type Meta = {
  modelId: string;
  modelName: string;
  layers: number;
  hidden: number;
  heads: number;
  kvHeads: number;
  vocab: number;
  corpus: number;
  tokens: number;
  generatedAt: string;
  tied: boolean;
  intermediate?: number;
  modelType?: string;
  params?: number;
  evidence: string;
};

export type Prompt = {
  id: string;
  text: string;
  target: string;
  targetStr: string;
  tokens: string[];
  n: number;
  finalTop1: string;
  top1Prob: number;
  targetProb: number;
  group: string;
};

export type LensEntry = {
  tokens: string[];
  /** [layer][k] -> [token string, probability] */
  top: [string, number][][];
  entropy: number[];
  targetProb: number[];
  targetRank: number[];
  top1Prob: number[];
  kl: number[];
  finalTop1: string;
  targetStr: string;
  river:
    | {
        top1: string[];
        p: number[];
        pt: number[];
        ent: number;
      }[]
    | null;
};

export type AttnEntry = {
  tokens: string[];
  shape: [number, number, number];
  heads: number[][][];
  mean: number[][];
};

export type DlaEntry = {
  attn: number[];
  mlp: number[];
  resid: number[];
  targetStr: string;
};

export type NormsEntry = {
  tokens: string[];
  norm: number[][];
  delta: number[][];
  attn: number[][];
  cosine: number[][];
  n: number;
};

export type HeadStat = {
  l: number;
  h: number;
  prev: number;
  self: number;
  sink: number;
  ent: number;
  ind: number;
  copy: number;
};

export type AblationEntry = {
  nll: number;
  delta: number;
  perPrompt: number[];
};
export type Ablation = {
  baseline: number;
  prompts: string[];
  kinds: string[];
  attn: Record<string, AblationEntry>;
  mlp: Record<string, AblationEntry>;
  block: Record<string, AblationEntry>;
};

export type Probes = {
  properties: { key: string; label: string; chance: number }[];
  layers: number[];
  byProp: Record<string, number[]>;
  std: Record<string, number[]>;
};

export type Geometry = {
  cka: number[][];
  effectiveRank: number[];
  drift: number[];
  pca: Record<string, { tokens: string[]; points: [number, number][][] }>;
  centroids: [number, number][];
};

export type GraphNode = {
  id: string;
  layer: number;
  pos: number;
  token: string;
  attr: number;
};
export type GraphEdge = {
  s: string;
  t: string;
  w: number;
  kind: "attn" | "mlp";
  layer: number;
  q?: number;
  k?: number;
};
export type GraphEntry = {
  tokens: string[];
  target: string;
  attr: number[][];
  dlaAttn: number[][];
  dlaMlp: number[][];
  nodes: GraphNode[];
  edges: GraphEdge[];
  L: number;
};

export type Neuron = {
  layer: number;
  neuron: number;
  max: number;
  mean: number;
  density: number;
  contexts: {
    prompt: string;
    pos: number;
    token: string;
    before: string;
    act: number;
  }[];
};

export type Atlas = {
  meta: Meta;
  prompts: Prompt[];
  lens: Record<string, LensEntry>;
  attention: Record<string, AttnEntry>;
  dla: Record<string, DlaEntry>;
  norms: Record<string, NormsEntry>;
  heads: HeadStat[];
  ablation: Ablation;
  probes: Probes;
  geometry: Geometry;
  graph: Record<string, GraphEntry>;
  neurons: Neuron[];
};

/* --------------------------------------------------------------- loading */
const FILES = [
  "meta",
  "prompts",
  "lens",
  "attention",
  "dla",
  "norms",
  "heads",
  "ablation",
  "probes",
  "geometry",
  "graph",
  "neurons",
] as const;

export function useAtlas() {
  const [atlas, setAtlas] = useState<Atlas | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(0);

  useEffect(() => {
    let live = true;
    (async () => {
      try {
        const results: Record<string, unknown> = {};
        await Promise.all(
          FILES.map(async (f) => {
            const res = await fetch(`${import.meta.env.BASE_URL}data/${f}.json`);
            if (!res.ok) throw new Error(`${f}.json -> HTTP ${res.status}`);
            results[f] = await res.json();
            if (live) setLoaded((n) => n + 1);
          }),
        );
        if (live) setAtlas(results as unknown as Atlas);
      } catch (e) {
        if (live) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      live = false;
    };
  }, []);

  return { atlas, error, loaded, total: FILES.length };
}

/* ---------------------------------------------------------------- routing */
export const CHAPTERS = [
  { slug: "", n: "—", title: "Overview", short: "Overview", desc: "What this is, how it was measured." },
  { slug: "architecture", n: "00", title: "The architecture", short: "Architecture", desc: "The Qwen block, exactly, with every tensor shape." },
  { slug: "lens", n: "01", title: "The lens", short: "Logit lens", desc: "The answer forming, layer by layer." },
  { slug: "anatomy", n: "02", title: "Anatomy of a depth", short: "Layer anatomy", desc: "What each layer adds to the residual stream." },
  { slug: "attention", n: "03", title: "Attention atlas", short: "Attention", desc: "Where every head looks, and what kind of head it is." },
  { slug: "circuit", n: "04", title: "The circuit", short: "Attribution graph", desc: "A traceable path from tokens to a logit." },
  { slug: "features", n: "05", title: "Feature catalogue", short: "Features", desc: "Neurons that fire, and the contexts that fire them." },
  { slug: "emergence", n: "06", title: "Capability emergence", short: "Probing", desc: "When each property becomes linearly readable." },
  { slug: "geometry", n: "07", title: "Geometry of thought", short: "Geometry", desc: "Representations as a shape, not a list." },
  { slug: "ablation", n: "08", title: "Ablation lab", short: "Ablation", desc: "Knock a layer out and measure the damage." },
  { slug: "landscape", n: "09", title: "Probability landscape", short: "Landscape", desc: "Full distributions, not just the argmax." },
  { slug: "flow", n: "10", title: "Attention budget", short: "Attention budget", desc: "Where the mass goes, by layer and by distance." },
  { slug: "waterfall", n: "11", title: "The logit waterfall", short: "Waterfall", desc: "Every layer's signed contribution, accumulated." },
  { slug: "commitment", n: "12", title: "Commitment atlas", short: "Commitment", desc: "At which layer each answer becomes the argmax." },
  { slug: "function", n: "13", title: "The function map", short: "Synthesis", desc: "All measurements, one account of depth." },
] as const;

export function useRoute() {
  const read = () => {
    const h = window.location.hash.replace(/^#\/?/, "").split("?")[0];
    return h || "";
  };
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => {
      setRoute(read());
      window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export function href(slug: string) {
  return `#/${slug}`;
}

/* --------------------------------------------------------------- measures */
export function useMeasure<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [w, setW] = useState(720);
  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    setW(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  return { ref, width: w };
}

/* ------------------------------------------------------------------ utils */
export const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export function fmt(v: number, d = 2): string {
  if (!Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 100) return v.toFixed(1);
  return v.toFixed(d);
}
export function pct(v: number, d = 1): string {
  return `${(v * 100).toFixed(d)}%`;
}

/** Hex lerp between two colours. */
export function mix(a: string, b: string, t: number): string {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  const c = pa.map((v, i) => Math.round(lerp(v, pb[i], clamp(t, 0, 1))));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export function ramp(t: number, stops: [number, string][]): string {
  const x = clamp(t, 0, 1);
  for (let i = 0; i < stops.length - 1; i++) {
    const [p0, c0] = stops[i];
    const [p1, c1] = stops[i + 1];
    if (x >= p0 && x <= p1) return mix(c0, c1, (x - p0) / (p1 - p0 || 1));
  }
  return stops[stops.length - 1][1];
}

export const SEQ: [number, string][] = [
  [0, "#f4f1e8"],
  [0.2, "#e6d6b6"],
  [0.4, "#d3b177"],
  [0.6, "#bb8642"],
  [0.8, "#8f5620"],
  [1, "#4a2410"],
];

export const DIV: [number, string][] = [
  [0, "#1f5c6b"],
  [0.42, "#dce8e6"],
  [0.5, "#f7f5f0"],
  [0.58, "#f0dbcd"],
  [1, "#a8431f"],
];

/** Categorical ink set for series that must stay distinguishable in print. */
export const INK = {
  rust: "#a8431f",
  teal: "#1f5c6b",
  indigo: "#43478c",
  olive: "#6f6420",
  plum: "#6b3f6e",
  blue: "#2b5f8a",
  green: "#2f6b3f",
  sand: "#b58a4a",
};

export function quantile(vals: number[], q: number): number {
  if (!vals.length) return 0;
  const s = [...vals].sort((a, b) => a - b);
  const i = clamp(Math.floor(q * (s.length - 1)), 0, s.length - 1);
  return s[i];
}

export function mean(v: number[]): number {
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0;
}

export function tokenLabel(t: string): string {
  return t.replace(/Ġ/g, "␣").replace(/Ċ/g, "⏎").replace(/^##/, "");
}

/** Token text with whitespace markers stripped, for equality checks. */
export function bare(t: string): string {
  return tokenLabel(t).replace(/[␣⏎]/g, "").trim().toLowerCase();
}

export function useHover<T>(initial: T | null = null) {
  const [hover, setHover] = useState<T | null>(initial);
  return { hover, setHover };
}

/** Selection state shared by pages: a prompt id, with fallback. */
export function usePrompt(atlas: Atlas) {
  const params = useMemo(() => new URLSearchParams(window.location.hash.split("?")[1] ?? ""), []);
  // Default to the prompt whose answer rises most from the first layers to the
  // last one: the clearest example of the phenomenon chapter 01 is about. This
  // is computed from the measured lens, not hand-picked.
  const demo = useMemo(() => {
    let best = atlas.prompts[0];
    let bestGain = -Infinity;
    for (const p of atlas.prompts) {
      const l = atlas.lens[p.id];
      if (!l || p.n < 7 || !l.targetProb.length) continue;
      const early = Math.max(...l.targetProb.slice(0, 3));
      const gain = l.targetProb[l.targetProb.length - 1] - early;
      if (gain > bestGain) {
        bestGain = gain;
        best = p;
      }
    }
    return best;
  }, [atlas]);
  const [id, setId] = useState(params.get("p") || demo.id);
  const prompt = atlas.prompts.find((p) => p.id === id) ?? atlas.prompts[0];
  return [prompt, setId] as const;
}
