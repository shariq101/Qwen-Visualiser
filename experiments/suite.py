"""
Depth Atlas experiment suite.

Runs a real Qwen model over a probe corpus and records everything ten
different visualisations need:

  capture   one forward pass per prompt -> residual stream norms, logit-lens
            distributions, per-head attention, per-layer direct logit
            attribution (DLA), MLP neuron maxima, residual activations
  ablate    causal knockout of each layer's attention / MLP / whole block,
            measured as the change in next-token NLL on a short subset
  analyse   linear probes per layer (GroupKFold by prompt), CKA geometry,
            effective rank, token trajectories
  graph     gradient x activation attribution + attention-flow edges for the
            deep-dive prompts (an attribution graph)
  build     assembles compact JSON for the site

Usage:
    python experiments/suite.py all
    python experiments/suite.py capture
    MODEL_ID=Qwen/Qwen2.5-3B-Instruct python experiments/suite.py all

Everything is cached under /tmp/qwen_cache so stages can be rerun on their own.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import string
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import ABLATION_IDS, CORPUS, DEEP_DIVE_IDS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("ATLAS_OUT", str(ROOT / "public" / "data")))
CACHE = Path(os.environ.get("QWEN_CACHE", "/tmp/qwen_cache"))
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = os.environ.get("DTYPE", "float16" if DEVICE == "cuda" else "float32")
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_grad_enabled(False)

PROMPT_BY_ID = {p["id"]: p for p in CORPUS}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def r(x, n=4):
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, n)
    return x


def arr(x, n=4):
    return [r(float(v), n) for v in np.asarray(x).ravel()]


def arr2(x, n=3):
    """Round while preserving the 2-D shape (layers x positions)."""
    a = np.asarray(x)
    if a.ndim == 1:
        return [r(float(v), n) for v in a]
    return [[r(float(v), n) for v in row] for row in a]


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class Model:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        log(f"loading {MODEL_ID} ...")
        self.m = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=getattr(torch, DTYPE), attn_implementation="eager"
        ).to(DEVICE)
        self.m.eval()
        self.cfg = self.m.config
        self.L = self.cfg.num_hidden_layers
        self.H = self.cfg.num_attention_heads
        self.d = self.cfg.hidden_size
        self.I = self.cfg.intermediate_size
        self.V = self.cfg.vocab_size
        self.W_U = self.m.lm_head.weight.detach()  # [V, d]
        self.gamma = self.m.model.norm.weight.detach()  # [d]
        self.head_dim = self.d // self.H
        log(
            f"ready in {time.time() - t0:.0f}s on {DEVICE}/{DTYPE}: {self.L} layers, "
            f"d={self.d}, {self.H} heads, vocab={self.V}"
        )

    def encode(self, text: str):
        ids = self.tok(text, return_tensors="pt").input_ids.to(DEVICE)
        toks = self.tok.convert_ids_to_tokens(ids[0].tolist())
        return ids, toks

    def unembed(self, h: torch.Tensor) -> torch.Tensor:
        """Final-norm + unembedding, i.e. the logit lens."""
        return self.m.lm_head(self.m.model.norm(h))

    def u(self, tid: int) -> torch.Tensor:
        """Unembedding direction, scaled by the final RMSNorm gain."""
        return self.gamma * self.W_U[tid]


# ----------------------------------------------------------------------------
# stage: capture
# ----------------------------------------------------------------------------
def capture():
    CACHE.mkdir(parents=True, exist_ok=True)
    M = Model()
    m = M.m
    store: dict[int, dict[str, torch.Tensor]] = {}
    handles = []

    def make_hooks(i):
        def attn_hook(mod, inp, out):
            store.setdefault(i, {})["attn"] = out[0].detach()

        def down_pre(mod, inp):
            store.setdefault(i, {})["mlp_act"] = inp[0].detach()

        def block_hook(mod, inp, out):
            store.setdefault(i, {})["block"] = out[0].detach()

        return attn_hook, down_pre, block_hook

    for i, layer in enumerate(m.model.layers):
        a, d, b = make_hooks(i)
        handles.append(layer.self_attn.register_forward_hook(a))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(d))
        handles.append(layer.register_forward_hook(b))

    records = []
    t_start = time.time()
    for idx, p in enumerate(CORPUS):
        store.clear()
        ids, toks = M.encode(p["text"])
        T = ids.shape[1]
        t0 = time.time()
        out = m(ids, output_attentions=True, output_hidden_states=True)
        hs = out.hidden_states  # L+1 x [1,T,d]
        logits_last = out.logits[0, -1]

        # ---- target id (first token of the continuation) -------------------
        tgt_ids = M.tok(p["target"], add_special_tokens=False).input_ids
        target_id = int(tgt_ids[0])

        # ---- logit lens at the final position ------------------------------
        lens_all = []
        for h in hs:
            lg = M.unembed(h[0, -1:])[0]
            lens_all.append(lg)
        lens_all = torch.stack(lens_all)  # [L+1, V]
        probs = torch.softmax(lens_all, dim=-1)
        topv, topi = probs.topk(12, dim=-1)
        final_probs = probs[-1]
        final_top1 = int(final_probs.argmax())
        kl_to_final = []
        target_prob, target_rank, ent, top1_prob = [], [], [], []
        for l in range(M.L + 1):
            pl = probs[l].clamp_min(1e-12)
            ent.append(float(-(pl * pl.log()).sum()))
            target_prob.append(float(probs[l, target_id]))
            rank = int((probs[l] > probs[l, target_id]).sum())
            target_rank.append(rank)
            top1_prob.append(float(topv[l, 0]))
            kl = float(
                (final_probs.clamp_min(1e-12) * (final_probs.clamp_min(1e-12).log() - pl.log())).sum()
            )
            kl_to_final.append(kl)

        # ---- "river": logit lens for every position (deep dives only) ------
        river = None
        if p["id"] in DEEP_DIVE_IDS:
            rows = []
            for l in range(M.L + 1):
                lg = M.unembed(hs[l][0])  # [T, V]
                pr = torch.softmax(lg, dim=-1)
                tp, ti = pr.topk(1, dim=-1)
                rows.append(
                    {
                        "top1": [M.tok.decode([int(v)]) for v in ti[:, 0]],
                        "p": [r(float(v), 4) for v in tp[:, 0]],
                        "pt": [r(float(v), 4) for v in pr[:, target_id]],
                        "ent": r(float(-(pr.clamp_min(1e-12) * pr.clamp_min(1e-12).log()).sum(-1).mean()), 4),
                    }
                )
            river = rows

        # ---- direct logit attribution --------------------------------------
        h_final = hs[-1][0, -1]
        rms = float(torch.sqrt(h_final.pow(2).mean() + 1e-6))
        u = M.u(target_id)
        attn_dla, mlp_dla, resid_dla = [], [], []
        for i in range(M.L):
            a = store[i]["attn"][0, -1]
            blk = store[i]["block"][0, -1]
            mlp = blk - hs[i][0, -1] - a
            attn_dla.append(float(torch.dot(a, u) / rms))
            mlp_dla.append(float(torch.dot(mlp, u) / rms))
        for l in range(M.L + 1):
            resid_dla.append(float(torch.dot(hs[l][0, -1], u) / rms))

        # ---- attention, last query position ---------------------------------
        last_attn = torch.stack([a[0, :, -1, :] for a in out.attentions])  # [L,H,T]
        mean_attn = last_attn.mean(1)  # [L,T]

        # ---- per-head behaviour over every query position -------------------
        tok_ids = ids[0].tolist()
        head_stats = []
        if T >= 2:
            for a in out.attentions:  # [1,H,T,T]
                A = a[0].float().cpu().numpy()
                for h in range(A.shape[0]):
                    Mh = A[h]
                    prev = float(np.mean(np.diag(Mh, -1)))
                    self_ = float(np.mean(np.diag(Mh)))
                    sink = float(np.mean(Mh[:, 0]))
                    pw = np.clip(Mh, 1e-9, 1)
                    ent_h = float(np.mean(-(pw * np.log(pw)).sum(-1)))
                    ind_vals, copy_vals = [], []
                    for t in range(1, T):
                        earlier = [s for s in range(t) if tok_ids[s] == tok_ids[t]]
                        if earlier:
                            s0 = earlier[-1]
                            if s0 + 1 < t:
                                ind_vals.append(float(Mh[t, s0 + 1]))
                            copy_vals.append(float(Mh[t, s0]))
                    head_stats.append({
                        "prev": r(prev), "self": r(self_), "sink": r(sink),
                        "ent": r(ent_h),
                        "ind": r(float(np.mean(ind_vals)) if ind_vals else 0.0),
                        "copy": r(float(np.mean(copy_vals)) if copy_vals else 0.0),
                    })
        # ---- neuron maxima ---------------------------------------------------
        nmeta = []
        for i in range(M.L):
            act = store[i]["mlp_act"][0]  # [T, I]
            mx, am = act.max(0)
            nmeta.append({"max": mx.cpu().numpy(), "arg": am.cpu().numpy().astype(np.int16)})

        # ---- residual stream norms -------------------------------------------
        norms = np.stack([h[0].norm(dim=-1).cpu().numpy() for h in hs])  # [L+1,T]
        delta = np.stack(
            [(store[i]["block"][0] - hs[i][0]).norm(dim=-1).cpu().numpy() for i in range(M.L)]
        )
        attn_norm = np.stack([store[i]["attn"][0].norm(dim=-1).cpu().numpy() for i in range(M.L)])
        cosine = np.stack(
            [
                torch.nn.functional.cosine_similarity(hs[l][0], hs[l + 1][0], dim=-1).cpu().numpy()
                for l in range(M.L)
            ]
        )

        rec = {
            "id": p["id"],
            "tokens": toks,
            "text": p["text"],
            "target": p["target"],
            "targetId": target_id,
            "targetStr": M.tok.decode([target_id]),
            "n": T,
            "finalTop1": final_top1,
            "finalTop1Str": M.tok.decode([final_top1]),
            "lens": {
                "top": [
                    [[int(i), r(float(v))] for v, i in zip(topv[l].tolist(), topi[l].tolist())]
                    for l in range(M.L + 1)
                ],
                "entropy": [r(e) for e in ent],
                "targetProb": [r(v) for v in target_prob],
                "targetRank": target_rank,
                "top1Prob": [r(v) for v in top1_prob],
                "kl": [r(v) for v in kl_to_final],
            },
            "river": river,
            "dla": {"attn": [r(v) for v in attn_dla], "mlp": [r(v) for v in mlp_dla],
                    "resid": [r(v) for v in resid_dla]},
            "attnLast": arr(last_attn.float().cpu().numpy(), 4),
            "attnShape": [M.L, M.H, T],
            "residNorm": arr2(norms, 3),
            "blockDelta": arr2(delta, 3),
            "attnNorm": arr2(attn_norm, 3),
            "cosine": arr2(cosine, 4),
            "seconds": r(time.time() - t0, 2),
        }
        rec["headStats"] = head_stats
        records.append(rec)

        np.savez_compressed(
            CACHE / f"act_{p['id']}.npz",
            hs=np.stack([h[0].float().cpu().numpy() for h in hs]).astype(np.float16),
            attn=last_attn.float().cpu().numpy().astype(np.float16),
            nmax=[n["max"] for n in nmeta],
            narg=[n["arg"] for n in nmeta],
        )
        # checkpoint after every prompt so a long capture can be resumed
        (CACHE / "capture.json").write_text(json.dumps(records))
        log(f"capture {idx + 1}/{len(CORPUS)} {p['id']:14s} T={T:2d} {rec['seconds']:5.2f}s")

    for h in handles:
        h.remove()
    (CACHE / "capture.json").write_text(json.dumps(records))
    log(f"capture done in {time.time() - t_start:.0f}s, {len(records)} prompts")


# ----------------------------------------------------------------------------
# stage: ablate
# ----------------------------------------------------------------------------
def ablate():
    """Knock out one component at every position and measure NLL damage."""
    CACHE.mkdir(parents=True, exist_ok=True)
    M = Model()
    m = M.m
    keep = [p for p in CORPUS if p["id"] in ABLATION_IDS]
    texts = [p["text"] for p in keep]
    targets = [M.tok(p["target"], add_special_tokens=False).input_ids[0] for p in keep]

    batch = M.tok(texts, return_tensors="pt", padding=True)
    ids, mask = batch.input_ids.to(DEVICE), batch.attention_mask.to(DEVICE)
    last_idx = (mask.sum(1) - 1).cpu()  # index of the final real token per row
    tgt = torch.tensor(targets).to(DEVICE)
    log(f"ablation batch {tuple(ids.shape)}, {len(keep)} prompts")

    def nll():
        out = m(ids, attention_mask=mask)
        rows = torch.arange(len(keep), device=DEVICE)
        lg = out.logits[rows, last_idx.to(DEVICE)]
        lp = torch.log_softmax(lg, dim=-1)
        per = -lp[rows, tgt]
        return float(per.mean()), per.float().cpu().tolist()

    base, base_per = nll()
    log(f"baseline NLL {base:.4f}")

    rows = {k: {} for k in ("attn", "mlp", "block")}
    handles = []

    def zero_attn_hook(mod, inp, out):
        return (torch.zeros_like(out[0]),) + tuple(out[1:])

    def zero_mlp_hook(mod, inp, out):
        return torch.zeros_like(out)

    def zero_block_hook(mod, inp, out):
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + tuple(out[1:])
        return torch.zeros_like(out)

    for i in range(M.L):
        layer = m.model.layers[i]
        for kind, mod, fn in (
            ("attn", layer.self_attn, zero_attn_hook),
            ("mlp", layer.mlp, zero_mlp_hook),
            ("block", layer, zero_block_hook),
        ):
            h = mod.register_forward_hook(fn)
            v, per = nll()
            h.remove()
            rows[kind][str(i)] = {"nll": r(v), "delta": r(v - base),
                                  "perPrompt": [r(x - b, 4) for x, b in zip(per, base_per)]}
        log(f"ablate layer {i:2d}  attn {rows['attn'][str(i)]['delta']:+.3f}  "
            f"mlp {rows['mlp'][str(i)]['delta']:+.3f}  block {rows['block'][str(i)]['delta']:+.3f}")

    payload = {
        "baseline": r(base),
        "prompts": [p["id"] for p in keep],
        "kinds": ["attn", "mlp", "block"],
        **rows,
    }
    (CACHE / "ablation.json").write_text(json.dumps(payload))
    log("ablation done")


# ----------------------------------------------------------------------------
# stage: analyse (probes + geometry)
# ----------------------------------------------------------------------------
WORD_RE = re.compile(r"[A-Za-z0-9]+")


def token_labels(text: str, tokens: list[str], id_to_piece):
    """Shallow, objectively-defined labels for per-token linear probes."""
    words = [w.lower() for w in WORD_RE.findall(text)]
    seen = set()
    labels = {k: [] for k in
              ("punct", "digit", "space_start", "repeat", "next_repeat",
               "capitalized", "long_word")}
    for t, tok in enumerate(tokens):
        piece = id_to_piece[t]
        core = piece.replace("Ġ", " ").replace("Ċ", "\n")
        stripped = core.strip()
        labels["punct"].append(int(bool(stripped) and all(c in string.punctuation for c in stripped)))
        labels["digit"].append(int(any(c.isdigit() for c in stripped)))
        labels["space_start"].append(int(piece.startswith("Ġ")))
        labels["capitalized"].append(int(bool(stripped) and stripped[0].isupper()))
        labels["long_word"].append(int(len(stripped) > 4))
        word = stripped.lower()
        labels["repeat"].append(int(bool(word) and word in seen))
        if word:
            seen.add(word)
        nxt = id_to_piece[t + 1].replace("Ġ", " ").strip().lower() if t + 1 < len(tokens) else ""
        earlier = set(
            id_to_piece[j].replace("Ġ", " ").strip().lower() for j in range(t)
        )
        labels["next_repeat"].append(int(bool(nxt) and nxt in earlier))
    return labels


def analyse():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cap = json.loads((CACHE / "capture.json").read_text())

    # rebuild tokenizer only (cheap)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    hs_by_layer = None
    feats = []
    groups = []
    labels_all = {k: [] for k in
                  ("punct", "digit", "space_start", "repeat", "next_repeat",
                   "capitalized", "long_word")}
    token_meta = []

    for rec in cap:
        z = np.load(CACHE / f"act_{rec['id']}.npz")
        hs = z["hs"].astype(np.float32)  # [L+1, T, d]
        if hs_by_layer is None:
            hs_by_layer = [[] for _ in range(hs.shape[0])]
        for l in range(hs.shape[0]):
            hs_by_layer[l].append(hs[l])
        labels = token_labels(rec["text"], rec["tokens"], rec["tokens"])
        for k in labels_all:
            labels_all[k].extend(labels[k])
        groups.extend([rec["id"]] * hs.shape[1])
        token_meta.extend([{"p": rec["id"], "t": t} for t in range(hs.shape[1])])
        feats.append(hs)

    X_layers = [np.concatenate(v, axis=0) for v in hs_by_layer]  # each [N, d]
    groups = np.array(groups)
    log(f"probe matrix {X_layers[0].shape}, L+1={len(X_layers)}")

    props = [
        ("punct", "punctuation"),
        ("digit", "contains a digit"),
        ("space_start", "starts a word"),
        ("capitalized", "capitalised"),
        ("long_word", "token longer than 4 chars"),
        ("repeat", "token seen earlier in prompt"),
        ("next_repeat", "next token seen earlier"),
    ]
    gkf = GroupKFold(n_splits=5)
    probe_out = {"properties": [], "layers": list(range(len(X_layers))), "byProp": {}}
    for key, label in props:
        y = np.array(labels_all[key])
        # Balanced accuracy: mean per-class recall. Chance is 0.5 for a binary
        # property, and unlike raw accuracy it is not dominated by the majority
        # class when the property is rare.
        chance = 0.5
        accs, stds = [], []
        for l, X in enumerate(X_layers):
            acc_folds = []
            for tr, te in gkf.split(X, y, groups):
                if len(np.unique(y[tr])) < 2:
                    continue
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=400, C=0.1, class_weight="balanced"),
                )
                with np.errstate(all="ignore"):
                    clf.fit(X[tr], y[tr])
                pred = clf.predict(X[te])
                recalls = [float((pred[y[te] == c] == c).mean()) for c in (0, 1) if (y[te] == c).any()]
                acc_folds.append(float(np.mean(recalls)))
            accs.append(r(float(np.mean(acc_folds)), 4))
            stds.append(r(float(np.std(acc_folds)), 4))
        probe_out["byProp"][key] = {"acc": accs, "std": stds}
        probe_out["properties"].append({"key": key, "label": label, "chance": r(chance),
                                        "support": r(float(y.mean()), 3)})
        log(f"probe {key:12s} L0={accs[0]:.2f} Lmid={accs[len(accs)//2]:.2f} "
            f"Ltop={accs[-1]:.2f} chance={chance:.2f}")

    (CACHE / "probes.json").write_text(json.dumps(probe_out))

    # ---- geometry ----------------------------------------------------------
    def center(X):
        return X - X.mean(0, keepdims=True)

    C = [center(X) for X in X_layers]
    n = len(C)
    cka = np.zeros((n, n))
    gram = {}
    for i in range(n):
        for j in range(i, n):
            a, b = C[i], C[j]
            num = float(np.linalg.norm(a.T @ b, "fro") ** 2)
            den = float(np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro"))
            v = num / den if den > 0 else 0.0
            cka[i, j] = cka[j, i] = v

    # effective rank = exp(entropy of normalised singular values)
    eff = []
    for l in range(n):
        s = np.linalg.svd(X_layers[l], compute_uv=False)
        p = s / (s.sum() + 1e-9)
        eff.append(r(float(np.exp(-(p * np.log(p + 1e-12)).sum())), 3))
        log(f"effective rank L{l}: {eff[-1]}")

    # drift: cosine(last token state at layer l, final state)
    drift = []
    for l in range(n):
        cs = []
        for i, rec in enumerate(cap):
            T = len(rec["tokens"])
            a = X_layers[l][sum(len(c["tokens"]) for c in cap[:i]) + T - 1]
            b = X_layers[-1][sum(len(c["tokens"]) for c in cap[:i]) + T - 1]
            cs.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        drift.append(r(float(np.mean(cs)), 4))

    # PCA shared across layers on deep-dive prompts
    deep = [i for i, rec in enumerate(cap) if rec["id"] in DEEP_DIVE_IDS]
    idx_map = {}
    off = 0
    for i, rec in enumerate(cap):
        idx_map[rec["id"]] = (off, off + len(rec["tokens"]))
        off += len(rec["tokens"])
    Xp = np.concatenate([X_layers[l][np.concatenate([np.arange(*idx_map[cap[i]["id"]]) for i in deep])]
                         for l in range(n)], axis=0)
    Xp = Xp - Xp.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xp, full_matrices=False)
    basis = Vt[:2].T  # [d, 2]
    pca = {}
    for i in deep:
        rec = cap[i]
        a, b = idx_map[rec["id"]]
        pts = []
        for t in range(b - a):
            pts.append([[r(float(X_layers[l][a + t] @ basis[:, 0]), 2),
                         r(float(X_layers[l][a + t] @ basis[:, 1]), 2)] for l in range(n)])
        pca[rec["id"]] = {"tokens": rec["tokens"], "points": pts}
    centroids = []
    allidx = np.arange(off)
    for l in range(n):
        z = X_layers[l][allidx] @ basis
        centroids.append([r(float(z[:, 0].mean()), 2), r(float(z[:, 1].mean()), 2)])

    geo = {
        "cka": [[r(v) for v in row] for row in cka],
        "effectiveRank": eff,
        "drift": drift,
        "pca": pca,
        "centroids": centroids,
    }
    (CACHE / "geometry.json").write_text(json.dumps(geo))
    log("analysis done")


# ----------------------------------------------------------------------------
# stage: graph
# ----------------------------------------------------------------------------
def graph():
    M = Model()
    m = M.m
    cap = {rec["id"]: rec for rec in json.loads((CACHE / "capture.json").read_text())}
    out_graphs = {}

    for pid in DEEP_DIVE_IDS:
        rec = cap[pid]
        ids, toks = M.encode(rec["text"])
        M_ids = ids
        store: dict[int, dict[str, torch.Tensor]] = {}
        handles = []
        for i, layer in enumerate(m.model.layers):
            def mk(i):
                def a_hook(mod, inp, out):
                    store.setdefault(i, {})["attn"] = out[0]
                def b_hook(mod, inp, out):
                    store.setdefault(i, {})["block"] = out[0] if isinstance(out, tuple) else out
                return a_hook, b_hook
            ah, bh = mk(i)
            handles.append(layer.self_attn.register_forward_hook(ah))
            handles.append(layer.register_forward_hook(bh))

        torch.set_grad_enabled(True)
        out = m(ids, output_hidden_states=True, output_attentions=True)
        hs = out.hidden_states
        tgt = rec["targetId"]
        logit = out.logits[0, -1, tgt]
        grads = torch.autograd.grad(logit, hs)
        torch.set_grad_enabled(False)
        for h in handles:
            h.remove()

        T = ids.shape[1]
        h_final = hs[-1][0, -1]
        rms = float(torch.sqrt(h_final.pow(2).mean() + 1e-6))
        u = M.u(tgt)

        attr = np.zeros((M.L + 1, T))
        for l in range(M.L + 1):
            attr[l] = (hs[l][0] * grads[l][0]).sum(-1).detach().float().cpu().numpy()

        dla_attn = np.zeros((M.L, T))
        dla_mlp = np.zeros((M.L, T))
        for i in range(M.L):
            a = store[i]["attn"][0]  # [T, d]
            blk = store[i]["block"][0]
            mlp = blk - hs[i][0] - a
            dla_attn[i] = (a @ u / rms).detach().float().cpu().numpy()
            dla_mlp[i] = (mlp @ u / rms).detach().float().cpu().numpy()

        attns = torch.stack([a[0] for a in out.attentions]).mean(1).float().cpu().numpy()  # [L,T,T] mean head

        # nodes: (layer, position)
        nodes = {}
        for l in range(M.L + 1):
            for t in range(T):
                nodes[(l, t)] = {
                    "id": f"{l}:{t}", "layer": l, "pos": t,
                    "token": rec["tokens"][t],
                    "attr": r(float(attr[l, t]), 4),
                }
        edges = []
        for i in range(M.L):
            for t in range(T):
                for s in range(t + 1):
                    w = float(attns[i, t, s]) * float(dla_attn[i, t])
                    if abs(w) > 1e-4:
                        edges.append({"s": f"{i}:{s}", "t": f"{i + 1}:{t}",
                                      "w": r(w, 4), "kind": "attn", "layer": i,
                                      "q": t, "k": s})
            for t in range(T):
                w = float(dla_mlp[i, t])
                if abs(w) > 1e-4:
                    edges.append({"s": f"{i}:{t}", "t": f"{i + 1}:{t}",
                                  "w": r(w, 4), "kind": "mlp", "layer": i})
        edges.sort(key=lambda e: -abs(e["w"]))
        keep = edges[:160]
        used = {e["s"] for e in keep} | {e["t"] for e in keep}
        node_list = [nodes[k] for k in nodes if k[0] in range(M.L + 1) and
                     (nodes[k]["id"] in used or abs(nodes[k]["attr"]) > 0.05)]
        out_graphs[pid] = {
            "tokens": rec["tokens"],
            "target": rec["target"],
            "attr": [[r(float(v), 4) for v in row] for row in attr],
            "dlaAttn": [[r(float(v), 4) for v in row] for row in dla_attn],
            "dlaMlp": [[r(float(v), 4) for v in row] for row in dla_mlp],
            "nodes": node_list,
            "edges": keep,
            "L": M.L,
        }
        log(f"graph {pid}: {len(node_list)} nodes, {len(keep)} edges")

    (CACHE / "graph.json").write_text(json.dumps(out_graphs))


# ----------------------------------------------------------------------------
# stage: build  (compact JSON for the site)
# ----------------------------------------------------------------------------
def build():
    OUT.mkdir(parents=True, exist_ok=True)
    cap = json.loads((CACHE / "capture.json").read_text())
    abl = json.loads((CACHE / "ablation.json").read_text())
    probes = json.loads((CACHE / "probes.json").read_text())
    geo = json.loads((CACHE / "geometry.json").read_text())
    graphs = json.loads((CACHE / "graph.json").read_text())

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    L = len(cap[0]["dla"]["attn"])
    meta = {
        "modelId": MODEL_ID,
        "modelName": MODEL_ID.split("/")[-1],
        "layers": L,
        "dModel": 0,
        "hidden": 0,
        "vocab": tok.vocab_size,
        "corpus": len(cap),
        "tokens": sum(len(c["tokens"]) for c in cap),
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kinds": sorted({c["id"].split("-")[0] for c in cap}),
        "suite": "depth-atlas/1.0",
        "evidence": "real forward/backward passes; no synthetic values",
    }

    # architecture facts, read from the checkpoint's own config
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(MODEL_ID)
        meta["hidden"] = cfg.hidden_size
        meta["dModel"] = cfg.hidden_size
        meta["heads"] = cfg.num_attention_heads
        meta["kvHeads"] = cfg.num_key_value_heads
        meta["intermediate"] = cfg.intermediate_size
        meta["vocab"] = cfg.vocab_size
        meta["tied"] = cfg.tie_word_embeddings
        meta["modelType"] = cfg.model_type
    except Exception as e:  # pragma: no cover
        log(f"config read failed: {e}")
    (OUT / "meta.json").write_text(json.dumps(meta))

    prompts = []
    for c in cap:
        prompts.append({
            "id": c["id"], "text": c["text"], "target": c["target"],
            "targetStr": c["targetStr"], "tokens": c["tokens"],
            "n": len(c["tokens"]),
            "finalTop1": c["finalTop1Str"],
            "top1Prob": c["lens"]["top1Prob"][-1],
            "targetProb": c["lens"]["targetProb"][-1],
            "group": next(p["group"] for p in CORPUS if p["id"] == c["id"]),
        })
    (OUT / "prompts.json").write_text(json.dumps(prompts))
    cap_by_id = {c["id"]: c for c in cap}

    def pick(d, keys):
        return {k: d[k] for k in keys}

    lens = {c["id"]: {
        "tokens": c["tokens"],
        "top": [[[tok.decode([t]), r(v)] for t, v in layer] for layer in c["lens"]["top"]],
        "entropy": c["lens"]["entropy"],
        "targetProb": c["lens"]["targetProb"],
        "targetRank": c["lens"]["targetRank"],
        "top1Prob": c["lens"]["top1Prob"],
        "kl": c["lens"]["kl"],
        "finalTop1": c["finalTop1Str"],
        "targetStr": c["targetStr"],
        "river": c["river"],
    } for c in cap}
    (OUT / "lens.json").write_text(json.dumps(lens))

    # attention: last-query rows per layer/head, plus mean rows
    attn = {}
    for c in cap:
        Lh, H, T = c["attnShape"]
        flat = np.array(c["attnLast"]).reshape(Lh, H, T)
        attn[c["id"]] = {
            "tokens": c["tokens"],
            "shape": c["attnShape"],
            "heads": [[[r(float(v), 4) for v in flat[l, h]] for h in range(H)] for l in range(Lh)],
            "mean": [[r(float(flat[l].mean(0)[t]), 4) for t in range(T)] for l in range(Lh)],
        }
    (OUT / "attention.json").write_text(json.dumps(attn))

    dla = {c["id"]: {"attn": c["dla"]["attn"], "mlp": c["dla"]["mlp"],
                     "resid": c["dla"]["resid"], "targetStr": c["targetStr"]} for c in cap}
    (OUT / "dla.json").write_text(json.dumps(dla))

    norms = {c["id"]: {"tokens": c["tokens"], "norm": c["residNorm"],
                       "delta": c["blockDelta"], "attn": c["attnNorm"],
                       "cosine": c["cosine"], "n": c["n"]} for c in cap}
    (OUT / "norms.json").write_text(json.dumps(norms))

    # ---- head taxonomy: mean behaviour over all query positions ------------
    H = meta.get("heads", 14)
    head_stats = []
    for l in range(L):
        for h in range(H):
            vals = [c["headStats"][l * H + h] for c in cap if len(c.get("headStats", [])) == L * H]
            if not vals:
                continue
            agg = {k: r(float(np.mean([v[k] for v in vals])), 4) for k in vals[0]}
            head_stats.append({"l": l, "h": h, **agg})
    (OUT / "heads.json").write_text(json.dumps(head_stats))

    (OUT / "ablation.json").write_text(json.dumps(abl))

    probes_out = {
        "properties": probes["properties"],
        "layers": probes["layers"],
        "byProp": {k: v["acc"] for k, v in probes["byProp"].items()},
        "std": {k: v["std"] for k, v in probes["byProp"].items()},
    }
    (OUT / "probes.json").write_text(json.dumps(probes_out))
    (OUT / "geometry.json").write_text(json.dumps(geo))
    (OUT / "graph.json").write_text(json.dumps(graphs))

    # ---- neurons -----------------------------------------------------------
    nmax, narg = [], []
    for c in cap:
        z = np.load(CACHE / f"act_{c['id']}.npz")
        nmax.append(np.stack([np.array(v) for v in z["nmax"]]))  # [L, I]
        narg.append(np.stack([np.array(v) for v in z["narg"]]))
    nmax = np.stack(nmax)  # [P, L, I]
    narg = np.stack(narg)
    neurons = []
    for l in range(L):
        Mx = nmax[:, l, :]
        Arg = narg[:, l, :]
        gmax = Mx.max(0)
        order = np.argsort(-gmax)[:14]
        for ni in order:
            ni = int(ni)
            best = np.argsort(-Mx[:, ni])[:6]
            ctx = []
            for bi in best:
                c = cap[int(bi)]
                pos = int(Arg[bi, ni])
                ctx.append({
                    "prompt": c["id"],
                    "pos": pos,
                    "token": c["tokens"][pos],
                    "before": "".join(tok.convert_tokens_to_string(c["tokens"][max(0, pos - 7):pos])),
                    "act": r(float(Mx[bi, ni]), 3),
                })
            density = float((Mx[:, ni] > 0.1 * gmax[ni]).mean())
            neurons.append({
                "layer": l, "neuron": ni,
                "max": r(float(gmax[ni]), 3),
                "mean": r(float(Mx[:, ni].mean()), 3),
                "density": r(density, 3),
                "contexts": ctx,
            })
        log(f"neurons layer {l}")
    (OUT / "neurons.json").write_text(json.dumps(neurons))
    log("build done")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["capture", "ablate", "analyse", "graph", "build", "all"])
    a = ap.parse_args()
    stages = ["capture", "ablate", "analyse", "graph", "build"] if a.stage == "all" else [a.stage]
    for s in stages:
        log(f"===== {s} =====")
        {"capture": capture, "ablate": ablate, "analyse": analyse,
         "graph": graph, "build": build}[s]()


if __name__ == "__main__":
    main()
