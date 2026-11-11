import React, { useMemo } from "react";
import { Atlas, INK, pct, useMeasure } from "../lib/data";
import {
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

const BINS = [
  { max: 0, label: "top-1", c: "#4a2410" },
  { max: 4, label: "2–5", c: "#8f5620" },
  { max: 19, label: "6–20", c: "#bb8642" },
  { max: 99, label: "21–100", c: "#d3b177" },
  { max: Infinity, label: ">100", c: "#e6d6b6" },
];

function binOf(rank: number) {
  for (const b of BINS) if (rank <= b.max) return b;
  return BINS[BINS.length - 1];
}

export default function Commitment({ atlas }: { atlas: Atlas }) {
  const L = atlas.meta.layers;
  const rows = useMemo(() => {
    return atlas.prompts
      .map((p) => {
        const lens = atlas.lens[p.id];
        let commit = -1;
        for (let l = 0; l < lens.targetRank.length; l++) {
          if (lens.targetRank[l] === 0) {
            commit = l;
            break;
          }
        }
        return { ...p, commit, ranks: lens.targetRank, final: p.targetProb };
      })
      .sort((a, b) => {
        const ca = a.commit < 0 ? 999 : a.commit;
        const cb = b.commit < 0 ? 999 : b.commit;
        return ca - cb || a.group.localeCompare(b.group);
      });
  }, [atlas]);

  const committed = rows.filter((r) => r.commit >= 0);
  const never = rows.length - committed.length;
  const median = (() => {
    const v = committed.map((r) => r.commit).sort((a, b) => a - b);
    return v.length ? v[Math.floor(v.length / 2)] : -1;
  })();

  const hist = Array.from({ length: L }, (_, l) => committed.filter((r) => r.commit === l).length);
  const maxHist = Math.max(...hist, 1);
  const survival = Array.from({ length: L }, (_, l) => committed.filter((r) => r.commit > l).length / rows.length);

  const groups = Array.from(new Set(rows.map((r) => r.group))).map((g) => {
    const rs = rows.filter((r) => r.group === g);
    const cs = rs.filter((r) => r.commit >= 0).map((r) => r.commit).sort((a, b) => a - b);
    return { g, n: rs.length, median: cs.length ? cs[Math.floor(cs.length / 2)] : -1, mean: rs.reduce((a, r) => a + r.final, 0) / rs.length };
  });

  const left = 118;
  const top = 22;
  const { ref, width } = useMeasure<HTMLDivElement>();
  const cell = Math.max(6, Math.min(15, Math.floor((Math.max(width, 320) - left - 76) / (L + 1))));
  const stripRight = left + (L + 1) * cell;
  const svgW = stripRight + 56;

  return (
    <>
      <ChapterHead
        n="12"
        kicker="Commitment atlas"
        title={<>When the answer <em>wins</em></>}
        standfirst={
          <>
            The lens can hold a token at rank 40 for twenty layers. The interesting
            event is not when it becomes probable but when it becomes{" "}
            <b>the argmax</b> — and stays there. This chapter measures that crossing
            for every prompt in the corpus, and shows that it happens at a
            characteristic depth.
          </>
        }
        chips={
          <>
            <Chip tone="amber">{rows.length} prompts</Chip>
            <Chip>{committed.length} commit · {never} never</Chip>
            <Chip tone="cyan">median L{median}</Chip>
          </>
        }
      />

      <Panel
        title={<><b>Figure 12.1</b> · rank of the target token, every prompt × layer</>}
        tools={<Legend items={BINS.map((b) => ({ c: b.c, label: b.label }))} />}
        note="Rows are prompts, sorted by the layer at which their target token first becomes the argmax. Cells are the token's rank in the lens distribution at that layer. A dark cell means the answer is already winning; pale means it is far from the top."
      >
        <div ref={ref} style={{ overflowX: "auto", maxWidth: "100%" }}>
          <svg width={svgW} height={top + rows.length * cell + 30} style={{ display: "block" }}>
            {Array.from({ length: L + 1 }, (_, l) =>
              l % 4 === 0 ? (
                <text key={l} x={left + l * cell + cell / 2} y={12} textAnchor="middle" className="ticklabel">
                  {l}
                </text>
              ) : null,
            )}
            {rows.map((r, i) => (
              <g key={r.id}>
                <text x={left - 6} y={top + i * cell + 10} textAnchor="end" className="ticklabel" style={{ fill: "var(--ink-2)" }}>
                  {r.id}
                </text>
                {Array.from({ length: L + 1 }, (_, l) => {
                  const rank = r.ranks[l] ?? 999;
                  return (
                    <rect
                      key={l}
                      x={left + l * cell}
                      y={top + i * cell}
                      width={cell - 1}
                      height={cell - 1}
                      fill={binOf(rank).c}
                      stroke={r.commit === l ? "var(--rust)" : "transparent"}
                      strokeWidth={r.commit === l ? 1.4 : 0}
                    >
                      <title>{`${r.id} · L${l} · rank ${rank}`}</title>
                    </rect>
                  );
                })}
                <rect x={stripRight + 8} y={top + i * cell + 2} width={Math.max(1, r.final * 40)} height={cell - 5} fill="var(--teal)" />
              </g>
            ))}
            <text x={stripRight + 8} y={12} className="ticklabel">P</text>
            <text x={left} y={top + rows.length * cell + 16} className="ticklabel">layer →</text>
          </svg>
        </div>
      </Panel>

      <div className="grid2" style={{ alignItems: "start" }}>
        <Panel
          title={<><b>Figure 12.2</b> · distribution of commit layers</>}
          note="Counts of prompts whose target first becomes the argmax at each layer. The distribution is not uniform: it concentrates in a band rather than spreading across depth."
        >
          <svg viewBox="0 0 470 190" style={{ display: "block", width: "100%", height: "auto" }}>
            {hist.map((v, l) => {
              const w = 430 / L;
              return (
                <g key={l}>
                  <rect
                    x={30 + l * w}
                    y={160 - (v / maxHist) * 120}
                    width={w - 1.5}
                    height={(v / maxHist) * 120}
                    fill={v ? INK.rust : "var(--rule)"}
                    opacity={v ? 0.9 : 0.4}
                  >
                    <title>{`L${l}: ${v}`}</title>
                  </rect>
                  {l % 6 === 0 && (
                    <text x={30 + l * w + w / 2} y={175} textAnchor="middle" className="ticklabel">{l}</text>
                  )}
                </g>
              );
            })}
            <line x1={30} x2={460} y1={160} y2={160} stroke="var(--ink)" />
            {[0, maxHist].map((v) => (
              <text key={v} x={24} y={160 - (v / maxHist) * 120 + 3} textAnchor="end" className="ticklabel">{v}</text>
            ))}
          </svg>
        </Panel>
        <div className="stack">
          <Panel
            title={<><b>Figure 12.3</b> · fraction still undecided</>}
            note="Survival curve: share of prompts whose target has not yet reached rank 1 at that layer."
          >
            <svg viewBox="0 0 440 150" style={{ display: "block", width: "100%", height: "auto" }}>
              {survival.map((v, l) => {
                const w = 400 / L;
                return (
                  <g key={l}>
                    <rect x={26 + l * w} y={120 - v * 96} width={w} height={v * 96} fill="var(--sand)" opacity={0.9}>
                      <title>{`after L${l}: ${pct(v, 0)} uncommitted`}</title>
                    </rect>
                  </g>
                );
              })}
              <line x1={26} x2={426} y1={120} y2={120} stroke="var(--ink)" />
              <text x={26} y={12} className="ticklabel">100%</text>
              <text x={26} y={134} className="ticklabel">0%</text>
            </svg>
          </Panel>
          <div className="grid2">
            <Stat k="Median commit" v={`L${median}`} s="of prompts that commit" />
            <Stat k="Never commit" v={never} s="answer is never the argmax" />
          </div>
        </div>
      </div>

      <h2 className="sec">By prompt family</h2>
      <Panel
        title={<><b>Figure 12.4</b> · commit layer by task type</>}
        note="Median over prompts in each family. A family with a late median needs more depth to resolve its answer; one with an early median is decided while the surface form is still being built."
        tight
      >
        <table className="tbl">
          <thead>
            <tr>
              <th>family</th>
              <th>prompts</th>
              <th>median commit</th>
              <th>mean final P(target)</th>
              <th>never</th>
            </tr>
          </thead>
          <tbody>
            {groups.map((g) => (
              <tr key={g.g}>
                <td className="k">{g.g}</td>
                <td>{g.n}</td>
                <td>{g.median < 0 ? "—" : `L${g.median}`}</td>
                <td>{pct(g.mean, 1)}</td>
                <td>{rows.filter((r) => r.group === g.g && r.commit < 0).length}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>

      <Prose>
        <p>
          Commitment is not spread evenly across depth. For this checkpoint the
          median prompt commits at <strong>L{median}</strong>, and the histogram is
          concentrated rather than uniform — most prompts cross at a similar band,
          with a tail that commits almost immediately and a handful that never do.
        </p>
        <p>
          The "never" cases are not failures to report: they are prompts where the
          model's answer differs from the hand-written target, so the tracked token
          is never the argmax. They are counted honestly rather than dropped,
          because dropping them would bias the histogram toward confident examples.
        </p>
      </Prose>

      <Method>
        <p>
          For each prompt and layer, the logit lens produces a distribution over the
          vocabulary; the rank of the target token is the number of tokens with
          higher probability. A prompt is said to commit at the first layer where
          that rank is 0. This is a statement about the lens at the final query
          position, not about the model's sampled output.
        </p>
        <p>
          Rows in Figure 12.1 are sorted by commit layer, with never-committing
          prompts last. The white outline marks the crossing cell itself.
        </p>
      </Method>

      <Footnotes
        items={[
          "Rank is computed on the full 151,936-token distribution, not on the stored top-12; the top-12 is only used for display elsewhere.",
          "A commit is not irreversible in principle, but in this corpus the target stays at rank 1 once it arrives in the large majority of cases.",
          `Median computed over the ${committed.length} prompts that commit; the ${never} that never do are excluded from that statistic and reported separately.`,
        ]}
      />
      <PageFoot index={13} />
    </>
  );
}
