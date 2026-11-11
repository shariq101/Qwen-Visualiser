import React, { ReactNode } from "react";
import { CHAPTERS, href, tokenLabel } from "../lib/data";

/* ------------------------------------------------------------------ chrome */
export function ChapterHead({
  n,
  kicker,
  title,
  standfirst,
  chips,
}: {
  n: string;
  kicker: string;
  title: ReactNode;
  standfirst: ReactNode;
  chips?: ReactNode;
}) {
  return (
    <header className="chapter-head">
      <div className="kicker">
        <span>
          {n} — {kicker}
        </span>
      </div>
      <h1 className="title">{title}</h1>
      <p className="standfirst">{standfirst}</p>
      {chips && <div className="byline">{chips}</div>}
    </header>
  );
}

export function Panel({
  title,
  tools,
  children,
  note,
  tight,
  style,
}: {
  title?: ReactNode;
  tools?: ReactNode;
  children: ReactNode;
  note?: ReactNode;
  tight?: boolean;
  style?: React.CSSProperties;
}) {
  return (
    <figure className="panel" style={style}>
      {(title || tools) && (
        <div className="panel-head">
          <div className="panel-title">{title}</div>
          <div className="spacer" />
          {tools}
        </div>
      )}
      <div className={"panel-body" + (tight ? " tight" : "")}>{children}</div>
      {note && <figcaption className="panel-note">{note}</figcaption>}
    </figure>
  );
}

export function Caption({ children }: { children: ReactNode }) {
  return <p className="caption">{children}</p>;
}

export function Method({ label = "How this is measured", children }: { label?: string; children: ReactNode }) {
  return (
    <div className="method">
      <div className="label">{label}</div>
      {children}
    </div>
  );
}

export function Prose({ children }: { children: ReactNode }) {
  return <div className="prose">{children}</div>;
}

export function Stat({ k, v, s }: { k: string; v: ReactNode; s?: ReactNode }) {
  return (
    <div className="stat">
      <div className="k">{k}</div>
      <div className="v">{v}</div>
      {s && <div className="s">{s}</div>}
    </div>
  );
}

export function Chip({ children, tone = "" }: { children: ReactNode; tone?: "" | "amber" | "cyan" }) {
  return <span className={"chip " + tone}>{children}</span>;
}

export function Seg<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { v: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="seg">
      {options.map((o) => (
        <button key={o.v} className={o.v === value ? "on" : ""} onClick={() => onChange(o.v)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Legend({ items }: { items: { c: string; label: string; line?: boolean }[] }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span className="legend-item" key={it.label}>
          <span className={"swatch" + (it.line ? " line" : "")} style={{ background: it.c }} />
          {it.label}
        </span>
      ))}
    </div>
  );
}

/** Numeric colour key for a heat map. */
export function Colorbar({
  stops,
  min,
  max,
  label,
  mid,
  width = 150,
  format,
}: {
  stops: [number, string][];
  min: number;
  max: number;
  label?: string;
  mid?: number;
  width?: number;
  format?: (v: number) => string;
}) {
  const grad = `linear-gradient(90deg, ${stops.map(([p, c]) => `${c} ${p * 100}%`).join(", ")})`;
  return (
    <div className="ramp">
      <span>{format ? format(min) : min}</span>
      <span className="bar" style={{ background: grad, width }} />
      <span>{format ? format(max) : max}</span>
      {label && <span style={{ marginLeft: 4, textTransform: "uppercase", letterSpacing: "0.08em" }}>{label}</span>}
      {mid !== undefined && <span style={{ marginLeft: 4 }}>mid {mid}</span>}
    </div>
  );
}

export function Tok({ t, style, title }: { t: string; style?: React.CSSProperties; title?: string }) {
  return (
    <span className="token" style={style} title={title ?? t}>
      {tokenLabel(t)}
    </span>
  );
}

export function PageFoot({ index }: { index: number }) {
  const prev = CHAPTERS[index - 1];
  const next = CHAPTERS[index + 1];
  return (
    <nav className="pagefoot">
      {prev ? (
        <a href={href(prev.slug)}>
          <span className="lbl">Previous</span>
          <span>
            {prev.n} · {prev.title}
          </span>
        </a>
      ) : (
        <span />
      )}
      {next ? (
        <a href={href(next.slug)} className="next">
          <span className="lbl">Next</span>
          <span>
            {next.n} · {next.title}
          </span>
        </a>
      ) : (
        <span />
      )}
    </nav>
  );
}

export function Footnotes({ items }: { items: (string | ReactNode)[] }) {
  return (
    <div className="footnotes">
      <div style={{ fontFamily: "var(--mono)", fontSize: 8.5, letterSpacing: "0.14em", textTransform: "uppercase" }}>
        Notes &amp; provenance
      </div>
      <ol>
        {items.map((t, i) => (
          <li key={i}>{typeof t === "string" ? <span dangerouslySetInnerHTML={{ __html: t }} /> : t}</li>
        ))}
      </ol>
    </div>
  );
}

/* ------------------------------------------------------------------ charts */
/**
 * Axes with real ticks, tick marks and unit labels. Every chart on the site is
 * expected to use this so that a reader can read values off the page.
 */
export function Axis({
  x,
  y,
  w,
  h,
  xTicks,
  yTicks,
  xLabel,
  yLabel,
  xFormat,
  yFormat,
}: {
  x: (v: number) => number;
  y: (v: number) => number;
  w: number;
  h: number;
  xTicks: number[];
  yTicks: number[];
  xLabel?: string;
  yLabel?: string;
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
}) {
  const x0 = x(xTicks[0] ?? 0);
  return (
    <g className="axis">
      {yTicks.map((t) => (
        <g key={"y" + t}>
          <line x1={x0} x2={w} y1={y(t)} y2={y(t)} className="gridline" />
          <line x1={x0 - 3} x2={x0} y1={y(t)} y2={y(t)} stroke="var(--rule-2)" />
          <text x={x0 - 6} y={y(t) + 3} textAnchor="end">
            {yFormat ? yFormat(t) : t}
          </text>
        </g>
      ))}
      <line x1={x0} x2={w} y1={h} y2={h} stroke="var(--rule-2)" />
      {xTicks.map((t, i) => (
        <g key={"x" + t}>
          <line x1={x(t)} x2={x(t)} y1={h} y2={h + 3} stroke="var(--rule-2)" />
          <text x={x(t)} y={h + 12} textAnchor={i === 0 ? "start" : i === xTicks.length - 1 ? "end" : "middle"}>
            {xFormat ? xFormat(t) : t}
          </text>
        </g>
      ))}
      {xLabel && (
        <text x={w} y={h + 26} textAnchor="end" className="axislabel">
          {xLabel}
        </text>
      )}
      {yLabel && (
        <text x={x0 - 6} y={-8} textAnchor="end" className="axislabel">
          {yLabel}
        </text>
      )}
    </g>
  );
}

export function LayerRuler({
  layers,
  x,
  y,
  w,
  active,
  onPick,
  every = 1,
  color = "var(--rule-2)",
}: {
  layers: number;
  x: (l: number) => number;
  y: number;
  w: number;
  active?: number;
  onPick?: (l: number) => void;
  every?: number;
  color?: string;
}) {
  return (
    <g>
      <line x1={x(0)} x2={x(layers)} y1={y} y2={y} stroke="var(--ink)" />
      {Array.from({ length: layers + 1 }, (_, l) => (
        <g key={l}>
          <line
            x1={x(l)}
            x2={x(l)}
            y1={y - 3}
            y2={y + 3}
            stroke={active === l ? "var(--rust)" : color}
            strokeWidth={active === l ? 1.8 : 1}
          />
          {l % every === 0 && (
            <text x={x(l)} y={y + 13} textAnchor="middle" className="ticklabel">
              {l}
            </text>
          )}
          {onPick && (
            <rect
              x={x(l) - w / (2 * (layers + 1))}
              y={y - 7}
              width={w / (layers + 1)}
              height={22}
              fill="transparent"
              style={{ cursor: "pointer" }}
              onClick={() => onPick(l)}
            />
          )}
        </g>
      ))}
    </g>
  );
}
