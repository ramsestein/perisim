"""Self-contained HTML for aggregate reports (no patient data)."""

from __future__ import annotations

import base64
import html
import io
from typing import Iterable

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px auto;max-width:1180px;
color:#1d2433;background:#fff;line-height:1.45;padding:0 16px}
h1{font-size:1.6rem;margin-bottom:.2rem}h2{font-size:1.2rem;margin-top:2rem;border-bottom:1px solid #dde3ea;
padding-bottom:.3rem}h3{font-size:1rem;margin-top:1.4rem}
table{border-collapse:collapse;margin:.6rem 0 1rem;font-size:.86rem;display:block;overflow-x:auto}
th,td{border:1px solid #dde3ea;padding:4px 8px;text-align:right;white-space:nowrap}
th{background:#f4f6f9;text-align:center}td:first-child,th:first-child{text-align:left}
.note{background:#f6f8fb;border-left:4px solid #6b7c93;padding:.6rem .9rem;font-size:.9rem}
.warn{background:#fff6e5;border-left:4px solid #e0a100;padding:.6rem .9rem}
.bad{background:#fdecec;border-left:4px solid #c62828;padding:.6rem .9rem}
.ok{background:#eaf6ec;border-left:4px solid #2e7d32;padding:.6rem .9rem}
img{max-width:100%;height:auto;border:1px solid #eef1f5}
.muted{color:#6b7c93;font-size:.85rem}
"""


def esc(x) -> str:
    return html.escape(str(x))


def fmt(v, nd: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:
            return "—"
        return f"{v:.{nd}f}"
    return esc(v)


def table(headers: list[str], rows: Iterable[Iterable], nd: int = 3) -> str:
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{fmt(c, nd)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def box(text: str, kind: str = "note") -> str:
    return f'<div class="{kind}">{text}</div>'


def fig_img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return f'<img alt="figure" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"/>'


def page(title: str, sections: list[str], subtitle: str = "") -> str:
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{esc(title)}</title><style>{CSS}</style></head>"
            f"<body><h1>{esc(title)}</h1><p class='muted'>{esc(subtitle)}</p>{''.join(sections)}</body></html>")


def suppress(n: int | None, min_n: int) -> str | int | None:
    if n is None:
        return None
    return f"<{min_n}" if 0 < n < min_n else int(n)
