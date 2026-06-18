"""Interactive view of the public LB.

Two stacked panels:
  1. Score-density bar chart — one bar per unique score, height = teams sharing
     that score. Hover shows the team list. Reveals ties / score plateaus.
  2. Score-vs-rank curve — the long descending shape, with our point marked.

Usage:
    .venv/bin/python scripts/plot_leaderboard.py
"""

from __future__ import annotations

import glob
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
COMP = "playground-series-s6e5"
OUR_USER = "mcathala"
TARGET_SCORE = 0.9544
DENSE_LO = 0.940  # zoom window for the density panel
OUT_HTML = ROOT / "notebooks" / "leaderboard.html"

GRAY = "#94a3b8"
RED = "#e63946"
GREEN = "#10b981"
AMBER = "#f59e0b"
INK = "#0f172a"


def fetch_lb() -> pd.DataFrame:
    tmp = Path(tempfile.mkdtemp(prefix="lb_"))
    subprocess.run(
        [".venv/bin/kaggle", "competitions", "leaderboard", "-c", COMP, "--download", "-p", str(tmp)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    zf = next(tmp.glob("*.zip"))
    with zipfile.ZipFile(zf) as z:
        z.extractall(tmp)
    csv = glob.glob(str(tmp / "*.csv"))[0]
    df = pd.read_csv(csv)
    shutil.rmtree(tmp, ignore_errors=True)
    # Kaggle inserts a rank-0 baseline-submission row; drop it.
    return df[df["Rank"] >= 1].reset_index(drop=True)


def build_figure(df: pd.DataFrame) -> go.Figure:
    df = df.sort_values("Rank").reset_index(drop=True)

    is_us = df["TeamMemberUserNames"].fillna("").str.contains(OUR_USER, case=False)
    our_row = df[is_us].iloc[0]
    our_rank = int(our_row["Rank"])
    our_score = float(our_row["Score"])

    top_score = float(df["Score"].max())
    n_teams = len(df)

    # Per-unique-score aggregation: count + sample team list for hover.
    def sample_teams(s: pd.Series) -> str:
        names = s.tolist()
        head = ", ".join(names[:5])
        return head + (f"  +{len(names) - 5} more" if len(names) > 5 else "")

    agg = (
        df.groupby("Score")
        .agg(count=("Rank", "size"), teams=("TeamName", sample_teams), best_rank=("Rank", "min"))
        .reset_index()
        .sort_values("Score")
    )
    max_count = int(agg["count"].max())
    our_count_at_score = int(agg.loc[agg["Score"] == our_score, "count"].iloc[0])

    # Density panel — bar per unique score, our score highlighted in red.
    bar_colors = [RED if abs(s - our_score) < 1e-9 else GRAY for s in agg["Score"]]
    hover = [
        f"<b>score {s:.5f}</b><br>{c} team(s) · best rank {br}<br>{t}"
        for s, c, br, t in zip(agg["Score"], agg["count"], agg["best_rank"], agg["teams"])
    ]

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.13,
        subplot_titles=(
            "<b>Score density</b> — one bar per unique AUC, height = teams sharing it",
            "<b>Score vs rank</b> — full leaderboard shape",
        ),
    )

    fig.add_trace(
        go.Bar(
            x=agg["Score"],
            y=agg["count"],
            marker=dict(color=bar_colors, line=dict(width=0)),
            width=0.000009,  # bin precision is 5 decimals → 1e-5; slightly less avoids overlap
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Reference verticals on the density panel.
    y_top = max_count * 1.18
    for x, color, label in [
        (top_score, GREEN, f"top  {top_score:.5f}"),
        (TARGET_SCORE, AMBER, f"target  {TARGET_SCORE:.4f}"),
        (our_score, RED, f"us  {our_score:.5f}"),
    ]:
        fig.add_shape(
            type="line",
            x0=x,
            x1=x,
            y0=0,
            y1=max_count * 1.05,
            line=dict(color=color, dash="dash", width=1.2),
            xref="x",
            yref="y",
        )
        fig.add_annotation(
            x=x,
            y=y_top,
            text=label,
            showarrow=False,
            font=dict(color=color, size=11),
            xref="x",
            yref="y",
        )

    # Rank-vs-score panel.
    fig.add_trace(
        go.Scatter(
            x=df["Rank"],
            y=df["Score"],
            mode="lines",
            line=dict(color="#64748b", width=1.4),
            hovertemplate="rank %{x}<br>score %{y:.5f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[1, our_rank],
            y=[top_score, our_score],
            mode="markers+text",
            marker=dict(
                color=[GREEN, RED],
                size=[12, 14],
                line=dict(color="white", width=2),
                symbol=["diamond", "circle"],
            ),
            text=[f"top · {top_score:.5f}", f"us · rank {our_rank} · {our_score:.5f}"],
            textposition=["middle right", "middle right"],
            textfont=dict(size=11, color=INK),
            hoverinfo="skip",
            showlegend=False,
            cliponaxis=False,
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(
        title="public score (AUC)",
        range=[DENSE_LO, top_score + 0.0007],
        showgrid=True,
        gridcolor="#e2e8f0",
        zeroline=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title="# teams at that score",
        range=[0, max_count * 1.25],
        showgrid=True,
        gridcolor="#e2e8f0",
        zeroline=False,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title="rank",
        showgrid=True,
        gridcolor="#e2e8f0",
        zeroline=False,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title="score (AUC)",
        showgrid=True,
        gridcolor="#e2e8f0",
        zeroline=False,
        row=2,
        col=1,
    )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>playground-series-s6e5 — public LB</b><br>"
                f"<span style='font-size:13px;color:#475569'>"
                f"{n_teams} teams · top {top_score:.5f} · us rank {our_rank} "
                f"({our_score:.5f}, {our_count_at_score} team{'s' if our_count_at_score != 1 else ''} at this score) · "
                f"gap to top {top_score - our_score:+.5f} · gap to target {TARGET_SCORE - our_score:+.5f}"
                f"</span>"
            ),
            x=0.02,
            xanchor="left",
        ),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        bargap=0,
        height=780,
        margin=dict(l=70, r=40, t=110, b=60),
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=INK),
    )

    return fig


def print_tie_summary(df: pd.DataFrame, n: int = 10) -> None:
    """Console summary of the worst score-plateaus (most-tied scores)."""
    ties = (
        df.groupby("Score")
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["count", "Score"], ascending=[False, False])
        .head(n)
    )
    print("\nmost-tied scores (top {} clusters):".format(n))
    for _, r in ties.iterrows():
        print(f"  {r['Score']:.5f}  · {int(r['count'])} teams")


def main() -> None:
    df = fetch_lb()
    fig = build_figure(df)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT_HTML, include_plotlyjs="cdn", full_html=True)
    print(f"wrote {OUT_HTML.relative_to(ROOT)}")
    print_tie_summary(df)


if __name__ == "__main__":
    main()
