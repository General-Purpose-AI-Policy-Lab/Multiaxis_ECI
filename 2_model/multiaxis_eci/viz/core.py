"""Core figures: save_fig, trace/posterior grids, forests, the
capability/difficulty timeline."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from multiaxis_eci.viz.style import DASHBOARD, FigureStyle, apply_fonts

CHAIN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# Shared palette: AI models, human tiers, display anchors, and the
# passed/future split on forecast crossover markers. The single source every
# figure script reads, so no caller carries its own hex.
AI_COLOR = "#4c78a8"
HUMAN_COLOR = "#e8890c"
ANCHOR_COLOR = "#888888"
PASSED_COLOR = "#2ca02c"
FUTURE_COLOR = "#d62728"


_AXIS_KEY = re.compile(r"^(timeline|loadings|forecast)_(\d+)_(.+)$")
_KEY_SUFFIXES = {"_when": "_crossover_dates"}


def figure_filename(key: str) -> str:
    """The on-disk name of a figure key.

    Figure keys stay short because the dashboard and the blog post address panels
    by them; files get the explicit form: `timeline_2_reasoning` ->
    `timeline_axis2_reasoning`, `forecast_1_math_when` ->
    `forecast_axis1_math_crossover_dates`. A slug that merely repeats the axis
    (`forecast_1_axis1_when`, the unnamed-axis case) is dropped rather than
    doubled: `forecast_axis1_crossover_dates`. Other keys are unchanged.
    """
    m = _AXIS_KEY.match(key)
    if m is None:
        return key
    kind, k, rest = m.groups()
    for short, long in _KEY_SUFFIXES.items():
        if rest.endswith(short):
            rest = rest[: -len(short)] + long
            break
    if rest == f"axis{k}":
        return f"{kind}_axis{k}"
    if rest.startswith(f"axis{k}_"):
        rest = rest[len(f"axis{k}_"):]
    return f"{kind}_axis{k}_{rest}"


# Vendor tokens whose casing a naive .title() would mangle.
_CASE = {"gpt": "GPT", "glm": "GLM", "o1": "o1", "o3": "o3", "o4": "o4",
         "ai": "AI", "xhigh": "xhigh", "xh": "xhigh", "sol": "Sol",
         "minimax": "MiniMax", "deepseek": "DeepSeek", "terra": "Terra"}


def pretty_model_name(raw: str) -> str:
    """Readable label for a raw model id: effort suffix in parentheses, trailing release date
    dropped, vendor casing fixed ("gpt-5.6-sol_max" -> "GPT 5.6 Sol (max)")."""
    name, *effort = raw.split("_", 1)
    name = re.sub(r"-\d{4}-\d{2}(-\d{2})?$", "", name)
    toks = [_CASE.get(t, t.capitalize() if t.isalpha() else t) for t in name.split("-")]
    label = " ".join(toks)
    if effort:
        e = effort[0].replace("_", " ")
        label += f" ({_CASE.get(e, e)})"
    return label


def two_column_layout(fig: go.Figure, col_domains, panel_titles, n_panels: int = 4) -> None:
    """Set a two-column grid's x domains directly and recentre each panel title.

    At print scale the right column's y tick labels live in the inter-column gutter and need
    more room than `make_subplots`' fractional spacing gives. `panel_titles` are the texts that
    identify a panel-title annotation among the figure's other annotations.
    """
    axes = ["xaxis"] + [f"xaxis{n}" for n in range(2, n_panels + 1)]
    for i, ax in enumerate(axes):
        fig.layout[ax].domain = col_domains[i % 2]
    titles = set(panel_titles)
    panels = [a for a in fig.layout.annotations if a.text in titles]
    for i, ann in enumerate(panels):
        ann.x = sum(col_domains[i % 2]) / 2


def save_fig(fig: go.Figure, name: str, figures_dir: Path) -> None:
    """One PNG in `figures_dir`, its interactive twin in `figures_dir/html/`."""
    html_dir = figures_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_dir / f"{name}.html")
    try:
        fig.write_image(figures_dir / f"{name}.png", scale=2)
    except Exception as e:
        print(f"  PNG export skipped for {name} ({type(e).__name__}: {e})")


# Print defaults, applied per property only where the figure left it unset, so a
# figure that carries its own type sizes or background keeps them. Dotted paths
# are resolved on the layout; `None` there means "the figure never set it".
# Deliberately just the two backgrounds: a printed figure must not inherit a
# dark or transparent canvas. No `font.family` here — the template's font is a
# visual choice, and forcing Arial re-renders every glyph of the committed LW
# exports (measured: 2.7% of pixels differ, dimensions unchanged).
_PRINT_LAYOUT = {"paper_bgcolor": "white",
                 "plot_bgcolor": "white"}


def _print_copy(fig: go.Figure, width: int | None = None,
                height: int | None = None) -> go.Figure:
    """A COPY of `fig` with the print defaults filled in (never the caller's
    figure, so the same figure can also go to HTML unchanged)."""
    out = go.Figure(fig)
    for dotted, value in _PRINT_LAYOUT.items():
        obj = out.layout
        for part in dotted.split(".")[:-1]:
            obj = obj[part]
        if obj[dotted.split(".")[-1]] is None:
            obj[dotted.split(".")[-1]] = value
    if width:
        out.layout.width = width
    if height:
        out.layout.height = height
    return out


def save_print(fig: go.Figure, path, *, width: int | None = None,
               height: int | None = None, scale: int = 2,
               pdf: bool = True) -> Path:
    """Write `fig` as a print-quality PNG (and PDF) at `path` (suffix ignored).

    The layout patch lands on a COPY: the caller's figure is never mutated, so
    the same figure can also go to HTML unchanged. `scale=2` is the pixel
    dimension every committed LW export was made at; changing it changes them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = _print_copy(fig, width, height)

    png = path.with_suffix(".png")
    out.write_image(png, scale=scale)
    if pdf:
        # Vector: `scale` is a raster notion and does not apply.
        out.write_image(path.with_suffix(".pdf"))
    return png


def save_html(fig: go.Figure, path) -> Path:
    """Write the interactive HTML twin of a `save_print` export.

    It lands in an `html/` folder BESIDE `path`, not next to the PNG/PDF: the
    figure directory stays a flat list of the print files the post embeds.
    """
    path = Path(path)
    out = path.parent / "html" / f"{path.stem}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out)
    return out


def save_svg(fig: go.Figure, path, *, width: int | None = None,
             height: int | None = None) -> Path:
    """Write the vector SVG twin of a `save_print` export.

    Same print layout as the PNG/PDF (same copy-and-patch, same width/height),
    and like `save_html` it lands in a sibling folder, `svg/` beside `path`, so
    the figure directory stays the flat list of files the post embeds. Vector:
    `save_print`'s `scale` is a raster notion and does not apply.
    """
    path = Path(path)
    out = path.parent / "svg" / f"{path.stem}.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    _print_copy(fig, width, height).write_image(out)
    return out


def subplot_grid(figs, titles=None, ncols: int = 2, vertical_spacing: float = 0.09,
                 horizontal_spacing: float = 0.09, **layout) -> go.Figure:
    """Lay finished figures out as the panels of one grid.

    What carries over per panel: the traces, the horizontal reference lines
    (`add_hline` shapes) and the two axis titles. What does not: per-figure
    width/height/title, since the grid sets its own. A legend name is shown on
    its FIRST panel only, so N panels of the same series read as one legend
    entry — which assumes a name means the same thing in every panel.

    Traces are copied, so the source figures stay usable (e.g. for HTML).
    """
    figs = list(figs)
    nrows = -(-len(figs) // ncols)
    grid = make_subplots(rows=nrows, cols=ncols, vertical_spacing=vertical_spacing,
                         horizontal_spacing=horizontal_spacing,
                         subplot_titles=list(titles) if titles else None)
    seen: set[str] = set()
    for i, f in enumerate(figs):
        row, col = i // ncols + 1, i % ncols + 1
        for tr in f.data:
            if getattr(tr, "x", None) is not None and len(tr.x) == 0:
                continue                # an empty panel trace is legend noise
            tr = tr.__class__(tr)
            if tr.name:
                tr.showlegend = tr.name not in seen
                seen.add(tr.name)
            grid.add_trace(tr, row=row, col=col)
        for sh in f.layout.shapes:
            if sh.type == "line" and sh.y0 == sh.y1:
                grid.add_hline(y=sh.y0, line=sh.line, layer=sh.layer, row=row, col=col)
        grid.update_xaxes(title_text=f.layout.xaxis.title.text, row=row, col=col)
        grid.update_yaxes(title_text=f.layout.yaxis.title.text, row=row, col=col)
    grid.update_layout(template="plotly_white", **layout)
    return grid


# ── Trace + posterior pair ────────────────────────────────────────────────
def trace_posterior_grid(items: list[tuple[str, np.ndarray]],
                          title: str,
                          row_height: int = 170,
                          width: int = 1000) -> go.Figure:
    n = len(items)
    fig = make_subplots(
        rows=n, cols=2, column_widths=[0.62, 0.38],
        horizontal_spacing=0.09, vertical_spacing=0.18,
        subplot_titles=[s for t, _ in items
                        for s in (f"{t}: trace", f"{t}: posterior")],
    )
    for r, (_name, samples) in enumerate(items, start=1):
        for ch in range(samples.shape[0]):
            fig.add_trace(go.Scatter(
                y=samples[ch], mode="lines",
                line=dict(width=0.7, color=CHAIN_COLORS[ch % 4]),
                opacity=0.65, name=f"chain {ch}",
                showlegend=(r == 1), legendgroup=f"chain{ch}",
            ), row=r, col=1)
        flat = samples.flatten()
        fig.add_trace(go.Histogram(
            x=flat, nbinsx=60, marker_color="#4682B4",
            opacity=0.75, showlegend=False, histnorm="probability density",
        ), row=r, col=2)
        fig.add_vline(x=float(flat.mean()), line_dash="dash",
                      line_color="crimson", row=r, col=2)
    fig.update_layout(
        height=row_height * n, width=width,
        title=dict(text=title, x=0.5, font=dict(size=13)),
        margin=dict(l=55, r=20, t=60, b=35),
        template="plotly_white",
    )
    return fig


def raw_scores_by_date_fig(raw: pd.DataFrame, initial: int = 3) -> go.Figure:
    """Observed scores vs the taker's release date, one legend entry per
    benchmark. All but the `initial` best-covered benchmarks start hidden
    (`legendonly`): click legend entries to overlay any set of benchmarks in
    the same axes. Raw data, no model — undated takers (humans included) are
    absent, since they have no x.

    A row's own `release_date` is often blank (the SEAL / RAND feeds carry
    none), so each taker's date is filled from its earliest dated row — the same
    fill the era filters use."""
    df = raw.dropna(subset=["score"]).copy()
    d = pd.to_datetime(df["release_date"], errors="coerce")
    d = d.fillna(df["model_version"].map(d.groupby(df["model_version"]).min()))
    df["release_date"] = d.dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["release_date"])
    fig = go.Figure()
    for i, b in enumerate(df["benchmark"].value_counts().index):
        d = df[df["benchmark"] == b]
        fig.add_trace(go.Scatter(
            x=d["release_date"], y=d["score"], mode="markers", name=b,
            text=d["model_version"], legendgroup=b,
            marker=dict(size=6, opacity=0.75),
            hovertemplate="%{text}<br>%{x} · %{y:.3f}<extra>" + b + "</extra>",
            visible=True if i < initial else "legendonly"))
    fig.update_layout(
        title=dict(text="Observed scores by release date", x=0.5,
                   font=dict(size=13)),
        xaxis=dict(title="release date"),
        yaxis=dict(title="score", range=[-0.02, 1.02]),
        height=560, margin=dict(l=55, r=20, t=60, b=45),
        legend=dict(font=dict(size=9), itemsizing="constant"),
        template="plotly_white", hovermode="closest",
    )
    return fig


def hyperparams_fig(trace) -> go.Figure:
    """Trace + posterior for the global scale hyperparameters.

    The discrimination/loading scale is `tau_alpha` on a 1D trace and
    `tau_A_normal` / `tau_A_signed` on a MIRT trace,
    depending on the loading prior.
    """
    post = trace.posterior
    items = [("tau_CD", post["tau_CD"].values)]
    for name in ("tau_alpha", "tau_A_normal", "tau_A_signed"):
        if name in post:
            items.append((name, post[name].values))
            break
    return trace_posterior_grid(items, title="Hyperparameter traces")


# ── Forest plot of a per-benchmark variable ───────────────────────────────
def forest_fig(stats_df: pd.DataFrame,
                var_name: str,
                title: str,
                color: str = "steelblue",
                reference: float | None = None,
                ref_label: str | None = None,
                row_px: int = 16) -> go.Figure:
    names = stats_df["name"].tolist()
    means = stats_df["mean"].values
    err_pos = stats_df["hdi_high"].values - means
    err_neg = means - stats_df["hdi_low"].values

    fig = go.Figure(go.Scatter(
        x=means, y=names, mode="markers",
        marker=dict(color="crimson", size=7, symbol="diamond"),
        error_x=dict(type="data", symmetric=False,
                     array=err_pos, arrayminus=err_neg,
                     color=color, thickness=1.8, width=4),
        hovertemplate=f"<b>%{{y}}</b><br>{var_name} = %{{x:.3f}}<extra></extra>",
    ))
    if reference is not None:
        fig.add_vline(x=reference, line_dash="dot", line_color="gray",
                      annotation_text=ref_label or "",
                      annotation_position="top")
    fig.update_layout(
        title=title, xaxis_title=var_name,
        # Force EVERY row label to render (Plotly auto-skips ticks on long lists,
        # which dropped ~half the benchmark names) and grow the height with the
        # row count so labels never crowd; small forests keep a sensible floor.
        yaxis=dict(categoryorder="array", categoryarray=names,
                   tickmode="array", tickvals=names, ticktext=names,
                   tickfont=dict(size=9), automargin=True,
                   showgrid=True, gridcolor="rgba(0,0,0,0.04)", zeroline=False),
        template="plotly_white",
        height=max(400, row_px * len(names)), width=950,
        margin=dict(l=260, r=40, t=55, b=45),
    )
    return fig


# ── All-models forest (very tall — one row per model) ────────────────────
def all_models_forest_fig(stats_df: pd.DataFrame,
                           highlight: set[str] | None = None,
                           row_px: int = 18,
                           anchors: list[tuple[float, str]] | None = None,
                           metric: str = "ECI") -> go.Figure:
    """One marker + HDI bar per model. Sorted ascending by mean.

    `highlight` — names rendered in red (e.g. SOTA / anchor models).
    `metric`    — axis/title label, "ECI" or "C".
    `anchors`   — (value, label) pairs displayed in the title (NOT as vlines —
                  in-plot anchor lines crowded the row labels).

    Two traces (non-highlighted blue, highlighted red diamonds) so the error bar
    colors can differ. Both traces share an explicit y-axis category order
    locked to `stats_df["name"]`, otherwise Plotly stacks the second trace's
    categories above the first's and breaks the ascending sort.
    """
    highlight = highlight or set()
    names   = stats_df["name"].tolist()
    means   = stats_df["mean"].values
    err_pos = stats_df["hdi_high"].values - means
    err_neg = means - stats_df["hdi_low"].values
    is_hl   = stats_df["name"].isin(highlight).values
    hover   = f"<b>%{{y}}</b><br>{metric} = %{{x:.2f}}<extra></extra>"

    fig = go.Figure()
    if (~is_hl).any():
        fig.add_trace(go.Scatter(
            x=means[~is_hl], y=stats_df["name"].values[~is_hl], mode="markers",
            marker=dict(color="#1f77b4", size=4),
            error_x=dict(type="data", symmetric=False,
                         array=err_pos[~is_hl], arrayminus=err_neg[~is_hl],
                         color="rgba(70,130,180,0.45)", thickness=1.0, width=0),
            hovertemplate=hover,
            showlegend=False,
        ))
    if is_hl.any():
        fig.add_trace(go.Scatter(
            x=means[is_hl], y=stats_df["name"].values[is_hl], mode="markers",
            marker=dict(color="crimson", size=8, symbol="diamond"),
            error_x=dict(type="data", symmetric=False,
                         array=err_pos[is_hl], arrayminus=err_neg[is_hl],
                         color="crimson", thickness=1.8, width=4),
            hovertemplate=hover,
            showlegend=False,
        ))

    n = len(stats_df)
    title = f"All {n} models: posterior median {metric} with 95% interval"
    if anchors:
        anchor_str = ", ".join(label for _, label in anchors)
        title = f"{title}<br><sub>anchored: {anchor_str}</sub>"

    fig.update_layout(
        title=title,
        xaxis=dict(
            title=metric, showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False,
        ),
        # Force every model name to render (default auto-skipping was dropping
        # ~half the labels, breaking label↔dot association). Smaller font + faint
        # horizontal gridlines per row make each lane visually unambiguous.
        yaxis=dict(
            categoryorder="array", categoryarray=names,
            tickmode="array", tickvals=names, ticktext=names,
            tickfont=dict(size=9),
            showgrid=True, gridcolor="rgba(0,0,0,0.04)", zeroline=False,
        ),
        template="plotly_white",
        height=max(600, row_px * n),
        width=1100,
        margin=dict(l=320, r=40, t=70, b=45),
    )
    return fig


# ── SOTA forest ───────────────────────────────────────────────────────────
def sota_forest_fig(sota_df: pd.DataFrame,
                     x_col: str,
                     title: str,
                     xaxis_title: str,
                     bar_color: str = "steelblue",
                     anchors: list[tuple[float, str]] | None = None,
                     reference: float | None = None,
                     ref_label: str | None = None) -> go.Figure:
    labels = [f"{m} ({d})" for m, d in zip(sota_df["model"], sota_df["release_date"])]
    means = sota_df[f"{x_col}_mean"].values
    los   = sota_df[f"{x_col}_hdi_low"].values
    his   = sota_df[f"{x_col}_hdi_high"].values

    fig = go.Figure(go.Scatter(
        x=means, y=labels, mode="markers",
        marker=dict(color="crimson", size=10, symbol="diamond"),
        error_x=dict(type="data", symmetric=False,
                     array=his - means, arrayminus=means - los,
                     color=bar_color, thickness=2.4, width=5),
        hovertemplate=f"<b>%{{y}}</b><br>{x_col} = %{{x:.3f}}<extra></extra>",
    ))
    if reference is not None:
        fig.add_vline(x=reference, line_dash="dot", line_color="gray",
                      annotation_text=ref_label or "", annotation_position="top")
    if anchors:
        for val, label in anchors:
            fig.add_vline(x=val, line_dash="dot", line_color="#666",
                          annotation_text=label, annotation_position="top left")
    fig.update_layout(
        title=title, xaxis_title=xaxis_title,
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=520, width=950,
        margin=dict(l=240, r=40, t=55, b=45),
    )
    return fig


# ── Capability / difficulty timeline ──────────────────────────────────────────
HUMAN_LEVEL_LABELS_FR = {
    "Committee of Average Humans":  "Comité d'humains moyens",
    "Average Human":                "Humain moyen",
    "Committee of Skilled Generalists": "Comité de généralistes qualifiés",
    "Skilled Generalist":           "Généraliste qualifié",
    "Domain Expert":                "Expert du domaine",
    "Committee of Domain Experts":  "Comité d'experts du domaine",
    "Best Performer":               "Meilleur performeur",
    "Top Performer":                "Meilleur performeur",
    "High School Qualifier":        "Lycéen qualifié",
    "High School Top Performer":    "Lycéen, meilleur performeur",
}
# Tier display names per language; English keeps the raw tier names.
HUMAN_LEVEL_LABELS = {"en": {}, "fr": HUMAN_LEVEL_LABELS_FR}

# Every string the timeline draws, per language. English is the default of every
# figure; the French render goes to a `fr/` folder beside it (3_fit/fit.py writes both).
TIMELINE_TEXT = {
    "en": {"title": "AI capability, human baselines and benchmark difficulty",
           "xaxis": "Release date", "yaxis": "Estimated capability / difficulty",
           "humans": "Human tiers", "bench": "Benchmark difficulty", "models": "AI models",
           "interval": lambda p: f"{p:.0%} interval"},
    "fr": {"title": "Capacités des IA, références humaines et difficulté des benchmarks",
           "xaxis": "Date de sortie", "yaxis": "Capacité / difficulté estimée",
           "humans": "Niveaux humains", "bench": "Difficulté des benchmarks",
           "models": "Modèles d'IA",
           "interval": lambda p: f"intervalle à {p * 100:.0f} %"},
}
# The two tiers whose interval the 1D ECI-H figures draw as a band by default:
# the bottom and the top of the human ladder, so the reader gets its bounds.
DEFAULT_HUMAN_BANDS = ("Average Human", "Top Performer")
BENCH_COLOR = "#d63384"
MODEL_COLOR = "#20a39e"


def _rgba(color: str, alpha: float) -> str:
    """'rgb(R,G,B)' or '#rrggbb' → 'rgba(R,G,B,alpha)'. Plotly's sample_colorscale returns
    rgb()-form strings, the palette constants are hex; add_hrect and error bars need rgba() to
    honor alpha."""
    if color.startswith("#"):
        h = color.lstrip("#")
        inner = ",".join(str(int(h[i:i + 2], 16)) for i in (0, 2, 4))
    else:
        inner = color[color.index("(") + 1 : color.index(")")]
    return f"rgba({inner},{alpha})"


def human_tier_palette(n: int) -> list[str]:
    """Sequential Blues, strongest tier darkest; both ends avoided (too light is
    invisible on white, too dark too bold under translucent band overlap)."""
    import plotly.colors as pc
    fractions = np.linspace(0.92, 0.35, n) if n > 1 else [0.7]
    return [pc.sample_colorscale("Blues", float(f))[0] for f in fractions]


def capability_timeline_fig(timeline_df: pd.DataFrame,
                             human_stats: pd.DataFrame | None = None,
                             annotate_benchmarks: list[str] | None = None,
                             annotate_models: list[str] | None = None,
                             human_labels: dict | None = None,
                             *,
                             lang: str = "en",
                             hdi_prob: float | None = None,
                             y_label: str | None = None,
                             human_bands: bool | tuple[str, ...] | None = None,
                             human_band_alpha: float = 0.06,
                             tier_names_at_right: bool = False,
                             style: FigureStyle = DASHBOARD,
                             x_min: str | None = None,
                             y_range: tuple[float, float] | None = None) -> go.Figure:
    """Capability (models) + difficulty (benchmarks) vs release date.

    The one builder behind the canonical ECI-H figure, the blog post's figure 1
    and the dashboard's per-axis timelines. Latent scale on Y, release date on X.
    Models in teal, benchmarks in pink, both with interval bars at the width the
    caller's stats frame carries; pass `hdi_prob` to name that width in the legend
    (the dashboard titles its 50% timelines itself and leaves it unset).

    Human tiers are dashed horizontal median lines, strongest darkest. `human_bands`
    adds the tier's interval as a faint band under every point: True for every
    tier (the post), a tuple of tier names for a subset (`DEFAULT_HUMAN_BANDS`, the
    canonical ECI-H figure), None for lines only.

    `lang` picks the labels ("en" default, "fr"); `human_labels` overrides the
    tier display names. `tier_names_at_right` writes each tier's name at the
    right end of its line instead of the legend (the post's arrangement).
    `x_min` drops models and benchmarks released before that date; `y_label`
    overrides the y-axis title (e.g. "ECI-H" when the frame is on that scale);
    `y_range` pins the y-axis (the post uses 30 to 220 on ECI-H, so one benchmark
    with a very uncertain difficulty does not squash the cloud)."""
    text = TIMELINE_TEXT[lang]
    labels = HUMAN_LEVEL_LABELS[lang] if human_labels is None else human_labels
    annotate_benchmarks = set(annotate_benchmarks or [])
    annotate_models = set(annotate_models or [])
    interval = f" ({text['interval'](hdi_prob)})" if hdi_prob else ""

    # Cast to date strings — plotly auto-detects as time-axis, kaleido can JSON-serialize.
    # (Timestamps break kaleido PNG export; .dt.to_pydatetime() trips a benches-only NaN
    # bug in the trace builder. Strings sidestep both.)
    tl = timeline_df.copy()
    tl["release_date"] = pd.to_datetime(tl["release_date"])
    if x_min is not None:
        tl = tl[tl["release_date"] >= pd.Timestamp(x_min)]
    tl["release_date"] = tl["release_date"].dt.strftime("%Y-%m-%d")

    models = tl[tl["kind"] == "model"]
    benches = tl[tl["kind"] == "benchmark"]

    fig = go.Figure()

    # Human posterior reference levels (drawn first so AI/benchmark points sit
    # on top): a dashed median line per tier, the interval band for the tiers
    # `human_bands` names. Tier names live in the legend on the right, or at the
    # right end of the lines with `tier_names_at_right`.
    if human_stats is not None and len(human_stats):
        rows = human_stats.sort_values("mean", ascending=False).reset_index(drop=True)
        palette = human_tier_palette(len(rows))
        if human_bands is True:
            banded = set(rows["name"])
        else:
            banded = set(human_bands or ())
        for i, (_, r) in enumerate(rows.iterrows()):
            if r["name"] in banded:
                fig.add_hrect(y0=float(r["hdi_low"]), y1=float(r["hdi_high"]),
                              fillcolor=_rgba(palette[i], human_band_alpha),
                              line_width=0, layer="below")
        for i, (_, r) in enumerate(rows.iterrows()):
            label_name = labels.get(r["name"], r["name"])
            line_color = _rgba(palette[i], 0.85)
            fig.add_hline(y=r["mean"],
                          line=dict(color=line_color, dash="dash", width=1.3),
                          layer="below")
            if tier_names_at_right:
                continue
            legend_label = f"{label_name} (n={r['n_obs']})"
            if r["name"] in banded and hdi_prob:
                legend_label += f", {text['interval'](hdi_prob)}"
            # Invisible Scatter trace solely to surface this group in the
            # Plotly legend (add_hrect/add_hline don't produce legend entries).
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=line_color, dash="dash", width=1.8),
                name=legend_label,
                legendgroup="humans",
                legendgrouptitle_text=text["humans"] if i == 0 else None,
                hoverinfo="skip",
            ))
        if tier_names_at_right:
            # Crowded tiers are nudged apart vertically (top-down), so a name can
            # sit a little off its line.
            lo = min(tl["hdi_low"].min(), rows["hdi_low"].min())
            hi = max(tl["hdi_high"].max(), rows["hdi_high"].max())
            gap = 0.038 * (hi - lo)
            prev = np.inf
            for i, (_, r) in enumerate(rows.iterrows()):
                y = min(float(r["mean"]), prev - gap)
                prev = y
                fig.add_annotation(
                    x=1.005, y=y, xref="paper", yref="y",
                    text=labels.get(r["name"], r["name"]), showarrow=False,
                    xanchor="left", font=dict(size=style.font_tier, color=palette[i]))

    # Benchmarks (difficulty)
    fig.add_trace(go.Scatter(
        x=benches["release_date"], y=benches["mean"], mode="markers",
        marker=dict(color=BENCH_COLOR, size=style.marker + 1, line=dict(width=0)),
        error_y=dict(type="data", symmetric=False,
                     array=(benches["hdi_high"] - benches["mean"]).values,
                     arrayminus=(benches["mean"] - benches["hdi_low"]).values,
                     color=BENCH_COLOR, thickness=style.errbar, width=style.errbar * 1.5),
        name=text["bench"] + interval, legendgroup="benchmarks",
        text=benches["name"],
        hovertemplate="<b>%{text}</b><br>D = %{y:.2f}<br>%{x|%Y-%m-%d}<extra></extra>",
    ))

    # Models (capability)
    fig.add_trace(go.Scatter(
        x=models["release_date"], y=models["mean"], mode="markers",
        marker=dict(color=MODEL_COLOR, size=style.marker, opacity=0.85, line=dict(width=0)),
        error_y=dict(type="data", symmetric=False,
                     array=(models["hdi_high"] - models["mean"]).values,
                     arrayminus=(models["mean"] - models["hdi_low"]).values,
                     color="rgba(32,163,158,0.35)", thickness=style.errbar * 0.75, width=0),
        name=text["models"] + interval, legendgroup="models",
        text=models["name"],
        hovertemplate="<b>%{text}</b><br>C = %{y:.2f}<br>%{x|%Y-%m-%d}<extra></extra>",
    ))

    # Inline name labels for selected points
    for _, r in benches.iterrows():
        if r["name"] in annotate_benchmarks:
            fig.add_annotation(x=r["release_date"], y=r["mean"],
                               text=r["name"], showarrow=False, yshift=-12,
                               font=dict(size=style.font_note - 2, color="#888"))
    for _, r in models.iterrows():
        if r["name"] in annotate_models:
            fig.add_annotation(x=r["release_date"], y=r["mean"],
                               text=pretty_model_name(r["name"]), showarrow=False, yshift=-12,
                               font=dict(size=style.font_note - 2, color="#666"))

    fig.update_layout(
        title=dict(text=text["title"], x=0.5),
        # type="date" is explicit because we add legend-only dummy traces
        # (x=[None]) before the real data traces; without it Plotly defaults
        # to numeric and silently drops the date-string scatters.
        xaxis=dict(type="date", title=text["xaxis"],
                   showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(title=y_label or text["yaxis"],
                   showgrid=True, gridcolor="rgba(0,0,0,0.06)",
                   **({"range": list(y_range)} if y_range else {})),
        template="plotly_white",
        height=int(style.height_per_row * 1.35), width=int(style.width * 1.25),
        # Right margin holds the legend (human groups + the two data series):
        # the group title + 7 entries need ~290px at dashboard scale before clipping.
        margin=dict(l=int(style.font_tick * 6), r=int(style.font_legend * 24),
                    t=int(style.font_title * 4.5), b=int(style.font_tick * 4.5)),
        legend=dict(
            orientation="v",
            yanchor="top",   y=0.99,
            xanchor="left",  x=1.01,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
    )
    return apply_fonts(fig, style)
