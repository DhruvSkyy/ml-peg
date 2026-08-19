"""
Layout and figures for the ML-PEG Model Explorer page.

The Explorer is built around ML-PEG's identity as a *performance and extrapolation
guide* across the breadth of chemistry. Everything is derived from the shipped benchmark
data (see :mod:`ml_peg.app.utils.explorer_data`), warmed at startup so navigation is
instant:

* **Model finder** -- state your chemistry (elements, benchmark areas, extrapolation
  tolerance) and get a ranked shortlist of models with the evidence behind each.
* **Model report card** -- pick a model to see its score across every chemistry domain
  (with the all-model mean and the model's rank for context), and whether each test is
  in-domain or an extrapolation; physicality is shown separately as soundness.
* **Head-to-head** -- two models compared structure by structure on one readable axis:
  which model was closer to the reference, and by how much, per category. Clicking a
  category opens a per-structure scatter; clicking a point inspects its 3D geometry and
  every model's error.
* **Coverage rose** -- a polar overview of the field: mean model score per benchmark
  area, so solved areas and open problems are visible at a glance.

Each view is built on an established method; the on-page "Methods & references" panel
(:func:`_build_references_panel`, driven by :data:`_REFERENCES`) cites the source for
each -- e.g. the finder's structure score is HELM's Mean Win Rate (Liang et al. 2022)
and the head-to-head is Demšar's (2006) sign test.

Figures are built server-side; the page holds no data of its own.

Contents (this module mixes three layers; a follow-up PR should split them into
``explorer_scoring`` / ``explorer_figures`` / ``explorer_layout`` with this file as a
re-export shim -- kept together here so the review diff stays readable):

1. Component ids + option/style constants.
2. SCORING (pure, no Dash/Plotly): ``_filter_rows``, ``report_card_summary``,
   ``head_to_head_stats``, ``model_finder_rankings``, ``structure_fails_ranking``,
   ``benchmark_error_medians``, ``_wilson_interval``, ``representative_models`` ...
3. FIGURES (``build_*_figure`` + ``_*_empty_figure``): one per view.
4. LAYOUT (``build_explorer_body`` + ``_build_*_panel`` + intro/nav/references).

Section banners (``# === ... ===``) below mark each layer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import math
import statistics
from typing import Any

from dash import dcc, html
import plotly.graph_objects as go

# Data loaders live in explorer_data so this module and the callbacks can share them
# without an import cycle. Re-exported here for callers/tests importing from here.
from ml_peg.app.utils.explorer_data import (
    category_page_paths,
    explorer_data_ready,
    load_benchmark_scores,
    load_error_matrix,
    load_hardness,
    load_model_benchmark_scores,
    load_similarity,
    model_taxonomy,
    split_structure_key,
    structure_model_ranks,
)

__all__ = [
    "load_benchmark_scores",
    "load_error_matrix",
    "load_hardness",
    "load_model_benchmark_scores",
    "load_similarity",
    "model_taxonomy",
    "build_coverage_rose_figure",
    "build_head_to_head_figure",
    "build_h2h_detail_figure",
    "benchmark_error_medians",
    "head_to_head_stats",
    "structure_fails_ranking",
    "build_structure_fails_figure",
    "build_similarity_heatmap_figure",
    "model_finder_rankings",
    "build_finder_figure",
    "report_card_summary",
    "build_report_profile_figure",
    "representative_models",
    "build_explorer_body",
    "build_explorer_layout",
]

# URL path for the Explorer page.
EXPLORER_PATH = "/explorer"

# Component ids (shared with register_explorer_callbacks).
REPORT_MODEL = "explorer-report-model"
REPORT_SUMMARY = "explorer-report-summary"
REPORT_PROFILE = "explorer-report-profile"
REPORT_TABLE = "explorer-report-table"
STRUCT_VIEWER = "explorer-struct-viewer"
STRUCT_TABLE = "explorer-struct-table"
H2H_MODEL_A = "explorer-h2h-model-a"
H2H_MODEL_B = "explorer-h2h-model-b"
H2H_GRAPH = "explorer-h2h-graph"
H2H_DETAIL_GRAPH = "explorer-h2h-detail-graph"
H2H_SUMMARY = "explorer-h2h-summary"
COVERAGE_GRAPH = "explorer-coverage-graph"
COVERAGE_GRANULARITY = "explorer-coverage-granularity"
FINDER_ELEMENTS = "explorer-finder-elements"
FINDER_CATEGORIES = "explorer-finder-categories"
FINDER_TOLERANCE = "explorer-finder-tolerance"
FINDER_PRESET = "explorer-finder-preset"
FINDER_GRAPH = "explorer-finder-graph"
FINDER_PICKS = "explorer-finder-picks"
STRUCT_FAILS_CATEGORIES = "explorer-fails-categories"
STRUCT_FAILS_MODEL = "explorer-fails-model"
STRUCT_FAILS_SORT = "explorer-fails-sort"
STRUCT_FAILS_GRAPH = "explorer-fails-graph"
SIMILARITY_GRAPH = "explorer-similarity-graph"

# Granularity for the coverage rose: one wedge per category, or one per benchmark.
COVERAGE_GRANULARITY_OPTIONS = [
    {"label": "By category", "value": "category"},
    {"label": "By benchmark", "value": "benchmark"},
]

# How the structure-fails explorer ranks the structures models struggle on. When a
# specific model is picked the ranking switches to "where this model fails worst"
# (its error rank vs peers), so this toggle only drives the all-models view.
STRUCT_FAILS_SORT_OPTIONS = [
    {"label": "Hardest (highest error)", "value": "hardest"},
    {"label": "Most contested (models disagree)", "value": "contested"},
]
# The failure bars read red (bad), matching the app-wide bad/good score palette.
_FAILS_BAR_COLORSCALE = "Reds"

# Red -> amber -> green (theme --mlpeg-bad / --mlpeg-warn / --mlpeg-good): a low mean
# score (more model development needed) reads red, a high one (solved) reads green.
COVERAGE_COLORSCALE = [[0.0, "#e5484d"], [0.5, "#d97706"], [1.0, "#16a34a"]]

# Level-of-theory classes for the report card: colour + short label. Blues/purples/grey
# only -- red and green are reserved app-wide for bad/good *scores*, and a red "tested
# against a different functional" bar would read as "the model failed here".
LOT_CLASS_STYLE = {
    "in_domain": {"color": "#0d8ff0", "label": "in-domain"},
    # Mid-lightness values on purpose: these paint plot bars and legend dots in
    # BOTH themes, so each must clear ~3:1 against white and the dark surface
    # (#475569 slate was near-invisible on dark).
    "dft": {"color": "#8b5cf6", "label": "functional mismatch"},
    "high_level": {"color": "#d946ef", "label": "vs CCSD(T)/MP2"},
    "experimental": {"color": "#64748b", "label": "vs experiment"},
    "none": {"color": "#9aa0a6", "label": "soundness"},
}

# Head-to-head winner colours (A = blue, B = amber, tie = grey).
_H2H_COLOUR_A = "#4c6ef5"
_H2H_COLOUR_B = "#f59f00"
_H2H_COLOUR_TIE = "#b0b6bd"

# Model-finder extrapolation tolerance -> admissible LoT classes.
FINDER_TOLERANCE_OPTIONS = [
    {"label": "In-domain evidence only", "value": "in_domain"},
    {"label": "Allow DFT-functional mismatch", "value": "dft"},
    {"label": "Use all evidence", "value": "any"},
]
_TOLERANCE_CLASSES = {
    "in_domain": {"in_domain"},
    "dft": {"in_domain", "dft"},
    "any": {"in_domain", "dft", "high_level", "experimental", "none"},
}

# Model-finder experiment presets: researchers think in experiments, not benchmark
# category names, so each preset selects the relevant category bundle. The category
# multi-select stays editable; touching it switches the preset to "custom" (None).
FINDER_EXPERIMENTS = [
    {"label": "Anything", "value": "any"},
    {"label": "Bulk materials & crystals", "value": "bulk"},
    {"label": "Surfaces, defects & reactions", "value": "surfaces"},
    {"label": "Molecules & non-covalent chemistry", "value": "molecular"},
    {"label": "Metals & f-block chemistry", "value": "metals"},
]
FINDER_EXPERIMENT_CATEGORIES: dict[str, list[str]] = {
    "any": [],
    "bulk": ["bulk_crystal", "molecular_crystal", "molecular_dynamics"],
    "surfaces": ["surfaces", "defect", "nebs", "molecular_reactions"],
    "molecular": [
        "molecular",
        "conformers",
        "isomers",
        "thermochemistry",
        "non_covalent_interactions",
        "supramolecular",
    ],
    "metals": ["tm_complexes", "lanthanides"],
}

# A model whose physicality (soundness) score falls below this gets a warning in the
# finder: accuracy means little if bonds break unphysically mid-simulation.
_PHYSICALITY_WARN = 0.5

# Minimum admissible benchmarks for a model to be ranked at all: fewer than this is too
# thin to trust. This is an *absolute* floor. It replaces an earlier "at least half the
# broadest-covered model's benchmarks" rule which, because the shipped pool is
# molecular-heavy, silently hid materials-only potentials (MACE-MP-0, MatterSim, ...)
# from the default view even though they carry ample materials evidence.
_MIN_FINDER_BENCH = 3
# Coverage is folded into the ranking by empirical-Bayes shrinkage, not a hard cut:
# each model's benchmark mean is pulled towards the pooled mean with this many
# pseudo-observations, so a model that aced a handful of benchmarks can't outrank one
# solid across many. Larger -> stronger shrinkage of low-coverage models to the average.
_FINDER_PRIOR_STRENGTH = 4.0


# ===========================================================================
# SCORING & FIGURES (interleaved by view: each view's pure stats then its
# figure builder). Pure-stats fns (report_card_summary, head_to_head_stats,
# model_finder_rankings, structure_fails_ranking, benchmark_error_medians,
# _wilson_interval, representative_models, _filter_rows, ...) take/return plain
# data and are unit-tested without Dash; build_*_figure turn them into Plotly.
# ===========================================================================


def _categories(rows: list[dict[str, Any]]) -> list[str]:
    """
    List the distinct benchmark categories present, sorted.

    Parameters
    ----------
    rows
        Hardness rows.

    Returns
    -------
    list[str]
        Sorted unique category names.
    """
    return sorted({row["category"] for row in rows})


def _element_options(rows: list[dict[str, Any]]) -> list[str]:
    """
    List the distinct element symbols present across all structures, sorted.

    Parameters
    ----------
    rows
        Hardness rows.

    Returns
    -------
    list[str]
        Sorted unique element symbols (empty element sets contribute nothing).
    """
    elements: set[str] = set()
    for row in rows:
        elements.update(row.get("elements") or [])
    return sorted(elements)


def _filter_rows(
    rows: list[dict[str, Any]],
    categories: list[str] | None,
    elements: list[str] | None,
) -> list[dict[str, Any]]:
    """
    Filter hardness rows by category and chemistry (used by the model finder).

    Parameters
    ----------
    rows
        Hardness rows.
    categories
        Keep only these categories (empty/None = all).
    elements
        Keep only structures containing at least one of these elements (empty/None =
        all). Structures with an unknown element set are excluded once a filter is set.

    Returns
    -------
    list[dict[str, Any]]
        The filtered rows.
    """
    out = rows
    if categories:
        allowed = set(categories)
        out = [row for row in out if row["category"] in allowed]
    if elements:
        wanted = set(elements)
        out = [row for row in out if wanted & set(row.get("elements") or [])]
    return out


# --- Model report card: per-model profile across chemistry + extrapolation ----------


@lru_cache(maxsize=128)
def _model_scores(model: str | None) -> list[dict[str, Any]]:
    """
    Return the per-benchmark score rows for one model.

    Memoized on ``model`` (a scan of the process-stable
    :func:`load_model_benchmark_scores` singleton): the report card rebuilds this for
    the summary, profile figure, and table on every selection, so caching the shared
    scan makes repeat selections and the three co-rendered builders effectively free.
    Callers must treat the returned list as read-only (they sort copies / iterate).

    Parameters
    ----------
    model
        Model display name.

    Returns
    -------
    list[dict[str, Any]]
        Rows from :func:`load_model_benchmark_scores` for that model.
    """
    if not model:
        return []
    return [row for row in load_model_benchmark_scores() if row["model"] == model]


def _agg(rows: list[dict[str, Any]]) -> tuple[float | None, int]:
    """
    Mean score and count for a set of report rows.

    Parameters
    ----------
    rows
        Report rows.

    Returns
    -------
    tuple[float | None, int]
        ``(mean, n)`` -- mean is ``None`` when empty.
    """
    if not rows:
        return None, 0
    return sum(row["score"] for row in rows) / len(rows), len(rows)


def report_card_summary(model: str | None) -> dict[str, Any]:
    """
    Summarise a model's accuracy in-domain vs extrapolated, plus physicality.

    Accuracy benchmarks (those with a reference level, excluding physicality) are split
    by whether the model's training level of theory matches the reference (in-domain) or
    not (extrapolated). Physicality (soundness) is reported separately.

    Parameters
    ----------
    model
        Model display name.

    Returns
    -------
    dict[str, Any]
        ``{model, training_lot, n_benchmarks, in_domain, extrapolated, physicality}``
        where each of the last three is ``(mean, n)``.
    """
    rows = _model_scores(model)
    accuracy = [
        r for r in rows if r["lot_class"] != "none" and r["domain"] != "physicality"
    ]
    in_domain = [r for r in accuracy if r["lot_class"] == "in_domain"]
    extrapolated = [r for r in accuracy if r["lot_class"] != "in_domain"]
    kinds = Counter(r["lot_class"] for r in extrapolated)
    physicality = [r for r in rows if r["domain"] == "physicality"]
    return {
        "model": model,
        "training_lot": rows[0]["model_lot"] if rows else None,
        "n_benchmarks": len(rows),
        "in_domain": _agg(in_domain),
        "extrapolated": _agg(extrapolated),
        "extrapolated_kinds": dict(kinds),
        "physicality": _agg(physicality),
    }


def _category_aggregates(model: str | None) -> list[dict[str, Any]]:
    """
    Aggregate a model's scores to one row per category (mean + LoT class breakdown).

    The category's ``lot_class`` is the *most common* class among its benchmarks (ties
    broken towards in-domain), with the full breakdown kept alongside -- colouring a
    category with 8 in-domain tests and 1 mismatch by the mismatch would misrepresent
    the evidence.

    Parameters
    ----------
    model
        Model display name.

    Returns
    -------
    list[dict[str, Any]]
        ``{category, domain, mean, lot_class, lot_breakdown, n}`` per category the
        model covers; ``lot_breakdown`` is a hover-ready string like
        ``"in-domain 8/9 · functional mismatch 1/9"``.
    """
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _model_scores(model):
        by_cat[row["category"]].append(row)
    out: list[dict[str, Any]] = []
    for category, rows in by_cat.items():
        counts = Counter(r["lot_class"] for r in rows)
        # Most common class wins; "in_domain" wins ties (it sorts before mismatches in
        # LOT_CLASS_STYLE's insertion order, which we use as the tie-break order).
        order = list(LOT_CLASS_STYLE)
        dominant = max(counts, key=lambda cls: (counts[cls], -order.index(cls)))
        breakdown = " · ".join(
            f"{LOT_CLASS_STYLE.get(cls, LOT_CLASS_STYLE['none'])['label']} "
            f"{count}/{len(rows)}"
            for cls, count in counts.most_common()
        )
        out.append(
            {
                "category": category,
                "domain": rows[0]["domain"],
                "mean": sum(r["score"] for r in rows) / len(rows),
                "lot_class": dominant,
                "lot_breakdown": breakdown,
                "n": len(rows),
            }
        )
    return out


@lru_cache(maxsize=1)
def _category_model_means() -> dict[str, dict[str, float]]:
    """
    Mean score per (category, model) across every model's covered benchmarks.

    The report card uses this for comparative context: the all-model mean per category
    (the "how does everyone do here?" tick) and the selected model's rank within the
    cohort covering that category.

    Returns
    -------
    dict[str, dict[str, float]]
        ``category -> {model: mean score}``.
    """
    sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in load_model_benchmark_scores():
        sums[row["category"]][row["model"]].append(row["score"])
    return {
        category: {model: sum(scores) / len(scores) for model, scores in models.items()}
        for category, models in sums.items()
    }


def build_report_profile_figure(model: str | None) -> go.Figure:
    """
    Build the report-card profile: the model's mean score per category, in context.

    Horizontal bars sorted best-to-worst, coloured by the category's most common
    level-of-theory class (full breakdown in hover). Each bar also shows the all-model
    mean for that category as a grey tick and the model's rank in the covering cohort
    ("3rd of 14"), so "0.55 on surfaces" reads as strong or weak *relative to the
    field*, not as a bare number.

    Parameters
    ----------
    model
        Model display name.

    Returns
    -------
    go.Figure
        A horizontal bar figure (empty if the model has no scores).
    """
    # Physicality is soundness, not accuracy -- it has its own summary chip, so keep it
    # off the accuracy profile to avoid conflating the two on one 0-1 axis.
    cats = [c for c in _category_aggregates(model) if c["domain"] != "physicality"]
    if not cats:
        return go.Figure()
    cats.sort(key=lambda c: c["mean"])
    labels = [c["category"] for c in cats]
    values = [c["mean"] for c in cats]
    colours = [
        LOT_CLASS_STYLE.get(c["lot_class"], LOT_CLASS_STYLE["none"])["color"]
        for c in cats
    ]

    cohort = _category_model_means()
    cohort_means: list[float | None] = []
    ranks: list[str] = []
    for c in cats:
        means = cohort.get(c["category"], {})
        cohort_means.append(sum(means.values()) / len(means) if means else None)
        if model in means:
            rank = 1 + sum(1 for v in means.values() if v > means[model])
            ranks.append(f"{_ordinal(rank)} of {len(means)}")
        else:
            ranks.append("")

    customdata = [
        [c["lot_breakdown"], c["n"], rank, cohort_mean or 0]
        for c, rank, cohort_mean in zip(cats, ranks, cohort_means, strict=True)
    ]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colours},
            text=ranks,
            textposition="outside",
            # No colour: inherit the themed layout font so labels read in dark mode.
            textfont={"size": 11},
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>mean score %{x:.2f} · rank %{customdata[2]}<br>"
                "all-model mean %{customdata[3]:.2f}<br>"
                "%{customdata[0]} · %{customdata[1]} benchmarks<extra></extra>"
            ),
        )
    )
    # All-model mean per category as a grey tick, so solved-vs-open areas are visible
    # on the same bars (this is what the old coverage rose was trying to say).
    tick_x = [m for m in cohort_means if m is not None]
    tick_y = [
        label for label, m in zip(labels, cohort_means, strict=True) if m is not None
    ]
    figure.add_trace(
        go.Scatter(
            x=tick_x,
            y=tick_y,
            mode="markers",
            marker={
                "symbol": "line-ns-open",
                "size": 14,
                # Mid-grey reference tick; reads on both plot backgrounds (was #6c757d,
                # which dimmed on dark).
                "color": "#8a8f98",
                "line": {"width": 2},
            },
            name="all-model mean",
            hovertemplate="all-model mean %{x:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=max(320, 26 * len(cats) + 90),
        margin={"l": 170, "r": 60, "t": 10, "b": 44},
        xaxis={"title": "mean score (0–1)", "range": [0, 1.08]},
        yaxis={"tickfont": {"size": 11}, "automargin": True},
        bargap=0.35,
        showlegend=False,
    )
    return figure


def _ordinal(n: int) -> str:
    """
    Format ``n`` as an English ordinal ("1st", "2nd", "3rd", "4th"...).

    Parameters
    ----------
    n
        A positive integer rank.

    Returns
    -------
    str
        The ordinal string.
    """
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def representative_models(names: list[str]) -> list[str]:
    """
    Reduce a list of model variants to one representative per architecture family.

    Keeps the broadest-coverage variant of each family, so cross-model views
    (similarity, committee) show real architectural relationships instead of a model
    versus its own
    ``+D3`` twin.

    Parameters
    ----------
    names
        Model display names.

    Returns
    -------
    list[str]
        One model per family, in the input order of first appearance.
    """
    variants = model_taxonomy()["variants"]
    wanted = set(names)
    coverage: dict[str, int] = defaultdict(int)
    for row in load_model_benchmark_scores():
        if row["model"] in wanted:
            coverage[row["model"]] += 1
    by_family: dict[str, list[str]] = defaultdict(list)
    for name in names:
        family = variants.get(name, {}).get("family", name)
        by_family[family].append(name)
    reps = {
        max(models, key=lambda m: coverage.get(m, 0)) for models in by_family.values()
    }
    return [name for name in names if name in reps]


def _summary_chip(
    label: str, agg: tuple[float | None, int], lot_class: str, sub: str
) -> html.Div:
    """
    Render one report-card summary chip (headline number + context).

    Parameters
    ----------
    label
        Chip title.
    agg
        ``(mean, n)`` from :func:`report_card_summary`.
    lot_class
        LoT class whose accent colour the value takes.
    sub
        One-line explanation beneath the value.

    Returns
    -------
    html.Div
        The chip.
    """
    mean, n = agg
    colour = LOT_CLASS_STYLE.get(lot_class, LOT_CLASS_STYLE["none"])["color"]
    value = f"{mean:.2f}" if mean is not None else "—"
    return html.Div(
        [
            html.Div(label, className="mlpeg-report-chip-label"),
            html.Div(
                value, className="mlpeg-report-chip-value", style={"color": colour}
            ),
            html.Div(f"{n} benchmarks · {sub}", className="mlpeg-report-chip-sub"),
        ],
        className="mlpeg-report-chip",
    )


def render_report_summary(model: str | None) -> html.Div:
    """
    Render the report-card extrapolation summary (training level + the three chips).

    Parameters
    ----------
    model
        Model display name.

    Returns
    -------
    html.Div
        The summary chip row.
    """
    summary = report_card_summary(model)
    training = html.Div(
        [
            html.Div("Trained at", className="mlpeg-report-chip-label"),
            html.Div(
                summary["training_lot"] or "—",
                className="mlpeg-report-chip-value mlpeg-report-chip-value--lot",
            ),
            html.Div(
                f"{summary['n_benchmarks']} benchmarks evaluated",
                className="mlpeg-report-chip-sub",
            ),
        ],
        className="mlpeg-report-chip",
    )
    # Spell out what the extrapolated mean pools, so "0.63" isn't read as one kind of
    # test when it mixes functional-mismatch, CCSD(T)/MP2 and experimental references.
    kind_bits = " / ".join(
        f"{count} {LOT_CLASS_STYLE[cls]['label']}"
        for cls, count in sorted(
            summary["extrapolated_kinds"].items(), key=lambda kv: -kv[1]
        )
    )
    chips = html.Div(
        [
            training,
            _summary_chip(
                "In-domain accuracy",
                summary["in_domain"],
                "in_domain",
                "reference = training level",
            ),
            _summary_chip(
                "Extrapolated accuracy",
                summary["extrapolated"],
                "high_level",
                kind_bits or "reference ≠ training level",
            ),
            _summary_chip(
                "Physicality",
                summary["physicality"],
                "none",
                "sensible physical behaviour, independent of accuracy",
            ),
        ],
        className="mlpeg-report-summary",
    )
    # The methodology note sits under the chip row, not inside it, so it never reads
    # as a fifth (empty) chip.
    caption = html.Div(
        "Chip values are unweighted means of 0–1 benchmark scores.",
        className="mlpeg-report-chip-sub",
        style={"margin": "4px 0 8px", "color": "var(--mlpeg-ink-3)"},
    )
    return html.Div([chips, caption])


def render_report_table(model: str | None) -> html.Table:
    """
    Render the per-benchmark table for the report card (best score first).

    Parameters
    ----------
    model
        Model display name.

    Returns
    -------
    html.Table
        Benchmark · category · score · reference level · test-type (LoT class) rows.
    """
    rows = sorted(_model_scores(model), key=lambda r: r["score"], reverse=True)
    # Score sits right after the benchmark name: long category names can push later
    # columns behind the horizontal scroll, and the score is the column that matters.
    header = html.Tr(
        [
            html.Th("Benchmark"),
            html.Th("Score"),
            html.Th("Category"),
            html.Th("Reference"),
            html.Th("Test"),
        ]
    )
    paths = category_page_paths()
    body = []
    for row in rows:
        style = LOT_CLASS_STYLE.get(row["lot_class"], LOT_CLASS_STYLE["none"])
        reference = ", ".join(row["ref_levels"]) if row["ref_levels"] else "—"
        path = paths.get(row["category"])
        category_cell = (
            dcc.Link(row["category"], href=path) if path else row["category"]
        )
        body.append(
            html.Tr(
                [
                    html.Td(row["benchmark"].split("/")[-1]),
                    html.Td(f"{row['score']:.2f}"),
                    html.Td(category_cell),
                    html.Td(reference, style={"fontSize": "11px"}),
                    html.Td(
                        [
                            html.Span("● ", style={"color": style["color"]}),
                            style["label"],
                        ]
                    ),
                ]
            )
        )
    return html.Table(
        [html.Thead(header), html.Tbody(body)],
        className="mlpeg-explorer-struct-table",
    )


def _report_default_model() -> str | None:
    """
    Choose the report card's default model.

    Prefers the broadest-coverage model that has at least one in-domain test, so the
    opening view illustrates the in-domain vs extrapolated contrast rather than a model
    (like an ωB97M one) that only ever extrapolates against ML-PEG's references.

    Returns
    -------
    str | None
        Model display name, or ``None`` if there are no scores.
    """
    coverage: dict[str, int] = defaultdict(int)
    has_in_domain: set[str] = set()
    for row in load_model_benchmark_scores():
        coverage[row["model"]] += 1
        if row["lot_class"] == "in_domain":
            has_in_domain.add(row["model"])
    if not coverage:
        return None
    candidates = has_in_domain or set(coverage)
    return max(candidates, key=lambda m: coverage[m])


def _model_options() -> list[dict[str, str]]:
    """
    List the options for the single searchable report-card model picker.

    One option per model, grouped visually by sorting on architecture family and
    prefixed with the family label, so "show me MACE-MP-0" is one search-and-click
    instead of a family-then-variant two-step.

    Returns
    -------
    list[dict[str, str]]
        One option per model, family-sorted.
    """
    taxonomy = model_taxonomy()
    variants = taxonomy["variants"]
    options: list[dict[str, str]] = []
    for family_meta in taxonomy["families"].values():
        for m in family_meta["models"]:
            lot = variants.get(m, {}).get("training_lot") or "?"
            options.append(
                {"label": f"{family_meta['label']}  ·  {m}  ·  {lot}", "value": m}
            )
    return options


def head_to_head_stats(
    model_a: str, model_b: str, error_matrix: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """
    Compare two models structure-by-structure over the structures they both cover.

    A model "wins" a structure when its absolute error there is smaller. Both models are
    measured on the *same* structure (same units), so a per-structure win/loss is a
    fair, unit-independent comparison.

    Win rates are **equal-weighted across benchmarks**, not pooled over raw structures:
    each benchmark (``category/benchmark/metric``) contributes one decisive win rate,
    a category averages its benchmarks, and the overall verdict averages the categories.
    Pooling raw structure counts would let one high-volume benchmark dominate the
    verdict -- the sign test / mean-win-rate this view cites weight *scenarios*, not
    instances.
    Ties (exactly equal ``|error|``) are kept for display but excluded from every
    win-rate denominator, so a rate is symmetric (``win_rate_a + win_rate_b == 1``).

    Parameters
    ----------
    model_a, model_b
        Model names to compare.
    error_matrix
        ``structure_key -> {model: signed_residual}`` from :func:`load_error_matrix`.

    Returns
    -------
    dict[str, Any]
        ``{"n_shared", "wins_a", "wins_b", "ties", "overall_win_rate_a",
        "per_category"}``. Each ``per_category`` row is ``{"category", "n", "wins_a",
        "wins_b", "ties", "decisive", "n_benchmarks", "win_rate_a"}`` where
        ``win_rate_a`` is the benchmark-averaged rate and ``n`` the structure count
        (rows sorted by descending structure count).
    """
    # Per benchmark (category/benchmark/metric): [wins_a, wins_b, ties].
    per_bench: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    bench_category: dict[str, str] = {}
    for key, resids in error_matrix.items():
        if model_a not in resids or model_b not in resids:
            continue
        ea, eb = abs(resids[model_a]), abs(resids[model_b])
        category, bench, _ = split_structure_key(key)
        bench_category[bench] = category
        bucket = per_bench[bench]
        if ea < eb:
            bucket[0] += 1
        elif eb < ea:
            bucket[1] += 1
        else:
            bucket[2] += 1

    # Roll benchmarks up into categories: each benchmark's win rate counts once.
    cat_bench_rates: dict[str, list[float]] = defaultdict(list)
    cat_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # a, b, ties
    for bench, (win_a, win_b, tie) in per_bench.items():
        category = bench_category[bench]
        decisive = win_a + win_b
        if decisive:
            cat_bench_rates[category].append(win_a / decisive)
        counts = cat_counts[category]
        counts[0] += win_a
        counts[1] += win_b
        counts[2] += tie

    per_category = []
    for category, (win_a, win_b, tie) in cat_counts.items():
        rates = cat_bench_rates.get(category, [])
        per_category.append(
            {
                "category": category,
                "n": win_a + win_b + tie,  # structures compared (incl. ties)
                "wins_a": win_a,
                "wins_b": win_b,
                "ties": tie,
                "decisive": win_a + win_b,  # for the Wilson whisker
                "n_benchmarks": len(rates),
                # Equal-weight the benchmarks; an all-tie category reads even.
                "win_rate_a": sum(rates) / len(rates) if rates else 0.5,
            }
        )
    per_category.sort(key=lambda row: row["n"], reverse=True)

    # Overall verdict: average the category win rates (each area counts equally).
    cat_rates = [row["win_rate_a"] for row in per_category if row["decisive"]]
    overall_win_rate_a = sum(cat_rates) / len(cat_rates) if cat_rates else 0.5

    wins_a = sum(bucket[0] for bucket in per_bench.values())
    wins_b = sum(bucket[1] for bucket in per_bench.values())
    ties = sum(bucket[2] for bucket in per_bench.values())
    return {
        "n_shared": wins_a + wins_b + ties,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "overall_win_rate_a": overall_win_rate_a,
        "per_category": per_category,
    }


@lru_cache(maxsize=128)
def cached_head_to_head_stats(model_a: str, model_b: str) -> dict[str, Any]:
    """
    Memoized :func:`head_to_head_stats` over the production error matrix.

    The error matrix is a process-stable singleton (:func:`load_error_matrix`), so the
    ``(model_a, model_b)`` pair is a complete cache key. This removes the head-to-head
    view's redundant double scan of ~13k structures per update -- the verdict figure and
    the callback headline both need the stats -- and makes revisiting a pair instant.
    Callers passing their own matrix (tests) call :func:`head_to_head_stats` directly.
    Mirrors :func:`cached_finder_rankings`; the returned dict is read-only.

    Parameters
    ----------
    model_a, model_b
        Model names to compare.

    Returns
    -------
    dict[str, Any]
        The (possibly cached) head-to-head stats (see :func:`head_to_head_stats`).
    """
    return head_to_head_stats(model_a, model_b, load_error_matrix())


def benchmark_error_medians(
    error_matrix: dict[str, dict[str, float]],
) -> dict[str, float]:
    """
    Median absolute error per benchmark, for unit-free head-to-head normalisation.

    Errors are in each benchmark's own units, so pooling raw magnitudes across
    benchmarks is meaningless. Dividing each error by its benchmark's median |error|
    makes the head-to-head axes unitless and comparable, while leaving the per-structure
    win/loss unchanged (both models share a structure's normaliser).

    Parameters
    ----------
    error_matrix
        The per-structure error matrix.

    Returns
    -------
    dict[str, float]
        Mapping ``category/benchmark/metric -> median |error|`` (over all models and
        structures in that benchmark).
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for key, resids in error_matrix.items():
        _, bench, _ = split_structure_key(key)
        for resid in resids.values():
            groups[bench].append(abs(resid))
    return {
        bench: statistics.median(values) for bench, values in groups.items() if values
    }


def _short(name: str | None, limit: int = 14) -> str:
    """
    Truncate a model name for axis ticks and annotations.

    Parameters
    ----------
    name
        Model display name.
    limit
        Maximum length before an ellipsis.

    Returns
    -------
    str
        The (possibly truncated) name.
    """
    name = name or "?"
    return name if len(name) <= limit else name[: limit - 1] + "…"


# A category is a "practical tie" when the win rate sits within ±5% of even. The band is
# our own display heuristic -- a region of practical equivalence (cf. Benavoli et al.
# 2017; Kruschke 2018) to avoid over-reading near-50% splits -- not a standardised
# threshold borrowed from any leaderboard.
_H2H_TIE_BAND = 0.05


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion ``k / n``.

    Preferred over the normal approximation for small ``n`` and proportions near 0/1.
    Used to whisker the head-to-head win rate so a category won on a handful of
    structures is visibly less certain than one won on thousands. Note the structures
    within a benchmark are correlated, so this (independence-assuming) interval is
    optimistic -- it is a floor on the uncertainty, not a calibrated bound.

    Parameters
    ----------
    k
        Number of successes (wins for model A).
    n
        Number of trials (shared structures in the category).
    z
        Normal quantile (1.96 = 95%).

    Returns
    -------
    tuple[float, float]
        ``(low, high)`` bounds in ``[0, 1]``.
    """
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def _h2h_empty_figure(height: int = 460) -> go.Figure:
    """
    Build the placeholder figure for a model pair that shares no structures.

    Parameters
    ----------
    height
        Figure height in px.

    Returns
    -------
    go.Figure
        An annotated empty figure.
    """
    figure = go.Figure()
    figure.update_layout(
        height=height,
        template="plotly_white",
        annotations=[
            {
                "text": "These two models share no structures.",
                "showarrow": False,
                "font": {"size": 13},
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
            }
        ],
    )
    return figure


def build_head_to_head_figure(
    model_a: str,
    model_b: str,
    error_matrix: dict[str, dict[str, float]] | None = None,
) -> go.Figure:
    """
    Build the head-to-head verdict: diverging per-category win-rate bars.

    The signed-difference diverging bar is the standard idiom for "who wins, where,
    by how much" (FT Visual Vocabulary; cf. Chatbot Arena's 0.5-centred win-rate
    convention): one bar per category, extending right (blue) when ``model_a`` was
    closer to the reference on more structures and left (amber) when ``model_b`` was,
    sorted into a tornado so the crossover is visible at a glance. A grey centre band
    marks practical ties (within ±5% of even). Clicking a bar opens that category's
    structure-level scatter (:func:`build_h2h_detail_figure`).

    Parameters
    ----------
    model_a, model_b
        Model names to compare (A = right of centre, B = left).
    error_matrix
        ``structure_key -> {model: signed_residual}``. Omitted in production so the
        cached :func:`cached_head_to_head_stats` is used; tests pass their own matrix
        to compute against it directly.

    Returns
    -------
    go.Figure
        A horizontal bar figure keyed by category on the y axis (so ``clickData``
        identifies the category to drill into).
    """
    # Production omits the matrix and hits the process-wide cache; tests pass their own
    # matrix and compute directly.
    stats = (
        head_to_head_stats(model_a, model_b, error_matrix)
        if error_matrix is not None
        else cached_head_to_head_stats(model_a, model_b)
    )
    ordered = sorted(
        stats["per_category"], key=lambda row: row["win_rate_a"], reverse=True
    )
    if not ordered:
        return _h2h_empty_figure()

    short_a, short_b = _short(model_a), _short(model_b)
    margins = [row["win_rate_a"] - 0.5 for row in ordered]
    # 95% Wilson interval per category, so a bar from a few structures reads as less
    # certain than one from thousands (whiskers are optimistic -- see _wilson_interval).
    # The interval is sized on the *decisive* comparisons (ties excluded) and drawn
    # symmetrically around the equal-weighted bar; its half-widths still shrink with n,
    # so it stays a sample-size cue even though the bar is a benchmark-averaged rate.
    err_plus = []
    err_minus = []
    for row in ordered:
        decisive = row["decisive"]
        if decisive:
            lo, hi = _wilson_interval(row["wins_a"], decisive)
            pooled = row["wins_a"] / decisive
            err_plus.append(hi - pooled)
            err_minus.append(pooled - lo)
        else:
            err_plus.append(0.0)
            err_minus.append(0.0)
    colours = []
    for margin in margins:
        if margin > _H2H_TIE_BAND:
            colours.append(_H2H_COLOUR_A)
        elif margin < -_H2H_TIE_BAND:
            colours.append(_H2H_COLOUR_B)
        else:
            colours.append(_H2H_COLOUR_TIE)

    figure = go.Figure(
        go.Bar(
            x=margins,
            y=[row["category"] for row in ordered],
            orientation="h",
            marker={"color": colours},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": err_plus,
                "arrayminus": err_minus,
                "thickness": 1,
                "width": 3,
                # Mid-grey (matches the reference lines) so the whisker reads on
                # both plot backgrounds; the dark 80/80/80 vanished on dark.
                "color": "rgba(138,143,152,0.65)",
            },
            text=[
                f"{100 * row['win_rate_a']:.0f}% · n={row['n']:,}" for row in ordered
            ],
            textposition="outside",
            # Inherit the themed layout font so the win-rate labels read in dark mode.
            textfont={"size": 11},
            cliponaxis=False,
            customdata=[
                [100 * row["win_rate_a"], row["n"], row["n_benchmarks"]]
                for row in ordered
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{short_a} more accurate in %{{customdata[0]:.0f}}% "
                "(evenly weighted across %{customdata[2]} benchmark(s),<br>"
                "%{customdata[1]:,} structures)<br>"
                "click to see every structure<extra></extra>"
            ),
        )
    )
    # Practical-tie band and centre line, so near-even bars aren't over-read.
    figure.add_shape(
        type="rect",
        x0=-_H2H_TIE_BAND,
        x1=_H2H_TIE_BAND,
        xref="x",
        y0=0,
        y1=1,
        yref="paper",
        fillcolor="#9aa0a6",
        opacity=0.08,
        line={"width": 0},
        layer="below",
    )
    figure.add_vline(x=0, line={"color": "#8a8f98", "width": 1})
    figure.add_annotation(
        x=0.27,
        y=1.05,
        xref="x",
        yref="paper",
        showarrow=False,
        text=f"<b>{short_a} more accurate →</b>",
        font={"size": 11, "color": _H2H_COLOUR_A},
    )
    figure.add_annotation(
        x=-0.27,
        y=1.05,
        xref="x",
        yref="paper",
        showarrow=False,
        text=f"<b>← {short_b} more accurate</b>",
        font={"size": 11, "color": _H2H_COLOUR_B},
    )

    figure.update_layout(
        margin={"l": 170, "r": 120, "t": 42, "b": 55},
        height=600,
        template="plotly_white",
        showlegend=False,
        xaxis={
            "title": (f"{short_a} win rate (benchmark-averaged, per area)"),
            "range": [-0.56, 0.56],
            "tickvals": [-0.25, 0, 0.25],
            "ticktext": ["25%", "50% · even", "75%"],
            "tickfont": {"size": 10},
        },
        yaxis={
            "tickfont": {"size": 11},
            "autorange": "reversed",  # biggest A-wins on top (tornado order)
            "automargin": True,  # widen the left margin to fit long model names
        },
        bargap=0.3,
    )
    return figure


def build_h2h_detail_figure(
    model_a: str,
    model_b: str,
    category: str,
    error_matrix: dict[str, dict[str, float]],
    meta: dict[str, dict[str, Any]],
) -> go.Figure:
    """
    Build one category's structure-level A-vs-B error scatter (the drilldown).

    The classic pairwise-comparison scatter (Demšar 2006; aeon's
    ``plot_pairwise_scatter``): x = model A's error, y = model B's error, one point per
    shared structure, dashed ``y = x`` diagonal. Points above the diagonal are
    structures A predicted more accurately (B's error is larger); colours repeat the
    winner encoding from the verdict bars. At a few hundred points per category this
    stays readable where the full 13k-point version did not. Errors are divided by
    their benchmark's median |error| so benchmarks with different units share axes.

    Parameters
    ----------
    model_a, model_b
        Model names on the x and y axes.
    category
        Benchmark category to restrict to.
    error_matrix
        ``structure_key -> {model: signed_residual}`` from :func:`load_error_matrix`.
    meta
        ``structure_key -> hardness row`` (from :func:`load_hardness`), used to attach
        benchmark/sid/geometry to each point for hover and click-to-inspect.

    Returns
    -------
    go.Figure
        A ``scattergl`` figure whose points carry ``customdata``
        (``[key, category, benchmark, sid, xyz]``) for the shared 3D drilldown.
    """
    medians = benchmark_error_medians(error_matrix)
    xs: list[float] = []
    ys: list[float] = []
    colours: list[str] = []
    customdata: list[list[Any]] = []
    wins_a = wins_b = ties = 0
    for key, resids in error_matrix.items():
        if model_a not in resids or model_b not in resids:
            continue
        row = meta.get(key)
        if row is None or row["category"] != category:
            continue
        med = medians.get(split_structure_key(key)[1]) or 1.0
        floor = 0.01 * med  # keep exact-zero errors on the log axes
        ea, eb = abs(resids[model_a]), abs(resids[model_b])
        xs.append(max(ea, floor) / med)
        ys.append(max(eb, floor) / med)
        if ea < eb:
            colours.append(_H2H_COLOUR_A)
            wins_a += 1
        elif eb < ea:
            colours.append(_H2H_COLOUR_B)
            wins_b += 1
        else:
            colours.append(_H2H_COLOUR_TIE)
            ties += 1
        customdata.append(
            [key, row["category"], row["benchmark"], row["sid"], row["xyz"]]
        )
    if not xs:
        return _h2h_empty_figure()

    short_a, short_b = _short(model_a), _short(model_b)
    lo, hi = min(min(xs), min(ys)), max(max(xs), max(ys))
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=xs,
            y=ys,
            mode="markers",
            marker={"size": 6, "color": colours, "opacity": 0.7, "line": {"width": 0}},
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[3]}</b> (%{customdata[2]})<br>"
                f"{short_a}: %{{x:.2f}}× benchmark median<br>"
                f"{short_b}: %{{y:.2f}}× benchmark median<br>"
                "click for 3D structure<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line={"dash": "dash", "color": "#8a8f98", "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # Winner halves in plain words: above the diagonal, B's error is larger.
    figure.add_annotation(
        x=0.03,
        y=0.97,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="left",
        text=f"<b>{short_a} more accurate</b>",
        font={"size": 11, "color": _H2H_COLOUR_A},
    )
    figure.add_annotation(
        x=0.97,
        y=0.03,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="right",
        text=f"<b>{short_b} more accurate</b>",
        font={"size": 11, "color": _H2H_COLOUR_B},
    )
    figure.update_layout(
        title={
            "text": (
                f"{category} — {short_a} wins {wins_a:,} · "
                f"{short_b} wins {wins_b:,} · {ties:,} ties"
            ),
            "font": {"size": 13},
        },
        margin={"l": 70, "r": 20, "t": 45, "b": 55},
        height=460,
        template="plotly_white",
        showlegend=False,
        xaxis={"title": f"{short_a} error (× benchmark median)", "type": "log"},
        yaxis={"title": f"{short_b} error (× benchmark median)", "type": "log"},
    )
    return figure


# --- Model finder: "which model for MY system?" --------------------------------------


def model_finder_rankings(
    categories: list[str] | None = None,
    elements: list[str] | None = None,
    tolerance: str = "any",
) -> list[dict[str, Any]]:
    """
    Rank models for a user-described use case, with the evidence behind each.

    Two unit-free evidence streams are combined:

    * **Benchmark evidence** -- the model's mean 0-1 score over benchmarks in the
      selected categories whose test kind passes the extrapolation tolerance.
    * **Structure evidence** (when elements are selected) -- the model's mean cohort
      standing (1 − normalised error rank, see :func:`structure_model_ranks`) over
      structures containing the requested elements.

    Models with fewer than :data:`_MIN_FINDER_BENCH` admissible benchmarks are too thin
    to rank and are dropped. Coverage otherwise enters through empirical-Bayes shrinkage
    (:data:`_FINDER_PRIOR_STRENGTH`): each benchmark mean is pulled towards the pooled
    mean by its coverage, so a model that aced its only few benchmarks cannot outrank
    one solid across many -- without silently hiding well-covered specialists (the
    earlier "half the broadest candidate" rule hid materials-only models by default).

    Parameters
    ----------
    categories
        Benchmark categories to restrict to (empty/None = all).
    elements
        Element symbols the user's system contains (empty/None = skip structure
        evidence).
    tolerance
        One of :data:`FINDER_TOLERANCE_OPTIONS` values -- how much extrapolation the
        benchmark evidence may include.

    Returns
    -------
    list[dict[str, Any]]
        Ranked rows ``{model, fit, bench_mean, n_bench, in_domain_frac, struct_score,
        n_struct, no_element_evidence, physicality, low_physicality, training_lot}``,
        best first.
    """
    admissible = _TOLERANCE_CLASSES.get(tolerance, _TOLERANCE_CLASSES["any"])
    wanted_cats = set(categories or [])
    bench: dict[str, list[float]] = defaultdict(list)
    in_domain_n: dict[str, int] = defaultdict(int)
    phys: dict[str, list[float]] = defaultdict(list)
    lots: dict[str, str | None] = {}
    for row in load_model_benchmark_scores():
        lots.setdefault(row["model"], row["model_lot"])
        # Physicality (soundness) is tracked for every model regardless of the
        # category/tolerance selection: an unphysical model is a bad pick everywhere.
        if row["domain"] == "physicality":
            phys[row["model"]].append(row["score"])
            continue
        if wanted_cats and row["category"] not in wanted_cats:
            continue
        if row["lot_class"] not in admissible:
            continue
        bench[row["model"]].append(row["score"])
        if row["lot_class"] == "in_domain":
            in_domain_n[row["model"]] += 1
    if not bench:
        return []
    candidates = {
        model: scores
        for model, scores in bench.items()
        if len(scores) >= _MIN_FINDER_BENCH
    }
    if not candidates:
        return []
    # Empirical-Bayes prior: the pooled mean over every admissible score. Low-coverage
    # models' benchmark means are shrunk towards this, so thin evidence can't fluke to
    # the top and coverage -- not just the point score -- shapes the ranking.
    pooled = [score for scores in candidates.values() for score in scores]
    prior = sum(pooled) / len(pooled)

    standings: dict[str, list[float]] = defaultdict(list)
    if elements:
        ranks = structure_model_ranks()
        rows = _filter_rows(load_hardness(), categories or None, elements)
        for row in rows:
            for model, rank in ranks.get(row["key"], {}).items():
                standings[model].append(1.0 - rank)

    results: list[dict[str, Any]] = []
    for model, scores in candidates.items():
        bench_mean = sum(scores) / len(scores)
        # Shrink the benchmark mean towards the prior in proportion to (in)coverage.
        bench_adj = (len(scores) * bench_mean + _FINDER_PRIOR_STRENGTH * prior) / (
            len(scores) + _FINDER_PRIOR_STRENGTH
        )
        struct = standings.get(model)
        struct_score = sum(struct) / len(struct) if struct else None
        if elements and struct_score is not None:
            fit = 0.6 * bench_adj + 0.4 * struct_score
        else:
            fit = bench_adj
        phys_scores = phys.get(model)
        physicality = sum(phys_scores) / len(phys_scores) if phys_scores else None
        results.append(
            {
                "model": model,
                "fit": fit,
                "bench_mean": bench_mean,
                "n_bench": len(scores),
                "in_domain_frac": in_domain_n[model] / len(scores),
                "struct_score": struct_score,
                "n_struct": len(struct) if struct else 0,
                "no_element_evidence": bool(elements) and not struct,
                "physicality": physicality,
                "low_physicality": (
                    physicality is not None and physicality < _PHYSICALITY_WARN
                ),
                "training_lot": lots.get(model),
            }
        )
    results.sort(key=lambda r: r["fit"], reverse=True)
    return results


@lru_cache(maxsize=64)
def cached_finder_rankings(
    categories: tuple[str, ...] | None,
    elements: tuple[str, ...] | None,
    tolerance: str = "any",
) -> list[dict[str, Any]]:
    """
    Memoized :func:`model_finder_rankings` keyed on hashable use-case controls.

    The ranking (empirical-Bayes shrinkage over every benchmark, plus optional
    structure evidence) is a pure function of these inputs and is re-run on every
    finder tweak; caching lets repeated selections return instantly. Consumers
    (:func:`build_finder_figure`, :func:`render_finder_picks`) only read the rows,
    so returning the cached list is safe.

    Parameters
    ----------
    categories, elements
        Selections as hashable tuples (``None`` for "all"/"skip").
    tolerance
        Extrapolation tolerance (see :func:`model_finder_rankings`).

    Returns
    -------
    list[dict[str, Any]]
        The (possibly cached) ranked rows.
    """
    return model_finder_rankings(
        categories=list(categories) if categories else None,
        elements=list(elements) if elements else None,
        tolerance=tolerance,
    )


def build_finder_figure(results: list[dict[str, Any]], top: int = 12) -> go.Figure:
    """
    Render the model-finder ranking as a horizontal bar chart with evidence in hover.

    Parameters
    ----------
    results
        Ranked rows from :func:`model_finder_rankings`.
    top
        How many models to show.

    Returns
    -------
    go.Figure
        Bars best-first (top of chart); clicking a bar loads that model's report card.
    """
    shown = results[:top]
    if not shown:
        figure = go.Figure()
        figure.update_layout(
            height=400,
            template="plotly_white",
            annotations=[
                {
                    "text": "No model has enough evidence for this selection — "
                    "widen the categories or tolerance.",
                    "showarrow": False,
                    "font": {"size": 13},
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                }
            ],
        )
        return figure

    shown = list(reversed(shown))  # best at the top of a horizontal bar chart
    hover: list[str] = []
    colours: list[str] = []
    for row in shown:
        bits = [
            f"benchmark score {row['bench_mean']:.2f} ({row['n_bench']} benchmarks)"
        ]
        if row["struct_score"] is not None:
            bits.append(
                f"element evidence {row['struct_score']:.2f} "
                f"({row['n_struct']:,} structures)"
            )
        if row["no_element_evidence"]:
            bits.append("⚠ no tested structures contain your elements")
        if row.get("physicality") is not None:
            bits.append(f"physicality {row['physicality']:.2f}")
        if row.get("low_physicality"):
            bits.append("⚠ low physicality — may behave unphysically")
        hover.append("<br>".join(bits))
        if row["no_element_evidence"]:
            colours.append("#9aa0a6")
        elif row.get("low_physicality"):
            colours.append("#d97706")
        else:
            colours.append("#0d8ff0")

    figure = go.Figure(
        go.Bar(
            x=[row["fit"] for row in shown],
            y=[row["model"] for row in shown],
            orientation="h",
            marker={"color": colours},
            text=[f"{row['fit']:.2f}" for row in shown],
            textposition="outside",
            # Inherit the themed layout font so the fit labels read in dark mode.
            textfont={"size": 11},
            cliponaxis=False,
            customdata=hover,
            hovertemplate="<b>%{y}</b><br>fit %{x:.2f}<br>%{customdata}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=400,  # fixed: the dcc.Graph wrapper matches, preventing overflow
        margin={"l": 190, "r": 50, "t": 10, "b": 44},
        xaxis={"title": "fit for your selection (0–1)", "range": [0, 1.08]},
        yaxis={"tickfont": {"size": 11}, "automargin": True},
        bargap=0.3,
    )
    return figure


def render_finder_picks(results: list[dict[str, Any]], top: int = 3) -> html.Div:
    """
    Render the finder's top picks as evidence cards ("why this model?").

    Each card spells out the evidence in sentences a researcher can quote: benchmark
    score with coverage and in-domain share, element-level standing, training level,
    and a physicality check with an explicit warning when it is poor. Cards are
    clickable (pattern-matched id) and load the model's report card.

    Parameters
    ----------
    results
        Ranked rows from :func:`model_finder_rankings`.
    top
        How many picks to show.

    Returns
    -------
    html.Div
        The card row (empty Div when there are no results).
    """
    if not results:
        return html.Div()
    cards = []
    for i, row in enumerate(results[:top]):
        lines: list[Any] = [
            html.Div(
                [
                    html.Span(f"#{i + 1}  ", style={"color": "var(--mlpeg-ink-3)"}),
                    html.Span(row["model"], style={"fontWeight": "700"}),
                    html.Span(
                        f"  ·  fit {row['fit']:.2f}",
                        style={"color": "var(--mlpeg-accent)", "fontWeight": "600"},
                    ),
                ],
                # Break long model ids inside the card so they don't force its width.
                style={
                    "fontSize": "14px",
                    "marginBottom": "6px",
                    "overflowWrap": "anywhere",
                },
            ),
            html.Div(
                f"Trained at {row['training_lot'] or 'unknown level'}",
                className="mlpeg-report-chip-sub",
            ),
            html.Div(
                f"Benchmark evidence: {row['bench_mean']:.2f} over "
                f"{row['n_bench']} benchmarks "
                f"({100 * row['in_domain_frac']:.0f}% in-domain)",
                className="mlpeg-report-chip-sub",
            ),
        ]
        if row["struct_score"] is not None:
            lines.append(
                html.Div(
                    f"Your elements: typically more accurate than "
                    f"{100 * row['struct_score']:.0f}% of models across "
                    f"{row['n_struct']:,} test structures",
                    className="mlpeg-report-chip-sub",
                )
            )
        elif row["no_element_evidence"]:
            lines.append(
                html.Div(
                    "⚠ No tested structures contain your elements",
                    className="mlpeg-report-chip-sub",
                    style={"color": "var(--mlpeg-warn)"},
                )
            )
        if row["physicality"] is not None:
            phys_style = (
                {"color": "var(--mlpeg-warn)"} if row["low_physicality"] else {}
            )
            phys_note = " ⚠ may behave unphysically" if row["low_physicality"] else " ✓"
            lines.append(
                html.Div(
                    f"Physicality: {row['physicality']:.2f}{phys_note}",
                    className="mlpeg-report-chip-sub",
                    style=phys_style,
                )
            )
        lines.append(
            html.Div(
                "Open report card →",
                style={
                    "fontSize": "11px",
                    "color": "var(--mlpeg-accent)",
                    "marginTop": "6px",
                },
            )
        )
        cards.append(
            html.Div(
                lines,
                id={"type": "explorer-finder-pick", "model": row["model"]},
                className="mlpeg-report-chip",
                # min-width: 0 lets the flex item shrink instead of a long id widening
                # the card and pushing its siblings around.
                style={
                    "cursor": "pointer",
                    "flex": "1 1 260px",
                    "maxWidth": "360px",
                    "minWidth": "0",
                },
                n_clicks=0,
            )
        )
    return html.Div(
        cards,
        style={
            "display": "flex",
            "gap": "12px",
            "flexWrap": "wrap",
            "marginTop": "10px",
        },
    )


# --- Coverage rose: the field's progress at a glance ---------------------------------


def _coverage_wedges(
    rows: list[dict[str, Any]], granularity: str
) -> list[dict[str, Any]]:
    """
    Reduce benchmark rows to the wedges the rose draws, ordered for a clean ring.

    Parameters
    ----------
    rows
        Benchmark rows from :func:`load_benchmark_scores`.
    granularity
        ``"category"`` collapses each category to one wedge (mean over its benchmarks);
        ``"benchmark"`` keeps one wedge per benchmark, ordered by category so a
        category's benchmarks sit contiguously on the ring.

    Returns
    -------
    list[dict[str, Any]]
        Wedges with ``label``, ``category``, ``mean_score`` and ``n``, ordered.
    """
    if granularity == "category":
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["category"], []).append(row)
        wedges = [
            {
                "label": category,
                "category": category,
                "mean_score": sum(r["mean_score"] for r in items) / len(items),
                "n": len(items),
            }
            for category, items in grouped.items()
        ]
        return sorted(wedges, key=lambda w: w["label"])

    return [
        {
            "label": row["benchmark"].split("/")[-1],
            "category": row["category"],
            "mean_score": row["mean_score"],
            "n": row["n_models"],
        }
        for row in sorted(rows, key=lambda r: (r["category"], r["benchmark"]))
    ]


def build_coverage_rose_figure(
    rows: list[dict[str, Any]], granularity: str = "category"
) -> go.Figure:
    """
    Build the coverage rose: a polar bar chart of mean model score per area.

    A Nightingale-rose overview of the *field*, not any one model: each wedge's length
    and colour show the mean score over every evaluated model, so long green wedges are
    areas current MLIPs already handle and short red wedges are open problems.

    Parameters
    ----------
    rows
        Benchmark rows from :func:`load_benchmark_scores`.
    granularity
        ``"category"`` (one wedge per category) or ``"benchmark"`` (one per benchmark).

    Returns
    -------
    go.Figure
        A ``Barpolar`` figure; empty if no scores are available.
    """
    wedges = _coverage_wedges(rows, granularity)
    if not wedges:
        return go.Figure()

    labels = [w["label"] for w in wedges]
    scores = [w["mean_score"] for w in wedges]
    n_unit = "models" if granularity == "benchmark" else "benchmarks"
    customdata = [[w["category"], w["n"]] for w in wedges]

    figure = go.Figure(
        go.Barpolar(
            r=scores,
            theta=labels,
            marker={
                "color": scores,
                "colorscale": COVERAGE_COLORSCALE,
                "cmin": 0.0,
                "cmax": 1.0,
                "colorbar": {
                    "title": {"text": "mean score, all models", "side": "right"},
                    "tickvals": [0, 0.5, 1],
                    "ticktext": ["0 · open problem", "0.5", "1 · solved"],
                },
                "line": {"color": "#ffffff", "width": 1},
            },
            customdata=customdata,
            hovertemplate=(
                "<b>%{theta}</b> (%{customdata[0]})<br>"
                "mean score across models: %{r:.2f}<br>"
                "%{customdata[1]} " + n_unit + "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=560,
        margin={"l": 40, "r": 40, "t": 20, "b": 20},
        showlegend=False,
        polar={
            "radialaxis": {
                "range": [0, 1],
                "tickvals": [0.25, 0.5, 0.75, 1.0],
                "angle": 90,
                # No tick colour / gridcolor: theme_plots.js sets both from tokens.
                "tickfont": {"size": 11},
            },
            "angularaxis": {
                "rotation": 90,
                "direction": "clockwise",
                "tickfont": {"size": 11},
            },
        },
    )
    return figure


def _chip(text: str) -> html.Div:
    """
    Render a small info chip matching the app's styling.

    Parameters
    ----------
    text
        Chip label.

    Returns
    -------
    html.Div
        The chip element.
    """
    return html.Div(text, className="mlpeg-chip")


def _default_h2h_models(
    models: list[str], error_matrix: dict[str, dict[str, float]]
) -> tuple[str | None, str | None]:
    """
    Pick the two most broadly-evaluated models as the head-to-head default pair.

    The similarity model list is alphabetical, and its first two models can be from
    disjoint domains (e.g. an organic-only model vs a materials model) that share no
    structures. Defaulting to the two broadest-coverage models guarantees a populated
    comparison on first view.

    Parameters
    ----------
    models
        Model names.
    error_matrix
        The per-structure error matrix.

    Returns
    -------
    tuple[str | None, str | None]
        The two highest-coverage model names (or ``None`` where unavailable).
    """
    coverage: dict[str, int] = defaultdict(int)
    for resids in error_matrix.values():
        for model in resids:
            coverage[model] += 1
    ranked = sorted(models, key=lambda m: coverage.get(m, 0), reverse=True)
    first = ranked[0] if ranked else None
    second = ranked[1] if len(ranked) > 1 else None
    return first, second


def _dropdown_control(label: str, dropdown: dcc.Dropdown) -> html.Div:
    """
    Wrap a labelled dropdown in the app's control styling.

    Parameters
    ----------
    label
        Control label.
    dropdown
        The dropdown component.

    Returns
    -------
    html.Div
        The labelled control.
    """
    return html.Div([html.Label(label, className="mlpeg-control-label"), dropdown])


def _a11y_graph(graph: dcc.Graph, label: str) -> html.Div:
    """
    Wrap an interactive graph with an accessible name for assistive technology.

    A ``dcc.Graph`` renders an opaque Plotly canvas with no accessible name, so a
    screen reader announces nothing. The wrapper carries ``role="img"`` and an
    ``aria-label`` describing the chart (the standard text-alternative pattern for a
    complex image); the labelled dropdowns beside each chart remain the
    keyboard-navigable path to the same data.

    Parameters
    ----------
    graph
        The graph to wrap (kept inside a ``dcc.Loading`` spinner).
    label
        The accessible name announced for the chart.

    Returns
    -------
    html.Div
        The loading-wrapped graph carrying an accessible name.
    """
    return html.Div(dcc.Loading(graph), role="img", **{"aria-label": label})


def _panel_help(*paragraphs: str, summary: str = "How this works") -> html.Details:
    """
    Build a collapsed "how this works" disclosure for a panel's methodology prose.

    Each panel leads with its heading and chart; the long explanation lives here so the
    page reads as data first, prose on demand (styling via ``.mlpeg-explorer-help``).

    Parameters
    ----------
    *paragraphs
        One or more prose paragraphs to reveal when expanded.
    summary
        The always-visible toggle label.

    Returns
    -------
    html.Details
        The collapsed help block.
    """
    return html.Details(
        [html.Summary(summary), *[html.P(text) for text in paragraphs]],
        className="mlpeg-explorer-help",
    )


def _explorer_empty_figure(
    message: str = "Nothing to show for this selection yet.", height: int = 360
) -> go.Figure:
    """
    Build a blank placeholder figure with a short centred message.

    Parameters
    ----------
    message
        The text to show.
    height
        Figure height in pixels.

    Returns
    -------
    go.Figure
        An axis-free figure with the message centred.
    """
    figure = go.Figure()
    figure.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=message,
        font={"size": 13},
    )
    figure.update_layout(
        height=height,
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return figure


def structure_fails_ranking(
    rows: list[dict[str, Any]],
    categories: list[str] | None,
    model: str | None,
    sort: str,
    top: int = 15,
) -> list[dict[str, Any]]:
    """
    Rank the structures models fail on hardest, or where one model fails worst.

    With no model selected the ranking is field-wide: by difficulty percentile
    (``"hardest"``) or by how much the models disagree (``"contested"``). Picking a
    model instead surfaces the structures where *that* model has the worst error rank
    among the models run on it (its per-structure evidence, after HELM's mean win rate),
    so a developer sees exactly where their model loses ground.

    Both field-wide keys are unit-free (a within-benchmark percentile and a coefficient
    of variation), so the ranking is comparable across benchmarks with different units;
    raw ``median_err`` is not, which is why it is shown but never sorted on.

    Parameters
    ----------
    rows
        Hardness rows from :func:`load_hardness`.
    categories
        Restrict to these benchmark categories (empty/None = all).
    model
        If set, rank by this model's per-structure error rank instead of difficulty.
    sort
        ``"hardest"`` or ``"contested"`` (ignored when ``model`` is set).
    top
        Keep at most this many structures.

    Returns
    -------
    list[dict[str, Any]]
        The worst-first structures, each a hardness row plus ``fail_score`` (the ranking
        value in ``[0, 1]``) and, in per-model mode, ``model_rank`` and ``model_err``.
    """
    filtered = _filter_rows(rows, categories, None)
    if model:
        ranks = structure_model_ranks()
        errors = load_error_matrix()
        scored: list[dict[str, Any]] = []
        for row in filtered:
            rank = ranks.get(row["key"], {}).get(model)
            if rank is None:  # model was not evaluated on this structure
                continue
            resid = errors.get(row["key"], {}).get(model)
            scored.append(
                {
                    **row,
                    "fail_score": rank,
                    "model_rank": rank,
                    "model_err": abs(resid) if resid is not None else None,
                }
            )
        scored.sort(key=lambda r: r["fail_score"], reverse=True)
        return scored[:top]
    key = "disagreement" if sort == "contested" else "hardness_pct"
    ordered = sorted(filtered, key=lambda r: r.get(key) or 0.0, reverse=True)
    return [{**row, "fail_score": row.get(key) or 0.0} for row in ordered[:top]]


@lru_cache(maxsize=128)
def cached_structure_fails_ranking(
    categories: tuple[str, ...] | None,
    model: str | None,
    sort: str,
    top: int = 15,
) -> list[dict[str, Any]]:
    """
    Memoized :func:`structure_fails_ranking` over the production hardness rows.

    The hardness rows are a process-stable singleton (:func:`load_hardness`), so the
    ``(categories, model, sort, top)`` controls form a complete cache key. The panel
    re-filters and re-sorts thousands of rows on every area/model/sort toggle, so a
    repeated selection becomes instant. Callers passing their own rows (tests) call
    :func:`structure_fails_ranking` directly. Mirrors :func:`cached_finder_rankings`.

    Parameters
    ----------
    categories
        Benchmark areas as a hashable tuple (``None`` for all).
    model
        Focus model, or ``None`` for the field-wide ranking.
    sort
        ``"hardest"`` or ``"contested"`` (ignored when a model is focused).
    top
        Maximum rows to return.

    Returns
    -------
    list[dict[str, Any]]
        The (possibly cached) ranked rows (see :func:`structure_fails_ranking`).
    """
    return structure_fails_ranking(
        load_hardness(), list(categories) if categories else None, model, sort, top
    )


def build_structure_fails_figure(
    rows: list[dict[str, Any]] | None = None,
    categories: list[str] | None = None,
    model: str | None = None,
    sort: str = "hardest",
    top: int = 15,
) -> go.Figure:
    """
    Build the structure-fails bars: the hardest structures, worst at the top.

    Each bar is one structure; its length is the ranking value (difficulty percentile,
    disagreement, or -- with a model picked -- that model's error rank vs peers). Bars
    read red (bad), and clicking one opens its 3D geometry and every model's error in
    the shared detail panel below.

    Parameters
    ----------
    rows
        Hardness rows. Omitted in production so the cached
        :func:`cached_structure_fails_ranking` is used; tests pass their own rows to
        rank against them directly.
    categories, model, sort, top
        Forwarded to :func:`structure_fails_ranking`.

    Returns
    -------
    go.Figure
        A horizontal bar figure whose points carry
        ``[key, category, benchmark, sid, xyz, ...]`` in ``customdata`` (the first five
        match the head-to-head detail scatter, so both feed one click handler).
    """
    # Production omits rows and hits the process-wide cache; tests pass their own rows
    # and rank directly.
    ranked = (
        structure_fails_ranking(rows, categories, model, sort, top)
        if rows is not None
        else cached_structure_fails_ranking(
            tuple(categories) if categories else None, model, sort, top
        )
    )
    if not ranked:
        return _explorer_empty_figure("No structures match this selection.", height=360)
    if model:
        axis_title = f"{_short(model)} error rank vs peers (1 = worst)"
    elif sort == "contested":
        axis_title = "model disagreement (coefficient of variation)"
    else:
        axis_title = "difficulty percentile within benchmark"
    xvals = [row["fail_score"] for row in ranked]
    # A leading rank number keeps y labels unique even when two benchmarks share a sid.
    labels = [
        f"{i + 1}. {_short(row['sid'], 22)} · {row['category']}"
        for i, row in enumerate(ranked)
    ]
    customdata = [
        [
            row["key"],
            row["category"],
            row["benchmark"],
            row["sid"],
            row["xyz"],
            row["median_err"],
            row["disagreement"],
            row["worst_model"],
            row["n_models"],
        ]
        for row in ranked
    ]
    figure = go.Figure(
        go.Bar(
            x=xvals,
            y=labels,
            orientation="h",
            marker={
                "color": xvals,
                "colorscale": _FAILS_BAR_COLORSCALE,
                "cmin": 0,
                "cmax": max(xvals) or 1.0,
            },
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[3]}</b><br>"
                "%{customdata[1]} · %{customdata[2]}<br>"
                "median |error| across %{customdata[8]} models: "
                "%{customdata[5]:.4g}<br>"
                "disagreement (CoV): %{customdata[6]:.2f}<br>"
                "worst model here: %{customdata[7]}<br>"
                "click to inspect in 3D<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        margin={"l": 230, "r": 30, "t": 20, "b": 46},
        height=max(340, 26 * len(ranked) + 90),
        template="plotly_white",
        showlegend=False,
        xaxis={"title": axis_title, "tickfont": {"size": 10}},
        yaxis={
            "tickfont": {"size": 10},
            "automargin": True,
            "autorange": "reversed",  # worst (rank 1) at the top
        },
        bargap=0.28,
    )
    return figure


def build_similarity_heatmap_figure() -> go.Figure:
    """
    Build the who-fails-alike heatmap: model x model error correlation, clustered.

    Signed residuals are z-scored within each benchmark, then correlated pairwise over
    shared structures (:func:`load_similarity`); warm cells are model pairs that err in
    the same direction on the same structures -- i.e. that tend to fail alike, so an
    ensemble of them adds little. Rows and columns follow the payload's hierarchical
    clustering order, so behavioural families sit together.

    Returns
    -------
    go.Figure
        The heatmap, or a short placeholder when too few models share evaluations.
    """
    sim = load_similarity()
    models = sim.get("models", [])
    corr = sim.get("corr", [])
    if len(models) < 2 or not corr:
        return _explorer_empty_figure(
            "Not enough shared evaluations to compare models yet.", height=360
        )
    order = sim.get("order") or list(range(len(models)))
    labels = [models[k] for k in order]  # full names stay unique (heatmap axes need it)
    z = [[corr[oi][oj] for oj in order] for oi in order]
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="RdBu",
            reversescale=True,  # high correlation -> red (these two fail alike)
            zmid=0.0,
            zmin=-1.0,
            zmax=1.0,
            hovertemplate=(
                "%{y}<br>vs %{x}<br>error correlation: %{z:.2f}<extra></extra>"
            ),
            colorbar={"title": "corr", "thickness": 12, "tickfont": {"size": 9}},
        )
    )
    figure.update_layout(
        margin={"l": 150, "r": 20, "t": 20, "b": 150},
        height=max(380, 22 * len(labels) + 190),
        template="plotly_white",
        xaxis={"tickangle": -45, "tickfont": {"size": 9}},
        yaxis={"tickfont": {"size": 9}, "autorange": "reversed"},
    )
    return figure


# ===========================================================================
# PAGE LAYOUT: page assembly, the seven panels, intro + jump nav, references.
# Everything below is Dash components; it consumes the scoring/figures above.
# ===========================================================================


def build_explorer_layout() -> html.Div:
    """
    Return the Explorer page.

    The data is warmed at app startup (:func:`warm_explorer_data`), so this is instant.
    If a data issue left the cache unbuilt, a short guidance message is shown rather
    than computing on the routing callback (which would freeze the UI).

    Returns
    -------
    html.Div
        The page content for the ``/explorer`` route.
    """
    if not explorer_data_ready():
        return html.Div(
            [
                html.H1("ML-PEG Model Explorer"),
                html.P(
                    "The explorer data could not be built from the shipped benchmarks. "
                    "Check the server logs and reload."
                ),
            ]
        )
    return build_explorer_body()


def build_explorer_body() -> html.Div:
    """
    Build the full Explorer panels (assumes the data is computed and cached).

    Returns
    -------
    html.Div
        The finder, report-card, head-to-head (with its shared 3D structure detail),
        and coverage-rose panels. A guidance message if no benchmark data is shipped.
    """
    rows = load_hardness()
    if not rows:
        return html.Div(
            [
                html.H1("ML-PEG Model Explorer"),
                html.P(
                    "No benchmark data is available to explore yet. Add benchmark "
                    "results under ml_peg/app/data and reload."
                ),
            ]
        )

    similarity = load_similarity()
    models = similarity.get("models", [])
    n_models = len(models)
    # Default the comparison to representatives of two *different* families -- a model
    # vs its own +D3 twin is a near-degenerate first view.
    reps = representative_models(models) or models
    default_a, default_b = _default_h2h_models(reps, load_error_matrix())
    categories = _categories(rows)
    element_options = _element_options(rows)
    # Label head-to-head models with their training level of theory so a cross-level
    # comparison (itself informative) is explicit.
    variant_meta = model_taxonomy()["variants"]
    h2h_options = [
        {
            "label": f"{m}  ·  {variant_meta.get(m, {}).get('training_lot') or '?'}",
            "value": m,
        }
        for m in models
    ]

    return html.Div(
        [
            _build_intro(len(rows), n_models),
            html.Div(
                _build_finder_panel(categories, element_options),
                id="explorer-sec-finder",
            ),
            html.Div(_build_report_panel(), id="explorer-sec-report"),
            html.Div(
                _build_h2h_panel(h2h_options, default_a, default_b),
                id="explorer-sec-h2h",
            ),
            # Both the structure-fails bars and the head-to-head drilldown feed the one
            # 3D detail panel, so they bracket it.
            html.Div(
                _build_structure_fails_panel(categories, _model_options()),
                id="explorer-sec-fails",
            ),
            html.Div(_build_detail_panel(), id="explorer-sec-detail"),
            html.Div(_build_similarity_panel(), id="explorer-sec-alike"),
            html.Div(_build_coverage_panel(), id="explorer-sec-coverage"),
            _build_references_panel(),
        ],
        className="mlpeg-explorer",
    )


# The Explorer's section jump nav: (label, anchor id) for each panel wrapper above.
_EXPLORER_JUMP_NAV: list[tuple[str, str]] = [
    ("Find a model", "explorer-sec-finder"),
    ("Report card", "explorer-sec-report"),
    ("Head-to-head", "explorer-sec-h2h"),
    ("Where models fail", "explorer-sec-fails"),
    ("Who fails alike", "explorer-sec-alike"),
    ("Coverage", "explorer-sec-coverage"),
]


def _build_intro(n_rows: int, n_models: int) -> html.Div:
    """
    Build the Explorer intro: heading, stat chips, and orientation copy.

    Parameters
    ----------
    n_rows
        Number of benchmark structures.
    n_models
        Number of evaluated models.

    Returns
    -------
    html.Div
        The intro section.
    """
    return html.Div(
        [
            html.H1("ML-PEG Model Explorer"),
            html.Div(
                [
                    _chip(f"{n_rows} structures"),
                    _chip(f"{n_models} models"),
                    _chip("every model × every structure"),
                ],
                style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
            ),
            html.P(
                "Describe your system in the finder to get a ranked shortlist, or "
                "pick a model's report card to see where it works across chemistry — "
                "and whether each test is in-domain or an extrapolation. Then compare "
                "two models head-to-head, dig into the structures models fail on, or "
                "see which models fail alike.",
                style={
                    "maxWidth": "72ch",
                    "color": "var(--mlpeg-ink-2)",
                    "fontSize": "14px",
                },
            ),
            html.Nav(
                [
                    html.A(
                        label,
                        href=f"#{anchor}",
                        className="mlpeg-explorer-jump",
                    )
                    for label, anchor in _EXPLORER_JUMP_NAV
                ],
                className="mlpeg-explorer-nav",
                **{"aria-label": "Explorer sections"},
            ),
        ]
    )


def _build_finder_panel(categories: list[str], element_options: list[str]) -> html.Div:
    """
    Build the model-finder panel (use-case controls, ranking chart, top picks).

    Parameters
    ----------
    categories
        Benchmark categories for the fine-tune filter.
    element_options
        Element symbols for the chemistry filter.

    Returns
    -------
    html.Div
        The finder panel.
    """
    finder_results = model_finder_rankings()
    return html.Div(
        [
            html.H3(
                "Model finder: which model for your experiment?",
                className="mlpeg-explorer-header",
            ),
            _panel_help(
                "Tell it what you're working on — the kind of experiment, the "
                "elements involved, and how much extrapolation you're willing to "
                "trust — and get a ranked shortlist with the evidence behind each "
                "pick. Click any bar or card to open the model's full report card "
                "below.",
            ),
            html.Div(
                [
                    html.Label(
                        "What are you simulating?", className="mlpeg-control-label"
                    ),
                    dcc.RadioItems(
                        id=FINDER_PRESET,
                        options=FINDER_EXPERIMENTS,
                        value="any",
                        inline=True,
                        style={"fontSize": "13px"},
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "16px"},
                    ),
                ],
                style={"marginBottom": "10px"},
            ),
            html.Div(
                [
                    _dropdown_control(
                        "Elements in your system",
                        dcc.Dropdown(
                            id=FINDER_ELEMENTS,
                            options=[{"label": e, "value": e} for e in element_options],
                            value=[],
                            multi=True,
                            placeholder="Any chemistry",
                            style={"minWidth": "min(240px, 100%)", "fontSize": "13px"},
                        ),
                    ),
                    _dropdown_control(
                        "Benchmark areas (fine-tune)",
                        dcc.Dropdown(
                            id=FINDER_CATEGORIES,
                            options=[{"label": c, "value": c} for c in categories],
                            value=[],
                            multi=True,
                            placeholder="All areas",
                            style={"minWidth": "min(240px, 100%)", "fontSize": "13px"},
                        ),
                    ),
                    _dropdown_control(
                        "Evidence to trust",
                        dcc.RadioItems(
                            id=FINDER_TOLERANCE,
                            options=FINDER_TOLERANCE_OPTIONS,
                            value="any",
                            inline=True,
                            style={"fontSize": "13px"},
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "14px"},
                        ),
                    ),
                ],
                className="mlpeg-explorer-controls",
            ),
            _a11y_graph(
                dcc.Graph(
                    id=FINDER_GRAPH,
                    figure=build_finder_figure(finder_results),
                    responsive=True,
                    style={"height": "min(400px, 78vh)"},
                    config={"displaylogo": False},
                ),
                "Model finder ranking bar chart. The labelled dropdowns above are the "
                "keyboard-navigable controls; clicking a bar opens that model's report "
                "card below.",
            ),
            html.Div(id=FINDER_PICKS, children=render_finder_picks(finder_results)),
            html.Details(
                [
                    html.Summary(
                        "How the ranking works",
                        style={
                            "fontSize": "12px",
                            "color": "var(--mlpeg-ink-3)",
                            "cursor": "pointer",
                            "marginTop": "8px",
                        },
                    ),
                    html.P(
                        "Each model's fit combines two independent evidence streams. "
                        "Benchmark evidence is its mean 0–1 score over the selected "
                        "areas (only tests passing your extrapolation tolerance); each "
                        "score is anchored to that benchmark's own good/bad error "
                        "thresholds, so it is a relative grade, not an absolute error. "
                        "When you pick elements, structure evidence is added: on "
                        "every test structure containing those elements, models are "
                        "ranked against each other by error, and the model's average "
                        "win rate over the others (a mean win rate, after HELM) is "
                        "blended 60% benchmark + 40% structure into the fit — so it "
                        "shifts with the set of models compared. "
                        "Models with fewer than three admissible benchmarks are too "
                        "thin to rank; the rest have their mean gently pulled towards "
                        "the field average in proportion to how few benchmarks back "
                        "it, so a model can't win on a couple of lucky results (hover "
                        "shows the benchmark count behind each). Physicality (does the "
                        "model behave physically sensibly?) is always checked and "
                        "flagged when poor — accuracy means little if a run blows up.",
                        style={
                            "fontSize": "12px",
                            "color": "var(--mlpeg-ink-3)",
                            "maxWidth": "76ch",
                        },
                    ),
                ]
            ),
        ],
        className="mlpeg-explorer-panel",
    )


def _build_report_panel() -> html.Div:
    """
    Build the model report-card panel (model picker, summary, profile, table).

    Returns
    -------
    html.Div
        The report-card panel, initialised to the broadest-coverage model.
    """
    default_model = _report_default_model()
    return html.Div(
        [
            html.H3(
                "Model report card: where it works, and where it extrapolates",
                className="mlpeg-explorer-header",
            ),
            _panel_help(
                "Pick a model to see its score across every chemistry domain, with "
                "the all-model mean (grey tick) and the model's rank for context. "
                "Each category is coloured by whether the model is mostly tested "
                "in-domain (reference matches its training level of theory) or "
                "extrapolated — against a different functional, a higher-level "
                "reference (CCSD(T)/MP2), or experiment; hover for the exact "
                "breakdown. Scores are graded against each benchmark's own good/bad "
                "thresholds (not absolute errors), and a rank's cohort size varies by "
                "category, so treat close ranks as ties.",
            ),
            html.Div(
                _dropdown_control(
                    "Model",
                    dcc.Dropdown(
                        id=REPORT_MODEL,
                        options=_model_options(),
                        value=default_model,
                        clearable=False,
                        # min() keeps the roomy 420px on desktop (long model names)
                        # but shrinks to the container on phones so it never widens
                        # the page.
                        style={"minWidth": "min(420px, 100%)", "fontSize": "13px"},
                    ),
                ),
                style={"marginBottom": "12px"},
            ),
            html.Div(id=REPORT_SUMMARY, children=render_report_summary(default_model)),
            # Chart full-width, table full-width below: side-by-side the five-column
            # table overflowed its half of the card.
            _a11y_graph(
                dcc.Graph(
                    id=REPORT_PROFILE,
                    figure=build_report_profile_figure(default_model),
                    responsive=True,
                    style={"height": "min(560px, 78vh)"},
                    config={"displaylogo": False},
                ),
                "Report card profile: bar chart of the selected model's mean score in "
                "each chemistry area, against the all-model average.",
            ),
            html.Details(
                [
                    html.Summary(
                        "Every benchmark for this model",
                        style={
                            "fontSize": "13px",
                            "fontWeight": "600",
                            "cursor": "pointer",
                            "margin": "8px 0 6px",
                        },
                    ),
                    html.Div(
                        id=REPORT_TABLE,
                        children=render_report_table(default_model),
                        className="mlpeg-report-table-wrap",
                    ),
                ],
                open=True,
            ),
        ],
        className="mlpeg-explorer-panel",
    )


def _build_detail_panel() -> html.Div:
    """
    Build the selected-structure panel (3D viewer + per-model error table slots).

    Returns
    -------
    html.Div
        The structure-detail panel, empty until a point is clicked.
    """
    return html.Div(
        [
            html.H3("Selected structure", className="mlpeg-explorer-header"),
            html.Div(
                "Open a head-to-head comparison above and click a category bar, then "
                "click any structure in that scatter to view its 3D geometry and "
                "every model's error here.",
                id=STRUCT_TABLE,
                style={"fontSize": "13px", "color": "var(--mlpeg-ink-2)"},
            ),
            html.Div(id=STRUCT_VIEWER),
        ],
        className="mlpeg-explorer-panel",
    )


def _build_h2h_panel(
    h2h_options: list[dict[str, str]],
    default_a: str | None,
    default_b: str | None,
) -> html.Div:
    """
    Build the head-to-head panel (model pickers, verdict bars, hidden drilldown).

    Parameters
    ----------
    h2h_options
        Model dropdown options, labelled with each model's training level of theory.
    default_a, default_b
        The initially compared models.

    Returns
    -------
    html.Div
        The head-to-head panel.
    """
    return html.Div(
        [
            html.H3(
                "Head-to-head: model vs model",
                className="mlpeg-explorer-header",
            ),
            _panel_help(
                "One bar per benchmark area: it points towards the model that "
                "was closer to the reference more often, and its length is the margin. "
                "Each area's win rate averages its benchmarks equally, so a benchmark "
                "with thousands of structures can't swamp the verdict. Grey bars are "
                "too close to call, and the whiskers show a 95% interval (optimistic — "
                "structures within an area are correlated). Win rate counts wins, not "
                "how big each win was; click a bar to open that area's "
                "structure-by-structure comparison, where the scatter shows the "
                "magnitudes — and click any structure to see it in 3D.",
            ),
            html.Div(
                [
                    _dropdown_control(
                        "Model A",
                        dcc.Dropdown(
                            id=H2H_MODEL_A,
                            options=h2h_options,
                            value=default_a,
                            clearable=False,
                            style={"minWidth": "min(260px, 100%)", "fontSize": "13px"},
                        ),
                    ),
                    _dropdown_control(
                        "Model B",
                        dcc.Dropdown(
                            id=H2H_MODEL_B,
                            options=h2h_options,
                            value=default_b,
                            clearable=False,
                            style={"minWidth": "min(260px, 100%)", "fontSize": "13px"},
                        ),
                    ),
                ],
                className="mlpeg-explorer-controls",
            ),
            html.Div(id=H2H_SUMMARY, style={"marginBottom": "10px"}),
            _a11y_graph(
                dcc.Graph(
                    id=H2H_GRAPH,
                    figure=build_head_to_head_figure(
                        default_a, default_b, load_error_matrix()
                    ),
                    responsive=True,
                    style={"height": "min(600px, 78vh)"},
                    config={"displaylogo": False},
                ),
                "Head-to-head verdict: diverging bar chart of which of the two chosen "
                "models is more accurate in each area, and by how much. Clicking a bar "
                "drills into that area's structures.",
            ),
            # The per-category drilldown scatter: hidden until a bar is clicked.
            dcc.Graph(
                id=H2H_DETAIL_GRAPH,
                figure=go.Figure(),
                responsive=True,
                style={"height": "min(460px, 78vh)", "display": "none"},
                config={"displaylogo": False},
            ),
        ],
        className="mlpeg-explorer-panel",
    )


def _build_structure_fails_panel(
    categories: list[str], model_options: list[dict[str, str]]
) -> html.Div:
    """
    Build the structure-fails explorer (hardest structures + where a model fails).

    Parameters
    ----------
    categories
        Benchmark categories for the area filter.
    model_options
        Options for the "focus on one model" picker (family-sorted).

    Returns
    -------
    html.Div
        The structure-fails panel.
    """
    rows = load_hardness()
    return html.Div(
        [
            html.H3(
                "Where models fail: the hardest structures",
                className="mlpeg-explorer-header",
            ),
            _panel_help(
                "The structures the field gets most wrong — ranked by difficulty "
                "(median error, as a within-benchmark percentile so units don't "
                "matter) or by how much the models disagree. Pick a single model to "
                "switch the ranking to where that model fails worst relative to its "
                "peers — the quickest way to see a model's weak spots. Click any bar "
                "to open its 3D geometry and every model's error in the panel below.",
            ),
            html.Div(
                [
                    _dropdown_control(
                        "Benchmark areas",
                        dcc.Dropdown(
                            id=STRUCT_FAILS_CATEGORIES,
                            options=[{"label": c, "value": c} for c in categories],
                            value=[],
                            multi=True,
                            placeholder="All areas",
                            style={"minWidth": "min(240px, 100%)", "fontSize": "13px"},
                        ),
                    ),
                    _dropdown_control(
                        "Focus on one model",
                        dcc.Dropdown(
                            id=STRUCT_FAILS_MODEL,
                            options=model_options,
                            value=None,
                            clearable=True,
                            placeholder="All models (rank by difficulty)",
                            style={"minWidth": "min(360px, 100%)", "fontSize": "13px"},
                        ),
                    ),
                    _dropdown_control(
                        "Rank by",
                        dcc.RadioItems(
                            id=STRUCT_FAILS_SORT,
                            options=STRUCT_FAILS_SORT_OPTIONS,
                            value="hardest",
                            inline=True,
                            style={"fontSize": "13px"},
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "14px"},
                        ),
                    ),
                ],
                className="mlpeg-explorer-controls",
            ),
            _a11y_graph(
                dcc.Graph(
                    id=STRUCT_FAILS_GRAPH,
                    figure=build_structure_fails_figure(rows),
                    responsive=True,
                    style={"height": "min(520px, 78vh)"},
                    config={"displaylogo": False},
                ),
                "Structure failures bar chart: the hardest or most-contested "
                "structures for the current selection. Clicking a bar opens the "
                "structure in 3D with every model's error.",
            ),
        ],
        className="mlpeg-explorer-panel",
    )


def _build_similarity_panel() -> html.Div:
    """
    Build the who-fails-alike panel (model x model error-correlation heatmap).

    Returns
    -------
    html.Div
        The similarity panel.
    """
    return html.Div(
        [
            html.H3(
                "Who fails alike: models that make the same mistakes",
                className="mlpeg-explorer-header",
            ),
            _panel_help(
                "Each model's per-structure errors are z-scored within every benchmark "
                "and correlated against every other model's over the structures they "
                "share. Warm (red) cells are pairs that err in the same direction on "
                "the same structures — they tend to fail alike, so combining them into "
                "an ensemble adds little; cool (blue) cells are complementary models. "
                "Rows and columns are ordered by clustering, so behavioural families "
                "group together.",
            ),
            _a11y_graph(
                dcc.Graph(
                    id=SIMILARITY_GRAPH,
                    figure=build_similarity_heatmap_figure(),
                    responsive=True,
                    style={"height": "min(560px, 78vh)"},
                    config={"displaylogo": False},
                ),
                "Model similarity heatmap: pairwise error correlation between models, "
                "clustered so models that fail alike group together.",
            ),
        ],
        className="mlpeg-explorer-panel",
    )


def _build_coverage_panel() -> html.Div:
    """
    Build the coverage-rose panel (field-wide progress overview).

    Returns
    -------
    html.Div
        The coverage panel.
    """
    return html.Div(
        [
            html.H3(
                "Coverage rose: what's solved, and what's open",
                className="mlpeg-explorer-header",
            ),
            _panel_help(
                "A bird's-eye view of the field. Each wedge is a benchmark area; its "
                "length and colour show the mean score across every evaluated model "
                "— so long green wedges are areas current MLIPs already handle, and "
                "short red wedges are open problems. This tracks the field as a "
                "whole, not the best model: use the finder or report card above to "
                "judge an individual model. Each wedge averages over whichever models "
                "were run in that area and is anchored to per-benchmark thresholds, so "
                "read lengths as a rough guide rather than exact cross-area "
                "comparisons; 'solved' means models meet that benchmark's target.",
            ),
            html.Div(
                _dropdown_control(
                    "Group by",
                    dcc.Dropdown(
                        id=COVERAGE_GRANULARITY,
                        options=COVERAGE_GRANULARITY_OPTIONS,
                        value="category",
                        clearable=False,
                        style={"width": "260px", "fontSize": "13px"},
                    ),
                ),
                style={"marginBottom": "10px"},
            ),
            _a11y_graph(
                dcc.Graph(
                    id=COVERAGE_GRAPH,
                    figure=build_coverage_rose_figure(load_benchmark_scores()),
                    responsive=True,
                    style={"height": "min(560px, 78vh)"},
                    config={"displaylogo": False},
                ),
                "Coverage rose: polar chart of the field's mean score per benchmark "
                "area — long green wedges are solved areas, short red wedges are open "
                "problems.",
            ),
        ],
        className="mlpeg-explorer-panel",
    )


# --- Methods & references: the analytics' provenance, cited ---------------------------

# Each view's methods, grounded in the literature. Entries are (what-it-grounds,
# citation, url); only independently-verified sources are listed (links checked to
# resolve). The 403s some publishers return to bots still resolve in a browser.
_REFERENCES: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "Model finder",
        [
            (
                "structure evidence = mean win rate",
                "Liang et al. (2022), HELM",
                "https://arxiv.org/abs/2211.09110",
            ),
            (
                "aggregating scores into an MLIP leaderboard",
                "Riebesell et al. (2023), Matbench Discovery",
                "https://arxiv.org/abs/2308.14920",
            ),
        ],
    ),
    (
        "Head-to-head",
        [
            (
                "per-structure win/loss = the sign test",
                "Demšar (2006), JMLR 7:1",
                "https://www.jmlr.org/papers/v7/demsar06a.html",
            ),
        ],
    ),
    (
        "Model report card",
        [
            (
                "per-model reporting disaggregated by condition",
                "Mitchell et al. (2019), Model Cards",
                "https://arxiv.org/abs/1810.03993",
            ),
        ],
    ),
    (
        "Who fails alike",
        [
            (
                "error correlation as ensemble (dis)agreement",
                "Kuncheva & Whitaker (2003), Machine Learning 51:181",
                "https://doi.org/10.1023/A:1022859003006",
            ),
        ],
    ),
]


def _ref(grounds: str, citation: str, href: str) -> html.Li:
    """
    Render one reference line: what it grounds, then a linked citation.

    Parameters
    ----------
    grounds
        The part of a view this reference underpins.
    citation
        Short citation text (author, year, venue).
    href
        A URL that resolves to the work (arXiv/DOI/venue page).

    Returns
    -------
    html.Li
        The list item.
    """
    return html.Li(
        [
            html.Span(f"{grounds} — ", style={"color": "var(--mlpeg-ink-3)"}),
            html.A(
                citation,
                href=href,
                target="_blank",
                rel="noopener",
                style={"color": "var(--mlpeg-accent)"},
            ),
        ],
        style={"marginBottom": "4px"},
    )


def _build_references_panel() -> html.Div:
    """
    Build the collapsed "Methods & references" panel at the page bottom.

    Surfaces the literature each view is built on, so the analytics read as grounded
    rather than ad hoc. Content is driven by :data:`_REFERENCES`.

    Returns
    -------
    html.Div
        The references panel.
    """
    groups: list[html.Div] = []
    for view, refs in _REFERENCES:
        groups.append(
            html.Div(
                [
                    html.Div(
                        view,
                        style={
                            "fontWeight": "600",
                            "fontSize": "13px",
                            "marginTop": "8px",
                        },
                    ),
                    html.Ul(
                        [_ref(*ref) for ref in refs],
                        style={
                            "margin": "2px 0 0",
                            "paddingLeft": "18px",
                            "fontSize": "12px",
                            "lineHeight": "1.5",
                        },
                    ),
                ]
            )
        )
    text = {
        "fontSize": "13px",
        "color": "var(--mlpeg-ink-2)",
        "maxWidth": "76ch",
    }
    return html.Div(
        html.Details(
            [
                html.Summary(
                    "How the scores work · methods & references",
                    style={
                        "fontSize": "16px",
                        "fontWeight": "600",
                        "cursor": "pointer",
                    },
                ),
                html.P(
                    [
                        "Every number on this page traces back to one per-model, "
                        "per-benchmark ",
                        html.B("score in 0–1"),
                        ". Each benchmark measures one or more raw errors (e.g. mean "
                        "absolute error, in meV/atom, GPa or kcal/mol). Each raw error "
                        "is mapped to 0–1 by linear interpolation between a ",
                        html.B("good"),
                        " threshold (→ 1) and a ",
                        html.B("bad"),
                        " threshold (→ 0), clipped outside — the good/bad thresholds "
                        "are set per benchmark by domain experts (hover any metric for "
                        "its values). A benchmark's score is the weighted average of "
                        "its metric scores.",
                    ],
                    style={**text, "marginBottom": "6px"},
                ),
                html.P(
                    "So 1 means 'as accurate as anyone needs here' and 0 means 'at or "
                    "beyond the point where the model should be avoided'. Everything "
                    "else is built from these scores: the report card and coverage "
                    "rose average them, the finder blends a benchmark-score mean with "
                    "a win-rate over other models, and the head-to-head counts "
                    "per-structure wins. Full detail is in the project's scoring & "
                    "normalisation guide.",
                    style={**text, "marginBottom": "6px"},
                ),
                html.P(
                    "Every view uses an established method — sources by view:",
                    style={**text, "marginBottom": "2px"},
                ),
                *groups,
            ],
            open=False,
        ),
        className="mlpeg-explorer-panel",
    )
