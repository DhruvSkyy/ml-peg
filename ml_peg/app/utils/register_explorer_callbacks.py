"""
Callbacks for the Model Behaviour Explorer page.

Wires the coordinated interactions: the model finder (rank models for a described use
case, click a bar to load its report card), the report card, the model-vs-model
head-to-head (click a category bar to drill into its structures), and clicking a
structure in that drilldown to inspect its 3D geometry and per-model errors.
"""

from __future__ import annotations

from typing import Any

from dash import ALL, Input, Output, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate
from dash.html import Iframe

from ml_peg.app.utils.build_explorer import (
    _H2H_COLOUR_A,
    _H2H_COLOUR_B,
    COVERAGE_GRANULARITY,
    COVERAGE_GRAPH,
    FINDER_CATEGORIES,
    FINDER_ELEMENTS,
    FINDER_EXPERIMENT_CATEGORIES,
    FINDER_GRAPH,
    FINDER_PICKS,
    FINDER_PRESET,
    FINDER_TOLERANCE,
    H2H_DETAIL_GRAPH,
    H2H_GRAPH,
    H2H_MODEL_A,
    H2H_MODEL_B,
    H2H_SUMMARY,
    REPORT_MODEL,
    REPORT_PROFILE,
    REPORT_SUMMARY,
    REPORT_TABLE,
    STRUCT_FAILS_CATEGORIES,
    STRUCT_FAILS_GRAPH,
    STRUCT_FAILS_MODEL,
    STRUCT_FAILS_SORT,
    STRUCT_TABLE,
    STRUCT_VIEWER,
    build_coverage_rose_figure,
    build_finder_figure,
    build_h2h_detail_figure,
    build_head_to_head_figure,
    build_report_profile_figure,
    build_structure_fails_figure,
    cached_finder_rankings,
    cached_head_to_head_stats,
    load_benchmark_scores,
    load_error_matrix,
    load_hardness,
    render_finder_picks,
    render_report_summary,
    render_report_table,
)
from ml_peg.app.utils.explorer_data import category_page_paths, split_structure_key
from ml_peg.app.utils.weas import generate_weas_html


def _struct_table(key: str) -> html.Table:
    """
    Build a per-model error table for one structure, best model first.

    Parameters
    ----------
    key
        Structure key into the error matrix.

    Returns
    -------
    html.Table
        Table of model and absolute error (with signed residual), sorted ascending.
    """
    resids = load_error_matrix().get(key, {})
    ordered = sorted(resids.items(), key=lambda kv: abs(kv[1]))
    header = html.Tr([html.Th("Model"), html.Th("Abs. error"), html.Th("Signed")])
    body = [
        html.Tr(
            [
                html.Td(model),
                html.Td(f"{abs(resid):.4g}"),
                html.Td(f"{resid:+.4g}"),
            ]
        )
        for model, resid in ordered
    ]
    return html.Table(
        [html.Thead(header), html.Tbody(body)],
        className="mlpeg-explorer-struct-table",
    )


def _structure_detail(
    key: str, benchmark: str, sid: str, xyz: str | None
) -> tuple[Any, Any]:
    """
    Render one structure's 3D geometry (WEAS) and its per-model error table.

    Shared by the structure-explorer and head-to-head click handlers.

    Parameters
    ----------
    key
        Structure key into the error matrix.
    benchmark
        Benchmark label for the heading.
    sid
        Structure id for the heading.
    xyz
        Geometry path relative to the data root, or ``None`` if none is shipped.

    Returns
    -------
    tuple[Any, Any]
        The viewer children and the detail-table children.
    """
    category, _, _ = split_structure_key(key)
    path = category_page_paths().get(category)
    heading: list[Any] = [f"{sid}  ·  {benchmark}"]
    if path:
        heading.extend(
            [
                "  ·  ",
                dcc.Link("full benchmark page", href=path, style={"fontSize": "12px"}),
            ]
        )
    table = html.Div(
        [
            html.Div(heading, style={"fontWeight": "600", "marginBottom": "6px"}),
            _struct_table(key),
        ]
    )
    if not xyz:
        viewer = html.Div(
            "No 3D geometry is shipped for this structure.",
            style={
                "fontSize": "13px",
                "color": "var(--mlpeg-ink-3)",
                "marginTop": "8px",
            },
        )
        return viewer, table
    viewer = Iframe(
        title=f"Interactive 3D structure viewer for {sid}",
        srcDoc=generate_weas_html(f"/assets/{xyz}", "struct", 0),
        style={
            "height": "560px",
            "width": "100%",
            "border": "1px solid var(--mlpeg-border-strong)",
            "borderRadius": "5px",
            "marginTop": "8px",
        },
    )
    return viewer, table


def _finder_pick_model(
    trigger: dict | str | None, click: dict | None, card_clicks: list[int]
) -> str:
    """
    Resolve which model a finder interaction points at.

    Kept module-level (pure) so the routing logic is unit-testable without a Dash
    callback context.

    Parameters
    ----------
    trigger
        ``ctx.triggered_id``: a pattern-matched dict for a top-pick card, or the finder
        graph's id string for a bar click.
    click
        Plotly click event from the finder bar chart.
    card_clicks
        Click counts from the pattern-matched top-pick cards.

    Returns
    -------
    str
        The clicked model name (bars are keyed by model on the y axis; cards carry it
        in their pattern-matched id).
    """
    if isinstance(trigger, dict):
        if not any(card_clicks):
            raise PreventUpdate  # cards re-rendered, not clicked
        return trigger["model"]
    if not click or not click.get("points"):
        raise PreventUpdate
    return click["points"][0]["y"]


def register_explorer_callbacks() -> None:
    """Register all Explorer page callbacks on the global Dash app."""

    @callback(
        Output(FINDER_GRAPH, "figure"),
        Output(FINDER_PICKS, "children"),
        Output(FINDER_CATEGORIES, "value"),
        Output(FINDER_PRESET, "value"),
        Input(FINDER_PRESET, "value"),
        Input(FINDER_ELEMENTS, "value"),
        Input(FINDER_CATEGORIES, "value"),
        Input(FINDER_TOLERANCE, "value"),
        prevent_initial_call=True,
    )
    def update_finder(
        preset: str | None,
        elements: list[str] | None,
        categories: list[str] | None,
        tolerance: str | None,
    ) -> tuple[Any, Any, Any, Any]:
        """
        Re-rank the model finder from its use-case controls.

        The experiment preset and the category multi-select share one callback so
        they stay in sync: choosing an experiment writes its category bundle into the
        multi-select, while hand-editing the categories switches the preset to
        "custom" (``None``).

        Parameters
        ----------
        preset
            Selected experiment preset (``None`` after a manual category edit).
        elements
            Elements the user's system contains.
        categories
            Benchmark areas the user cares about.
        tolerance
            How much extrapolation the benchmark evidence may include.

        Returns
        -------
        tuple[Any, Any, Any, Any]
            The ranking figure, the top-pick cards, and the (possibly rewritten)
            categories and preset values.
        """
        trigger = ctx.triggered_id
        if trigger == FINDER_PRESET and preset:
            categories = FINDER_EXPERIMENT_CATEGORIES.get(preset, [])
        elif trigger == FINDER_CATEGORIES:
            preset = None
        results = cached_finder_rankings(
            tuple(sorted(categories)) if categories else None,
            tuple(sorted(elements)) if elements else None,
            tolerance or "any",
        )
        return (
            build_finder_figure(results),
            render_finder_picks(results),
            categories,
            preset,
        )

    @callback(
        Output(REPORT_MODEL, "value"),
        Input(FINDER_GRAPH, "clickData"),
        Input({"type": "explorer-finder-pick", "model": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def open_report_from_finder(click: dict | None, card_clicks: list[int]) -> str:
        """
        Load the clicked finder bar's or pick-card's model in the report card.

        Parameters
        ----------
        click
            Plotly click event from the finder bar chart.
        card_clicks
            Click counts from the pattern-matched top-pick cards.

        Returns
        -------
        str
            The clicked model name (bars are keyed by model on the y axis; cards
            carry it in their pattern-matched id).
        """
        return _finder_pick_model(ctx.triggered_id, click, card_clicks)

    @callback(
        Output(REPORT_SUMMARY, "children"),
        Output(REPORT_PROFILE, "figure"),
        Output(REPORT_TABLE, "children"),
        Input(REPORT_MODEL, "value"),
        prevent_initial_call=True,
    )
    def update_report_card(model: str | None) -> tuple[Any, Any, Any]:
        """
        Rebuild the report card (summary, profile, table) for the selected model.

        Parameters
        ----------
        model
            Selected model.

        Returns
        -------
        tuple[Any, Any, Any]
            The summary chips, the profile figure, and the per-benchmark table.
        """
        return (
            render_report_summary(model),
            build_report_profile_figure(model),
            render_report_table(model),
        )

    @callback(
        Output(COVERAGE_GRAPH, "figure"),
        Input(COVERAGE_GRANULARITY, "value"),
        prevent_initial_call=True,
    )
    def update_coverage_rose(granularity: str | None) -> Any:
        """
        Rebuild the coverage rose at the requested grouping.

        Parameters
        ----------
        granularity
            ``"category"`` or ``"benchmark"`` (defaults to ``"category"``).

        Returns
        -------
        Any
            The rebuilt polar figure.
        """
        return build_coverage_rose_figure(
            load_benchmark_scores(), granularity=granularity or "category"
        )

    @callback(
        Output(STRUCT_FAILS_GRAPH, "figure"),
        Input(STRUCT_FAILS_CATEGORIES, "value"),
        Input(STRUCT_FAILS_MODEL, "value"),
        Input(STRUCT_FAILS_SORT, "value"),
        prevent_initial_call=True,
    )
    def update_structure_fails(
        categories: list[str] | None, model: str | None, sort: str | None
    ) -> Any:
        """
        Rebuild the structure-fails bars from the area/model/sort controls.

        Parameters
        ----------
        categories
            Benchmark areas to restrict to (empty/None = all).
        model
            Focus model; when set, the ranking becomes where that model fails worst.
        sort
            ``"hardest"`` or ``"contested"`` (ignored when a model is focused).

        Returns
        -------
        Any
            The rebuilt structure-fails bar figure.
        """
        return build_structure_fails_figure(
            categories=categories, model=model, sort=sort or "hardest"
        )

    @callback(
        Output(STRUCT_VIEWER, "children"),
        Output(STRUCT_TABLE, "children"),
        Input(H2H_DETAIL_GRAPH, "clickData"),
        Input(STRUCT_FAILS_GRAPH, "clickData"),
        prevent_initial_call=True,
    )
    def show_structure(
        h2h_click: dict | None, fails_click: dict | None
    ) -> tuple[Any, Any]:
        """
        Render a clicked structure in 3D with its per-model error table.

        Fired from either the head-to-head per-category detail scatter or the
        structure-fails bars; both carry ``[key, category, benchmark, sid, xyz]`` in
        their points' ``customdata``, so one handler serves both. ``ctx.triggered_id``
        picks the source that actually fired.

        Parameters
        ----------
        h2h_click
            Plotly click event from the head-to-head per-category detail scatter.
        fails_click
            Plotly click event from the structure-fails bars.

        Returns
        -------
        tuple[Any, Any]
            The viewer children and the detail-table children.
        """
        click = fails_click if ctx.triggered_id == STRUCT_FAILS_GRAPH else h2h_click
        if not click or not click.get("points"):
            raise PreventUpdate
        customdata = click["points"][0].get("customdata")
        if customdata is None:
            raise PreventUpdate  # e.g. the y = x reference line
        key, _category, benchmark, sid, xyz = customdata[:5]
        return _structure_detail(key, benchmark, sid, xyz)

    @callback(
        Output(H2H_GRAPH, "figure"),
        Output(H2H_SUMMARY, "children"),
        Output(H2H_DETAIL_GRAPH, "figure"),
        Output(H2H_DETAIL_GRAPH, "style"),
        Input(H2H_MODEL_A, "value"),
        Input(H2H_MODEL_B, "value"),
        Input(H2H_GRAPH, "clickData"),
    )
    def update_head_to_head(
        model_a: str | None, model_b: str | None, bar_click: dict | None
    ) -> Any:
        """
        Rebuild the head-to-head verdict bars, headline, and category drilldown.

        One callback covers both interactions: switching models rebuilds the verdict
        bars (and hides any open drilldown, since it belongs to the old pair), while
        clicking a category bar opens that category's structure-level scatter.

        Parameters
        ----------
        model_a, model_b
            Selected models.
        bar_click
            Plotly click event from the verdict bars (bars are keyed by category).

        Returns
        -------
        Any
            The bars figure, the headline children, the drilldown figure, and the
            drilldown visibility style.
        """
        figure = build_head_to_head_figure(model_a, model_b)
        detail_hidden = {"height": "min(460px, 78vh)", "display": "none"}
        if not model_a or not model_b or model_a == model_b:
            return figure, "Pick two different models to compare.", {}, detail_hidden

        stats = cached_head_to_head_stats(model_a, model_b)
        if not stats["n_shared"]:
            return figure, "These two models share no structures.", {}, detail_hidden

        # Category verdicts follow the same practical-tie band as the bars.
        cats_a = sum(1 for c in stats["per_category"] if c["win_rate_a"] > 0.55)
        cats_b = sum(1 for c in stats["per_category"] if c["win_rate_a"] < 0.45)
        cats_tie = len(stats["per_category"]) - cats_a - cats_b
        summary = html.Div(
            [
                html.Span(
                    f"{model_a} ",
                    style={"fontWeight": "600", "color": _H2H_COLOUR_A},
                ),
                html.Span(f"leads in {cats_a} areas  ·  "),
                html.Span(
                    f"{model_b} ",
                    style={"fontWeight": "600", "color": _H2H_COLOUR_B},
                ),
                html.Span(
                    f"in {cats_b}  ·  {cats_tie} too close to call  —  "
                    f"averaged evenly across areas, {model_a} was more "
                    f"accurate {100 * stats['overall_win_rate_a']:.0f}% of the time "
                    f"(over {stats['n_shared']:,} shared structures)"
                ),
            ],
            style={"fontSize": "13px"},
        )

        if ctx.triggered_id == H2H_GRAPH and bar_click and bar_click.get("points"):
            category = bar_click["points"][0]["y"]
            meta = {row["key"]: row for row in load_hardness()}
            detail = build_h2h_detail_figure(
                model_a, model_b, category, load_error_matrix(), meta
            )
            return figure, summary, detail, {"height": "min(460px, 78vh)"}
        return figure, summary, {}, detail_hidden
