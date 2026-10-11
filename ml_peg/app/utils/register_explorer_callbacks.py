"""
Callbacks for the Model Explorer page.

Wires the coordinated interactions: the model finder (rank models for a described use
case; a bar or pick card loads its report card), the report card, the model-vs-model
head-to-head (a category bar or the area picker opens its structures), the failure
bars, and opening a structure from any of those in the shared 3D detail panel.
"""

from __future__ import annotations

from typing import Any

from dash import ALL, Input, Output, callback, ctx, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from ml_peg.app.utils.explorer_data import (
    load_benchmark_scores,
    load_error_matrix,
    load_hardness_by_key,
)
from ml_peg.app.utils.explorer_figures import (
    build_coverage_rose_figure,
    build_finder_figure,
    build_h2h_detail_figure,
    build_head_to_head_figure,
    build_report_profile_figure,
    build_structure_fails_figure,
    graph_style,
    render_fails_list,
    render_finder_picks,
    render_h2h_summary,
    render_report_summary,
    render_report_table,
    render_structure_detail,
)
from ml_peg.app.utils.explorer_ids import (
    COVERAGE_GRANULARITY,
    COVERAGE_GRAPH,
    FAILS_PICK_TYPE,
    FINDER_CATEGORIES,
    FINDER_ELEMENTS,
    FINDER_EXPERIMENT_CATEGORIES,
    FINDER_GRAPH,
    FINDER_PICK_TYPE,
    FINDER_PICKS,
    FINDER_PRESET,
    FINDER_TOLERANCE,
    H2H_DETAIL_CATEGORY,
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
    STRUCT_FAILS_LIST,
    STRUCT_FAILS_MODEL,
    STRUCT_FAILS_SORT,
    STRUCT_TABLE,
    STRUCT_VIEWER,
)
from ml_peg.app.utils.explorer_scoring import (
    cached_benchmark_error_medians,
    cached_finder_rankings,
    cached_structure_fails_ranking,
    h2h_pair_stats,
)


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
    # Dash also fires this callback (with no trigger) when pattern-matched cards are
    # removed; only a genuine bar click may act on the graph's stale clickData.
    if trigger != FINDER_GRAPH or not click or not click.get("points"):
        raise PreventUpdate
    return click["points"][0]["y"]


def _clicked_structure_key(
    trigger: dict | str | None,
    h2h_click: dict | None,
    fails_click: dict | None,
    list_clicks: list[int],
) -> str:
    """
    Resolve which structure a detail-panel interaction points at.

    Parameters
    ----------
    trigger
        ``ctx.triggered_id``: a pattern-matched dict for a list button, or the id of
        the graph that was clicked.
    h2h_click
        Plotly click event from the head-to-head detail scatter.
    fails_click
        Plotly click event from the failure bars.
    list_clicks
        Click counts from the pattern-matched structure-list buttons.

    Returns
    -------
    str
        The structure key. Both graphs carry it first in their points' ``customdata``;
        the buttons carry it in their id.
    """
    if isinstance(trigger, dict):
        if not any(list_clicks):
            raise PreventUpdate  # list re-rendered, not clicked
        return trigger["key"]
    # Dash also fires this callback (with no trigger) when pattern-matched buttons
    # are removed; only a genuine graph click may act on stale clickData.
    if trigger == STRUCT_FAILS_GRAPH:
        click = fails_click
    elif trigger == H2H_DETAIL_GRAPH:
        click = h2h_click
    else:
        raise PreventUpdate
    if not click or not click.get("points"):
        raise PreventUpdate
    customdata = click["points"][0].get("customdata")
    if not customdata:
        raise PreventUpdate  # e.g. the y = x reference line
    return customdata[0]


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
        wanted_categories: list[str] | None,
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
        wanted_categories
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
        categories_out: Any = no_update
        preset_out: Any = no_update
        if trigger == FINDER_PRESET and preset:
            wanted_categories = FINDER_EXPERIMENT_CATEGORIES.get(preset, [])
            categories_out = wanted_categories
        elif trigger == FINDER_CATEGORIES:
            preset_out = None
        results = cached_finder_rankings(
            tuple(sorted(wanted_categories)) if wanted_categories else None,
            tuple(sorted(elements)) if elements else None,
            tolerance or "any",
        )
        return (
            build_finder_figure(results),
            render_finder_picks(results),
            categories_out,
            preset_out,
        )

    @callback(
        Output(REPORT_MODEL, "value"),
        Input(FINDER_GRAPH, "clickData"),
        Input({"type": FINDER_PICK_TYPE, "model": ALL}, "n_clicks"),
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
            The clicked model name.
        """
        return _finder_pick_model(ctx.triggered_id, click, card_clicks)

    @callback(
        Output(REPORT_SUMMARY, "children"),
        Output(REPORT_PROFILE, "figure"),
        Output(REPORT_PROFILE, "style"),
        Output(REPORT_TABLE, "children"),
        Input(REPORT_MODEL, "value"),
        prevent_initial_call=True,
    )
    def update_report_card(model: str | None) -> tuple[Any, Any, Any, Any]:
        """
        Rebuild the report card (summary, profile, table) for the selected model.

        Parameters
        ----------
        model
            Selected model.

        Returns
        -------
        tuple[Any, Any, Any, Any]
            The summary chips, the profile figure and its wrapper style (the figure's
            height follows the category count), and the per-benchmark table.
        """
        profile = build_report_profile_figure(model)
        return (
            render_report_summary(model),
            profile,
            graph_style(profile),
            render_report_table(model),
        )

    @callback(
        Output(COVERAGE_GRAPH, "figure"),
        Input(COVERAGE_GRANULARITY, "value"),
        prevent_initial_call=True,
    )
    def update_coverage_rose(granularity: str | None) -> go.Figure:
        """
        Rebuild the coverage rose at the requested grouping.

        Parameters
        ----------
        granularity
            ``"category"`` or ``"benchmark"`` (defaults to ``"category"``).

        Returns
        -------
        go.Figure
            The rebuilt polar figure.
        """
        return build_coverage_rose_figure(
            load_benchmark_scores(), granularity=granularity or "category"
        )

    @callback(
        Output(STRUCT_FAILS_GRAPH, "figure"),
        Output(STRUCT_FAILS_GRAPH, "style"),
        Output(STRUCT_FAILS_LIST, "children"),
        Input(STRUCT_FAILS_CATEGORIES, "value"),
        Input(STRUCT_FAILS_MODEL, "value"),
        Input(STRUCT_FAILS_SORT, "value"),
        prevent_initial_call=True,
    )
    def update_structure_fails(
        wanted_categories: list[str] | None, model: str | None, sort: str | None
    ) -> tuple[Any, Any, Any]:
        """
        Rebuild the structure-fails bars and their list from the controls.

        Parameters
        ----------
        wanted_categories
            Benchmark areas to restrict to (empty/None = all).
        model
            Focus model; when set, the ranking becomes where that model fails worst.
        sort
            ``"hardest"`` or ``"contested"`` (ignored when a model is focused).

        Returns
        -------
        tuple[Any, Any, Any]
            The bar figure, its wrapper style, and the keyboard-operable list.
        """
        sort = sort or "hardest"
        ranked = cached_structure_fails_ranking(
            tuple(sorted(wanted_categories)) if wanted_categories else None,
            model,
            sort,
        )
        figure = build_structure_fails_figure(ranked, model=model, sort=sort)
        return figure, graph_style(figure), render_fails_list(ranked)

    @callback(
        Output(STRUCT_VIEWER, "children"),
        Output(STRUCT_TABLE, "children"),
        Input(H2H_DETAIL_GRAPH, "clickData"),
        Input(STRUCT_FAILS_GRAPH, "clickData"),
        Input({"type": FAILS_PICK_TYPE, "key": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def show_structure(
        h2h_click: dict | None, fails_click: dict | None, list_clicks: list[int]
    ) -> tuple[Any, Any]:
        """
        Render the chosen structure in 3D with its per-model error table.

        Fired from the head-to-head detail scatter, the failure bars, or the buttons
        in the failure list; all three identify the structure by its key, so one
        handler serves them.

        Parameters
        ----------
        h2h_click
            Plotly click event from the head-to-head detail scatter.
        fails_click
            Plotly click event from the failure bars.
        list_clicks
            Click counts from the pattern-matched structure-list buttons.

        Returns
        -------
        tuple[Any, Any]
            The viewer children and the detail-table children.
        """
        key = _clicked_structure_key(
            ctx.triggered_id, h2h_click, fails_click, list_clicks
        )
        row = load_hardness_by_key().get(key)
        if row is None:
            raise PreventUpdate
        return render_structure_detail(key, row)

    @callback(
        Output(H2H_GRAPH, "figure"),
        Output(H2H_SUMMARY, "children"),
        Output(H2H_DETAIL_GRAPH, "figure"),
        Output(H2H_DETAIL_GRAPH, "style"),
        Output(H2H_DETAIL_CATEGORY, "value"),
        Input(H2H_MODEL_A, "value"),
        Input(H2H_MODEL_B, "value"),
        Input(H2H_GRAPH, "clickData"),
        Input(H2H_DETAIL_CATEGORY, "value"),
        prevent_initial_call=True,
    )
    def update_head_to_head(
        model_a: str | None,
        model_b: str | None,
        bar_click: dict | None,
        picked_category: str | None,
    ) -> tuple[Any, Any, Any, Any, Any]:
        """
        Rebuild the head-to-head verdict, or open a category's drilldown.

        One callback covers both interactions so they cannot disagree: switching a
        model rebuilds the verdict bars and headline and closes any open drilldown
        (it belonged to the old pair), while clicking a category bar or picking an
        area opens that category's structure-level scatter and keeps the bars as
        they are.

        Parameters
        ----------
        model_a, model_b
            Selected models.
        bar_click
            Plotly click event from the verdict bars (bars are keyed by category).
        picked_category
            The area chosen in the drilldown picker (the keyboard route).

        Returns
        -------
        tuple[Any, Any, Any, Any, Any]
            The bars figure, the headline, the drilldown figure and style, and the
            picker value (kept in step with a clicked bar).
        """
        stats = h2h_pair_stats(model_a, model_b)
        trigger = ctx.triggered_id
        if trigger in (H2H_MODEL_A, H2H_MODEL_B):
            detail = go.Figure()
            return (
                build_head_to_head_figure(model_a, model_b, stats),
                render_h2h_summary(model_a, model_b, stats),
                detail,
                {**graph_style(detail), "display": "none"},
                None,
            )

        if trigger == H2H_GRAPH:
            if not bar_click or not bar_click.get("points"):
                raise PreventUpdate
            category = bar_click["points"][0]["y"]
        else:
            category = picked_category
        if not category or stats is None:
            detail = go.Figure()
            return (
                no_update,
                no_update,
                detail,
                {**graph_style(detail), "display": "none"},
                None if stats is None else no_update,
            )
        detail = build_h2h_detail_figure(
            model_a,
            model_b,
            category,
            load_error_matrix(),
            load_hardness_by_key(),
            cached_benchmark_error_medians(),
        )
        return no_update, no_update, detail, graph_style(detail), category
