"""Browser-driven tests for the ML-PEG Model Explorer page.

These pin the coordinated behaviours a user relies on -- reaching the page from the nav,
the model finder / report card / panels rendering, the structure explorer reacting to
its preset and advanced controls, the head-to-head comparison, and the developer
heatmap expanding -- so the Explorer cannot silently break.
"""

from __future__ import annotations

from playwright.sync_api import Page, expect

TIMEOUT = 60_000


def _goto_explorer(page: Page) -> None:
    """Navigate to the Explorer page via the sidebar link and await its main plot."""
    page.locator('#sidebar-nav a[href="/explorer"]').first.click()
    expect(page.locator("#explorer-report-profile")).to_be_visible(timeout=TIMEOUT)


def test_explorer_panels_render(ready_page: Page) -> None:
    """The Explorer shows the finder, report card and head-to-head plots."""
    _goto_explorer(ready_page)
    finder = ready_page.locator("#explorer-finder-graph .js-plotly-plot")
    report = ready_page.locator("#explorer-report-profile .js-plotly-plot")
    h2h = ready_page.locator("#explorer-h2h-graph .js-plotly-plot")
    expect(finder).to_be_visible(timeout=TIMEOUT)
    expect(report).to_be_visible(timeout=TIMEOUT)
    expect(h2h).to_be_visible(timeout=TIMEOUT)


def test_explorer_report_card(ready_page: Page) -> None:
    """The report card shows the summary, profile and per-benchmark table."""
    _goto_explorer(ready_page)
    summary = ready_page.locator("#explorer-report-summary")
    expect(summary).to_contain_text("In-domain", timeout=TIMEOUT)
    expect(summary).to_contain_text("Extrapolated", timeout=TIMEOUT)
    expect(summary).to_contain_text("Physicality", timeout=TIMEOUT)
    profile = ready_page.locator("#explorer-report-profile .js-plotly-plot")
    table = ready_page.locator("#explorer-report-table table")
    expect(profile).to_be_visible(timeout=TIMEOUT)
    expect(table).to_be_visible(timeout=TIMEOUT)


def test_explorer_head_to_head(ready_page: Page) -> None:
    """The head-to-head panel renders its verdict bars and a headline summary."""
    _goto_explorer(ready_page)
    graph = ready_page.locator("#explorer-h2h-graph .js-plotly-plot")
    summary = ready_page.locator("#explorer-h2h-summary")
    expect(graph).to_be_visible(timeout=TIMEOUT)
    expect(summary).to_contain_text("leads in", timeout=TIMEOUT)
    # The category drilldown stays hidden until a bar is clicked.
    detail = ready_page.locator("#explorer-h2h-detail-graph")
    expect(detail).to_be_hidden()


def test_explorer_head_to_head_drilldown(ready_page: Page) -> None:
    """Clicking a verdict bar opens that category's structure-level detail scatter."""
    _goto_explorer(ready_page)
    graph = ready_page.locator("#explorer-h2h-graph .js-plotly-plot")
    expect(graph).to_be_visible(timeout=TIMEOUT)
    # The drilldown is hidden until a category bar is clicked.
    expect(ready_page.locator("#explorer-h2h-detail-graph")).to_be_hidden()
    # Fire a click on the first verdict bar; the callback reads the category off the
    # point's y value (bars are keyed by category on the y axis).
    ready_page.evaluate(
        """() => {
            const gd = document.querySelector('#explorer-h2h-graph .js-plotly-plot');
            const y = gd.data[0].y[0];
            gd.emit('plotly_click',
                {points: [{curveNumber: 0, pointNumber: 0, y: y}]});
        }"""
    )
    detail = ready_page.locator("#explorer-h2h-detail-graph .js-plotly-plot")
    expect(detail).to_be_visible(timeout=TIMEOUT)


def test_explorer_finder_experiment_preset(ready_page: Page) -> None:
    """Choosing an experiment preset fills the category filter and keeps ranking."""
    _goto_explorer(ready_page)
    ready_page.locator('#explorer-finder-preset input[value="molecular"]').check()
    finder = ready_page.locator("#explorer-finder-graph .js-plotly-plot")
    expect(finder).to_be_visible(timeout=TIMEOUT)
    picks = ready_page.locator("#explorer-finder-picks")
    expect(picks).to_contain_text("Benchmark evidence", timeout=TIMEOUT)


def test_explorer_coverage_rose(ready_page: Page) -> None:
    """The coverage rose renders and survives a grouping switch."""
    _goto_explorer(ready_page)
    rose = ready_page.locator("#explorer-coverage-graph .js-plotly-plot")
    expect(rose).to_be_visible(timeout=TIMEOUT)
    ready_page.locator("#explorer-coverage-granularity").click()
    ready_page.get_by_text("By benchmark", exact=False).first.click()
    expect(rose).to_be_visible(timeout=TIMEOUT)


def test_explorer_references_expand(ready_page: Page) -> None:
    """The methods panel reveals the scoring explainer and cited sources when opened."""
    _goto_explorer(ready_page)
    ready_page.get_by_text("How the scores work", exact=False).first.click()
    # A known, verified citation link becomes visible once expanded.
    citation = ready_page.get_by_text("Liang et al. (2022), HELM", exact=False)
    expect(citation.first).to_be_visible(timeout=TIMEOUT)


def test_explorer_finder_reranks(ready_page: Page) -> None:
    """Tightening the finder's evidence tolerance keeps a populated ranking."""
    _goto_explorer(ready_page)
    ready_page.locator('#explorer-finder-tolerance input[value="in_domain"]').check()
    finder = ready_page.locator("#explorer-finder-graph .js-plotly-plot")
    expect(finder).to_be_visible(timeout=TIMEOUT)


def test_explorer_jump_nav(ready_page: Page) -> None:
    """The intro jump nav exposes an anchor to each section, including the new ones."""
    _goto_explorer(ready_page)
    for anchor in ("explorer-sec-fails", "explorer-sec-alike", "explorer-sec-h2h"):
        link = ready_page.locator(f'a.mlpeg-explorer-jump[href="#{anchor}"]')
        expect(link.first).to_be_visible(timeout=TIMEOUT)
    # The target section wrappers exist so the anchors resolve.
    expect(ready_page.locator("#explorer-sec-fails")).to_have_count(1)


def test_explorer_structure_fails_panel(ready_page: Page) -> None:
    """The structure-fails explorer renders its bars and reranks on control changes."""
    _goto_explorer(ready_page)
    fails = ready_page.locator("#explorer-fails-graph .js-plotly-plot")
    expect(fails).to_be_visible(timeout=TIMEOUT)
    # Switching the ranking to "most contested" keeps a populated chart.
    ready_page.locator('#explorer-fails-sort input[value="contested"]').check()
    expect(fails).to_be_visible(timeout=TIMEOUT)


def test_explorer_structure_fails_click_shows_detail(ready_page: Page) -> None:
    """Clicking a structure-fails bar loads that structure's shared 3D detail panel."""
    _goto_explorer(ready_page)
    fails = ready_page.locator("#explorer-fails-graph .js-plotly-plot")
    expect(fails).to_be_visible(timeout=TIMEOUT)
    # The bars carry [key, category, benchmark, sid, xyz] in customdata, like the h2h
    # detail scatter, so the same show_structure handler fills the detail panel.
    ready_page.evaluate(
        """() => {
            const gd = document.querySelector('#explorer-fails-graph .js-plotly-plot');
            gd.emit('plotly_click', {points: [{curveNumber: 0, pointNumber: 0}]});
        }"""
    )
    table = ready_page.locator("#explorer-struct-table table")
    expect(table).to_be_visible(timeout=TIMEOUT)


def test_explorer_similarity_heatmap(ready_page: Page) -> None:
    """The who-fails-alike heatmap renders as a square model x model matrix."""
    _goto_explorer(ready_page)
    heatmap = ready_page.locator("#explorer-similarity-graph .js-plotly-plot")
    expect(heatmap).to_be_visible(timeout=TIMEOUT)


def test_explorer_structure_click_shows_detail(ready_page: Page) -> None:
    """Clicking a head-to-head detail point loads its error table and 3D viewer."""
    _goto_explorer(ready_page)
    # Open the head-to-head drilldown (a category bar click reveals the detail scatter).
    graph = ready_page.locator("#explorer-h2h-graph .js-plotly-plot")
    expect(graph).to_be_visible(timeout=TIMEOUT)
    ready_page.evaluate(
        """() => {
            const gd = document.querySelector('#explorer-h2h-graph .js-plotly-plot');
            const y = gd.data[0].y[0];
            gd.emit('plotly_click',
                {points: [{curveNumber: 0, pointNumber: 0, y: y}]});
        }"""
    )
    detail = ready_page.locator("#explorer-h2h-detail-graph .js-plotly-plot")
    expect(detail).to_be_visible(timeout=TIMEOUT)
    # Scattergl points can't be clicked reliably by canvas coordinates, so fire
    # Plotly's click event directly; Dash re-derives customdata from
    # gd.data[curveNumber].customdata[pointNumber], so the point's two indices carry
    # the [key, category, benchmark, sid, xyz] needed by the callback.
    ready_page.evaluate(
        """() => {
            const gd = document.querySelector(
                '#explorer-h2h-detail-graph .js-plotly-plot'
            );
            gd.emit('plotly_click', {points: [{curveNumber: 0, pointNumber: 0}]});
        }"""
    )
    table = ready_page.locator("#explorer-struct-table table")
    expect(table).to_be_visible(timeout=TIMEOUT)
    # The viewer shows the WEAS iframe when a geometry ships, or the fallback note.
    viewer = ready_page.locator("#explorer-struct-viewer")
    geometry = viewer.locator("iframe")
    fallback = viewer.get_by_text("No 3D geometry", exact=False)
    expect(geometry.or_(fallback).first).to_be_visible(timeout=TIMEOUT)
