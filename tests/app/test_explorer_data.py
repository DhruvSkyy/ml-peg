"""Unit tests for the Model Behaviour Explorer data and figure logic.

These are browser-free: they exercise the parity parser, the on-load data build, the
element/geometry resolver, the figure builders, the head-to-head stats, the model
finder, and the progress reporting directly, so the Explorer's data pipeline is guarded
without a full app + Playwright run.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from ml_peg.analysis.explorer import build_explorer_data
from ml_peg.analysis.explorer.build_explorer_data import (
    _iter_parity_traces,
    _xyz_element_union,
    build,
)
from ml_peg.app.utils import explorer_data as explorer_data_module
from ml_peg.app.utils.build_explorer import (
    benchmark_error_medians,
    build_coverage_rose_figure,
    build_finder_figure,
    build_h2h_detail_figure,
    build_head_to_head_figure,
    build_report_profile_figure,
    build_similarity_heatmap_figure,
    build_structure_fails_figure,
    cached_finder_rankings,
    cached_head_to_head_stats,
    cached_structure_fails_ranking,
    head_to_head_stats,
    load_benchmark_scores,
    load_error_matrix,
    load_hardness,
    load_model_benchmark_scores,
    load_similarity,
    model_finder_rankings,
    model_taxonomy,
    report_card_summary,
    representative_models,
    structure_fails_ranking,
)
from ml_peg.app.utils.explorer_data import (
    _benchmark_lot_class,
    category_page_paths,
    split_structure_key,
    structure_model_ranks,
)
from ml_peg.app.utils.register_explorer_callbacks import _finder_pick_model
from ml_peg.app.utils.utils import classify_level_of_theory


def test_iter_parity_traces_signed_residual() -> None:
    """A parity trace yields signed residuals and skips the y=x reference line."""
    figure = {
        "data": [
            {
                "name": "model-a",
                "mode": "markers",
                "x": [1.0, 3.0],
                "y": [1.5, 2.0],
                "customdata": [["s1"], ["s2"]],
            },
            # Reference line: no name, no customdata -> must be skipped.
            {"name": None, "mode": "lines", "x": [0, 1], "y": [0, 1]},
        ]
    }
    traces = list(_iter_parity_traces(figure))
    assert len(traces) == 1
    model, ids, residuals = traces[0]
    assert model == "model-a"
    assert ids == ["s1", "s2"]
    assert residuals == pytest.approx([-0.5, 1.0])


def test_xyz_element_union_multiframe(tmp_path) -> None:
    """The element parser unions symbols across all frames of an .xyz file."""
    path = tmp_path / "reaction.xyz"
    path.write_text(
        "2\ncomment\nC 0 0 0\nH 1 0 0\n"  # frame 1: C, H
        "1\ncomment\nO 0 0 0\n"  # frame 2: O
    )
    assert _xyz_element_union(path) == ["C", "H", "O"]
    assert _xyz_element_union(tmp_path / "missing.xyz") is None


def test_explorer_data_consistent_with_new_fields() -> None:
    """The on-load hardness/error-matrix/similarity data agree and carry new fields."""
    rows = load_hardness()
    error_matrix = load_error_matrix()
    similarity = load_similarity()
    assert rows and error_matrix and similarity

    row = rows[0]
    for field in (
        "key",
        "median_err",
        "disagreement",
        "hardness_pct",
        "n_models",
        "elements",
        "n_elements",
        "spread_abs",
        "best_model",
        "worst_model",
    ):
        assert field in row
    assert 0.0 <= row["hardness_pct"] <= 1.0
    assert row["n_elements"] == len(row["elements"])
    # Every hardness row has an error-matrix entry with the recorded model count.
    assert row["key"] in error_matrix
    assert len(error_matrix[row["key"]]) == row["n_models"]

    corr = similarity["corr"]
    n = len(similarity["models"])
    assert len(corr) == n
    assert all(corr[i][i] == 1.0 for i in range(n)), "correlation diagonal must be 1"


def test_elements_recovered_for_most_structures() -> None:
    """The dynamic resolver recovers elements for the large majority of structures."""
    rows = load_hardness()
    with_elements = sum(1 for row in rows if row["elements"])
    assert with_elements / len(rows) > 0.8


def test_build_subset_writes_expected_shape(tmp_path, monkeypatch) -> None:
    """``build`` on a small subset writes artifacts (redirected off the real data)."""
    monkeypatch.setattr(build_explorer_data, "OUT_DIR", tmp_path)
    summary = build(limit=6)
    assert summary["structures"] > 0
    assert summary["models"] > 0
    assert summary["benchmarks"] > 0
    assert (tmp_path / "hardness.json").exists()
    assert (tmp_path / "error_matrix.json").exists()
    assert (tmp_path / "model_similarity.json").exists()


def _broadest_pair() -> tuple[str, str]:
    """Return the two most broadly-evaluated model names."""
    coverage: dict[str, int] = defaultdict(int)
    for resids in load_error_matrix().values():
        for model in resids:
            coverage[model] += 1
    ranked = sorted(coverage, key=coverage.get, reverse=True)
    return ranked[0], ranked[1]


def test_head_to_head_stats_and_verdict_bars() -> None:
    """Head-to-head wins are consistent and the verdict bars encode them."""
    model_a, model_b = _broadest_pair()
    error_matrix = load_error_matrix()
    stats = head_to_head_stats(model_a, model_b, error_matrix)
    assert stats["n_shared"] > 0
    assert stats["wins_a"] + stats["wins_b"] + stats["ties"] == stats["n_shared"]
    # Per-category win counts sum to the overall shared count.
    assert sum(c["n"] for c in stats["per_category"]) == stats["n_shared"]

    figure = build_head_to_head_figure(model_a, model_b, error_matrix)
    bars = figure.data[0]
    # One diverging bar per shared category, keyed by category for click-to-drill.
    assert sorted(bars.y) == sorted(c["category"] for c in stats["per_category"])
    # Bars encode win-rate margin around even (0.5): margin is within ±0.5.
    per_cat = {c["category"]: c for c in stats["per_category"]}
    for category, margin in zip(bars.y, bars.x, strict=True):
        assert margin == pytest.approx(per_cat[category]["win_rate_a"] - 0.5)


def test_head_to_head_detail_scatter() -> None:
    """The category drilldown scatter is clickable and matches the category count."""
    model_a, model_b = _broadest_pair()
    error_matrix = load_error_matrix()
    stats = head_to_head_stats(model_a, model_b, error_matrix)
    meta = {row["key"]: row for row in load_hardness()}
    # Pick the largest category; meta may lack a few keys, so allow <=.
    category = stats["per_category"][0]["category"]
    figure = build_h2h_detail_figure(model_a, model_b, category, error_matrix, meta)
    points = figure.data[0]
    assert 0 < len(points.x) <= stats["per_category"][0]["n"]
    assert len(points.customdata[0]) == 5  # key,cat,bench,sid,xyz -> 3D drilldown
    assert all(cd[1] == category for cd in points.customdata)
    # Log axes with a y=x reference line as the second trace.
    assert figure.layout.xaxis.type == "log"
    assert figure.data[1].mode == "lines"


def test_head_to_head_disjoint_models_empty() -> None:
    """Two models that share no structures give an empty (but valid) figure."""
    error_matrix = {"cat/b/m::s": {"only_a": 0.1}}
    stats = head_to_head_stats("only_a", "only_b", error_matrix)
    assert stats["n_shared"] == 0
    figure = build_head_to_head_figure("only_a", "only_b", error_matrix)
    assert not figure.data


def test_head_to_head_equal_weight_across_benchmarks() -> None:
    """A high-volume benchmark cannot dominate the verdict; ties leave the denominator.

    In one category, benchmark ``big`` (100 structures) is a clean win for A and
    benchmark ``small`` (2 structures) a clean win for B. Pooling raw structures would
    read as a ~98% A win; weighting each benchmark equally makes it an even 50%.
    """
    error_matrix: dict[str, dict[str, float]] = {}
    for i in range(100):
        error_matrix[f"cat/big/mae::{i}"] = {"A": 0.1, "B": 0.2}  # A closer
    for i in range(2):
        error_matrix[f"cat/small/mae::{i}"] = {"A": 0.2, "B": 0.1}  # B closer
    error_matrix["cat/tie/mae::0"] = {"A": 0.3, "B": 0.3}  # exact tie

    stats = head_to_head_stats("A", "B", error_matrix)
    # Overall and the single category both average the two decisive benchmarks (1, 0).
    assert stats["overall_win_rate_a"] == pytest.approx(0.5)
    (row,) = stats["per_category"]
    assert row["win_rate_a"] == pytest.approx(0.5)
    # Raw counts are retained for context; the tie is counted but not in the win rate.
    assert (row["wins_a"], row["wins_b"], row["ties"]) == (100, 2, 1)
    assert row["decisive"] == 102 and row["n_benchmarks"] == 2
    assert stats["n_shared"] == 103


def test_head_to_head_win_rate_symmetric_in_ties() -> None:
    """Excluding ties keeps the win rate symmetric: rate(A) + rate(B) == 1."""
    error_matrix = {
        "cat/b/mae::0": {"A": 0.1, "B": 0.2},  # A wins
        "cat/b/mae::1": {"A": 0.2, "B": 0.1},  # B wins
        "cat/b/mae::2": {"A": 0.5, "B": 0.5},  # tie (must not skew either way)
    }
    rate_a = head_to_head_stats("A", "B", error_matrix)["overall_win_rate_a"]
    rate_b = head_to_head_stats("B", "A", error_matrix)["overall_win_rate_a"]
    assert rate_a == pytest.approx(0.5)
    assert rate_a + rate_b == pytest.approx(1.0)


def test_structure_fails_ranking_modes() -> None:
    """Difficulty, disagreement, and per-model ranking each order the rows correctly."""
    rows = load_hardness()
    assert rows

    hardest = structure_fails_ranking(rows, None, None, "hardest", top=10)
    assert 0 < len(hardest) <= 10
    scores = [r["fail_score"] for r in hardest]
    assert scores == sorted(scores, reverse=True)  # worst-first
    assert all(r["fail_score"] == r["hardness_pct"] for r in hardest)

    contested = structure_fails_ranking(rows, None, None, "contested", top=10)
    c_scores = [r["fail_score"] for r in contested]
    assert c_scores == sorted(c_scores, reverse=True)
    assert all(r["fail_score"] == r["disagreement"] for r in contested)

    # Per-model mode ranks by the model's own error rank vs peers (1 = worst).
    model = next(iter(next(iter(load_error_matrix().values()))))
    per_model = structure_fails_ranking(rows, None, model, "hardest", top=10)
    assert per_model, "focus model should have failing structures"
    m_scores = [r["model_rank"] for r in per_model]
    assert m_scores == sorted(m_scores, reverse=True)
    assert all(0.0 <= r["model_rank"] <= 1.0 for r in per_model)

    # Category filtering restricts the rows to the requested area.
    category = rows[0]["category"]
    only = structure_fails_ranking(rows, [category], None, "hardest", top=50)
    assert only and all(r["category"] == category for r in only)


def test_build_structure_fails_figure_clickable() -> None:
    """The fails bars carry the shared [key, cat, bench, sid, xyz] click payload."""
    rows = load_hardness()
    figure = build_structure_fails_figure(rows, sort="hardest", top=8)
    bars = figure.data[0]
    assert 0 < len(bars.y) <= 8
    # First five customdata fields match the head-to-head detail scatter, so both feed
    # the one show_structure handler.
    assert all(len(cd) >= 5 for cd in bars.customdata)
    key, category, benchmark, sid, _xyz = bars.customdata[0][:5]
    assert key in load_error_matrix()
    parsed_category, bench_prefix, parsed_sid = split_structure_key(key)
    assert parsed_category == category and parsed_sid == sid
    assert bench_prefix.split("/")[1] == benchmark
    # Empty selection yields a valid placeholder figure, not a crash.
    empty = build_structure_fails_figure([], sort="hardest")
    assert empty.layout.xaxis.visible is False


def test_similarity_heatmap_shape_and_order() -> None:
    """The heatmap is a square matrix ordered by the clustering, with unique labels."""
    sim = load_similarity()
    models = sim.get("models", [])
    if len(models) < 2:
        pytest.skip("too few models with shared evaluations to compare")
    figure = build_similarity_heatmap_figure()
    heat = figure.data[0]
    assert len(heat.x) == len(heat.y) == len(models)
    # Labels are unique (heatmap axes collapse duplicate category strings otherwise).
    assert len(set(heat.x)) == len(heat.x)
    # Axis order follows the payload's hierarchical-clustering leaf order.
    order = sim.get("order") or list(range(len(models)))
    assert list(heat.x) == [models[k] for k in order]
    # Diagonal is self-correlation (1.0).
    assert heat.z[0][0] == pytest.approx(1.0)


def test_benchmark_scores_shape() -> None:
    """Each aggregated benchmark row has a category and an in-range mean score."""
    rows = load_benchmark_scores()
    assert rows, "no *_metrics_table.json files found under app/data"
    for row in rows:
        assert row["category"] and row["benchmark"]
        assert 0.0 <= row["mean_score"] <= 1.0
        assert row["n_models"] > 0


def test_coverage_rose_granularity() -> None:
    """The rose has one wedge per category, or one per benchmark, with aligned data."""
    rows = load_benchmark_scores()
    n_categories = len({r["category"] for r in rows})

    by_category = build_coverage_rose_figure(rows, "category")
    by_benchmark = build_coverage_rose_figure(rows, "benchmark")

    cat_trace = by_category.data[0]
    bench_trace = by_benchmark.data[0]
    assert len(cat_trace.r) == len(cat_trace.theta) == n_categories
    assert len(bench_trace.r) == len(bench_trace.theta) == len(rows)
    # Colour drives the red->green encoding, so it must align with the bar lengths.
    assert list(cat_trace.marker.color) == list(cat_trace.r)


def test_model_finder_rankings() -> None:
    """The finder ranks broadly-covered models with both evidence streams."""
    all_results = model_finder_rankings()
    assert all_results
    fits = [r["fit"] for r in all_results]
    assert fits == sorted(fits, reverse=True)
    assert all(0.0 <= r["fit"] <= 1.0 for r in all_results)

    # Element evidence engages the structure stream (or flags its absence).
    rows = load_hardness()
    element = next(e for r in rows for e in (r["elements"] or []))
    with_elements = model_finder_rankings(elements=[element])
    assert with_elements
    assert any(r["n_struct"] > 0 for r in with_elements)
    for r in with_elements:
        assert r["no_element_evidence"] == (r["n_struct"] == 0)

    figure = build_finder_figure(all_results)
    assert len(figure.data[0].y) == min(12, len(all_results))


def test_structure_model_ranks_normalised() -> None:
    """Per-structure ranks span 0 (best) to 1 (worst) over the evaluated cohort."""
    ranks = structure_model_ranks()
    assert ranks
    key, per_model = next(iter(ranks.items()))
    values = sorted(per_model.values())
    assert values[0] == 0.0 and values[-1] == 1.0
    assert set(per_model) <= set(load_error_matrix()[key])


def test_category_page_paths_route_shape() -> None:
    """Every scored category maps to a main-app category route."""
    paths = category_page_paths()
    categories = {row["category"] for row in load_model_benchmark_scores()}
    assert set(paths) == categories
    assert all(path.startswith("/category/") for path in paths.values())
    # Titles (not directory names) drive the slug, matching build_app's routing.
    if "bulk_crystal" in paths:
        assert paths["bulk_crystal"] == "/category/bulk-crystals"


def test_head_to_head_normalization_is_unitless() -> None:
    """Head-to-head axes are relative errors (structure error / benchmark median)."""
    error_matrix = load_error_matrix()
    medians = benchmark_error_medians(error_matrix)
    assert medians and all(m >= 0 for m in medians.values())
    # A benchmark median exists for every structure key's benchmark prefix.
    sample_key = next(iter(error_matrix))
    assert sample_key.rsplit("::", 1)[0] in medians


def test_compute_progress_is_monotonic() -> None:
    """``compute_explorer_data`` reports a monotonic 0->1 progress fraction."""
    seen: list[float] = []
    build_explorer_data.compute_explorer_data(limit=8, progress=seen.append)
    assert seen and seen == sorted(seen)
    assert 0.0 <= seen[0] and seen[-1] == 1.0


def test_classify_level_of_theory() -> None:
    """The LoT classifier matches the table warning scheme."""
    assert classify_level_of_theory("PBE", "PBE") == "in_domain"
    assert classify_level_of_theory("PBE", "r2SCAN") == "dft"
    assert classify_level_of_theory("PBE", "DLPNO-CCSD(T)/CBS") == "high_level"
    assert classify_level_of_theory("PBE", "Experimental") == "experimental"
    assert classify_level_of_theory("PBE", None) == "in_domain"


def test_model_benchmark_scores_shape() -> None:
    """Per-model benchmark scores carry the score, levels of theory, and LoT class."""
    rows = load_model_benchmark_scores()
    assert rows
    row = rows[0]
    for field in (
        "model",
        "base_id",
        "category",
        "benchmark",
        "domain",
        "score",
        "model_lot",
        "ref_levels",
        "lot_class",
    ):
        assert field in row
    assert 0.0 <= row["score"] <= 1.0
    assert row["lot_class"] in {
        "in_domain",
        "dft",
        "high_level",
        "experimental",
        "none",
    }
    # Physicality benchmarks carry no reference level -> "none".
    phys = [r for r in rows if r["domain"] == "physicality"]
    assert phys and all(r["lot_class"] == "none" for r in phys)


def test_model_taxonomy_groups_variants() -> None:
    """Dispersion/head variants collapse under one architecture family."""
    tax = model_taxonomy()
    assert tax["variants"] and tax["families"]
    # A base model and its +D3 twin share a family.
    variants = tax["variants"]
    if "mace-mp-0a" in variants and "mace-mp-0a-D3" in variants:
        assert variants["mace-mp-0a"]["family"] == variants["mace-mp-0a-D3"]["family"]
    # Every family lists at least one model, ordered by coverage.
    assert all(fam["models"] for fam in tax["families"].values())


def test_report_card_summary_splits_in_domain_and_physicality() -> None:
    """A PBE model has in-domain PBE tests and separate physicality; profile bars."""
    summary = report_card_summary("mace-mp-0a")
    assert summary["training_lot"] == "PBE"
    assert summary["n_benchmarks"] > 0
    in_mean, in_n = summary["in_domain"]
    assert in_n > 0 and in_mean is not None  # PBE-referenced benchmarks exist
    assert summary["physicality"][1] > 0
    figure = build_report_profile_figure("mace-mp-0a")
    assert len(figure.data[0].y) > 0


def test_representative_models_collapses_families() -> None:
    """Representatives keep one variant per family, fewer than the full list."""
    models = load_similarity()["models"]
    reps = representative_models(models)
    assert 0 < len(reps) < len(models)
    families = {model_taxonomy()["variants"].get(m, {}).get("family", m) for m in reps}
    assert len(families) == len(reps)  # one per family


def test_finder_pick_model_routing() -> None:
    """The finder click router resolves cards vs bars and guards re-renders."""
    from dash.exceptions import PreventUpdate

    card_trigger = {"type": "explorer-finder-pick", "model": "mace-mp-0a"}
    # A card click routes to the card's model.
    assert _finder_pick_model(card_trigger, None, [0, 1, 0]) == "mace-mp-0a"
    # All-zero click counts mean the cards merely re-rendered: no update.
    with pytest.raises(PreventUpdate):
        _finder_pick_model(card_trigger, None, [0, 0, 0])
    # A bar click routes to the bar's y value (the model name).
    bar_click = {"points": [{"y": "orb-v3"}]}
    assert _finder_pick_model("explorer-finder-graph", bar_click, []) == "orb-v3"
    # An empty/missing click event is a no-op.
    with pytest.raises(PreventUpdate):
        _finder_pick_model("explorer-finder-graph", None, [])


def test_split_structure_key() -> None:
    """The key parser splits category, benchmark prefix, and sid; '/' in metric OK."""
    assert split_structure_key("surfaces/adsorb/energy::s1") == (
        "surfaces",
        "surfaces/adsorb/energy",
        "s1",
    )
    # A metric containing "/" stays part of the benchmark prefix.
    assert split_structure_key("molecular/conf/dE/atom::mol-7") == (
        "molecular",
        "molecular/conf/dE/atom",
        "mol-7",
    )


def test_warm_explorer_data_primes_all_loaders() -> None:
    """``warm_explorer_data`` primes the base data and every derived cached loader."""
    from ml_peg.app.utils import explorer_data

    derived = (
        explorer_data.load_benchmark_scores,
        explorer_data.load_model_benchmark_scores,
        explorer_data.structure_model_ranks,
        explorer_data.category_page_paths,
        explorer_data.model_taxonomy,
    )
    for fn in derived:
        fn.cache_clear()
    explorer_data.warm_explorer_data()
    assert explorer_data.explorer_data_ready()
    for fn in derived:
        assert fn.cache_info().currsize == 1, fn.__name__


def test_cached_finder_rankings_memoizes() -> None:
    """Identical use-case controls return the same cached ranking list."""
    first = cached_finder_rankings(None, None, "any")
    assert cached_finder_rankings(None, None, "any") is first
    assert isinstance(first, list) and first, "ranking should be non-empty"
    assert cached_finder_rankings(("molecular",), None, "any") is not first


def test_cached_head_to_head_stats_memoizes() -> None:
    """A model pair returns the same cached stats, equal to the direct computation."""
    model_a, model_b = _broadest_pair()
    first = cached_head_to_head_stats(model_a, model_b)
    assert cached_head_to_head_stats(model_a, model_b) is first
    # The wrapper matches head_to_head_stats over the production error matrix.
    assert first == head_to_head_stats(model_a, model_b, load_error_matrix())
    assert first["n_shared"] > 0


def test_cached_structure_fails_ranking_memoizes() -> None:
    """Identical controls return the same cached ranking, equal to the direct call."""
    first = cached_structure_fails_ranking(None, None, "hardest", 10)
    assert cached_structure_fails_ranking(None, None, "hardest", 10) is first
    assert first == structure_fails_ranking(load_hardness(), None, None, "hardest", 10)
    # A different sort key is a distinct cache entry.
    assert cached_structure_fails_ranking(None, None, "contested", 10) is not first


def test_a11y_graph_gives_charts_an_accessible_name() -> None:
    """Interactive graphs are wrapped with role=img + aria-label for screen readers."""
    from dash import dcc

    from ml_peg.app.utils.build_explorer import _a11y_graph

    wrapped = _a11y_graph(dcc.Graph(id="demo"), "Finder ranking chart")
    props = wrapped.to_plotly_json()["props"]
    assert props["role"] == "img"
    assert props["aria-label"] == "Finder ranking chart"


# --- Edge cases / robustness -------------------------------------------------------


def test_finder_unknown_elements_fall_back_to_benchmark_only() -> None:
    """An element with no tested structures still ranks models on benchmark evidence."""
    results = model_finder_rankings(elements=["Xx"])  # no structure carries "Xx"
    assert results, "finder should still rank on benchmark evidence"
    for row in results:
        assert row["n_struct"] == 0
        assert row["struct_score"] is None
        assert row["no_element_evidence"] is True
        # With no structure evidence, fit is the (benchmark-only) adjusted mean.
        assert 0.0 <= row["fit"] <= 1.0


def test_head_to_head_all_ties_reads_even() -> None:
    """A pair that ties on every shared structure reads 50/50, ties out of the rate."""
    error_matrix = {
        "cat/b/mae::0": {"A": 0.2, "B": 0.2},
        "cat/b/mae::1": {"A": 0.3, "B": 0.3},
    }
    stats = head_to_head_stats("A", "B", error_matrix)
    assert stats["wins_a"] == stats["wins_b"] == 0
    assert stats["ties"] == 2 and stats["n_shared"] == 2
    assert stats["overall_win_rate_a"] == pytest.approx(0.5)
    (row,) = stats["per_category"]
    assert row["decisive"] == 0 and row["win_rate_a"] == pytest.approx(0.5)
    # An all-tie comparison still yields a valid (non-empty) verdict figure.
    figure = build_head_to_head_figure("A", "B", error_matrix)
    assert figure.data


def test_benchmark_lot_class_clamps_unknown(monkeypatch) -> None:
    """An unexpected classifier output degrades to the neutral 'none' style."""
    monkeypatch.setattr(
        explorer_data_module, "classify_level_of_theory", lambda *_: "bogus_class"
    )
    assert _benchmark_lot_class("PBE", {"energy": "PBE"}) == "none"
    # A benchmark with no reference level is still "none" (unchanged behaviour).
    assert _benchmark_lot_class("PBE", {}) == "none"
