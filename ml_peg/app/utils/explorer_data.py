"""
Shared data layer for the Explorer page.

Loads the cross-model signals the Explorer renders. Everything is derived from the
benchmark data shipped under ``ml_peg/app/data`` (parity figures, geometries, and
metrics tables) and cached, so the Explorer reflects whatever benchmarks are present --
adding a new benchmark makes it appear with no build step.

Building the signals takes a few seconds, so :func:`warm_explorer_data` is called once
at app startup (before serving) to build and cache them, keeping that work off the
routing callback. Every loader then serves from that cache; non-UI callers (tests)
build it lazily on first use.

The Explorer holds no per-session state: its controls start from their defaults on
every visit and it is independent of the app-wide model and element filters, which
scope the benchmark tables rather than this cross-benchmark view.

This module holds only data loaders (no Dash/figure code) so it can be imported by the
figure/layout modules without an import cycle.
"""

from __future__ import annotations

from functools import lru_cache
import logging
import re
import threading
from typing import Any

from yaml import YAMLError, safe_load

from ml_peg.analysis.explorer.build_explorer_data import (
    compute_explorer_data,
    data_file_index,
    data_root,
    read_json,
)
from ml_peg.app import APP_ROOT
from ml_peg.app.utils.explorer_ids import LOT_CLASSES
from ml_peg.app.utils.utils import (
    WARNING_CATEGORY_PRIORITY,
    category_to_path,
    classify_level_of_theory,
)

logger = logging.getLogger(__name__)

# Category directory -> conceptual domain, used to order the report card and split
# physical soundness ("physicality") out from accuracy. Keys are the directory names
# under ml_peg/app; test_explorer_data checks they all exist.
CATEGORY_DOMAIN = {
    "alloy_metallurgy": "materials",
    "bulk_crystal": "materials",
    "surfaces": "materials",
    "defects": "materials",
    "molecular_crystal": "materials",
    "molecular_dynamics": "materials",
    "nebs": "materials",
    "electrolytes": "materials",
    "molecular": "molecular",
    "molecular_reactions": "molecular",
    "conformers": "molecular",
    "isomers": "molecular",
    "thermochemistry": "molecular",
    "non_covalent_interactions": "molecular",
    "supramolecular": "molecular",
    "tm_complexes": "molecular",
    "f_block": "molecular",
    "electric_field": "molecular",
    "biomolecules": "molecular",
    "physicality": "physicality",
}

# Name tokens dropped when reducing a model id to its architecture family, so a base
# model and its dispersion/head/size variants collapse to one family (e.g. mace-mp-0a,
# mace-mp-0a-D3, mace-mp-0b3 -> "mace-mp"). Version-like tokens (0a, 0b3, 5m, 1p1) are
# dropped by pattern; "v3"-style tokens are kept so ORB-v3 stays its own family.
_FAMILY_DROP_TOKENS = {"omat", "omol", "consv", "direct", "inf", "s", "m", "l", "xs"}
_VERSION_TOKEN = re.compile(r"^\d+[a-z]?\d*$|^\d+p\d+$")

# The (expensive) signals are computed once and cached. A lock makes the first build
# single-flight so concurrent callers never build it twice.
_lock = threading.Lock()
_data: dict[str, Any] | None = None


def _ensure_data() -> dict[str, Any]:
    """
    Return the cached Explorer data, computing it once on first use.

    Returns
    -------
    dict[str, Any]
        The cached ``{"hardness", "error_matrix", "similarity"}`` payload.
    """
    global _data
    if _data is not None:
        return _data
    with _lock:
        if _data is None:
            _data = compute_explorer_data()
    return _data


def warm_explorer_data() -> None:
    """
    Build and cache all Explorer data now (blocking), so navigating there is instant.

    Called once at app startup, before the server serves requests, so the build never
    lands on the routing callback. Besides the base signals, the loaders the page
    layout reads first are primed here. A failure is logged and otherwise swallowed
    so the app still boots; the Explorer then shows its no-data guidance.
    """
    try:
        _ensure_data()
        load_benchmark_scores()
        load_model_benchmark_scores()
        structure_model_ranks()
        category_page_paths()
        model_taxonomy()
    except Exception:
        logger.exception("Explorer data could not be built; the page will show no data")


def explorer_data_ready() -> bool:
    """
    Report whether the Explorer data is cached and ready.

    Returns
    -------
    bool
        ``True`` once the cache is populated.
    """
    return _data is not None


def split_structure_key(key: str) -> tuple[str, str, str]:
    """
    Split a structure key into its category, benchmark prefix, and structure id.

    Structure keys have the form ``"category/benchmark/metric::sid"``; this is the one
    place that format is parsed, so consumers never split on ``"/"``/``"::"`` ad hoc.

    Parameters
    ----------
    key
        A structure key from the error matrix / hardness rows.

    Returns
    -------
    tuple[str, str, str]
        ``(category, benchmark_prefix, sid)`` where ``benchmark_prefix`` is
        ``"category/benchmark/metric"``.
    """
    bench, _, sid = key.rpartition("::")
    return bench.split("/", 1)[0], bench, sid


def load_hardness() -> list[dict[str, Any]]:
    """
    Return the per-structure hardness rows (computed on first use, then cached).

    Returns
    -------
    list[dict[str, Any]]
        One row per benchmark structure; empty if no benchmark data is shipped.
    """
    return _ensure_data()["hardness"]


def load_error_matrix() -> dict[str, dict[str, float]]:
    """
    Return the ``structure_key -> {model: signed_residual}`` matrix.

    Returns
    -------
    dict[str, dict[str, float]]
        The error matrix, or an empty mapping if no benchmark data is shipped.
    """
    return _ensure_data()["error_matrix"]


def load_similarity() -> dict[str, Any]:
    """
    Return the model similarity payload (correlation matrix + clustering order).

    Returns
    -------
    dict[str, Any]
        ``{"models", "order", "corr"}``; all three empty if no benchmark data is
        shipped.
    """
    return _ensure_data()["similarity"]


@lru_cache(maxsize=1)
def load_hardness_by_key() -> dict[str, dict[str, Any]]:
    """
    Index the hardness rows by structure key for O(1) lookups in click handlers.

    Returns
    -------
    dict[str, dict[str, Any]]
        ``structure_key -> hardness row``.
    """
    return {row["key"]: row for row in load_hardness()}


@lru_cache(maxsize=1)
def load_benchmark_scores() -> list[dict[str, Any]]:
    """
    Aggregate each benchmark's mean model ``Score`` from the shipped metrics tables.

    Every benchmark ships a ``*_metrics_table.json`` holding one row per model with a
    ``Score`` in ``[0, 1]`` (1 = meets the "good" threshold, 0 = at/below "bad"). The
    mean over all scored models is the "how solved is this area?" signal the coverage
    rose renders. Derived from :func:`load_model_benchmark_scores` so the tables are
    read once.

    Returns
    -------
    list[dict[str, Any]]
        One row per benchmark with ``category``, ``benchmark``, ``mean_score`` and
        ``n_models`` (the count of models with a numeric score), in table order.
    """
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in load_model_benchmark_scores():
        grouped.setdefault((row["category"], row["benchmark"]), []).append(row["score"])
    return [
        {
            "category": category,
            "benchmark": benchmark,
            "mean_score": sum(scores) / len(scores),
            "n_models": len(scores),
        }
        for (category, benchmark), scores in grouped.items()
    ]


def _benchmark_lot_class(
    model_lot: str | None, metric_levels: dict[str, str | None]
) -> str:
    """
    Classify a model's test on one benchmark as in-domain or a kind of extrapolation.

    A benchmark may score several metrics against different references; the benchmark's
    class is the most severe mismatch across them (or ``"in_domain"`` if all match, or
    ``"none"`` when it carries no reference level -- e.g. physicality soundness checks).

    Parameters
    ----------
    model_lot
        The model's training level of theory.
    metric_levels
        Mapping of metric name to its reference level (``None`` = no reference).

    Returns
    -------
    str
        ``"in_domain"``, ``"dft"``, ``"high_level"``, ``"experimental"`` or ``"none"``.
    """
    references = [level for level in metric_levels.values() if level]
    if not references:
        return "none"
    classes = [classify_level_of_theory(model_lot, level) for level in references]
    mismatches = [cls for cls in classes if cls != "in_domain"]
    if not mismatches:
        return "in_domain"
    worst = max(mismatches, key=lambda cls: WARNING_CATEGORY_PRIORITY.get(cls, 0))
    # Guard the Explorer's colour/label lookup: an unexpected class (upstream change
    # in classify_level_of_theory) falls back to the neutral "none" style rather than
    # rendering a missing colour.
    if worst not in LOT_CLASSES:
        logger.warning("Unknown level-of-theory class %r; treating as 'none'", worst)
        return "none"
    return worst


@lru_cache(maxsize=1)
def load_model_benchmark_scores() -> list[dict[str, Any]]:
    """
    Per-model, per-benchmark scores joined to level-of-theory context.

    Reads every shipped ``*_metrics_table.json`` and, for each scored model row, records
    its benchmark ``Score`` alongside the model's training level of theory, the
    benchmark's reference level(s), and the extrapolation class of that pairing -- the
    backbone of the Explorer's model report card.

    Returns
    -------
    list[dict[str, Any]]
        Rows of ``{model, base_id, category, benchmark, domain, score, model_lot,
        ref_levels, lot_class}``. ``base_id`` is the dispersion-stripped model id.
    """
    rows: list[dict[str, Any]] = []
    for path in data_file_index().metrics_tables:
        try:
            table = read_json(path)
        except (ValueError, OSError) as exc:
            logger.warning("Skipping malformed metrics table %s: %s", path, exc)
            continue
        rel = path.relative_to(data_root()).parts  # (category, ...benchmark..., file)
        category = rel[0]
        benchmark = "/".join(rel[1:-1])
        metric_levels = table.get("metric_levels_of_theory", {}) or {}
        model_levels = table.get("model_levels_of_theory", {}) or {}
        name_map = table.get("model_name_map", {}) or {}
        ref_levels = sorted({lvl for lvl in metric_levels.values() if lvl})
        for row in table.get("data", []):
            name = row.get("MLIP")
            score = row.get("Score")
            if not isinstance(name, str) or not isinstance(score, (int, float)):
                continue
            model_lot = model_levels.get(name)
            rows.append(
                {
                    "model": name,
                    "base_id": row.get("id") or name_map.get(name) or name,
                    "category": category,
                    "benchmark": benchmark,
                    "domain": CATEGORY_DOMAIN.get(category, "materials"),
                    "score": float(score),
                    "model_lot": model_lot,
                    "ref_levels": ref_levels,
                    "lot_class": _benchmark_lot_class(model_lot, metric_levels),
                }
            )
    return rows


@lru_cache(maxsize=1)
def structure_model_ranks() -> dict[str, dict[str, float]]:
    """
    Rank each model's error within every structure's evaluated cohort.

    Per structure, models are ordered by absolute error and assigned
    ``rank / (n - 1)`` so 0 is the most accurate model on that structure and 1 the
    least. Ranks are unit-free and cohort-relative, so they can be pooled across
    benchmarks with different units -- the model finder's structure-level evidence.

    Returns
    -------
    dict[str, dict[str, float]]
        ``structure_key -> {model: normalised rank}``; structures evaluated by fewer
        than two models are omitted (a rank of a single model is meaningless).
    """
    ranks: dict[str, dict[str, float]] = {}
    for key, resids in load_error_matrix().items():
        if len(resids) < 2:
            continue
        ordered = sorted(resids, key=lambda m: abs(resids[m]))
        denom = len(ordered) - 1
        ranks[key] = {model: i / denom for i, model in enumerate(ordered)}
    return ranks


@lru_cache(maxsize=1)
def category_page_paths() -> dict[str, str]:
    """
    Map each category directory name to its main-app category page path.

    Uses the main app's routing rule (:func:`ml_peg.app.utils.utils.category_to_path`)
    on the category's display title, read from ``ml_peg/app/<category>/<category>.yml``
    and falling back to the directory name when no title is shipped -- exactly as
    ``build_app`` does, so the Explorer's deep links can never diverge from the pages.

    Returns
    -------
    dict[str, str]
        ``category_dir -> "/category/<slug>"`` for every category with shipped data.
    """
    paths: dict[str, str] = {}
    for category in sorted({row["category"] for row in load_model_benchmark_scores()}):
        title = category
        # The category ymls are source, not data, so they are found under APP_ROOT
        # rather than relative to the (test-redirectable) data root.
        yml = APP_ROOT / category / f"{category}.yml"
        try:
            info = safe_load(yml.read_text())
            title = (info or {}).get("title") or category
        except (OSError, YAMLError) as exc:
            logger.warning("No readable title for category %s: %s", category, exc)
        try:
            paths[category] = category_to_path(title)
        except ValueError as exc:
            logger.warning("Skipping category page link for %s: %s", category, exc)
    return paths


def _model_family(base_id: str) -> str:
    """
    Reduce a model id to its architecture family (dispersion/head/size collapsed).

    Parameters
    ----------
    base_id
        The dispersion-stripped model id (e.g. ``"uma-s-1p1-omat"``).

    Returns
    -------
    str
        The family key (e.g. ``"uma"``, ``"orb-v3"``, ``"mace-mp"``).
    """
    keep = [
        token
        for token in base_id.split("-")
        if token.lower() not in _FAMILY_DROP_TOKENS
        and not _VERSION_TOKEN.match(token.lower())
    ]
    return "-".join(keep) or base_id


@lru_cache(maxsize=1)
def model_coverage() -> dict[str, int]:
    """
    Count the benchmarks each model was scored on.

    Returns
    -------
    dict[str, int]
        ``model -> number of scored benchmarks``.
    """
    coverage: dict[str, int] = {}
    for row in load_model_benchmark_scores():
        coverage[row["model"]] = coverage.get(row["model"], 0) + 1
    return coverage


@lru_cache(maxsize=1)
def model_taxonomy() -> dict[str, dict[str, Any]]:
    """
    Group every evaluated model under its architecture family.

    Built from the shipped metrics tables (base id, training level of theory) so the
    report card and head-to-head can offer a family-first picker and the similarity
    view can collapse near-duplicate variants.

    Returns
    -------
    dict[str, Any]
        ``{"variants": {model: {family, family_label, training_lot}},
        "families": {family: {label, models}}}``. ``families`` is ordered by label;
        each family's ``models`` are ordered by descending benchmark coverage so the
        first is a sensible default.
    """
    coverage = model_coverage()
    info: dict[str, dict[str, Any]] = {}
    for row in load_model_benchmark_scores():
        name = row["model"]
        if name not in info:
            family = _model_family(row["base_id"])
            info[name] = {
                "family": family,
                "family_label": family.upper(),
                "training_lot": row["model_lot"],
            }

    families: dict[str, dict[str, Any]] = {}
    for name, meta in info.items():
        fam = families.setdefault(
            meta["family"], {"label": meta["family_label"], "models": []}
        )
        fam["models"].append(name)
    for fam in families.values():
        fam["models"].sort(key=lambda m: coverage.get(m, 0), reverse=True)
    ordered = dict(sorted(families.items(), key=lambda kv: kv[1]["label"]))
    return {"variants": info, "families": ordered}
