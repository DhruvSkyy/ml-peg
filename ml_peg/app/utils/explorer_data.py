"""
Shared data layer for the Explorer page.

Loads the cross-model signals the Explorer renders. Everything is derived from the
benchmark data shipped under ``ml_peg/app/data`` (parity figures, geometries, and
metrics tables) and cached, so the Explorer reflects whatever benchmarks are present --
adding a new benchmark makes it appear with no build step.

Building the signals takes several seconds, so :func:`warm_explorer_data` is called once
at app startup (before serving) to build and cache them -- keeping the heavy work off
the routing callback (which otherwise froze the UI) and off a worker thread that a busy
dev server would starve. Every loader then serves from that cache; non-UI callers
(tests, the CLI) build it lazily on first use.

This module holds only data loaders (no Dash/figure code) so it can be imported by the
figure/layout module without an import cycle.
"""

from __future__ import annotations

from functools import lru_cache
import json
import logging
from pathlib import Path
import re
import threading
from typing import Any

from yaml import safe_load

from ml_peg.analysis.explorer.build_explorer_data import compute_explorer_data
from ml_peg.app.utils.utils import (
    WARNING_CATEGORY_PRIORITY,
    classify_level_of_theory,
)

logger = logging.getLogger(__name__)

# The shipped benchmark data: one directory per category, holding the metrics tables
# and parity figures every Explorer signal is derived from.
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

# Category -> conceptual domain, used to order the report card and split physical
# soundness ("physicality") out from accuracy.
CATEGORY_DOMAIN = {
    "bulk_crystal": "materials",
    "surfaces": "materials",
    "defect": "materials",
    "molecular_crystal": "materials",
    "molecular_dynamics": "materials",
    "nebs": "materials",
    "molecular": "molecular",
    "molecular_reactions": "molecular",
    "conformers": "molecular",
    "isomers": "molecular",
    "thermochemistry": "molecular",
    "non_covalent_interactions": "molecular",
    "supramolecular": "molecular",
    "tm_complexes": "molecular",
    "lanthanides": "molecular",
    "electric_field": "molecular",
    "physicality": "physicality",
}

# The extrapolation classes the report card knows how to colour (mirrors the keys of
# LOT_CLASS_STYLE in build_explorer). Kept here to avoid a circular import.
_LOT_CLASSES = frozenset({"in_domain", "dft", "high_level", "experimental", "none"})

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

    Called once at app startup, before the server serves requests: this keeps the heavy
    build off the routing callback (which froze the UI) without needing a worker thread
    that a busy dev server would starve. Besides the base signals, the derived
    ``lru_cache`` loaders are also primed here -- together they cost seconds cold, which
    would otherwise land on the first Explorer render. Failures are swallowed so the app
    still boots; the Explorer then shows its no-data guidance.
    """
    try:
        _ensure_data()
        load_benchmark_scores()
        load_model_benchmark_scores()
        structure_model_ranks()
        category_page_paths()
        model_taxonomy()
    except Exception:  # noqa: BLE001 - never let a data issue block app startup
        pass


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
        The similarity payload, or an empty mapping if no benchmark data is shipped.
    """
    return _ensure_data()["similarity"]


@lru_cache(maxsize=1)
def load_benchmark_scores() -> list[dict[str, Any]]:
    """
    Aggregate each benchmark's mean model ``Score`` from the shipped metrics tables.

    Every benchmark ships a ``*_metrics_table.json`` holding one row per model with a
    ``Score`` in ``[0, 1]`` (1 = meets the "good" threshold, 0 = at/below "bad"). The
    mean over all scored models is the "how solved is this area?" signal the coverage
    rose renders, so this reads them all once (cached).

    Returns
    -------
    list[dict[str, Any]]
        One row per benchmark with ``category``, ``benchmark``, ``mean_score`` and
        ``n_models`` (the count of models with a numeric score). Benchmarks with no
        numeric scores are skipped.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(DATA_ROOT.glob("**/*_metrics_table.json")):
        try:
            table = json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            logger.warning("Skipping malformed metrics table %s: %s", path, exc)
            continue
        scores = [
            row["Score"]
            for row in table.get("data", [])
            if isinstance(row.get("Score"), (int, float))
        ]
        if not scores:
            continue
        rel = path.relative_to(DATA_ROOT).parts  # (category, ...benchmark..., file)
        rows.append(
            {
                "category": rel[0],
                "benchmark": "/".join(rel[1:-1]),
                "mean_score": sum(scores) / len(scores),
                "n_models": len(scores),
            }
        )
    return rows


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
    if worst not in _LOT_CLASSES:
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
    for path in sorted(DATA_ROOT.glob("**/*_metrics_table.json")):
        try:
            table = json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            logger.warning("Skipping malformed metrics table %s: %s", path, exc)
            continue
        rel = path.relative_to(DATA_ROOT).parts
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

    Replicates the main app's routing: the page slug is derived from the category's
    display title (read from ``ml_peg/app/<category>/<category>.yml``), falling back to
    the directory name when no title is shipped -- exactly as ``build_app`` does.

    Returns
    -------
    dict[str, str]
        ``category_dir -> "/category/<slug>"`` for every category with shipped data.
    """
    app_root = DATA_ROOT.parent  # ml_peg/app
    paths: dict[str, str] = {}
    for category in sorted({row["category"] for row in load_model_benchmark_scores()}):
        title = category
        yml = app_root / category / f"{category}.yml"
        try:
            info = safe_load(yml.read_text())
            title = (info or {}).get("title") or category
        except (OSError, ValueError):
            pass
        slug = "".join(c.lower() if c.isalnum() else "-" for c in title)
        slug = "-".join(part for part in slug.split("-") if part)
        if slug:
            paths[category] = f"/category/{slug}"
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
def model_taxonomy() -> dict[str, dict[str, Any]]:
    """
    Group every evaluated model under its architecture family, with variant metadata.

    Built from the shipped metrics tables (training level of theory, base id, dispersion
    suffix) so the report card and head-to-head can offer a family-first picker and the
    similarity/committee views can collapse near-duplicate variants.

    Returns
    -------
    dict[str, Any]
        ``{"variants": {model: {base_id, family, family_label, variant_label,
        training_lot}}, "families": {family: {label, models}}}``. ``families`` is
        ordered by label; each family's ``models`` are ordered by descending benchmark
        coverage so the first is a sensible default.
    """
    scores = load_model_benchmark_scores()
    coverage: dict[str, int] = {}
    info: dict[str, dict[str, Any]] = {}
    for row in scores:
        name = row["model"]
        coverage[name] = coverage.get(name, 0) + 1
        if name not in info:
            base_id = row["base_id"]
            family = _model_family(base_id)
            suffix = name[len(base_id) :].strip("-") if name.startswith(base_id) else ""
            info[name] = {
                "base_id": base_id,
                "family": family,
                "family_label": family.upper(),
                "variant_label": name,
                "training_lot": row["model_lot"],
                "dispersion": suffix,
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
