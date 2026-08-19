"""
Derive the cross-model Explorer signals from the shipped benchmark data.

Reconstructs a per-structure ``structure x model`` error matrix from the parity
figures already shipped in ``ml_peg/app/data`` (no re-computation of predictions),
then derives the signals the Explorer page renders:

* per-structure **hardness** rows -- the median error across models (its "hardness"),
  the cross-model disagreement, the structure's element set, and the relative path to a
  3D geometry for the WEAS viewer.
* the model x model **error-correlation** matrix (Pearson over per-structure errors,
  normalised within each benchmark so units don't dominate) together with a
  hierarchical-clustering leaf order for the heatmap/dendrogram.

The Explorer page calls :func:`compute_explorer_data` directly and memoises it, so the
signals are built **on load** from whatever benchmark data is present in
``ml_peg/app/data`` -- adding a new benchmark's ``figure_*.json`` (and its shipped
``.xyz`` geometries) makes it appear in the Explorer with no separate build step. The
CLI below is a convenience that also writes the same data to JSON::

    python -m ml_peg.analysis.explorer.build_explorer_data
    python -m ml_peg.analysis.explorer.build_explorer_data --limit 5  # quick subset

The error of a parity point is ``abs(x - y)``: parity axes are reference-vs-predicted
(orientation varies between benchmarks), so the absolute difference is the error
regardless of which axis is which.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from functools import cache
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
import typer

logger = logging.getLogger(__name__)

# Data root holds one directory per category, each with benchmark subdirectories.
DATA_ROOT = Path(__file__).resolve().parents[2] / "app" / "data"
OUT_DIR = DATA_ROOT / "explorer"

# Benchmarks whose structures span multiple models still need a minimum committee
# for a "hardness"/"disagreement" number to be meaningful.
MIN_MODELS = 3


def _iter_parity_traces(figure: dict[str, Any]):
    """
    Yield ``(model, structure_ids, errors)`` for each model trace in a parity figure.

    Parameters
    ----------
    figure
        Deserialised Plotly figure dictionary.

    Yields
    ------
    tuple[str, list[str], list[float]]
        Model name, per-point structure ids, and per-point **signed** residuals
        (``x - y``). The sign is kept so an ensemble's error can be modelled as the
        residual of the mean prediction (over- and under-predictions cancel); the
        magnitude is the per-structure error. Within one figure all models share the
        reference/predicted axis orientation, so signs are mutually comparable. Traces
        without markers, without a name, or without ``customdata`` (reference lines,
        density plots) are skipped.
    """
    for trace in figure.get("data", []):
        if "markers" not in (trace.get("mode") or ""):
            continue
        model = trace.get("name")
        customdata = trace.get("customdata")
        x = trace.get("x")
        y = trace.get("y")
        if not model or not isinstance(customdata, list) or x is None or y is None:
            continue
        if not (len(customdata) == len(x) == len(y)):
            continue
        ids: list[str] = []
        residuals: list[float] = []
        for cd, xi, yi in zip(customdata, x, y, strict=True):
            sid = cd[0] if isinstance(cd, list) else cd
            if xi is None or yi is None:
                continue
            ids.append(str(sid))
            residuals.append(float(xi) - float(yi))
        if ids:
            yield model, ids, residuals


def _benchmark_key(figure_path: Path) -> tuple[str, str, str]:
    """
    Split a ``figure_*.json`` path into ``(category, benchmark, metric)``.

    Parameters
    ----------
    figure_path
        Path to a parity figure under :data:`DATA_ROOT`.

    Returns
    -------
    tuple[str, str, str]
        Top-level category, benchmark (any intermediate directories joined by "/"),
        and the metric (the figure filename stem without the ``figure_`` prefix).
    """
    rel = figure_path.relative_to(DATA_ROOT)
    parts = rel.parts
    category = parts[0]
    benchmark = "/".join(parts[1:-1])
    metric = figure_path.stem.removeprefix("figure_")
    return category, benchmark, metric


def _xyz_element_union(path: Path) -> list[str] | None:
    """
    Read the set of element symbols in an ``.xyz``/``.extxyz`` file (all frames).

    A structure's geometry file may hold several frames (e.g. a reaction's reagents);
    the union of their element symbols is the structure's element set. Parses only the
    first token of each atom line, so it is fast and independent of ASE.

    Parameters
    ----------
    path
        Path to a shipped geometry file.

    Returns
    -------
    list[str] | None
        Sorted unique element symbols, or ``None`` if the file cannot be read or holds
        no recognisable symbols (e.g. atomic-number-indexed geometries).
    """
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    symbols: set[str] = set()
    i, n = 0, len(lines)
    while i < n:
        header = lines[i].strip()
        if not header:
            i += 1
            continue
        try:
            n_atoms = int(header)
        except ValueError:
            # Not a frame header (stray/blank line); skip and resync.
            i += 1
            continue
        # header, then a comment line, then n_atoms atom lines "<symbol> x y z ...".
        for k in range(i + 2, min(i + 2 + n_atoms, n)):
            tokens = lines[k].split()
            if tokens and tokens[0].isalpha():
                symbols.add(tokens[0])
        i = i + 2 + n_atoms
    return sorted(symbols) if symbols else None


@cache
def _benchmark_geometry_index(
    category: str, benchmark: str
) -> tuple[dict[str, Path], dict[str, Path]]:
    """
    Index a benchmark's shipped ``.xyz`` geometries by structure id, dynamically.

    Scans the first model subdirectory (geometries are identical across models) and
    keys each file by its stem. A parity point's structure id is sometimes a suffix of
    the geometry stem -- e.g. a GSCDB138 reaction id ``AE11_1`` is shipped as
    ``AE11_AE11_1.xyz`` -- so every ``_``-delimited tail of each stem is also indexed as
    a fallback. Cached per benchmark; rebuilt from disk each process, so newly-committed
    benchmark data is picked up on the next load.

    Parameters
    ----------
    category
        Benchmark category directory.
    benchmark
        Benchmark directory (relative to the category).

    Returns
    -------
    tuple[dict[str, Path], dict[str, Path]]
        ``(exact, suffix)`` maps from structure id to geometry path; ``exact`` wins.
    """
    bench_dir = DATA_ROOT / category / benchmark
    exact: dict[str, Path] = {}
    suffix: dict[str, Path] = {}
    if not bench_dir.is_dir():
        return exact, suffix
    model_dirs = [p for p in sorted(bench_dir.iterdir()) if p.is_dir()]
    if not model_dirs:
        return exact, suffix
    for file in model_dirs[0].rglob("*.xyz"):
        stem = file.stem
        exact.setdefault(stem, file)
        pos = stem.find("_")
        while pos != -1:
            suffix.setdefault(stem[pos + 1 :], file)
            pos = stem.find("_", pos + 1)
    return exact, suffix


def _resolve_structure(
    category: str, benchmark: str, sid: str
) -> tuple[str | None, list[str]]:
    """
    Resolve a structure's shipped geometry path and element set from the shipped data.

    Parameters
    ----------
    category
        Benchmark category directory.
    benchmark
        Benchmark directory (relative to the category).
    sid
        Structure id from the parity figure.

    Returns
    -------
    tuple[str | None, list[str]]
        The geometry path relative to :data:`DATA_ROOT` (POSIX style, for the WEAS
        viewer) or ``None`` if no geometry is shipped, and the sorted element set (empty
        when it cannot be recovered -- an explicit "unspecified" bucket the Explorer's
        chemistry filter shows, shrinking as more geometries are shipped).
    """
    exact, suffix = _benchmark_geometry_index(category, benchmark)
    file = exact.get(sid) or suffix.get(sid)
    if file is None:
        return None, []
    elements = _xyz_element_union(file) or []
    return file.relative_to(DATA_ROOT).as_posix(), elements


def compute_explorer_data(
    limit: int | None = None,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """
    Build the Explorer signals in memory from the shipped benchmark data.

    Reads every ``figure_*.json`` under :data:`DATA_ROOT`, reconstructs the
    error matrix, and derives the hardness rows and model-similarity payload. The
    Explorer page calls this (memoised) on load, so the result reflects whatever data is
    currently shipped -- no separate build step is required.

    Parameters
    ----------
    limit
        If given, only process the first ``limit`` parity figures (for quick local
        validation). Default is ``None`` (process everything).
    progress
        Optional callback invoked with a monotonic ``0.0 -> 1.0`` fraction as the
        figures are parsed and the similarity step finishes, so a UI can show real
        progress while this runs on a background thread.

    Returns
    -------
    dict[str, Any]
        ``{"hardness", "error_matrix", "similarity"}`` -- the hardness rows, the
        ``structure_key -> {model: residual}`` matrix, and the similarity payload.
    """
    figures = sorted(DATA_ROOT.glob("**/figure_*.json"))
    if limit is not None:
        figures = figures[:limit]

    # structure_key -> {"resids": {model: err}, metadata...}
    structures: dict[str, dict[str, Any]] = {}
    # (benchmark, model) error vectors, keyed by structure_key, for correlation.
    per_benchmark: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    all_models: set[str] = set()

    # Parsing the figures (with their element/geometry reads) is the bulk of the work;
    # reserve the last slice of the bar for the similarity clustering.
    total = len(figures)
    skipped = 0
    for done, figure_path in enumerate(figures):
        if progress is not None and total:
            progress(0.9 * done / total)
        try:
            figure = json.loads(figure_path.read_text())
            category, benchmark, metric = _benchmark_key(figure_path)
            bench_id = f"{category}/{benchmark}/{metric}"

            for model, ids, residuals in _iter_parity_traces(figure):
                all_models.add(model)
                # strict=True: a trace whose x/y/customdata lengths disagree is
                # malformed -- skip the whole figure (below) rather than silently
                # misalign structure ids with residuals.
                for sid, resid in zip(ids, residuals, strict=True):
                    key = f"{bench_id}::{sid}"
                    per_benchmark[bench_id][model][key] = resid
                    row = structures.get(key)
                    if row is None:
                        xyz, elements = _resolve_structure(category, benchmark, sid)
                        row = {
                            "key": key,
                            "sid": sid,
                            "category": category,
                            "benchmark": benchmark,
                            "metric": metric,
                            "elements": elements,
                            "xyz": xyz,
                            "resids": {},
                        }
                        structures[key] = row
                    row["resids"][model] = resid
        except (ValueError, OSError) as exc:
            # Malformed JSON, unreadable file, or a length-mismatched trace: drop this
            # one figure but keep building, and make the loss visible in the logs.
            skipped += 1
            logger.warning(
                "Skipping malformed Explorer figure %s: %s", figure_path, exc
            )
            continue
    if skipped:
        logger.warning(
            "Explorer data build skipped %d of %d parity figures", skipped, total
        )

    if progress is not None:
        progress(0.9)
    hardness_rows, error_matrix = _summarise_structures(structures)
    similarity = _model_similarity(per_benchmark, sorted(all_models))
    if progress is not None:
        progress(1.0)
    return {
        "hardness": hardness_rows,
        "error_matrix": error_matrix,
        "similarity": similarity,
    }


def build(limit: int | None = None) -> dict[str, Any]:
    """
    Compute the Explorer signals and also write them to JSON (CLI convenience).

    The app no longer needs these files -- it calls :func:`compute_explorer_data` on
    load -- but writing them is useful for inspection or offline caching.

    Parameters
    ----------
    limit
        If given, only process the first ``limit`` parity figures. Default ``None``.

    Returns
    -------
    dict[str, Any]
        Summary counts ``{"structures", "models", "benchmarks"}``.
    """
    data = compute_explorer_data(limit=limit)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "hardness.json").write_text(json.dumps(data["hardness"]))
    (OUT_DIR / "error_matrix.json").write_text(json.dumps(data["error_matrix"]))
    (OUT_DIR / "model_similarity.json").write_text(json.dumps(data["similarity"]))
    return {
        "structures": len(data["hardness"]),
        "models": len(data["similarity"].get("models", [])),
        "benchmarks": len({row["key"].rsplit("::", 1)[0] for row in data["hardness"]}),
    }


def _summarise_structures(
    structures: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """
    Reduce each structure's per-model errors to hardness and disagreement signals.

    Parameters
    ----------
    structures
        Mapping from structure key to its accumulated row (including ``errs``).

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, dict[str, float]]]
        The lightweight hardness rows (``median_err`` hardness, ``disagreement``
        coefficient of variation, ``n_models``, metadata and geometry path -- but no
        per-model errors, so the plot payload stays small), and the ``error_matrix``
        mapping ``structure_key -> {model: signed_residual}`` for O(1) click lookups
        and committee (ensemble) calculations. Rows with fewer than :data:`MIN_MODELS`
        models are dropped.
    """
    rows: list[dict[str, Any]] = []
    error_matrix: dict[str, dict[str, float]] = {}
    for row in structures.values():
        resids = row.pop("resids")
        if len(resids) < MIN_MODELS:
            continue
        abs_by_model = {m: abs(v) for m, v in resids.items()}
        magnitudes = np.array(list(abs_by_model.values()), dtype=float)
        median = float(np.median(magnitudes))
        mean = float(np.mean(magnitudes))
        std = float(np.std(magnitudes))
        # Coefficient of variation is a scale-free "how much do models disagree here",
        # comparable across benchmarks with different units. Guard the zero-mean case.
        disagreement = float(std / mean) if mean > 0 else 0.0
        row["median_err"] = round(median, 6)
        row["disagreement"] = round(disagreement, 6)
        row["n_models"] = len(resids)
        # Extra discovery-scatter fields (all derivable from the errors in hand).
        row["n_elements"] = len(row["elements"])
        row["spread_abs"] = round(float(magnitudes.max() - magnitudes.min()), 6)
        row["best_model"] = min(abs_by_model, key=abs_by_model.get)
        row["worst_model"] = max(abs_by_model, key=abs_by_model.get)
        rows.append(row)
        error_matrix[row["key"]] = {m: round(v, 6) for m, v in resids.items()}

    _add_hardness_percentile(rows)
    return rows, error_matrix


def _add_hardness_percentile(rows: list[dict[str, Any]]) -> None:
    """
    Annotate each row with ``hardness_pct``: its median-error percentile in-benchmark.

    Raw ``median_err`` is in each benchmark's own units, so it cannot be compared
    across benchmarks. The percentile rank of a structure's median error *within its
    own benchmark-metric* (0 = easiest, 1 = hardest) is unit-free and comparable, so
    the Explorer can place every structure on one "hardness" axis.

    Parameters
    ----------
    rows
        Hardness rows (mutated in place to add ``hardness_pct``).
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["category"], row["benchmark"], row["metric"])].append(row)
    for group in groups.values():
        order = sorted(group, key=lambda r: r["median_err"])
        denom = max(len(order) - 1, 1)
        for rank, row in enumerate(order):
            row["hardness_pct"] = round(rank / denom, 4)


def _model_similarity(
    per_benchmark: dict[str, dict[str, dict[str, float]]],
    models: list[str],
) -> dict[str, Any]:
    """
    Compute the model x model error-correlation matrix and a clustering leaf order.

    Signed residuals are z-scored within each benchmark before stacking, so a
    high-magnitude benchmark does not dominate the correlation and models that err in
    the same direction score as similar. Correlation is computed pairwise over the
    structures both models cover.

    Parameters
    ----------
    per_benchmark
        ``benchmark -> model -> {structure_key: error}``.
    models
        Sorted list of all model names.

    Returns
    -------
    dict[str, Any]
        ``{"models", "order", "corr", "n_structures"}`` where ``order`` is the model
        index order from hierarchical clustering (empty if too few models).
    """
    index = {m: i for i, m in enumerate(models)}
    # Build a long-form normalised error table: structure_key -> model -> z-score.
    normalised: dict[str, dict[str, float]] = defaultdict(dict)
    for model_errors in per_benchmark.values():
        for model, key_errs in model_errors.items():
            vals = np.array(list(key_errs.values()), dtype=float)
            if len(vals) < 2:
                continue
            mean = vals.mean()
            std = vals.std()
            if std == 0:
                continue
            for key, err in key_errs.items():
                normalised[key][model] = (err - mean) / std

    n = len(models)
    corr = np.full((n, n), np.nan)
    np.fill_diagonal(corr, 1.0)
    # Accumulate per-model vectors aligned on shared structures for each pair.
    columns: dict[str, dict[str, float]] = defaultdict(dict)
    for key, model_vals in normalised.items():
        for model, z in model_vals.items():
            columns[model][key] = z

    for i, mi in enumerate(models):
        for j in range(i + 1, n):
            mj = models[j]
            shared = columns[mi].keys() & columns[mj].keys()
            if len(shared) < 10:
                continue
            a = np.array([columns[mi][k] for k in shared])
            b = np.array([columns[mj][k] for k in shared])
            if a.std() == 0 or b.std() == 0:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            corr[i, j] = corr[j, i] = c

    order = _cluster_order(corr)
    corr_out = [
        [None if np.isnan(v) else round(float(v), 4) for v in row] for row in corr
    ]
    n_structures = len(normalised)
    return {
        "models": models,
        "order": order,
        "corr": corr_out,
        "n_structures": n_structures,
        "index": index,
    }


def _cluster_order(corr: np.ndarray) -> list[int]:
    """
    Order models by hierarchical clustering of ``1 - correlation`` distances.

    Parameters
    ----------
    corr
        Symmetric model x model correlation matrix (may contain NaNs).

    Returns
    -------
    list[int]
        Leaf order of model indices. Falls back to the identity order when there are
        fewer than three models or the distances are degenerate.
    """
    n = corr.shape[0]
    if n < 3:
        return list(range(n))
    # Undefined pairs (too few shared structures) are imputed at the *mean* observed
    # correlation, so a never-co-evaluated pair sits at average distance rather than
    # being treated as maximally dissimilar (which would let coverage, not behaviour,
    # drive the leaf order).
    off_diagonal = corr[~np.eye(n, dtype=bool)]
    defined = off_diagonal[~np.isnan(off_diagonal)]
    fill = float(defined.mean()) if defined.size else 0.0
    dist = 1.0 - np.where(np.isnan(corr), fill, corr)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    # scipy expects the condensed upper triangle.
    condensed = dist[np.triu_indices(n, k=1)]
    if not np.any(condensed):
        return list(range(n))
    linkage_matrix = linkage(condensed, method="average")
    return [int(i) for i in leaves_list(linkage_matrix)]


def main(limit: int | None = None) -> None:
    """
    Typer entry point: build the artifacts and print a short summary.

    Parameters
    ----------
    limit
        Optional cap on the number of parity figures to process.
    """
    summary = build(limit=limit)
    typer.echo(
        f"Wrote Explorer data to {OUT_DIR}: "
        f"{summary['structures']} structures, {summary['models']} models, "
        f"{summary['benchmarks']} benchmark-metrics."
    )


if __name__ == "__main__":
    typer.run(main)
