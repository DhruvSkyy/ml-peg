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

The Explorer page calls :func:`compute_explorer_data` once at startup and memoises it,
so the signals are built **on load** from whatever benchmark data is present in
``ml_peg/app/data`` -- adding a new benchmark's ``figure_*.json`` (and its shipped
``.xyz`` geometries) makes it appear in the Explorer with no separate build step.

The data tree is walked exactly once (:func:`data_file_index`): a benchmark directory
is one that holds a metrics table or parity figures, and its subdirectories are the
per-model outputs (thousands of geometry files), which are never searched for figures.
Cross-model parity figures therefore live in benchmark directories; a figure inside a
model directory belongs to that model alone and is not a cross-model comparison.

The error of a parity point is ``abs(x - y)``: parity axes are reference-vs-predicted
(orientation varies between benchmarks), so the absolute difference is the error
regardless of which axis is which.
"""

from __future__ import annotations

from collections import defaultdict
from functools import cache
import json
import logging
import os
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import orjson
from scipy.cluster.hierarchy import leaves_list, linkage

logger = logging.getLogger(__name__)

# Data root holds one directory per category, each with benchmark subdirectories.
DATA_ROOT = Path(__file__).resolve().parents[2] / "app" / "data"

# Benchmarks whose structures span multiple models still need a minimum committee
# for a "hardness"/"disagreement" number to be meaningful.
MIN_MODELS = 3

# A model pair needs this many co-evaluated structures before its error correlation is
# reported; fewer and the coefficient is noise.
MIN_SHARED_STRUCTURES = 10

# Rounding applied when serialising derived numbers, so the payload shipped to the
# browser stays compact without losing meaningful precision.
_ERROR_DECIMALS = 6
_RATIO_DECIMALS = 4


def data_root() -> Path:
    """
    Return the directory the Explorer reads shipped benchmark data from.

    Every read goes through this rather than binding :data:`DATA_ROOT` at import
    time, so the tests can point the Explorer at an isolated fixture tree with a
    single ``monkeypatch.setattr`` on this module. A ``from ... import DATA_ROOT``
    elsewhere captures the value instead, and would keep reading the real tree.

    Returns
    -------
    Path
        Directory holding one subdirectory per category.
    """
    return DATA_ROOT


class DataFileIndex(NamedTuple):
    """The benchmark-level data files found by one walk of :func:`data_root`."""

    figures: tuple[Path, ...]
    """Sorted paths of every benchmark-level ``figure_*.json``."""
    metrics_tables: tuple[Path, ...]
    """Sorted paths of every ``*_metrics_table.json``."""
    model_dirs: dict[tuple[str, str], tuple[Path, ...]]
    """``(category, benchmark)`` -> sorted per-model output directories."""


@cache
def data_file_index() -> DataFileIndex:
    """
    Locate every benchmark-level data file with a single pruned walk of the data tree.

    The shipped data holds ~1.5 million files, almost all per-model geometries, so
    recursive globs for the handful of figures and metrics tables dominated the
    Explorer's startup cost. Walking once and pruning below each benchmark directory
    (a directory that holds a metrics table or parity figures) finds the same files in
    milliseconds. Cached per process, like every other Explorer loader.

    Returns
    -------
    DataFileIndex
        The figures, metrics tables, and per-benchmark model directories.
    """
    figures: list[Path] = []
    tables: list[Path] = []
    model_dirs: dict[tuple[str, str], tuple[Path, ...]] = {}
    root = data_root()
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        figs = [f for f in filenames if f.startswith("figure_") and f.endswith(".json")]
        tabs = [f for f in filenames if f.endswith("_metrics_table.json")]
        figures.extend(here / f for f in figs)
        tables.extend(here / f for f in tabs)
        if figs or tabs:
            rel = here.relative_to(root).parts
            if len(rel) >= 2:
                model_dirs[(rel[0], "/".join(rel[1:]))] = tuple(
                    sorted(here / d for d in dirnames)
                )
            # Everything below a benchmark directory is per-model output.
            dirnames[:] = []
    return DataFileIndex(tuple(sorted(figures)), tuple(sorted(tables)), model_dirs)


def read_json(path: Path) -> Any:
    """
    Parse a JSON file with orjson, falling back to the stdlib for bare ``NaN`` tokens.

    orjson is several times faster on the large parity figures but rejects the bare
    ``NaN``/``Infinity`` tokens the analysis stage sometimes writes, which the stdlib
    parser accepts. The same fallback lives in ``ml_peg.app.utils.load.read_json``;
    it is repeated here so this analysis module stays importable without Dash.

    Parameters
    ----------
    path
        Path to the JSON file.

    Returns
    -------
    Any
        The parsed document.
    """
    data = path.read_bytes()
    try:
        return orjson.loads(data)
    except orjson.JSONDecodeError:
        return json.loads(data)


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
        Path to a parity figure under :func:`data_root`.

    Returns
    -------
    tuple[str, str, str]
        Top-level category, benchmark (any intermediate directories joined by "/"),
        and the metric (the figure filename stem without the ``figure_`` prefix).
    """
    rel = figure_path.relative_to(data_root())
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
    except OSError as exc:
        logger.warning("Cannot read geometry %s: %s", path, exc)
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
    exact: dict[str, Path] = {}
    suffix: dict[str, Path] = {}
    model_dirs = data_file_index().model_dirs.get((category, benchmark), ())
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


@cache
def _benchmark_element_index(category: str, benchmark: str) -> dict[str, list[str]]:
    """
    Map a benchmark's geometry-file stems to their element sets via ``info.json``.

    The analysis stage writes each benchmark's ``info.json``; when it pairs a
    per-structure ``elements`` list-of-lists with an equal-length ``filenames`` list,
    the element sets are read from there instead of parsing every geometry file. Any
    other shape is not trusted: a benchmark-wide element union would wrongly assign
    every element to every structure, and multi-frame geometries make the two lists
    misaligned (``elements`` is per frame, ``filenames`` per file), so the geometry
    parser stays authoritative in both cases.

    Parameters
    ----------
    category
        Benchmark category directory.
    benchmark
        Benchmark directory (relative to the category).

    Returns
    -------
    dict[str, list[str]]
        ``geometry stem -> sorted element symbols``; empty when ``info.json`` is
        absent or not per-structure.
    """
    info_path = data_root() / category / benchmark / "info.json"
    try:
        payload = read_json(info_path)
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    elements = payload.get("elements")
    filenames = payload.get("filenames")
    if (
        not isinstance(elements, list)
        or not isinstance(filenames, list)
        or len(elements) != len(filenames)
        or not all(isinstance(els, list) and els for els in elements)
    ):
        return {}
    return {
        str(stem): sorted({str(symbol) for symbol in els})
        for stem, els in zip(filenames, elements, strict=True)
    }


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
        The geometry path relative to :func:`data_root` (POSIX style, for the WEAS
        viewer) or ``None`` if no geometry is shipped, and the sorted element set (empty
        when it cannot be recovered -- an explicit "unspecified" bucket the Explorer's
        chemistry filter shows, shrinking as more geometries are shipped).
    """
    exact, suffix = _benchmark_geometry_index(category, benchmark)
    file = exact.get(sid) or suffix.get(sid)
    if file is None:
        return None, []
    elements = _benchmark_element_index(category, benchmark).get(file.stem)
    if elements is None:
        elements = _xyz_element_union(file) or []
    return file.relative_to(data_root()).as_posix(), elements


def compute_explorer_data() -> dict[str, Any]:
    """
    Build the Explorer signals in memory from the shipped benchmark data.

    Reads every benchmark-level ``figure_*.json`` under :func:`data_root`, reconstructs
    the error matrix, and derives the hardness rows and model-similarity payload. The
    Explorer page calls this (memoised) at startup, so the result reflects whatever
    data is currently shipped -- no separate build step is required.

    Returns
    -------
    dict[str, Any]
        ``{"hardness", "error_matrix", "similarity"}`` -- the hardness rows, the
        ``structure_key -> {model: residual}`` matrix, and the similarity payload.
    """
    figures = data_file_index().figures

    # structure_key -> {"resids": {model: err}, metadata...}
    structures: dict[str, dict[str, Any]] = {}
    # (benchmark, model) error vectors, keyed by structure_key, for correlation.
    per_benchmark: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    all_models: set[str] = set()

    skipped = 0
    for figure_path in figures:
        try:
            figure = read_json(figure_path)
            category, benchmark, metric = _benchmark_key(figure_path)
            bench_id = f"{category}/{benchmark}/{metric}"

            for model, ids, residuals in _iter_parity_traces(figure):
                all_models.add(model)
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
        except (ValueError, TypeError, OSError, AttributeError) as exc:
            # Malformed JSON, an unreadable file, or a trace whose points are not
            # numbers: stop reading this figure (structures already taken from it
            # stay) and make the loss visible in the logs.
            skipped += 1
            logger.warning(
                "Skipping malformed Explorer figure %s: %s", figure_path, exc
            )
            continue
    if skipped:
        logger.warning(
            "Explorer data build skipped %d of %d parity figures", skipped, len(figures)
        )

    hardness_rows, error_matrix = _summarise_structures(structures)
    similarity = _model_similarity(per_benchmark, sorted(all_models))
    return {
        "hardness": hardness_rows,
        "error_matrix": error_matrix,
        "similarity": similarity,
    }


def _summarise_structures(
    structures: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """
    Reduce each structure's per-model errors to hardness and disagreement signals.

    Parameters
    ----------
    structures
        Mapping from structure key to its accumulated row (including ``resids``).

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, dict[str, float]]]
        The lightweight hardness rows (``median_err`` hardness, ``disagreement``
        coefficient of variation, ``n_models``, ``worst_model``, metadata and geometry
        path -- but no per-model errors, so the plot payload stays small), and the
        ``error_matrix`` mapping ``structure_key -> {model: signed_residual}`` for O(1)
        click lookups and committee (ensemble) calculations. Rows with fewer than
        :data:`MIN_MODELS` models are dropped.
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
        row["median_err"] = round(median, _ERROR_DECIMALS)
        row["disagreement"] = round(disagreement, _ERROR_DECIMALS)
        row["n_models"] = len(resids)
        row["worst_model"] = max(abs_by_model, key=abs_by_model.get)
        rows.append(row)
        error_matrix[row["key"]] = {
            m: round(v, _ERROR_DECIMALS) for m, v in resids.items()
        }

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
            row["hardness_pct"] = round(rank / denom, _RATIO_DECIMALS)


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
        ``{"models", "order", "corr"}`` where ``order`` is the model index order from
        hierarchical clustering (identity order if too few models) and ``corr`` the
        matrix with ``None`` for pairs sharing too few structures.
    """
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
            if len(shared) < MIN_SHARED_STRUCTURES:
                continue
            a = np.array([columns[mi][k] for k in shared])
            b = np.array([columns[mj][k] for k in shared])
            if a.std() == 0 or b.std() == 0:
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            corr[i, j] = corr[j, i] = c

    order = _cluster_order(corr)
    corr_out = [
        [None if np.isnan(v) else round(float(v), _RATIO_DECIMALS) for v in row]
        for row in corr
    ]
    return {"models": models, "order": order, "corr": corr_out}


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
