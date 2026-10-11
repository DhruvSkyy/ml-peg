"""Fixtures for the browser-free Explorer tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from ml_peg.analysis.explorer import build_explorer_data


@pytest.fixture(scope="session", autouse=True)
def benchmark_data(
    fixture_data: Path,
    fixture_benchmarks: tuple[tuple[str, str], ...],
) -> Iterator[tuple[tuple[str, str], ...]]:
    """
    Point the Explorer at the unpacked fixture for every test in this directory.

    Reading the fixture where it was unpacked, rather than from a copy placed
    under ``ml_peg/app/data``, is what makes these tests hermetic: a developer
    with the real data in place would otherwise be testing against it, and the
    assertions here are specific to the fixture (which benchmarks the finder can
    rank, how many categories the walk finds). Redirecting the one module
    attribute is enough because every read goes through
    :func:`ml_peg.analysis.explorer.build_explorer_data.data_root`.

    Parameters
    ----------
    fixture_data
        Root of the unpacked fixture (``tests/conftest.py``).
    fixture_benchmarks
        The ``(category, benchmark)`` pairs the fixture supplies
        (``tests/conftest.py``).

    Yields
    ------
    tuple[tuple[str, str], ...]
        The ``(category, benchmark)`` pairs the fixture supplies.
    """
    patches = pytest.MonkeyPatch()
    patches.setattr(build_explorer_data, "DATA_ROOT", fixture_data)
    # The loaders cache per process, so clear them either side of the redirect:
    # nothing should carry over from an earlier root, in or out.
    loaders = (
        build_explorer_data.data_file_index,
        build_explorer_data._benchmark_geometry_index,
        build_explorer_data._benchmark_element_index,
    )
    for loader in loaders:
        loader.cache_clear()
    try:
        yield fixture_benchmarks
    finally:
        patches.undo()
        for loader in loaders:
            loader.cache_clear()
