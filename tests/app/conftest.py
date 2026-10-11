"""Fixtures for browser-driven ML-PEG app tests.

A small subset of the app is built once per test session and served on an
ephemeral port via a background Werkzeug thread. This exercises the real Dash app
(layout + callbacks) while avoiding the full ~13s cold start of every benchmark.
The benchmark data is the published test fixture (see ``tests/conftest.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib import import_module
import logging
from pathlib import Path
import shutil
import threading
import warnings

from playwright.sync_api import Locator
import pytest
from werkzeug.serving import make_server

from ml_peg.analysis.explorer import build_explorer_data
from ml_peg.app import APP_ROOT

# Quieten the per-request werkzeug access log so test output stays readable.
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# The benchmark the interaction tests drive by id.
TEST_CATEGORY = "non_covalent_interactions"
TEST_BENCHMARK = "IONPI19"

# Generous ceiling for first hydration of the app under CI load; the test files
# mirror this value in their own TIMEOUT constants.
READY_TIMEOUT = 60_000


def wait_for_app_ready(page, timeout: int = READY_TIMEOUT) -> None:  # noqa: ANN001
    """
    Block until the app has hydrated and hidden its start-up mask.

    Waiting on ``#startup-mask`` being ``state="hidden"`` is not enough: the
    mask is part of the layout Dash fetches after ``load``, so it is briefly
    absent — and Playwright counts "detached" as hidden, making such a wait
    return immediately against a blank page. Hence exists *and* hidden.

    Parameters
    ----------
    page
        Playwright page to wait on.
    timeout
        Milliseconds to wait before failing.
    """
    page.wait_for_function(
        """() => {
          const mask = document.getElementById('startup-mask');
          return mask !== null && mask.style.display === 'none';
        }""",
        timeout=timeout,
    )


@pytest.fixture(scope="session")
def benchmark_data(
    fixture_data: Path,
    fixture_benchmarks: tuple[tuple[str, str], ...],
    tmp_path_factory,
) -> Iterator[Path]:
    """
    Make the fixture benchmarks' data available under the app's assets folder.

    Redirect benchmark module paths, Dash assets and the Explorer's data root to
    an isolated temporary tree. Local analysis outputs cannot change the dataset
    or be overwritten.

    Parameters
    ----------
    fixture_data
        Root of the unpacked fixture.
    fixture_benchmarks
        The ``(category, benchmark)`` pairs the fixture supplies.
    tmp_path_factory
        Session temporary-directory factory.

    Yields
    ------
    Path
        Isolated assets directory.
    """
    data_root = tmp_path_factory.mktemp("ui-assets")
    shutil.copytree(APP_ROOT / "data" / "ui", data_root / "ui")
    patches = pytest.MonkeyPatch()
    for category, benchmark in fixture_benchmarks:
        target = data_root / category / benchmark
        shutil.copytree(fixture_data / category / benchmark, target)
        module = import_module(f"ml_peg.app.{category}.{benchmark}.app_{benchmark}")
        # Redirect all module-level data paths, including derived info paths.
        for name, value in vars(module).copy().items():
            if isinstance(value, Path) and value.is_relative_to(APP_ROOT / "data"):
                patches.setattr(
                    module, name, data_root / value.relative_to(APP_ROOT / "data")
                )
    # The Explorer reads every benchmark through this root (via its data_root()
    # accessor) and caches what it finds per process, so clear the caches either
    # side of the redirect: nothing should carry over from an earlier root, in or
    # out.
    patches.setattr(build_explorer_data, "DATA_ROOT", data_root)
    loaders = (
        build_explorer_data.data_file_index,
        build_explorer_data._benchmark_geometry_index,
        build_explorer_data._benchmark_element_index,
    )
    for loader in loaders:
        loader.cache_clear()
    try:
        yield data_root
    finally:
        patches.undo()
        for loader in loaders:
            loader.cache_clear()


@pytest.fixture(scope="session")
def app_url(
    benchmark_data: Path, fixture_benchmarks: tuple[tuple[str, str], ...]
) -> Iterator[str]:
    """
    Build the ML-PEG app for the fixture benchmarks and serve it for the session.

    Discover only the explicitly selected benchmarks and fail if any are
    missing. Unrelated local benchmarks must not affect browser tests.

    Parameters
    ----------
    benchmark_data
        Ensures the benchmarks' data is in place before the app is built.
    fixture_benchmarks
        The ``(category, benchmark)`` pairs the fixture supplies.

    Yields
    ------
    str
        Base URL of the running app.
    """
    from dash import Dash

    from ml_peg.app import build_app
    from ml_peg.app.utils.head_scripts import inject_head_scripts

    app = Dash(
        __name__,
        assets_folder=str(benchmark_data),
        title="ML-PEG",
        update_title=None,
        suppress_callback_exceptions=True,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ],
    )
    inject_head_scripts(app)

    discover = build_app.get_all_tests

    def fixture_tests(**kwargs):
        result = ({}, {}, {}, {})
        for category, benchmark in fixture_benchmarks:
            selected = discover(category=category, test=benchmark)
            result[0].update(selected[0])
            for output, incoming in zip(result[1:], selected[1:], strict=True):
                for key, value in incoming.items():
                    output.setdefault(key, {}).update(value)
        assert set(result[0]) == {name for _, name in fixture_benchmarks}
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(build_app, "get_all_tests", fixture_tests)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_app.build_full_app(app, category="*", test="*")

    expected = {benchmark for _, benchmark in fixture_benchmarks}
    messages = [str(warning.message) for warning in caught]
    skipped = sorted(
        benchmark
        for benchmark in expected
        if any(f" {benchmark} in " in message for message in messages)
    )
    assert not skipped, (
        f"fixture benchmarks failed to build: {skipped}. The published test "
        "fixture is incomplete for them."
    )

    server = make_server("127.0.0.1", 0, app.server, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def ready_page(page, app_url):  # noqa: ANN001
    """
    Return a Playwright page with the app loaded and onboarding pre-dismissed.

    Parameters
    ----------
    page
        Playwright page fixture (function scoped).
    app_url
        Base URL of the running app.

    Returns
    -------
    Page
        Loaded, hydrated page ready for interaction.
    """
    # Mark the tutorial complete before load so its modal never overlays the page
    # (the modal is gated on the locally-persisted ``onboarding-state-store``).
    page.add_init_script(
        "try { window.localStorage.setItem('onboarding-state-store', "
        "JSON.stringify({completed: true})); } catch (e) {}"
    )
    page.goto(app_url)
    wait_for_app_ready(page)
    return page


def settle_locator(locator: Locator, timeout: int = READY_TIMEOUT) -> None:
    """
    Wait until a locator's bounding box stops moving.

    Pages keep reflowing for a moment after load while their Plotly figures autosize
    and re-theme; on a slow CI runner that shift can outlast Playwright's own
    actionability wait, so hovers/clicks time out with "element is not stable". Poll
    the box until it is unchanged across two ~120ms samples before interacting.

    Parameters
    ----------
    locator
        Element to wait on.
    timeout
        Maximum time to wait, in milliseconds.
    """
    page = locator.page
    locator.wait_for(state="visible", timeout=timeout)
    deadline = timeout
    prev = locator.bounding_box()
    while deadline > 0:
        page.wait_for_timeout(120)
        deadline -= 120
        curr = locator.bounding_box()
        if prev is not None and curr == prev:
            return
        prev = curr


@pytest.fixture
def settle() -> Callable[[Locator], None]:
    """
    Provide :func:`settle_locator` to tests.

    Returns
    -------
    Callable[[Locator], None]
        Waits until the given locator stops moving.
    """
    return settle_locator
