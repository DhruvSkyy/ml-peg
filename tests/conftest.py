"""Fixtures shared by the app (browser) tests and the browser-free Explorer tests.

The tests run against a small, fixed subset of the published benchmark data,
stored in the ml-peg-data bucket as ``tests/ui-fixture-<version>.zip``. The
``fixture_data`` fixture below fetches, verifies and unpacks it; each suite's own
conftest then redirects its data paths at the unpacked tree, so neither suite
reads or writes the real data directory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

from botocore.exceptions import BotoCoreError, ClientError
import pytest

from ml_peg.calcs.utils.utils import BENCHMARK_DATA_DIR
from ml_peg.data.data import download

# Stand-in for the published data tarball (~800 KB against ~9 GB): a handful of
# benchmarks, published once to the ml-peg-data bucket and never overwritten. A
# changed fixture is a new -v2 object plus a bump here. The bucket has no
# versioning, so the digest is what actually enforces that: a swapped object
# fails loudly instead of quietly changing what the suite tests. (The object as
# uploaded also carries macOS "__MACOSX/" resource-fork entries, which
# extraction skips.)
FIXTURE_VERSION = "v1"
FIXTURE_KEY = f"tests/ui-fixture-{FIXTURE_VERSION}.zip"
FIXTURE_SHA256 = "b0fab46961813f329702b6d464171efc02edf73503c7c4c5069db7f91935564f"
FIXTURE_ARCHIVE = BENCHMARK_DATA_DIR / f"ui-fixture-{FIXTURE_VERSION}.zip"

# Benchmarks the fixture supplies, chosen for coverage per KB rather than size:
#   IONPI19          one metric, a parity plot and a structure viewer; the
#                    benchmark every interaction test targets by id.
#   extensivity      physicality category (weight 1, so the overall score is a
#                    real weighted mean) and the only mace-multihead benchmark
#                    here, so /framework/mace-multihead exists.
#   oxidation_states three metrics with experimental levels of theory, and the
#                    per-cell plot dispatch rather than IONPI19's per-column one.
#   iron_properties  twelve metric columns and dropdown-driven figures, i.e. the
#                    widest weight/threshold grid and a custom callback shape.
#   ACONFL           the Explorer's model finder drops any model with fewer than
#   UpU46            MIN_FINDER_BENCH (3) admissible benchmarks, and physicality
#                    is scored separately rather than counting towards that, so
#                    IONPI19 and iron_properties alone leave every model one
#                    short and the finder ranks nothing at all. These two carry
#                    the count to three, and all three are categories the
#                    finder's "molecular" preset selects, so that preset ranks
#                    too. Both are conformers, the cheapest benchmarks that do
#                    this: one model's geometries each, which the element
#                    resolver needs. Their info.json pairs per-structure
#                    elements with filenames (the resolver's fast path), while
#                    IONPI19's has no filenames (so it exercises the .xyz
#                    fallback).
# Together they span four categories, which is what puts more than one slice in
# the overall summary table and the Explorer. Nothing is added to
# non_covalent_interactions on purpose: only the first benchmark card on a
# category page opens by default, and the interaction tests drive IONPI19.
FIXTURE_BENCHMARKS = (
    ("non_covalent_interactions", "IONPI19"),
    ("physicality", "extensivity"),
    ("physicality", "oxidation_states"),
    ("bulk_crystal", "iron_properties"),
    ("conformers", "ACONFL"),
    ("conformers", "UpU46"),
)


@pytest.fixture(scope="session")
def fixture_benchmarks() -> tuple[tuple[str, str], ...]:
    """
    Return the ``(category, benchmark)`` pairs the published fixture supplies.

    A fixture rather than an importable constant: pytest propagates fixtures from
    this root conftest into every suite, whereas reaching into another suite's
    conftest for a constant couples the suites to each other.

    Returns
    -------
    tuple[tuple[str, str], ...]
        The ``(category, benchmark)`` pairs the fixture supplies.
    """
    return FIXTURE_BENCHMARKS


def _sha256(path: Path) -> str:
    """
    Return the SHA-256 hex digest of a file.

    Parameters
    ----------
    path
        File to hash.

    Returns
    -------
    str
        Hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="session")
def fixture_data(tmp_path_factory) -> Path:
    """
    Fetch, verify and unpack the published test fixture, returning its root.

    ``download_s3_data`` is deliberately not used: it extracts before returning,
    so there is no point at which the archive can be checked before it is
    trusted, and it unpacks every member rather than just the fixture directory.
    The cached archive is re-hashed on every session, so a truncated or stale
    copy fails the same way a swapped object does. Behaviour is identical locally
    and in CI: no network means a failed session, not a silently skipped one.

    Returns
    -------
    Path
        Directory holding the fixture's ``<category>/<benchmark>`` tree.
    """
    if not FIXTURE_ARCHIVE.exists():
        try:
            download(key=FIXTURE_KEY, filename=str(FIXTURE_ARCHIVE))
        except (BotoCoreError, ClientError) as err:
            raise RuntimeError(
                f"Could not fetch the test fixture {FIXTURE_KEY} from the "
                "ml-peg-data bucket. Check the connection, or fetch it by hand: "
                f"ml_peg download --key {FIXTURE_KEY} --filename {FIXTURE_ARCHIVE}"
            ) from err

    digest = _sha256(FIXTURE_ARCHIVE)
    if digest != FIXTURE_SHA256:
        FIXTURE_ARCHIVE.unlink()
        raise RuntimeError(
            f"Test fixture {FIXTURE_ARCHIVE} has SHA-256 {digest}, expected "
            f"{FIXTURE_SHA256}. The archive has been deleted; rerun to fetch it "
            "again. If the published object itself changed, it should have been "
            "a new version rather than an overwrite."
        )

    extract_root = tmp_path_factory.mktemp("ui-fixture")
    prefix = f"ui-fixture-{FIXTURE_VERSION}/"
    try:
        with zipfile.ZipFile(FIXTURE_ARCHIVE) as archive:
            members = [name for name in archive.namelist() if name.startswith(prefix)]
            archive.extractall(extract_root, members=members)
    except zipfile.BadZipFile as err:
        raise RuntimeError(f"Test fixture {FIXTURE_ARCHIVE} is not a zip file") from err
    return extract_root / prefix
