"""Browser tests for the collapsible benchmark cards on category pages.

These pin the native ``<details>`` card behaviour: a card opens by default, its
scoring controls (weights & thresholds) are shown by default, the card collapses
on click without a server round-trip, and the summary table sits in an accent
card. The multi-card "first N open" default is covered by a unit assertion on
``CARDS_OPEN`` plus the layout builder.
"""

from __future__ import annotations

from playwright.sync_api import Page, expect

TIMEOUT = 60_000


def _goto_category(page: Page) -> None:
    """Open the test category page and wait for its benchmark card."""
    page.locator('#sidebar-nav a[href^="/category/"]').first.click()
    expect(page.locator(".mlpeg-bench-card").first).to_be_visible(timeout=TIMEOUT)


def test_benchmark_card_is_open_details(ready_page: Page) -> None:
    """The first benchmark renders as a native <details> that starts open."""
    _goto_category(ready_page)
    card = ready_page.locator(".mlpeg-bench-card").first
    assert card.evaluate("el => el.tagName.toLowerCase()") == "details"
    assert card.evaluate("el => el.open") is True, "first card should start expanded"
    expect(ready_page.locator("#IONPI19-table")).to_be_visible(timeout=TIMEOUT)


def test_scoring_controls_open_by_default(ready_page: Page) -> None:
    """Weights & thresholds are visible by default (review feedback item 4a)."""
    _goto_category(ready_page)
    controls = ready_page.locator(".mlpeg-controls-details").first
    assert controls.evaluate("el => el.open") is True, (
        "the weights/thresholds controls should be open by default"
    )


def test_card_collapses_on_click(ready_page: Page) -> None:
    """Clicking the card header collapses it natively (no server callback)."""
    _goto_category(ready_page)
    card = ready_page.locator(".mlpeg-bench-card").first
    assert card.evaluate("el => el.open") is True
    card.locator(".mlpeg-bench-header").first.click()
    # Native <details> toggles synchronously; the table is display:none once shut.
    expect(ready_page.locator("#IONPI19-table")).to_be_hidden(timeout=TIMEOUT)
    assert card.evaluate("el => el.open") is False


def test_summary_table_in_accent_card(ready_page: Page) -> None:
    """The category summary table sits in an accent scroll-card."""
    _goto_category(ready_page)
    expect(ready_page.locator(".mlpeg-summary-card").first).to_be_visible(
        timeout=TIMEOUT
    )
