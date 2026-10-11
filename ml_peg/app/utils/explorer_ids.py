"""
Component ids, option lists and shared style constants for the Explorer page.

Every Explorer layer (data, scoring, figures, layout, callbacks) imports from here and
this module imports from none of them -- and has no Dash import -- so it is the one
place the page's component ids and shared vocabulary are defined.
"""

from __future__ import annotations

__all__ = [
    "COVERAGE_GRANULARITY",
    "COVERAGE_GRANULARITY_OPTIONS",
    "COVERAGE_GRAPH",
    "EXPLORER_PATH",
    "FAILS_PICK_TYPE",
    "FINDER_CATEGORIES",
    "FINDER_ELEMENTS",
    "FINDER_EXPERIMENTS",
    "FINDER_EXPERIMENT_CATEGORIES",
    "FINDER_GRAPH",
    "FINDER_PICKS",
    "FINDER_PICK_TYPE",
    "FINDER_PRESET",
    "FINDER_TOLERANCE",
    "FINDER_TOLERANCE_OPTIONS",
    "H2H_COLOUR_A",
    "H2H_COLOUR_B",
    "H2H_COLOUR_TIE",
    "H2H_DETAIL_CATEGORY",
    "H2H_DETAIL_GRAPH",
    "H2H_GRAPH",
    "H2H_MODEL_A",
    "H2H_MODEL_B",
    "H2H_SUMMARY",
    "LOT_CLASSES",
    "LOT_CLASS_STYLE",
    "REPORT_MODEL",
    "REPORT_PROFILE",
    "REPORT_SUMMARY",
    "REPORT_TABLE",
    "SIMILARITY_GRAPH",
    "STRUCT_FAILS_CATEGORIES",
    "STRUCT_FAILS_GRAPH",
    "STRUCT_FAILS_LIST",
    "STRUCT_FAILS_MODEL",
    "STRUCT_FAILS_SORT",
    "STRUCT_FAILS_SORT_OPTIONS",
    "STRUCT_TABLE",
    "STRUCT_VIEWER",
]

# URL path for the Explorer page.
EXPLORER_PATH = "/explorer"

# Component ids: the layout renders them, register_explorer_callbacks wires them.
REPORT_MODEL = "explorer-report-model"
REPORT_SUMMARY = "explorer-report-summary"
REPORT_PROFILE = "explorer-report-profile"
REPORT_TABLE = "explorer-report-table"
STRUCT_VIEWER = "explorer-struct-viewer"
STRUCT_TABLE = "explorer-struct-table"
H2H_MODEL_A = "explorer-h2h-model-a"
H2H_MODEL_B = "explorer-h2h-model-b"
H2H_GRAPH = "explorer-h2h-graph"
H2H_DETAIL_CATEGORY = "explorer-h2h-detail-category"
H2H_DETAIL_GRAPH = "explorer-h2h-detail-graph"
H2H_SUMMARY = "explorer-h2h-summary"
COVERAGE_GRAPH = "explorer-coverage-graph"
COVERAGE_GRANULARITY = "explorer-coverage-granularity"
FINDER_ELEMENTS = "explorer-finder-elements"
FINDER_CATEGORIES = "explorer-finder-categories"
FINDER_TOLERANCE = "explorer-finder-tolerance"
FINDER_PRESET = "explorer-finder-preset"
FINDER_GRAPH = "explorer-finder-graph"
FINDER_PICKS = "explorer-finder-picks"
STRUCT_FAILS_CATEGORIES = "explorer-fails-categories"
STRUCT_FAILS_MODEL = "explorer-fails-model"
STRUCT_FAILS_SORT = "explorer-fails-sort"
STRUCT_FAILS_GRAPH = "explorer-fails-graph"
STRUCT_FAILS_LIST = "explorer-fails-list"
SIMILARITY_GRAPH = "explorer-similarity-graph"

# Pattern-matched id ``type`` values: the finder's top-pick buttons carry their model
# in ``{"type": FINDER_PICK_TYPE, "model": ...}``, and the keyboard-operable structure
# list under the failure bars carries a structure key in
# ``{"type": FAILS_PICK_TYPE, "key": ...}``.
FINDER_PICK_TYPE = "explorer-finder-pick"
FAILS_PICK_TYPE = "explorer-fails-pick"

# Granularity for the coverage rose: one wedge per category, or one per benchmark.
COVERAGE_GRANULARITY_OPTIONS = [
    {"label": "By category", "value": "category"},
    {"label": "By benchmark", "value": "benchmark"},
]

# How the structure-fails explorer ranks the structures models struggle on. When a
# specific model is picked the ranking switches to "where this model fails worst"
# (its error rank vs peers), so this toggle only drives the all-models view.
STRUCT_FAILS_SORT_OPTIONS = [
    {"label": "Hardest (highest error)", "value": "hardest"},
    {"label": "Most contested (models disagree)", "value": "contested"},
]

# Level-of-theory classes for the report card: colour + short label. Blues/purples/grey
# only -- red and green are reserved app-wide for bad/good *scores*, and a red "tested
# against a different functional" bar would read as "the model failed here". Insertion
# order is the severity order (in-domain first), used to break ties.
LOT_CLASS_STYLE = {
    # Mirrors --mlpeg-accent (theme.css): Plotly cannot read CSS variables server-side,
    # so keep the two in sync.
    "in_domain": {"color": "#0d8ff0", "label": "in-domain"},
    # Mid-lightness values on purpose: these paint plot bars and legend dots in
    # BOTH themes, so each must clear ~3:1 against white and the dark surface
    # (#475569 slate was near-invisible on dark).
    "dft": {"color": "#8b5cf6", "label": "functional mismatch"},
    "high_level": {"color": "#d946ef", "label": "vs CCSD(T)/MP2"},
    "experimental": {"color": "#64748b", "label": "vs experiment"},
    "none": {"color": "#9aa0a6", "label": "soundness"},
}
# The class names alone, for code that validates a class without needing its style.
LOT_CLASSES = tuple(LOT_CLASS_STYLE)

# Head-to-head winner colours (A = blue, B = amber, tie = grey). Semantic data colours,
# so theme_plots.js leaves them alone; each clears 3:1 on both plot backgrounds.
H2H_COLOUR_A = "#4c6ef5"
H2H_COLOUR_B = "#f59f00"
H2H_COLOUR_TIE = "#b0b6bd"

# Model-finder extrapolation tolerance options (values map to admissible LoT classes
# in explorer_scoring).
FINDER_TOLERANCE_OPTIONS = [
    {"label": "In-domain evidence only", "value": "in_domain"},
    {"label": "Allow DFT-functional mismatch", "value": "dft"},
    {"label": "Use all evidence", "value": "any"},
]

# Model-finder experiment presets: researchers think in experiments, not benchmark
# category names, so each preset selects the relevant category bundle. The category
# multi-select stays editable; touching it switches the preset to "custom" (None).
# Keys of the bundles are category directory names under ml_peg/app (a test checks
# every one exists, so a renamed category cannot silently empty a preset).
FINDER_EXPERIMENTS = [
    {"label": "Anything", "value": "any"},
    {"label": "Bulk materials & crystals", "value": "bulk"},
    {"label": "Surfaces, defects & reactions", "value": "surfaces"},
    {"label": "Molecules & non-covalent chemistry", "value": "molecular"},
    {"label": "Metals & f-block chemistry", "value": "metals"},
]
FINDER_EXPERIMENT_CATEGORIES: dict[str, list[str]] = {
    "any": [],
    "bulk": [
        "bulk_crystal",
        "molecular_crystal",
        "molecular_dynamics",
        "alloy_metallurgy",
    ],
    "surfaces": ["surfaces", "defects", "nebs", "molecular_reactions"],
    "molecular": [
        "molecular",
        "conformers",
        "isomers",
        "thermochemistry",
        "non_covalent_interactions",
        "supramolecular",
    ],
    "metals": ["tm_complexes", "f_block", "alloy_metallurgy"],
}
