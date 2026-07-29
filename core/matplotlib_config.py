"""Centralized, robust matplotlib configuration with font fallbacks.
Suppresses font warnings and configures fonts with fallback chains for
headless systems and CI environments.
"""

import logging
import os
import warnings

# Suppress warnings BEFORE importing matplotlib
warnings.filterwarnings("ignore")

# Configure matplotlib backend before any imports
os.environ["MPLBACKEND"] = "Agg"

# Now import matplotlib with warnings suppressed. The import MUST follow the
# MPLBACKEND assignment above — matplotlib reads the backend at import time, so
# hoisting this to the top of the file would select the wrong (interactive)
# backend and break headless figure generation in CI.
import matplotlib  # noqa: E402

# Set the backend explicitly
matplotlib.use("Agg")

# Disable font debugging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Suppress all warnings that might come from matplotlib or pyplot
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*")  # Catch all messages

# Configure matplotlib to be less verbose
matplotlib.rcParams["figure.max_open_warning"] = 50


def configure_matplotlib_fonts(use_tex: bool = False, font_family: str = "sans-serif") -> None:
    """Configure matplotlib with robust font fallback chain.

    Supports both TeX and non-TeX rendering with automatic fallbacks
    for systems where certain fonts are unavailable.

    Args:
        use_tex: Enable TeX rendering (slower, requires LaTeX installation)
        font_family: Primary font family ('sans-serif', 'serif', 'monospace')
    """
    # Disable TeX rendering by default (faster, more reliable on headless systems)
    matplotlib.rcParams["text.usetex"] = use_tex

    # Configure font family with fallback chain
    if font_family == "sans-serif":
        # Fallback chain: DejaVu Sans -> Liberation Sans -> Bitstream Vera Sans -> Default
        matplotlib.rcParams["font.sans-serif"] = [
            "DejaVu Sans",
            "Liberation Sans",
            "Bitstream Vera Sans",
            "Nimbus Sans L",
            "Helvetica",
            "Arial",
            "sans-serif",
        ]
    elif font_family == "serif":
        matplotlib.rcParams["font.serif"] = [
            "DejaVu Serif",
            "Liberation Serif",
            "Bitstream Vera Serif",
            "Nimbus Roman",
            "Times New Roman",
            "Times",
            "serif",
        ]
    elif font_family == "monospace":
        matplotlib.rcParams["font.monospace"] = [
            "DejaVu Sans Mono",
            "Liberation Mono",
            "Bitstream Vera Sans Mono",
            "Nimbus Mono L",
            "Courier New",
            "Courier",
            "monospace",
        ]

    # Set the default font family
    matplotlib.rcParams["font.family"] = font_family

    # Configure math text rendering
    if use_tex:
        matplotlib.rcParams["mathtext.fontset"] = "cm"
    else:
        matplotlib.rcParams["mathtext.fontset"] = "dejavusans"

    matplotlib.rcParams["mathtext.default"] = "regular"

    # Additional robustness settings
    matplotlib.rcParams["svg.fonttype"] = "none"  # Embed font names, not paths
    matplotlib.rcParams["pdf.fonttype"] = 42  # Type 42 (TrueType) fonts in PDF


def suppress_font_warnings() -> None:
    """Suppress all matplotlib and font-related warnings.
    Call this before plotting to silence warnings about missing fonts.
    """
    # Matplotlib font warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*findfont.*")
    warnings.filterwarnings("ignore", message=".*font.*not found.*")

    # Scipy/LAPACK warnings
    warnings.filterwarnings("ignore", message=".*DLASCL.*")
    warnings.filterwarnings("ignore", message=".*LAPACK.*")

    # Pandas/NumPy warnings
    warnings.filterwarnings("ignore", message=".*settingwithcopy.*", category=Warning)


# Auto-configure on module import
configure_matplotlib_fonts(use_tex=False, font_family="sans-serif")
suppress_font_warnings()
