"""APGI core module - initialize with font warning suppression."""

import logging
import os
import warnings

# Suppress matplotlib font warnings globally
warnings.filterwarnings("ignore", message=".*findfont.*")
warnings.filterwarnings("ignore", message=".*Falling back.*")
warnings.filterwarnings("ignore", message=".*Font family.*not found.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.*")

# Suppress LAPACK warnings
warnings.filterwarnings("ignore", message=".*DLASCL.*")
warnings.filterwarnings("ignore", message=".*LAPACK.*")

# Set matplotlib backend before any plotting
os.environ["MPLBACKEND"] = "Agg"

# Suppress matplotlib logger
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
