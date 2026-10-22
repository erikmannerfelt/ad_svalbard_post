from pathlib import Path
from typing import Final

PROJECT_DIR = Path(__file__).absolute().parent.parent
INPUT_DIR = PROJECT_DIR / "input"
CACHE_DIR = PROJECT_DIR / "cache"
FIGURE_DIR = PROJECT_DIR / "figures"
TABLE_DIR = PROJECT_DIR / "tables"
#: Optional local mirror of downloaded source data, consulted by `tools.fetch()`
#: before the network. Nothing is required to be present.
MIRROR_DIR: Final = PROJECT_DIR.parent / "resources"

CRS_EPSG = 32633
#: Rate (m a^-1) above which a pixel counts towards the positive area and positive volume.
#: About one median sigma_baseline, so near-zero rates are not counted.
POSITIVE_CHANGE_THRESHOLD: Final = 0.1
