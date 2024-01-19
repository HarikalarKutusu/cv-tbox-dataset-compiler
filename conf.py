"""Config file for cv-tbox Dataset Compiler"""

import os

# Your local configuration
# Modify the values to point to respective directories in your system

# This is where your split data is
SRC_BASE_DIR: str = os.path.join(
    "C:", os.sep, "GITREPO", "_HK_GITHUB", "common-voice-diversity-check", "experiments"
)

# This is where cache of API calls and clones are kept, common to cv-tbox repors
CV_TBOX_CACHE: str = os.path.join("C:", os.sep, "GITREPO", "_HK_GITHUB", "cv-tbox-cache")


# This is where your compressed splits will go (under "upload" and "uploaded") - so that we can upload it to Google Drive
COMPRESSED_RESULTS_BASE_DIR: str = os.path.join(
    "C:", os.sep, "GITREPO", "_HK_GITHUB", "cv-tbox-dataset-compiler", "data", "results"
)

# Regenerate the data or skip existing?
SKIP_TEXT_CORPORA: bool = False
SKIP_REPORTED: bool = False
SKIP_VOICE_CORPORA: bool = False

# Should we re-create old version even if they exist?
FORCE_CREATE_TC_STATS: bool = False
FORCE_CREATE_VC_STATS: bool = False
FORCE_CREATE_REPORTED_STATS: bool = False
# Should we re-create the compressed .tar files even if they exist?
FORCE_CREATE_COMPRESSED: bool = False

# Program parameters
VERBOSE: bool = False
FAIL_ON_NOT_FOUND: bool = True

# Debug & Limiters
DEBUG: bool = False
DEBUG_PROC_COUNT: int = 1
DEBUG_CV_VER: list[str] = ["16.1"]
DEBUG_CV_LC: list[str] = ["tr"]
