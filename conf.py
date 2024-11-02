"""Config file for cv-tbox Dataset Compiler"""

import os
import const as c

# Your local configuration
# Modify the values to point to respective directories in your system

# This is where your split data is
# T:\GITREPO_DATA\cv-tbox-split-maker\experiments
SRC_BASE_DIR: str = os.path.join(
    "T:",
    os.sep,
    "GITREPO_DATA",
    "cv-tbox-split-maker",
    "experiments",
)

# This is where your intermediate and final data will be saved
# T:\GITREPO_DATA\cv-tbox-dataset-compiler\data
DATA_BASE_DIR: str = os.path.join(
    "T:",
    os.sep,
    "GITREPO_DATA",
    "cv-tbox-dataset-compiler",
    "data",
)

# This is where extra metadata files exist in tbox monorepo
TBOX_META_DIR: str = os.path.join("T:", os.sep, "TBOX", "ds_meta")

# This is where cache of API calls and clones are kept, common to cv-tbox repos
TBOX_CLONES_DIR: str = os.path.join("T:", os.sep, "TBOX", ".clones")

# This is where cache of API calls and clones are kept, common to cv-tbox repos
TBOX_TSV_CACHE_DIR: str = os.path.join("T:", os.sep, "TBOX", "cache", "tsv")

# This is where your compressed splits will go (under "upload" and "uploaded") - so that we can upload it to Google Drive
COMPRESSED_RESULTS_BASE_DIR: str = os.path.join("T:", os.sep, "TBOX", "ds_split_share")

# AUDIO ANALYSIS THRESHOLDS
LOW_SNR_THRESHOLD: float = 0.0
LOW_POWER_THRESHOLD: float = 1e-06

#
# The following settings help development of new modules
# If not skipped - might take hours to re-calculate
#

# Regenerate the data or skip existing?
SKIP_TEXT_CORPORA: bool = False
SKIP_REPORTED: bool = False
SKIP_VOICE_CORPORA: bool = False
SKIP_SUPPORT_MATRIX: bool = False

# Should we re-create old version even if they exist?
FORCE_CREATE_TC_STATS: bool = False
FORCE_CREATE_REPORTED_STATS: bool = False
FORCE_CREATE_VC_STATS: bool = False
# Should we re-create the compressed .tar files even if they exist?
FORCE_CREATE_COMPRESSED: bool = False

# Program parameters
VERBOSE: bool = False
FAIL_ON_NOT_FOUND: bool = True
SAVE_LEVEL: int = c.SAVE_LEVEL_DEFAULT

# Debug & Limiters
DEBUG: bool = False
DEBUG_PROC_COUNT: int = 1
DEBUG_CV_VER: list[str] = ["19.0"]
DEBUG_CV_LC: list[str] = ["tr"]

# This is independent of debug value
# Create "not_found" files for text corpora?
CREATE_TS_NOT_FOUND: bool = False
# Create corrected & problems files
CREATE_REPORTED_PROBLEMS: bool = False

# Multi-processing limiters
PROCS_HARD_MAX: int = 60
HARD_MAX_TASK_PER_CHILD: int = 100
CHUNKS_HARD_MIN: int = 1
CHUNKS_HARD_MAX: int = 10
