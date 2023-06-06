# You local configuration
# Modify the values to point to respective directories in your system

# This is where your split data is
SRC_BASE_DIR: str = "D:\\GITREPO\\_HK_GITHUB\\common-voice-diversity-check\\experiments"

# Point to your common voice clone
CV_REPO: str = "D:\\GITREPO\\_AI_VOICE\\_CV\\common-voice\\server\\data"

# Regenerate the data or skip existing?
SKIP_TEXT_CORPUS: bool = False
SKIP_REPORTED: bool = False
SKIP_SPLITS: bool = False

FORCE_CREATE_SPLIT_STATS: bool = False
