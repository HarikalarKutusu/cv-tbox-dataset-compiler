"""Constants for cv-tbox Dataset Compiler"""
###########################################################################
# const.py
#
# Constants for scripts in this repository
# The values should be parallel to JS data structures
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-tbox-dataset-compiler
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################

from typedef import GitRec

#
# cv related
#

# cv-dataset repo base URL
CV_DATASET_BASE_URL: str = (
    "https://raw.githubusercontent.com/common-voice/cv-dataset/main/datasets"
)

# These values are as of v15.0 and should be updated with each version
# Warning: We leave out v2, v3 just came out with fixes afterwards
CV_VERSIONS: list[str] = [
    "1",
    "3",
    "4",
    "5.1",
    "6.1",
    "7.0",
    "8.0",
    "9.0",
    "10.0",
    "11.0",
    "12.0",
    "13.0",
    "14.0",
    "15.0",
    "16.1",
    # "17.0",
    # "18.0",
    # "19.0",
    # "20.0",
]

CV_DATES: list[str] = [
    "2019-02-25",
    "2019-06-24",
    "2019-12-10",
    "2020-06-22",
    "2020-12-11",
    "2021-07-21",
    "2022-01-19",
    "2022-04-27",
    "2022-07-04",
    "2022-09-21",
    "2022-12-07",
    "2023-03-09",
    "2023-06-23",
    "2023-09-08",
    "2023-12-06",
    # "2024-03-00",
    # "2024-06-00",
    # "2024-09-00",
    # "2024-12-00",
]

MAIN_BUCKETS: list[str] = ["validated", "invalidated", "other"]
EXTENDED_BUCKET_FILES: list[str] = [
    "validated.tsv",
    "invalidated.tsv",
    "other.tsv",
    "reported.tsv",
]
TRAINING_SPLITS: list[str] = ["train", "dev", "test"]
SPLIT_FILES: list[str] = ["train.tsv", "dev.tsv", "test.tsv"]

NODATA: str = "nodata"  # .isna cases replaced with this
CV_GENDERS: list[str] = ["male", "female", "other", NODATA]
CV_AGES: list[str] = [
    "teens",
    "twenties",
    "thirties",
    "fourties",
    "fifties",
    "sixties",
    "seventies",
    "eighties",
    "nineties",
    NODATA,
]

# COLUMNS FOR DATAFRAMES
CLIP_DURATIONS_FILE: str = "clip_durations.tsv"
COLS_CLIP_DURATIONS: list[str] = [
    "clip",
    "duration[ms]",
]


#
# cv-tbox related
#

ALGORITHMS: list[str] = ["s1", "s99", "v1", "vw", "vx"]

# SEPARATORS
SEP_ROW: str = "|"
SEP_COL: str = "#"
SEP_ALGO: str = "|"

#
# COLUMNS FOR DATAFRAMES
#

# COLS_SPLIT_STATS: list[str] = [
#     'file',
#     'sentence',
#     'chars',
# ]


#
# COLUMNS FOR TEXT-CORPUS RELATED
#


COLS_TEXT_CORPUS: list[str] = [
    "file",
    "sentence",
]

COLS_TC_STATS: list[str] = [
    "ver",
    "lc",
    "algo",
    "sp",
    "has_val",
    "s_cnt",
    "uq_s",
    "uq_n",
    "val",
    "c_sum",
    "c_avg",
    "c_med",
    "c_std",
    "c_freq",
    "w_sum",
    "w_avg",
    "w_med",
    "w_std",
    "w_freq",
    "t_sum",
    "t_avg",
    "t_med",
    "t_std",
    "t_freq",
    "g_freq",
    "p_freq",
]

COLS_TOKENS: list[str] = ["token", "count"]
COLS_GRAPHEMES: list[str] = ["grapheme", "count"]
COLS_PHONEMES: list[str] = ["phoneme", "count"]


#
# REPORTED SENTENCES
#
COLS_REPORTED_STATS: list[str] = [
    "ver",
    "lc",
    "rep_sum",
    "rep_sen",
    "rep_avg",
    "rep_med",
    "rep_std",
    "rep_freq",
    "rea_freq",
]


REPORTING_BASE: list[str] = [
    "offensive-language",
    "grammar-or-spelling",
    "different-language",
    "difficult-pronounce",
    # "other",                  # everything else will be other
]
REPORTING_ALL: list[str] = REPORTING_BASE.copy()
REPORTING_ALL.append("other")


#
# DIRECTORIES / FILENAMES
#
DATA_DIRNAME: str = "data"
RES_DIRNAME: str = "results"

TC_DIRNAME: str = "text-corpus"
VC_DIRNAME: str = "voice-corpus"
CD_DIRNAME: str = "clip-durations"
UPLOAD_DIRNAME: str = "upload"
UPLOADED_DIRNAME: str = "uploaded"
TSV_DIRNAME: str = "tsv"
JSON_DIRNAME: str = "json"

TEXT_CORPUS_FN: str = "$text_corpus"
TOKENS_FN: str = "$tokens"
GRAPHEMES_FN: str = "$graphemes"
PHONEMES_FN: str = "$phonemes"

TEXT_CORPUS_STATS_FN: str = "$text_corpus_stats"
REPORTED_STATS_FN: str = "$reported"
SUPPORT_MATRIX_FN: str = "$support_matrix"
CONFIG_FN: str = "$config"

CLONES_DIRNAME: str = "clones"
API_DIRNAME: str = "api"

#
# BINS
#

BINS_DURATION: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 999999]
BINS_VOICES: list[int] = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    999999,
]
BINS_SENTENCES: list[int] = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    20,
    30,
    40,
    50,
    100,
    999999,
]
BINS_CHARS: list[int] = [
    0,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    130,
    140,
    150,
    999999,
]
BINS_WORDS: list[int] = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    999999,
]
BINS_TOKENS: list[int] = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    999999,
]

BINS_VOTES_UP: list[int] = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    50,
    100,
    500,
    1000,
    999999,
]
BINS_VOTES_DOWN: list[int] = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    50,
    100,
    500,
    1000,
    999999,
]

BINS_REPORTED: list[int] = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    50,
    100,
    500,
    1000,
    999999,
]


#
# CLONING
#

GITHUB_BASE: str = "https://github.com/"
CV_GITREC: GitRec = GitRec(user="common-voice", repo="common-voice", branch="main")
CV_DATASET_GITREC: GitRec = GitRec(
    user="common-voice", repo="cv-dataset", branch="main"
)
CLONES: list[GitRec] = [CV_GITREC, CV_DATASET_GITREC]
