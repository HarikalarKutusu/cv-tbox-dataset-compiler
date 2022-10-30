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

# These values are as of v11.0 and should be updated with each version
CV_VERSIONS: "list[str]" = [
    '1', '2', '3', '4', '5.1',
    '6.1', '7.0', '8.0', '9.0', '10.0',
    '11.0'
]
CV_DATES: "list[str]" = [
    '2019-02-25', '2019-06-11', '2019-06-24', '2019-12-10', '2020-06-22',
    '2020-12-11', '2021-07-21', '2022-01-19', '2022-04-27', '2022-07-04',
    '2022-09-21'
]

ALGORITHMS: "list[str]" = ["s1", "s99", "v1"]
MAIN_SPLITS: "list[str]" = ["validated", "invalidated", "other"]
TRAINING_SPLITS: "list[str]" = ["train", "dev", "test"]

ALL_LOCALES: "list[str]" = [
    'ab', 'ar', 'as', 'ast', 'az',
    'ba', 'bas', 'be', 'bg', 'bn', 'br',
    'ca', 'ckb', 'cnh', 'cs', 'cv', 'cy',
    'da', 'de', 'dv',
    'el', 'en', 'eo', 'es', 'et', 'eu',
    'fa', 'fi', 'fr', 'fy-NL',
    'ga-IE', 'gl', 'gn',
    'ha', 'hi', 'hsb', 'hu', 'hy-AM',
    'ia', 'id', 'ig', 'it',
    'ja',
    'ka', 'kab', 'kk', 'kmr', 'ky',
    'lg', 'lt', 'lv',
    'mdf', 'mhr', 'mk', 'ml', 'mn', 'mr', 'mrj', 'mt', 'myv',
    'nan-tw', 'ne-NP', 'nl', 'nn-NO',
    'or',
    'pa-IN', 'pl', 'pt',
    'rm-sursilv', 'rm-vallader', 'ro', 'ru', 'rw',
    'sah', 'sat', 'sc', 'sk', 'skr', 'sl', 'sr', 'sv-SE', 'sw',
    'ta', 'th', 'ti', 'tig', 'tok', 'tr', 'tt', 'tw',
    'ug', 'uk', 'ur', 'uz',
    'vi', 'vot',
    'yue',
    'zh-CN', 'zh-HK', 'zh-TW',
]

NODATA: str = 'nodata'    # .isna cases replaced with this
CV_GENDERS: "list[str]" = ['male', 'female', 'other', NODATA]
CV_AGES: "list[str]" = [
    'teens', 'twenties', 'thirties', 'fourties', 'fifties',
    'sixties', 'seventies', 'eighties', 'nineties', NODATA
]

BINS_DURATION: "list[int]" = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 999999]
BINS_VOICES: "list[int]" = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536, 999999
]
BINS_SENTENCES: "list[int]" = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 999999]

# SEPARATORS
SEP_ROW: str = '|'
SEP_COL: str = '#'
SEP_ALGO: str = '|'

# COLUMNS FOR DATAFRAMES
COLS_CLIP_DURATIONS: "list[str]" = [
    'clip',
    'duration',
]

COLS_TEXT_CORPUS: "list[str]" = [
    'file',
    'sentence',
    'chars',
]

# COLS_SPLIT_STATS: "list[str]" = [
#     'file',
#     'sentence',
#     'chars',
# ]

COLS_TEXT_CORPUS: "list[str]" = [
    "file", "sentence", "lower", "normalized", "chars", "words", 'valid'
]

COLS_TOKENS: "list[str]" = [
    "token", "count"
]

# COL_TC_STATS: "list[str]" = [
#     "lc", "s_cnt", "uq_s", "uq_n", "has_val", "val",
#     "c_total", "c_mean", "c_median", "c_freq",
#     "w_total", "w_mean", "w_median", "w_freq",
#     "t_total", "t_mean", "t_median", "t_freq"
# ]

BINS_CHARS: "list[int]" = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 999999]
BINS_WORDS: "list[int]" = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 999999]
BINS_TOKENS: "list[int]" = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536, 999999
]
