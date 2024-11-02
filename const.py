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

# Standard Lib

# External dependencies
import pandas as pd

# Module
from typedef import (
    GitRec,
    dtype_pa_uint8,
    dtype_pa_int16,
    dtype_pa_uint32,
    dtype_pa_uint64,
    dtype_pa_float16,
    dtype_pa_float32,
    dtype_pa_float64,
    dtype_pa_str,
    dtype_pa_list_str,
    # dtype_pa_list_uint8,
    # dtype_pa_list_uint16,
    # dtype_pa_list_uint32,
    dtype_pa_list_uint64,
)


#
# Pandans 2 settings
#
pd.options.mode.copy_on_write = True


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
    "17.0",
    "18.0",
    "19.0",
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
    "2024-03-15",
    "2024-06-14",
    "2024-09-13",
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

FIELDS_BUCKETS_SPLITS: dict[str, pd.ArrowDtype] = {
    "client_id": dtype_pa_str,
    "path": dtype_pa_str,
    "sentence_id": dtype_pa_str,
    "sentence": dtype_pa_str,
    "sentence_domain": dtype_pa_str,
    "up_votes": dtype_pa_int16,
    "down_votes": dtype_pa_int16,
    "age": dtype_pa_str,
    "gender": dtype_pa_str,
    "accents": dtype_pa_str,
    "variant": dtype_pa_str,
    "locale": dtype_pa_str,
    "segment": dtype_pa_str,
}

FIELDS_REPORTED: dict[str, pd.ArrowDtype] = {
    "sentence_id": dtype_pa_str,
    "sentence": dtype_pa_str,
    "locale": dtype_pa_str,
    "reason": dtype_pa_str,
}

FIELDS_REPORTED_OLD: dict[str, pd.ArrowDtype] = {
    "sentence": dtype_pa_str,
    "sentence_id": dtype_pa_str,
    "locale": dtype_pa_str,
    "reason": dtype_pa_str,
}


CV_GENDERS: list[str] = ["male", "female", "other", NODATA]
# new gender definitions
CV_GENDERS_EXTENDED: list[str] = [
    "male_masculine",
    "female_feminine",
    "intersex",
    "transgender",
    "non-binary",
    "do_not_wish_to_say",
    NODATA,
]
# backmapping of new genders for backwards compatibility
CV_GENDER_MAPPER: dict[str, str] = {
    "male_masculine": "male",
    "female_feminine": "female",
    "intersex": "other",
    "transgender": "other",
    "non-binary": "other",
    "do_not_wish_to_say": "other",
}

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

CV_DOMAINS: list[str] = [
    "agriculture_food",
    "automotive_transport",
    "finance",
    "general",
    "healthcare",
    "history_law_government",
    "language_fundamentals",
    "media_entertainment",
    "nature_environment",
    "news_current_affairs",
    "service_retail",
    "technology_robotics",
]

CV_DOMAIN_MAPPER: dict[str, str] = {
    # v17 - v18 mapping
    "agriculture": "agriculture_food",
    "automotive": "automotive_transport",
    "food_service_retail": "service_retail",
    # v18
    "agriculture_food": "agriculture_food",
    "automotive_transport": "automotive_transport",
    "finance": "finance",
    "general": "general",
    "healthcare": "healthcare",
    "history_law_government": "history_law_government",
    "language_fundamentals": "language_fundamentals",
    "media_entertainment": "media_entertainment",
    "nature_environment": "nature_environment",
    "news_current_affairs": "news_current_affairs",
    "service_retail": "service_retail",
    "technology_robotics": "technology_robotics",
}


# CLIP DURATIONS
CLIP_DURATIONS_FILE: str = "clip_durations.tsv"
FIELDS_CLIP_DURATIONS: dict[str, pd.ArrowDtype] = {
    "clip": dtype_pa_str,
    "duration[ms]": dtype_pa_uint32,
}

#
# clip-errors from TBOX
#
FIELDS_CLIP_ERRORS: dict[str, pd.ArrowDtype] = {
    "path": dtype_pa_str,
    "source": dtype_pa_str,
    "ds": dtype_pa_str,
    "lc": dtype_pa_str,
    "ver": dtype_pa_str,
}


#
# SPLIT STATS
#
FIELDS_SPLIT_STATS: dict[str, pd.ArrowDtype] = {
    "ver": dtype_pa_str,
    "lc": dtype_pa_str,
    "alg": dtype_pa_str,
    "sp": dtype_pa_str,
    "clips": dtype_pa_uint32,
    "uq_v": dtype_pa_uint32,
    "uq_s": dtype_pa_uint32,
    "uq_sl": dtype_pa_uint32,
    # Duration
    "dur_total": dtype_pa_float64,
    "dur_avg": dtype_pa_float16,
    "dur_med": dtype_pa_float16,
    "dur_std": dtype_pa_float16,
    "dur_freq": dtype_pa_str,
    # Recordings per Voice
    "v_avg": dtype_pa_float16,
    "v_med": dtype_pa_float16,
    "v_std": dtype_pa_float16,
    "v_freq": dtype_pa_str,
    # Recordings per Sentence
    "s_avg": dtype_pa_float16,
    "s_med": dtype_pa_float16,
    "s_std": dtype_pa_float16,
    "s_freq": dtype_pa_str,
    # Votes (UpVotes, DownVotes)
    "uv_sum": dtype_pa_uint32,
    "uv_avg": dtype_pa_float16,
    "uv_med": dtype_pa_float16,
    "uv_std": dtype_pa_float16,
    "uv_freq": dtype_pa_str,
    "dv_sum": dtype_pa_uint32,
    "dv_avg": dtype_pa_float16,
    "dv_med": dtype_pa_float16,
    "dv_std": dtype_pa_float16,
    "dv_freq": dtype_pa_str,
    # Demographics distribution for recordings
    "dem_table": dtype_pa_str,
    "dem_uq": dtype_pa_str,
    "dem_fix_r": dtype_pa_str,
    "dem_fix_v": dtype_pa_str,
}


#
# TEXT-CORPUS RELATED
#
TC_BUCKETS: list[str] = ["validated_sentences", "unvalidated_sentences"]
TC_BUCKET_FILES: list[str] = ["validated_sentences.tsv", "unvalidated_sentences.tsv"]
TC_VALIDATED_FILE: str = TC_BUCKET_FILES[0]

FIELDS_TC_UNVALIDATED: dict[str, pd.ArrowDtype] = {
    "sentence_id": dtype_pa_str,
    "sentence": dtype_pa_str,
    "sentence_domain": dtype_pa_str,
    "source": dtype_pa_str,
}

FIELDS_TC_VALIDATED: dict[str, pd.ArrowDtype] = {
    "sentence_id": dtype_pa_str,
    "sentence": dtype_pa_str,
    "sentence_domain": dtype_pa_str,
    "source": dtype_pa_str,
    "is_used": dtype_pa_uint8,
    "clips_count": dtype_pa_int16,
}

FIELDS_TEXT_CORPUS: dict[str, pd.ArrowDtype] = {
    "sentence_id": dtype_pa_str,
    "sentence": dtype_pa_str,
    "sentence_domain": dtype_pa_str,
    "source": dtype_pa_str,
    "is_used": dtype_pa_uint8,
    "clips_count": dtype_pa_uint8,
    "normalized": dtype_pa_str,
    "phonemised": dtype_pa_str,
    "tokens": dtype_pa_list_str,
    "char_cnt": dtype_pa_int16,
    "word_cnt": dtype_pa_uint8,
    "valid": dtype_pa_uint8,
}

FIELDS_TC_STATS: dict[str, pd.ArrowDtype] = {
    "ver": dtype_pa_str,
    "lc": dtype_pa_str,
    "algo": dtype_pa_str,
    "sp": dtype_pa_str,
    "has_val": dtype_pa_uint8,
    "has_phon": dtype_pa_uint8,
    "s_cnt": dtype_pa_uint32,
    "uq_s": dtype_pa_uint32,
    "uq_n": dtype_pa_uint32,
    "val": dtype_pa_uint8,
    "c_sum": dtype_pa_uint32,
    "c_avg": dtype_pa_float32,
    "c_med": dtype_pa_float32,
    "c_std": dtype_pa_float32,
    "c_freq": dtype_pa_list_uint64,
    "w_sum": dtype_pa_uint32,
    "w_avg": dtype_pa_float32,
    "w_med": dtype_pa_float32,
    "w_std": dtype_pa_float32,
    "w_freq": dtype_pa_list_uint64,
    "t_sum": dtype_pa_uint32,
    "t_avg": dtype_pa_float32,
    "t_med": dtype_pa_float32,
    "t_std": dtype_pa_float32,
    "t_freq": dtype_pa_list_uint64,
    "g_cnt": dtype_pa_int16,
    "g_items": dtype_pa_str,  # dtype_pa_list_str,
    "g_freq": dtype_pa_list_uint64,
    "p_cnt": dtype_pa_int16,
    "p_items": dtype_pa_str,  # dtype_pa_list_str,
    "p_freq": dtype_pa_list_uint64,
    "dom_cnt": dtype_pa_int16,
    "dom_items": dtype_pa_list_str,
    "dom_freq": dtype_pa_list_uint64,
}

FIELDS_TOKENS: dict[str, pd.ArrowDtype] = {
    "token": dtype_pa_str,
    "count": dtype_pa_uint64,
}
FIELDS_GRAPHEMES: dict[str, pd.ArrowDtype] = {
    "grapheme": dtype_pa_str,
    "count": dtype_pa_uint64,
}
FIELDS_PHONEMES: dict[str, pd.ArrowDtype] = {
    "phoneme": dtype_pa_str,
    "count": dtype_pa_uint64,
}
FIELDS_SENTENCE_DOMAINS: dict[str, pd.ArrowDtype] = {
    "sentence_domain": dtype_pa_str,
    "count": dtype_pa_uint64,
}

#
# REPORTED SENTENCES
#
FIELDS_REPORTED_STATS: dict[str, pd.ArrowDtype] = {
    "ver": dtype_pa_str,
    "lc": dtype_pa_str,
    "rep_sum": dtype_pa_uint32,
    "rep_sen": dtype_pa_uint32,
    "rep_avg": dtype_pa_float32,
    "rep_med": dtype_pa_float32,
    "rep_std": dtype_pa_float32,
    "rep_freq": dtype_pa_list_uint64,
    "rea_freq": dtype_pa_list_uint64,
}

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
# AUDIO ANALYSIS - INCOMING FROM TBOX MONOREPO
#

# Define only what we use here
FIELDS_AUDIO_SPECS: dict[str, pd.ArrowDtype] = {
    "clip_id": dtype_pa_uint64,
    "orig_path": dtype_pa_str,
    "orig_encoding": dtype_pa_str,
    "orig_sample_rate": dtype_pa_float32,
    # "orig_num_frames": dtype_pa_uint64,
    "orig_num_channels": dtype_pa_uint8,
    "orig_bitrate_kbps": dtype_pa_uint8,
    # "orig_bits_per_sample": dtype_pa_uint8,
    # "tc_path": dtype_pa_str,
    # "tc_encoding": dtype_pa_str,
    # "tc_sample_rate": dtype_pa_float32,
    # "tc_num_frames": dtype_pa_uint64,
    # "tc_num_channels": dtype_pa_uint8,
    # "tc_bitrate_kbps": dtype_pa_uint8,
    # "tc_bits_per_sample": dtype_pa_uint8,
    "duration": dtype_pa_uint32,
    "speech_duration": dtype_pa_uint32,
    "speech_power": dtype_pa_float32,
    "silence_power": dtype_pa_float32,
    "est_snr": dtype_pa_float32,
    "ver": dtype_pa_float32,  # dtype_pa_str,
    "ds": dtype_pa_str,
    "lc": dtype_pa_str,
}

#
# cv-tbox related
#

ALGORITHMS: list[str] = ["s1", "s5", "s99", "v1", "vw", "vx"]

# SEPARATORS
SEP_ROW: str = "|"
SEP_COL: str = "#"
SEP_ALGO: str = "|"

#
# DIRECTORIES / FILENAMES
#
RES_DIRNAME: str = "results"

TC_DIRNAME: str = "text-corpus"
TC_ANALYSIS_DIRNAME: str = "text-analysis"
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
DOMAINS_FN: str = "$domains"

TEXT_CORPUS_STATS_FN: str = "tc_stats"
REPORTED_STATS_FN: str = "$reported"
SUPPORT_MATRIX_FN: str = "$support_matrix"
CONFIG_FN: str = "$config"

AUDIO_SPECS_FN: str = "audio_specs"

API_DIRNAME: str = "api"

#
# SAVE LEVELS
#

SAVE_LEVEL_NONE = 0  # do not save detailed text corpora analysis under data/text-corpus
SAVE_LEVEL_DEFAULT = 1  # save only the following: text-corpus & validated/train/dev/test analysis results for s1 algorithm
SAVE_LEVEL_DETAILED = 2  # save every calculated result for all algorithms

#
# BINS
#

BINS_DURATION: list[int] = [
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
    20,
    30,
    999999,
]

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
    131072,
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
    21,
    22,
    23,
    24,
    25,
    30,
    40,
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
    131072,
    262144,
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

# Sentence Length (measure by Python "len() function")
CHARS_BIN_THRESHOLD: int = 30
# For logogram languages (avg. sentence length < CHARS_BIN_THRESHOLD)
BINS_CHARS_SHORT: list[int] = [
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
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    60,
    70,
    80,
    999999,
]
# Regular languages
BINS_CHARS_LONG: list[int] = [
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
    160,
    170,
    180,
    190,
    200,
    250,
    999999,
]

# Character Speed
# Average CS to decide between bin types
CS_BIN_THRESHOLD: int = 300
# This one is for latin
BINS_CS_LOW: list[int] = [
    0,
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
    175,
    200,
    250,
    300,
    350,
    400,
    999999,
]

# This one usually is for logographic languages (one char = a word)
BINS_CS_HIGH: list[int] = [
    0,
    100,
    200,
    250,
    300,
    325,
    350,
    375,
    400,
    425,
    450,
    475,
    500,
    525,
    550,
    575,
    600,
    650,
    999999,
]

#
# AUDIO ANALYSIS RELATED BINS
#

# power values are 10^6
BINS_POWER: list[int] = [
    0,
    1,
    3,
    10,
    30,
    100,
    300,
    1_000,
    3_000,
    10_000,
    30_000,
    100_000,
    999999,
]

BINS_SNR: list[int] = [
    -9999999,
    -100,
    -30,
    -20,
    -10,
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
