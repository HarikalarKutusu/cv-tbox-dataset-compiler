"""Type Definitions for cv-tbox Dataset Compiler"""
###########################################################################
# typedef.py
#
# Type Definitions for scripts in this repository
# These should be parallel to JS data structures
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-tbox-dataset-compiler
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################

from dataclasses import dataclass


#
# Text Corpus
#


@dataclass
class TextCorpusRec:
    """Record definition for combined text-corpora"""

    file: str = ""  # filename in cv repo
    sentence: str = ""  # original sentence
    normalized: str = ""  # normalized sentence
    chars: int = 0  # number of characters
    words: int = 0  # number of words
    valid: int = 1  # is it a valid sentence according to coommonvoice-utils? 1=valid


@dataclass
class TextCorpusStatsRec:  # pylint: disable=too-many-instance-attributes
    """Record definition for text-corpus statistics"""

    lc: str = ""  # cv language code
    s_cnt: int = 0  # raw sentence count
    uq_s: int = 0  # unique sentence count
    uq_n: int = 0  # unique nomilized sentence count
    has_val: int = 0  # 1 if commonvoice-utils has validator for it
    val: int = 0
    # character statistics
    c_sum: int = 0  # total count
    c_avg: float = 0.0  # average (mean)
    c_med: float = 0.0  # median
    c_std: float = 0.0  # standard deviation
    c_freq: str = ""  # string encoded frequency distribution
    # word statistics
    w_sum: int = 0  # total count
    w_avg: float = 0.0  # average (mean)
    w_med: float = 0.0  # median
    w_std: float = 0.0  # standard deviation
    w_freq: str = ""  # string encoded frequency distribution
    # token statistics
    t_sum: int = 0  # total counts
    t_avg: float = 0.0  # average (mean)
    t_med: float = 0.0  # median
    t_std: float = 0.0  # standard deviation
    t_freq: str = ""  # string encoded frequency distribution


#
# Reported Sentences
#


@dataclass
class ReportedStatsRec:  # pylint: disable=too-many-instance-attributes
    """Record definition for reported sentences statistics"""

    ver: str = ""  # cv version code (internal format nn.n, see const.py)
    lc: str = ""  # cv language code
    rep_sum: int = 0  # total reports
    rep_sen: int = 0  # total sentences reported
    rep_avg: float = 0.0  # average (mean)
    rep_med: float = 0.0  # median
    rep_std: float = 0.0  # standard deviation
    rep_freq: str = ""  # string encoded frequency distribution for report per sentence
    rea_freq: str = ""  # string encoded frequency distribution for reporting reasons


#
# Dataset Split Statistics
#


@dataclass
class SplitStatsRec:  # pylint: disable=too-many-instance-attributes
    """Record definition for dataset split statistics"""

    ver: str = ""  # cv version code (internal format nn.n, see const.py)
    lc: str = ""  # cv language code
    alg: str = ""  # cv-tbox splitting algorithm (see const.py)
    sp: str = ""  # split name (blank, train, dev, test)
    clips: int = 0  # number of recordings
    uq_v: int = 0  # number of unique voices
    uq_s: int = 0  # number of unique sentences
    uq_sl: int = 0  # number of unique sentences (lower case)
    # Duration
    dur_total: float = 0.0  # total
    dur_avg: float = 0.0  # average (mean)
    dur_med: float = 0.0  # median
    dur_std: float = 0.0  # standard deviation
    dur_freq: str = ""  # string encoded frequency distribution
    # Recordings per Voice
    v_avg: float = 0.0  # average (mean)
    v_med: float = 0.0  # median
    v_std: float = 0.0  # standard deviation
    v_freq: str = ""  # string encoded frequency distribution
    # Recordings per Sentence
    s_avg: float = 0.0  # average (mean)
    s_med: float = 0.0  # median
    s_std: float = 0.0  # standard deviation
    s_freq: str = ""  # string encoded frequency distribution
    # Votes (UpVotes, DownVotes)
    uv_sum: int = 0  # total
    uv_avg: float = 0.0  # average (mean)
    uv_med: float = 0.0  # median
    uv_std: float = 0.0  # standard deviation
    uv_freq: str = ""  # string encoded frequency distribution
    dv_sum: int = 0  # total
    dv_avg: float = 0.0  # average (mean)
    dv_med: float = 0.0  # median
    dv_std: float = 0.0  # standard deviation
    dv_freq: str = ""  # string encoded frequency distribution
    # Demographics distribution for recordings
    dem_table: str = ""
    dem_uq: str = ""
    dem_fix_r: str = ""
    dem_fix_v: str = ""


#
# Config Record
#


@dataclass
class ConfigRec:  # pylint: disable=too-many-instance-attributes
    """Record definition for config"""

    date: str = ""
    cv_versions: list[str] = []
    cv_dates: list[str] = []
    cv_locales: list[str] = []
    algorithms: list[str] = []
    bins_duration: list[int] = []
    bins_voices: list[int] = []
    bins_votes_up: list[int] = []
    bins_votes_down: list[int] = []
    bins_sentences: list[int] = []
    bins_chars: list[int] = []
    bins_words: list[int] = []
    bins_tokens: list[int] = []
    bins_reported: list[int] = []
    bins_reasons: list[str] = []
