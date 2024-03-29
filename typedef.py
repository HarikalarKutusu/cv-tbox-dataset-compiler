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

# Standard Lib
from dataclasses import dataclass, field
from datetime import datetime

#
# Process
#


@dataclass
class Globals:  # pylint: disable=too-many-instance-attributes
    """Class to keep globals in one place"""

    total_ver: int = 0  # total count of versions
    total_lc: int = 0  # total count of languages in all versions
    total_algo: int = 0  # total count of algorithms
    total_splits: int = 0  # total count of algorithms

    processed_ver: int = 0  # counter for corpora processed
    processed_lc: int = 0  # counter for corpora processed
    processed_algo: int = 0  # counter for corpora processed

    skipped_exists: int = 0  # skipped because the destination already exists
    skipped_nodata: int = 0  # skipped because there is no data

    start_time: datetime = datetime.now()


#
# Language
#


@dataclass
class LanguageRec:
    """Record definition for the language"""

    lc: str = ""  # cv language code
    n_name: str = ""  # Native name
    e_name: str = ""  # Name in English
    w_url: str = ""  # Wikipedia English URL
    g_url: str = ""  # Glattolog URL
    g_code: str = ""  # Glattolog code


#
# GIT
#
@dataclass
class GitRec:
    """Record definition for github access"""

    user: str = ""
    repo: str = ""
    branch: str = ""


#
# Text Corpus
#


@dataclass
class TextCorpusRec:
    """Record definition for combined text-corpora"""

    file: str = ""  # filename in cv repo
    sentence: str = ""  # original sentence


@dataclass
class TextCorpusStatsRec:  # pylint: disable=too-many-instance-attributes
    """Record definition for text-corpus statistics"""

    ver: str = ""  # cv version code (internal format nn.n, see const.py)
    lc: str = ""  # cv language code
    algo: str = ""  # splitting algorithm the analysis based on (empty for buckets validated etc)
    sp: str = ""  # Source of the text-corpus (Empty if TC from server/data, else the bucket/split name)
    has_val: bool = False  # if commonvoice-utils has validator for it
    has_phon: bool = False  # if commonvoice-utils has phonemiser for it
    # sentence statistics
    s_cnt: int = 0  # raw sentence count
    uq_s: int = 0  # unique sentence count
    uq_n: int = 0  # unique nomilized sentence count
    val: int = 0  # How many of the sentences are validated with commonvoice-utils validator - if exists?
    # character statistics
    c_sum: int = 0  # total count
    c_avg: float = 0.0  # average (mean)
    c_med: float = 0.0  # median
    c_std: float = 0.0  # standard deviation
    c_freq: list[int] = field(default_factory=lambda: [])  # frequency distribution
    # word statistics
    w_sum: int = 0  # total count
    w_avg: float = 0.0  # average (mean)
    w_med: float = 0.0  # median
    w_std: float = 0.0  # standard deviation
    w_freq: list[int] = field(default_factory=lambda: [])  # frequency distribution
    # token statistics
    t_sum: int = 0  # total counts
    t_avg: float = 0.0  # average (mean)
    t_med: float = 0.0  # median
    t_std: float = 0.0  # standard deviation
    t_freq: list[int] = field(default_factory=lambda: [])  # frequency distribution
    # graphemes & phonemes
    g_cnt: int = 0  # count of different graphemes
    g_freq: list[list[str | int]] = field(default_factory=lambda: [])  # frequency distribution symbol-count
    p_cnt: int = 0  # count of different phonemes
    p_freq: list[list[str | int]] = field(default_factory=lambda: [])  # frequency distribution symbol-count


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
    rep_freq: list[int] = field(default_factory=lambda: [])  # frequency distribution for report per sentence
    rea_freq: list[int] = field(default_factory=lambda: [])  # frequency distribution for reporting reasons


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
    dur_freq: list[int] = field(default_factory=lambda: [])  # frequency distribution
    # Recordings per Voice
    v_avg: float = 0.0  # average (mean)
    v_med: float = 0.0  # median
    v_std: float = 0.0  # standard deviation
    v_freq: list[int] = field(default_factory=lambda: [])  # frequency distribution
    # Recordings per Sentence
    s_avg: float = 0.0  # average (mean)
    s_med: float = 0.0  # median
    s_std: float = 0.0  # standard deviation
    s_freq: list[int] = field(default_factory=lambda: [])  # frequency distribution
    # Votes (UpVotes, DownVotes)
    uv_sum: int = 0  # total
    uv_avg: float = 0.0  # average (mean)
    uv_med: float = 0.0  # median
    uv_std: float = 0.0  # standard deviation
    uv_freq: list[int] = field(default_factory=lambda: [])  # frequency distribution
    dv_sum: int = 0  # total
    dv_avg: float = 0.0  # average (mean)
    dv_med: float = 0.0  # median
    dv_std: float = 0.0  # standard deviation
    dv_freq: list[int] = field(default_factory=lambda: [])  # frequency distribution
    # Demographics distribution for recordings
    dem_table: list[list[int]] = field(default_factory=lambda: [])
    dem_uq: list[list[int]] = field(default_factory=lambda: [])
    dem_fix_r: list[int] = field(default_factory=lambda: [])
    dem_fix_v: list[int] = field(default_factory=lambda: [])


#
# Config Record
#


@dataclass
class ConfigRec:  # pylint: disable=too-many-instance-attributes
    """Record definition for config"""

    date: str = ""
    cv_versions: list[str] = field(default_factory=lambda: [])
    cv_dates: list[str] = field(default_factory=lambda: [])
    cv_locales: list[str] = field(default_factory=lambda: [])
    algorithms: list[str] = field(default_factory=lambda: [])
    bins_duration: list[int] = field(default_factory=lambda: [])
    bins_voices: list[int] = field(default_factory=lambda: [])
    bins_votes_up: list[int] = field(default_factory=lambda: [])
    bins_votes_down: list[int] = field(default_factory=lambda: [])
    bins_sentences: list[int] = field(default_factory=lambda: [])
    bins_chars: list[int] = field(default_factory=lambda: [])
    bins_words: list[int] = field(default_factory=lambda: [])
    bins_tokens: list[int] = field(default_factory=lambda: [])
    bins_reported: list[int] = field(default_factory=lambda: [])
    bins_reasons: list[str] = field(default_factory=lambda: [])
