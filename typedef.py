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
from typing import Optional

# External dependencies
import pyarrow as pa
import pandas as pd

#
# Pandas / ArrowDType definitions to use Arrow backend
# See: https://pandas.pydata.org/docs/user_guide/pyarrow.html#data-structure-integration
#
dtype_pa_int8 = pd.ArrowDtype(pa.int8())
dtype_pa_int16 = pd.ArrowDtype(pa.int16())
dtype_pa_int32 = pd.ArrowDtype(pa.int32())
dtype_pa_int64 = pd.ArrowDtype(pa.int64())

dtype_pa_uint8 = pd.ArrowDtype(pa.uint8())
dtype_pa_uint16 = pd.ArrowDtype(pa.uint16())
dtype_pa_uint32 = pd.ArrowDtype(pa.uint32())
dtype_pa_uint64 = pd.ArrowDtype(pa.uint64())

dtype_pa_float16 = pd.ArrowDtype(pa.float16())
dtype_pa_float32 = pd.ArrowDtype(pa.float32())
dtype_pa_float64 = pd.ArrowDtype(pa.float64())

dtype_pa_str = pd.ArrowDtype(pa.string())

dtype_pa_list_str = pd.ArrowDtype(pa.list_(pa.string()))
dtype_pa_list_uint8 = pd.ArrowDtype(pa.list_(pa.uint8()))
dtype_pa_list_uint16 = pd.ArrowDtype(pa.list_(pa.uint16()))
dtype_pa_list_uint32 = pd.ArrowDtype(pa.list_(pa.uint32()))
dtype_pa_list_uint64 = pd.ArrowDtype(pa.list_(pa.uint64()))

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
# Parameter Passing to MultiProcessing Handlers
#


@dataclass
class MultiProcessingParams:
    """Record definition for split/audio analysis MP parameters"""

    ds_path: str = ""  # source dataset path
    ver: str = ""  # cv version code
    lc: str = ""  # cv language code
    # audio specs dataframe coming from TBOX (only for splits)
    df_aspecs: Optional[pd.DataFrame] = None
    # clip-errors dataframe coming from TBOX (only for splits)
    df_clip_errors: Optional[pd.DataFrame] = None


#
# Text Corpus
#


@dataclass
class TextCorpusStatsRec:  # pylint: disable=too-many-instance-attributes
    """Record definition for text-corpus statistics"""

    ver: str = ""  # cv version code (internal format nn.n, see const.py)
    lc: str = ""  # cv language code
    algo: str = (
        ""  # splitting algorithm the analysis based on (empty for buckets validated etc)
    )
    sp: str = (
        ""  # Source of the text-corpus (Empty if TC from server/data, else the bucket/split name)
    )
    has_val: bool = False  # if commonvoice-utils has validator for it
    has_phon: bool = False  # if commonvoice-utils has phonemiser for it
    # data from validated_sentences.tsv & unvalidated_sentences.tsv
    ## recordable (validated_sentences.tsv is_used = 1)
    ## disabled (validated_sentences.tsv is_used = 0)
    ## disabled sentences recorded count (validated_sentences.tsv is_used = 0 & sum(clips_count))
    ## invalidated & not-yet validated (validated_sentences.tsv just num rows)
    # sentence statistics
    s_cnt: int = 0  # raw sentence count
    uq_s: int = 0  # unique sentence count
    uq_n: int = 0  # unique nomilized sentence count
    val: int = (
        0  # How many of the sentences are validated with commonvoice-utils validator - if exists?
    )
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
    # graphemes: count, items & frequency distribution
    g_cnt: int = 0
    g_items: str = ""  # list[str] = field(default_factory=lambda: [])
    g_freq: list[int] = field(default_factory=lambda: [])
    # phonemes: count, items & frequency distribution
    p_cnt: int = 0
    p_items: str = ""  # list[str] = field(default_factory=lambda: [])
    p_freq: list[int] = field(default_factory=lambda: [])
    # sentence domain statistics
    dom_cnt: int = 0
    dom_items: list[str] = field(default_factory=lambda: [])
    dom_freq: list[int] = field(default_factory=lambda: [])


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
    rep_freq: list[int] = field(
        default_factory=lambda: []
    )  # frequency distribution for report per sentence
    rea_freq: list[int] = field(
        default_factory=lambda: []
    )  # frequency distribution for reporting reasons


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


@dataclass
class CharSpeedRec:  # pylint: disable=too-many-instance-attributes
    """Record definition for dataset split statistics"""

    ver: str = ""  # cv version code (internal format nn.n, see const.py)
    lc: str = ""  # cv language code
    alg: str = ""  # cv-tbox splitting algorithm (see const.py)
    sp: str = ""  # split name (blank, train, dev, test)
    clips: int = 0  # number of recordings
    # Character Speed data
    cs_avg: float = 0.0  # average (mean)
    cs_med: float = 0.0  # median
    cs_std: float = 0.0  # standard deviation
    cs_freq: list[int] = field(default_factory=lambda: [])  # frequency distribution
    # CrossTabs
    cs_r: str = ""  # row labels for all crosstabs (from list of int)
    cs2s_c: str = ""  # col labels for sentence length (from list of int)

    cs2s: str = ""  # char-speed vs sentence length (from arr of int)
    cs2g: str = ""  # char-speed vs gender (columns are known) (from arr of int)
    cs2a: str = ""  # char-speed vs age (columns are known) (from arr of int)


@dataclass
class AudioAnalysisStatsRec:  # pylint: disable=too-many-instance-attributes
    """Record definition for dataset audio analysis statistics"""

    ver: str = ""  # cv version code (internal format nn.n, see const.py)
    lc: str = ""  # cv language code
    alg: str = ""  # cv-tbox splitting algorithm (see const.py)
    sp: str = ""  # split name (blank, train, dev, test)
    # basic counts/sums
    clips: int = 0  # number of recordings
    errors: int = 0  # total errors (all kind)
    dur: int = 0  # measured duration
    no_vad: int = 0  # Clip count where no speech is detected
    low_power: int = 0  # Clip count where speech power is low
    low_snr: int = 0  # Clip count where SNR is negative
    # basic audio property distributions
    enc_r: str = ""  # row values
    enc_freq: str = ""  # encodings
    chan_r: str = ""  # row values
    chan_freq: str = ""  # channels
    srate_r: str = ""  # row values
    srate_freq: str = ""  # sampling rate
    brate_r: str = ""  # row values
    brate_freq: str = ""  # bit rate
    # Errors
    err_r: str = ""  # row values = error sources
    err_freq: str = ""  # counts
    # Real Voice Durations
    vad_sum: int = 0  # total VAD duration
    vad_avg: float = 0.0  # average (mean)
    vad_med: float = 0.0  # median
    vad_std: float = 0.0  # standard deviation
    vad_freq: list[int] = field(default_factory=lambda: [])  # freq. distribution
    # Speech Power Statistics
    sp_pwr_avg: float = 0.0  # average (mean)
    sp_pwr_med: float = 0.0  # median
    sp_pwr_std: float = 0.0  # standard deviation
    sp_pwr_freq: list[int] = field(default_factory=lambda: [])  # freq. distribution
    # Silence Power Statistics
    sil_pwr_avg: float = 0.0  # average (mean)
    sil_pwr_med: float = 0.0  # median
    sil_pwr_std: float = 0.0  # standard deviation
    sil_pwr_freq: list[int] = field(default_factory=lambda: [])  # freq. distribution
    # SNR Statistics
    snr_avg: float = 0.0  # average (mean)
    snr_med: float = 0.0  # median
    snr_std: float = 0.0  # standard deviation
    snr_freq: list[int] = field(default_factory=lambda: [])  # freq. distribution


#
# Config Record
#


@dataclass
class ConfigRec:  # pylint: disable=too-many-instance-attributes
    """Record definition for config"""

    # general
    date: str = ""
    cv_versions: list[str] = field(default_factory=lambda: [])
    cv_dates: list[str] = field(default_factory=lambda: [])
    cv_locales: list[str] = field(default_factory=lambda: [])
    algorithms: list[str] = field(default_factory=lambda: [])
    # basic bins
    bins_duration: list[int] = field(default_factory=lambda: [])
    bins_voices: list[int] = field(default_factory=lambda: [])
    bins_votes_up: list[int] = field(default_factory=lambda: [])
    bins_votes_down: list[int] = field(default_factory=lambda: [])
    bins_sentences: list[int] = field(default_factory=lambda: [])
    # char speed
    cs_threshold: int = 0
    bins_cs_low: list[int] = field(default_factory=lambda: [])
    bins_cs_high: list[int] = field(default_factory=lambda: [])
    ch_threshold: int = 0
    bins_chars_short: list[int] = field(default_factory=lambda: [])
    bins_chars_long: list[int] = field(default_factory=lambda: [])
    # text-corpus
    bins_words: list[int] = field(default_factory=lambda: [])
    bins_tokens: list[int] = field(default_factory=lambda: [])
    # reported
    bins_reported: list[int] = field(default_factory=lambda: [])
    bins_reasons: list[str] = field(default_factory=lambda: [])
    # audio-analysis
    bins_aa_pwr: list[int] = field(default_factory=lambda: [])  # VAD power
    bins_aa_snr: list[int] = field(default_factory=lambda: [])  # SNR
