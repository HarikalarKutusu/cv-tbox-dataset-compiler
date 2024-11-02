"""cv-tbox Dataset Compiler - Final Compilation Phase - Processes"""

###########################################################################
# lib_final.py
#
# Used by final_compile.py
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-tbox-dataset-compiler
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################

# Standard Lib
from collections import Counter
from typing import Optional
from ast import literal_eval
import os
import sys
import gc

# External dependencies
import numpy as np
import pandas as pd
import cvutils as cvu  # type: ignore

# Module
import const as c
import conf
from typedef import (
    MultiProcessingParams,
    AudioAnalysisStatsRec,
    TextCorpusStatsRec,
    ReportedStatsRec,
    SplitStatsRec,
    CharSpeedRec,
    dtype_pa_str,
    dtype_pa_uint32,
)
from lib import (
    df_read,
    df_read_safe_reported,
    df_write,
    gender_backmapping,
    dec3,
    calc_dataset_prefix,
    arr2str,
    list2str,
)

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

cv: cvu.CV = cvu.CV()
VALIDATORS: list[str] = cv.validators()
PHONEMISERS: list[str] = cv.phonemisers()
# ALPHABETS: list[str] = [str(p).split(os.sep)[-2] for p in cv.alphabets()]
# SEGMENTERS: list[str] = [str(p).split(os.sep)[-2] for p in cv.segmenters()]


########################################################
# Text-Corpus Stats (Multi Processing Handler)
########################################################


def handle_text_corpus(ver_lc: str) -> list[TextCorpusStatsRec]:
    """Multi-Process text-corpus for a single locale"""

    def handle_df(
        df: pd.DataFrame, algo: str = "", sp: str = ""
    ) -> TextCorpusStatsRec | None:
        """Calculate stats given a dataframe containing only sentences"""

        if df.shape[0] == 0:
            if conf.VERBOSE:
                print(
                    f"WARN: Skipping empty data for: Ver: {ver} - LC: {lc} - Algo: {algo} - Split: {sp}"
                )
            return None

        # prep result record with default
        res: TextCorpusStatsRec = TextCorpusStatsRec(
            ver=ver,
            lc=lc,
            algo=algo,
            sp=sp,
            has_val=has_validator,
            has_phon=has_phonemiser,
        )

        # init counters
        token_counter: Counter = Counter()
        grapheme_counter: Counter = Counter()
        phoneme_counter: Counter = Counter()

        # decide on saving
        do_save: bool = False
        if conf.SAVE_LEVEL == c.SAVE_LEVEL_DETAILED:
            do_save = True
        elif conf.SAVE_LEVEL == c.SAVE_LEVEL_DEFAULT and algo == "" and sp == "":
            do_save = True
        elif (
            conf.SAVE_LEVEL == c.SAVE_LEVEL_DEFAULT
            and algo == "s1"
            and sp in ["validated", "train", "dev", "test"]
        ):
            do_save = True

        # see: https://github.com/pylint-dev/pylint/issues/3956
        _ser: pd.Series[int] = pd.Series()  # pylint: disable=unsubscriptable-object
        _df2: pd.DataFrame = pd.DataFrame()
        _arr: np.ndarray
        fn: str

        # validator dependent
        if has_validator:
            token_counter.update(
                w
                for ww in df["tokens"].dropna().apply(literal_eval).to_list()
                for w in ww
            )

            # word_cnt stats
            _ser = df["word_cnt"].dropna()
            if _ser.shape[0] > 0:
                res.w_sum = _ser.sum()
                res.w_avg = dec3(_ser.mean())
                res.w_med = _ser.median()
                res.w_std = dec3(_ser.std(ddof=0))
                # Calc word count distribution
                _arr = np.fromiter(
                    _ser.apply(int).reset_index(drop=True).to_list(), int
                )
                _hist = np.histogram(_arr, bins=c.BINS_WORDS)
                res.w_freq = _hist[0].tolist()[1:]

            # token_cnt stats
            _df2 = pd.DataFrame(token_counter.most_common(), columns=c.FIELDS_TOKENS)

            # "token", "count"
            res.t_sum = _df2.shape[0]
            _ser = _df2["count"].dropna()
            if _ser.shape[0] > 0:
                res.t_avg = dec3(_ser.mean())
                res.t_med = _ser.median()
                res.t_std = dec3(_ser.std(ddof=0))
                # Token/word repeat distribution
                _arr = np.fromiter(
                    _df2["count"].dropna().apply(int).reset_index(drop=True).to_list(),
                    int,
                )
                _hist = np.histogram(_arr, bins=c.BINS_TOKENS)
                res.t_freq = _hist[0].tolist()[1:]
            if do_save:
                fn = os.path.join(
                    tc_anal_dir,
                    f"{c.TOKENS_FN}_{algo}_{sp}.tsv".replace("__", "_").replace(
                        "_.", "."
                    ),
                )
                df_write(_df2, fn)

        # phonemiser dependent
        if has_phonemiser:
            _ = [phoneme_counter.update(p) for p in df["phonemised"].dropna().tolist()]

            # PHONEMES
            _df2 = pd.DataFrame(
                phoneme_counter.most_common(), columns=c.FIELDS_PHONEMES
            )
            _values = _df2.values.tolist()
            res.p_cnt = len(_values)
            _values = _values[:100]
            res.p_items = list2str([x[0] for x in _values])
            res.p_freq = [x[1] for x in _values]
            if do_save:
                fn = os.path.join(
                    tc_anal_dir,
                    f"{c.PHONEMES_FN}_{algo}_{sp}.tsv".replace("__", "_").replace(
                        "_.", "."
                    ),
                )
                df_write(_df2, fn)

        # simpler values which are independent
        res.s_cnt = df.shape[0]
        res.val = df["valid"].dropna().astype(int).sum()
        res.uq_s = df["sentence"].dropna().unique().shape[0]
        res.uq_n = df["normalized"].dropna().unique().shape[0]

        # char_cnt stats
        _ser = df["char_cnt"].dropna()
        if _ser.shape[0] > 0:
            res.c_sum = _ser.sum()
            res.c_avg = dec3(_ser.mean())
            res.c_med = _ser.median()
            res.c_std = dec3(_ser.std(ddof=0))
            # Calc character length distribution
            _arr = np.fromiter(_ser.apply(int).reset_index(drop=True).to_list(), int)
            _sl_bins: list[int] = (
                c.BINS_CHARS_SHORT
                if res.c_avg < c.CHARS_BIN_THRESHOLD
                else c.BINS_CHARS_LONG
            )
            _hist = np.histogram(_arr, bins=_sl_bins)
            res.c_freq = _hist[0].tolist()

        # GRAPHEMES
        _ = [grapheme_counter.update(s) for s in df["sentence"].dropna().tolist()]
        _df2 = pd.DataFrame(grapheme_counter.most_common(), columns=c.FIELDS_GRAPHEMES)
        _values = _df2.values.tolist()
        res.g_cnt = len(_values)
        _values = _values[:100]
        res.g_items = list2str([x[0] for x in _values])
        res.g_freq = [x[1] for x in _values]
        if do_save:
            fn = os.path.join(
                tc_anal_dir,
                f"{c.GRAPHEMES_FN}_{algo}_{sp}.tsv".replace("__", "_").replace(
                    "_.", "."
                ),
            )
            df_write(_df2, fn)

        # SENTENCE DOMAINS
        # for < CV v17.0, they will be "nodata", after that new items are added
        # for CV v17.0, there will be single instance (or empty)
        # [TODO] for CV >= v18.0, it can be comma delimited list of max 3 domains
        _df2 = (
            df["sentence_domain"]
            .astype(dtype_pa_str)
            # .fillna(c.NODATA)
            .dropna()
            .value_counts()
            .to_frame()
            .reset_index()
            .reindex(columns=c.FIELDS_SENTENCE_DOMAINS)
            # .sort_values("count", ascending=False)
        ).astype(c.FIELDS_SENTENCE_DOMAINS)

        # prep counters & loop for comma delimited multi-domains
        counters: dict[str, int] = {}
        # domain_list: list[str] = c.CV_DOMAINS_V17 if ver == "17.0" else c.CV_DOMAINS
        domain_list: list[str] = c.CV_DOMAINS
        for dom in domain_list:
            counters[dom] = 0
        for _, row in _df2.iterrows():
            domains: list[str] = row.iloc[0].split(",")
            for dom in domains:
                counters[c.CV_DOMAIN_MAPPER[dom]] += row.iloc[1]

        res.dom_cnt = len([tup[0] for tup in counters.items() if tup[1] > 0])
        res.dom_items = [tup[0] for tup in counters.items() if tup[1] > 0]
        res.dom_freq = [tup[1] for tup in counters.items() if tup[1] > 0]

        if do_save:
            fn = os.path.join(
                tc_anal_dir,
                f"{c.DOMAINS_FN}_{algo}_{sp}.tsv".replace("__", "_").replace("_.", "."),
            )
            df_write(_df2, fn)

        # return result
        return res

    def handle_tc_global(df_base_ver_tc: pd.DataFrame) -> None:
        """Calculate stats using the whole text corpus from server/data"""
        # res: TextCorpusStatsRec | None = handle_df(
        #     df_read(base_tc_file).reset_index(drop=True)[["sentence"]]
        # )
        _res: TextCorpusStatsRec | None = handle_df(df_base_ver_tc)
        if _res is not None:
            results.append(_res)

    def handle_tc_split(df_base_ver_tc: pd.DataFrame, sp: str, algo: str = "") -> None:
        """Calculate stats using sentence data in a algo - bucket/split"""
        _fn: str = os.path.join(conf.SRC_BASE_DIR, algo, ver_dir, lc, f"{sp}.tsv")
        if not os.path.isfile(_fn):
            if conf.VERBOSE:
                print(f"WARN: No such split file for: {ver} - {lc} - {algo} - {sp}")
            return
        _res: TextCorpusStatsRec | None = None
        df: pd.DataFrame
        if float(ver) >= 17.0:
            # For newer versions, just use the sentence_id
            sentence_id_list: list[str] = (
                df_read(_fn)
                .reset_index(drop=True)["sentence_id"]
                .dropna()
                .drop_duplicates()
                .to_list()
            )
            df = df_base_ver_tc[df_base_ver_tc["sentence_id"].isin(sentence_id_list)]
            _res = handle_df(df, algo=algo, sp=sp)
        else:
            # For older versions, use the sentence
            sentence_list: list[str] = (
                df_read(_fn)
                .reset_index(drop=True)["sentence"]
                .dropna()
                .drop_duplicates()
                .to_list()
            )
            df = df_base_ver_tc[df_base_ver_tc["sentence"].isin(sentence_list)]
            _res = handle_df(df, algo=algo, sp=sp)
        if _res is not None:
            results.append(_res)

    #
    # handle_df main
    #
    ver: str = ver_lc.split("|")[0]
    lc: str = ver_lc.split("|")[1]
    ver_dir: str = calc_dataset_prefix(ver)

    results: list[TextCorpusStatsRec] = []

    base_tc_file: str = os.path.join(
        conf.DATA_BASE_DIR, c.TC_DIRNAME, lc, f"{c.TEXT_CORPUS_FN}.tsv"
    )
    ver_tc_inx_file: str = os.path.join(
        conf.DATA_BASE_DIR, c.TC_DIRNAME, lc, f"{c.TEXT_CORPUS_FN}_{ver}.tsv"
    )
    if not os.path.isfile(base_tc_file):
        if conf.VERBOSE:
            print(f"WARN: No text-corpus file for: {lc}")
        return results
    if not os.path.isfile(ver_tc_inx_file):
        if conf.VERBOSE:
            print(f"WARN: No text-corpus index file for: {ver} - {lc}")
        return results

    tc_anal_dir: str = os.path.join(
        conf.DATA_BASE_DIR, c.TC_ANALYSIS_DIRNAME, ver_dir, lc
    )
    os.makedirs(tc_anal_dir, exist_ok=True)

    # cvu - do we have them?
    has_validator: bool = lc in VALIDATORS
    has_phonemiser: bool = lc in PHONEMISERS
    # tokeniser: cvu.Tokeniser = cvu.Tokeniser(lc)

    # get and filter text_corpus data
    df_base_ver_tc: pd.DataFrame = df_read(base_tc_file)
    # we only should use allowed ones
    df_base_ver_tc = df_base_ver_tc[df_base_ver_tc["is_used"] == 1]
    df_ver_inx: pd.DataFrame = df_read(ver_tc_inx_file)
    sentence_id_list: list[str] = df_ver_inx["sentence_id"].to_list()
    df_base_ver_tc = df_base_ver_tc[
        df_base_ver_tc["sentence_id"].isin(sentence_id_list)
    ]
    del df_ver_inx
    df_ver_inx = pd.DataFrame()

    handle_tc_global(df_base_ver_tc)
    for sp in ["validated", "invalidated", "other"]:
        handle_tc_split(df_base_ver_tc, sp, c.ALGORITHMS[0])
        # handle_tc_split(df_base_ver_tc, sp, "")

    for algo in c.ALGORITHMS:
        for sp in ["train", "dev", "test"]:
            handle_tc_split(df_base_ver_tc, sp, algo)
    # done
    return results


# END handle_text_corpus

# END - Text-Corpus Stats (Multi Processing Handler)


########################################################
# Reported Stats
########################################################


def handle_reported(ver_lc: str) -> ReportedStatsRec:
    """Process text-corpus for a single locale"""

    ver: str = ver_lc.split("|")[0]
    lc: str = ver_lc.split("|")[1]

    ver_dir: str = calc_dataset_prefix(ver)

    # Calc reported file
    rep_file: str = os.path.join(
        conf.DATA_BASE_DIR, c.VC_DIRNAME, ver_dir, lc, "reported.tsv"
    )

    # skip process if no such file or there can be empty files :/
    if not os.path.isfile(rep_file) or os.path.getsize(rep_file) == 0:
        return ReportedStatsRec(ver=ver, lc=lc)
    # read file in - Columns: sentence sentence_id locale reason
    # df: pd.DataFrame = df_read(rep_file)
    problem_list: list[str] = []
    fields: dict[str, pd.ArrowDtype] = (
        c.FIELDS_REPORTED if float(ver) >= 17.0 else c.FIELDS_REPORTED_OLD
    )
    df: pd.DataFrame = pd.DataFrame(columns=fields).astype(fields)
    df, problem_list = df_read_safe_reported(rep_file)

    # debug
    if conf.CREATE_REPORTED_PROBLEMS:
        # if ver == "17.0" and lc == "en":
        if ver == "17.0":
            df_write(
                df,
                os.path.join(conf.DATA_BASE_DIR, ".debug", f"{lc}_{ver}_reported.tsv"),
            )
            if len(problem_list) > 0:
                with open(
                    os.path.join(
                        conf.DATA_BASE_DIR, ".debug", f"{lc}_{ver}_problems.txt"
                    ),
                    mode="w",
                    encoding="utf8",
                ) as fd:
                    fd.write("\n".join(problem_list))

    if df.shape[0] == 0:  # skip those without records
        return ReportedStatsRec(ver=ver, lc=lc)

    # Now we have a file with some records in it...
    df = df.drop(["sentence", "locale"], axis=1).reset_index(drop=True)

    reported_total: int = df.shape[0]
    reported_sentences: int = len(df["sentence_id"].unique().tolist())

    # get a distribution of reasons/sentence & stats
    rep_counts: pd.DataFrame = (
        df["sentence_id"].value_counts().dropna().to_frame().reset_index()
    )
    # make others 'other'
    df.loc[~df["reason"].isin(c.REPORTING_BASE)] = "other"

    # Get statistics
    ser: pd.Series = rep_counts["count"]
    # sys.exit(0)
    rep_mean: float = ser.mean()
    rep_median: float = ser.median()
    rep_std: float = ser.std(ddof=0)
    # Calc report-per-sentence distribution
    arr: np.ndarray = np.fromiter(
        rep_counts["count"].dropna().apply(int).reset_index(drop=True).to_list(),
        int,
    )
    hist = np.histogram(arr, bins=c.BINS_REPORTED)
    rep_freq = hist[0].tolist()[1:]

    # Get reason counts
    reason_counts: pd.DataFrame = (
        df["reason"].value_counts().dropna().to_frame().reset_index()
    )
    reason_counts.set_index(keys="reason", inplace=True)
    reason_counts = reason_counts.reindex(index=c.REPORTING_ALL, fill_value=0)
    reason_freq = reason_counts["count"].to_numpy(int).tolist()

    res: ReportedStatsRec = ReportedStatsRec(
        ver=ver,
        lc=lc,
        rep_sum=reported_total,
        rep_sen=reported_sentences,
        rep_avg=dec3(rep_mean),
        rep_med=dec3(rep_median),
        rep_std=dec3(rep_std),
        rep_freq=rep_freq,
        rea_freq=reason_freq,
    )
    return res


# END - Reported Stats


########################################################
# Dataset Split Stats (inc. Audio Specs Stats) (MP Handler)
########################################################


def handle_dataset_splits(
    params: MultiProcessingParams,
) -> int:
    """Handle a single dataset (ver/lc)"""
    # Handle one split, this is where calculations happen
    # The default column structure of CV dataset splits is as follows [FIXME] variants?
    # client_id, path, sentence, up_votes, down_votes, age, gender, accents, locale, segment
    # we have as input:
    # 'version', 'locale', 'algo', 'split'

    path_list: list[str] = []

    def handle_split_audio_stats(
        ver: str,
        lc: str,
        algo: str,
        split: str,
        df_aspecs_sub: pd.DataFrame,
    ) -> AudioAnalysisStatsRec:
        """Processes a single split's audio specs statistics and return calculated values"""
        # columns:
        # clip_id
        # orig_path, orig_encoding, orig_sample_rate, orig_num_frames, orig_num_channels, orig_bitrate_kbps, orig_bits_per_sample
        # tc_path, tc_encoding, tc_sample_rate, tc_num_frames, tc_num_channels, tc_bitrate_kbps, tc_bits_per_sample
        # duration, speech_duration, speech_power, silence_power, est_snr
        # ver, ds, lc

        nonlocal path_list

        res: AudioAnalysisStatsRec = AudioAnalysisStatsRec(
            ver=ver, lc=lc, alg=algo, sp=split
        )
        if df_aspecs_sub is None or df_aspecs_sub.shape[0] == 0:
            # if no data, return an empty placeholder
            return res

        # Now get some statistics
        _ser: pd.Series[float]  # pylint: disable=unsubscriptable-object
        _df2: pd.DataFrame
        _df3: pd.DataFrame
        _df4: pd.DataFrame

        # Start with errors
        _df2 = df_clip_errors[df_clip_errors["path"].isin(path_list)]
        res.errors = _df2.shape[0]
        # [TODO] Detailed error stats - type vs count
        if _df2.shape[0] > 0:
            _df2 = (
                _df2["source"]
                .value_counts()
                .to_frame()
                .reset_index(drop=False)
                .astype({"source": dtype_pa_str, "count": dtype_pa_uint32})
                .sort_values(["source"])
            )
            res.err_r = list2str(_df2["source"].to_list())
            res.err_freq = list2str(_df2["count"].to_list())

        # general
        res.clips = df_aspecs_sub.shape[0]
        res.dur = round(df_aspecs_sub["duration"].dropna().sum() / 1000)  # seconds

        # vad stats (convert to secs)
        _ser = df_aspecs_sub["speech_duration"].dropna().apply(lambda x: x / 1000)
        if _ser.shape[0] > 0:
            res.vad_sum = int(_ser.sum())
            res.vad_avg = dec3(_ser.mean())
            res.vad_med = dec3(_ser.median())
            res.vad_std = dec3(_ser.std(ddof=0))
            # Calc word count distribution
            _arr = np.fromiter(_ser.apply(int).reset_index(drop=True).to_list(), int)
            _hist = np.histogram(_arr, bins=c.BINS_DURATION)
            res.vad_freq = _hist[0].tolist()

        # speech power stats (10^-6) we scale it
        _ser = df_aspecs_sub["speech_power"].dropna().apply(lambda x: x * 1_000_000)
        if _ser.shape[0] > 0:
            res.sp_pwr_avg = dec3(_ser.mean())
            res.sp_pwr_med = dec3(_ser.median())
            res.sp_pwr_std = dec3(_ser.std(ddof=0))
            # Calc word count distribution
            _arr = np.fromiter(_ser.apply(int).reset_index(drop=True).to_list(), int)
            _hist = np.histogram(_arr, bins=c.BINS_POWER)
            res.sp_pwr_freq = _hist[0].tolist()

        # silence power stats (10^-9) we scale it
        _ser = (
            df_aspecs_sub["silence_power"].dropna().apply(lambda x: x * 1_000_000_000)
        )
        if _ser.shape[0] > 0:
            res.sil_pwr_avg = dec3(_ser.mean())
            res.sil_pwr_med = dec3(_ser.median())
            res.sil_pwr_std = dec3(_ser.std(ddof=0))
            # Calc word count distribution
            _arr = np.fromiter(_ser.apply(int).reset_index(drop=True).to_list(), int)
            _hist = np.histogram(_arr, bins=c.BINS_POWER)
            res.sil_pwr_freq = _hist[0].tolist()

        # snr stats
        # no speech
        _df2 = df_aspecs_sub[df_aspecs_sub["est_snr"].isna()]
        res.no_vad = _df2.shape[0]
        # low snr
        _df3 = df_aspecs_sub[df_aspecs_sub["est_snr"] < conf.LOW_SNR_THRESHOLD]
        res.low_snr = _df3.shape[0]
        # low power
        _df4 = df_aspecs_sub[df_aspecs_sub["speech_power"] < conf.LOW_POWER_THRESHOLD]
        res.low_power = _df4.shape[0]
        # combine and save for main buckets
        if algo == "" and split in c.MAIN_BUCKETS:
            _df2 = pd.concat([_df2, _df3, _df4]).drop_duplicates()
            if _df2.shape[0] > 0:
                df_write(
                    df=_df3, fpath=os.path.join(ds_meta_dir, f"audio_bad_{split}.tsv")
                )

        # valid snr (where we detected speech)
        _ser = df_aspecs_sub[~df_aspecs_sub["est_snr"].isna()]["est_snr"]
        if _ser.shape[0] > 0:
            res.snr_avg = dec3(_ser.mean())
            res.snr_med = dec3(_ser.median())
            res.snr_std = dec3(_ser.std(ddof=0))
            # Calc word count distribution
            _arr = np.fromiter(_ser.apply(int).reset_index(drop=True).to_list(), int)
            _hist = np.histogram(_arr, bins=c.BINS_SNR)
            res.snr_freq = _hist[0].tolist()[1:]  # drop lower than -100 SNR

        # direct distributions (value counts)
        # encoding
        _df2 = (
            df_aspecs_sub["orig_encoding"]
            .astype(dtype_pa_str)
            .dropna()
            .value_counts()
            .to_frame()
            .reset_index(drop=False)
            .astype({"orig_encoding": dtype_pa_str, "count": dtype_pa_uint32})
            .sort_values(["orig_encoding"])
        )
        res.enc_r = list2str(_df2["orig_encoding"].to_list())
        res.enc_freq = list2str(_df2["count"].to_list())
        # channels
        _df2 = (
            df_aspecs_sub["orig_num_channels"]
            .astype(dtype_pa_str)
            .dropna()
            .value_counts()
            .to_frame()
            .reset_index(drop=False)
            .astype({"orig_num_channels": dtype_pa_str, "count": dtype_pa_uint32})
            .sort_values(["orig_num_channels"])
        )
        res.chan_r = list2str(_df2["orig_num_channels"].to_list())
        res.chan_freq = list2str(_df2["count"].to_list())
        # sample rate
        _df2 = (
            df_aspecs_sub["orig_sample_rate"]
            .astype(dtype_pa_str)
            .dropna()
            .value_counts()
            .to_frame()
            .reset_index(drop=False)
            .astype({"orig_sample_rate": dtype_pa_str, "count": dtype_pa_uint32})
            .sort_values(["orig_sample_rate"])
        )
        res.srate_r = list2str(_df2["orig_sample_rate"].to_list())
        res.srate_freq = list2str(_df2["count"].to_list())
        # bit rate
        _df2 = (
            df_aspecs_sub["orig_bitrate_kbps"]
            .astype(dtype_pa_str)
            .dropna()
            .value_counts()
            .to_frame()
            .reset_index(drop=False)
            .astype({"orig_bitrate_kbps": dtype_pa_str, "count": dtype_pa_uint32})
            .sort_values(["orig_bitrate_kbps"])
        )
        res.brate_r = list2str(_df2["orig_bitrate_kbps"].to_list())
        res.brate_freq = list2str(_df2["count"].to_list())

        return res

    # now, do calculate some statistics...
    def handle_split(
        ver: str, lc: str, algo: str, split: str, src_ds_dir: str
    ) -> tuple[SplitStatsRec, CharSpeedRec, AudioAnalysisStatsRec]:
        """Processes a single split and return calculated values"""

        nonlocal path_list

        # find_fixes
        # def find_fixes(df_split: pd.DataFrame) -> list[list[int]]:
        #     """Finds fixable demographic info from the split and returns a string"""

        #     # df is local dataframe which will keep records
        #     # only necessary columns with some additional columns
        #     df: pd.DataFrame = df_split.copy().reset_index(drop=True)
        #     df["v_enum"], _ = pd.factorize(
        #         df["client_id"]
        #     )  # add an enumaration column for client_id's, more memory efficient
        #     df["p_enum"], _ = pd.factorize(
        #         df["path"]
        #     )  # add an enumaration column for recordings, more memory efficient
        #     df = (
        #         df[["v_enum", "age", "gender", "p_enum"]]
        #         .fillna(c.NODATA)
        #         .reset_index(drop=True)
        #     )

        #     # prepare empty results
        #     fixes: pd.DataFrame = pd.DataFrame(columns=df.columns).reset_index(
        #         drop=True
        #     )
        #     dem_fixes_recs: list[int] = []
        #     dem_fixes_voices: list[int] = []

        #     # get unique voices with multiple demographic values
        #     df_counts: pd.DataFrame = (
        #         df[["v_enum", "age", "gender"]]
        #         .drop_duplicates()
        #         .copy()
        #         .groupby("v_enum")
        #         .agg({"age": "count", "gender": "count"})
        #     )
        #     df_counts.reset_index(inplace=True)
        #     df_counts = df_counts[
        #         (df_counts["age"].astype(int) == 2)
        #         | (df_counts["gender"].astype(int) == 2)
        #     ]  # reduce that to only processible ones
        #     v_processable: list[int] = df_counts["v_enum"].unique().tolist()

        #     # now, work only on problem voices & records.
        #     # For each voice, get related records and decide
        #     for v in v_processable:
        #         recs: pd.DataFrame = df[df["v_enum"] == v].copy()
        #         recs_blanks: pd.DataFrame = recs[
        #             (recs["gender"] == c.NODATA) | (recs["age"] == c.NODATA)
        #         ].copy()  # get full blanks
        #         # gender
        #         recs_w_gender: pd.DataFrame = recs[~(recs["gender"] == c.NODATA)].copy()
        #         if recs_w_gender.shape[0] > 0:
        #             val: str = recs_w_gender["gender"].tolist()[0]
        #             recs_blanks.loc[:, "gender"] = val
        #         # age
        #         recs_w_age: pd.DataFrame = recs[~(recs["age"] == c.NODATA)].copy()
        #         if recs_w_age.shape[0] > 0:
        #             val: str = recs_w_age["age"].tolist()[0]
        #             recs_blanks.loc[:, "age"] = val
        #         # now we can add them to the result fixed list
        #         fixes = pd.concat([fixes.loc[:], recs_blanks]).reset_index(drop=True)

        #     # Here, we have a df maybe with records of possible changes
        #     if fixes.shape[0] > 0:
        #         # records
        #         pt: pd.DataFrame = pd.pivot_table(
        #             fixes,
        #             values="p_enum",
        #             index=["age"],
        #             columns=["gender"],
        #             aggfunc="count",
        #             fill_value=0,
        #             dropna=False,
        #             margins=False,
        #         )
        #         # get only value parts : nodata is just negative sum of these, and TOTAL will be 0,
        #         # so we drop them for file size and leave computation to the client
        #         pt = (
        #             pt.reindex(c.CV_AGES, axis=0)
        #             .reindex(c.CV_GENDERS, axis=1)
        #             .fillna(value=0)
        #             .astype(int)
        #             .drop(c.NODATA, axis=0)
        #             .drop(c.NODATA, axis=1)
        #         )
        #         dem_fixes_recs = pt.to_numpy(int).tolist()

        #         # voices
        #         fixes = fixes.drop("p_enum", axis=1).drop_duplicates()
        #         pt: pd.DataFrame = pd.pivot_table(
        #             fixes,
        #             values="v_enum",
        #             index=["age"],
        #             columns=["gender"],
        #             aggfunc="count",
        #             fill_value=0,
        #             dropna=False,
        #             margins=False,
        #         )
        #         # get only value parts : nodata is just -sum of these, sum will be 0
        #         pt = (
        #             pt.reindex(c.CV_AGES, axis=0)
        #             .reindex(c.CV_GENDERS, axis=1)
        #             .fillna(value=0)
        #             .astype(int)
        #             .drop(c.NODATA, axis=0)
        #             .drop(c.NODATA, axis=1)
        #         )
        #         dem_fixes_voices = pt.to_numpy(int).tolist()

        #     return [dem_fixes_recs, dem_fixes_voices]

        # END - find_fixes

        #
        # === START ===
        #

        # Read in DataFrames
        sp_fpath: str
        df_orig: pd.DataFrame = pd.DataFrame()
        if split == "clips":  # build "clips" from val+inval+other
            sp_fpath = os.path.join(src_ds_dir, "validated.tsv")
            df_orig = df_read(sp_fpath)
            # add invalidated
            _df: pd.DataFrame = df_read(sp_fpath.replace("validated", "invalidated"))
            if _df.shape[0] > 0:
                df_orig = pd.concat([df_orig, _df]) if df_orig.shape[0] > 0 else _df
            # add other
            _df = df_read(sp_fpath.replace("validated", "other"))
            if _df.shape[0] > 0:
                df_orig = pd.concat([df_orig, _df]) if df_orig.shape[0] > 0 else _df
        else:
            sp_fpath = os.path.join(src_ds_dir, f"{split}.tsv")
            if os.path.isfile(sp_fpath):
                df_orig = df_read(sp_fpath)

        # Do nothing, if there is no data or no such split
        if df_orig.shape[0] == 0:
            return (
                SplitStatsRec(ver=ver, lc=lc, alg=algo, sp=split),
                CharSpeedRec(ver=ver, lc=lc, alg=algo, sp=split),
                AudioAnalysisStatsRec(ver=ver, lc=lc, alg=algo, sp=split),
            )

        # [TODO] Move these to split_compile: Make all confirm to current style?
        # Normalize data to the latest version's columns with typing
        # Replace NA with NODATA with some typing and conditionals
        # df: pd.DataFrame = df_orig.fillna(value=c.NODATA)
        df: pd.DataFrame = pd.DataFrame(
            columns=list(c.FIELDS_BUCKETS_SPLITS.keys())
        ).astype(c.FIELDS_BUCKETS_SPLITS)
        # these should exist
        df["client_id"] = df_orig["client_id"]
        df["path"] = df_orig["path"]
        df["sentence"] = df_orig["sentence"]
        df["up_votes"] = df_orig["up_votes"]
        df["down_votes"] = df_orig["down_votes"]
        # these exist, but can be NaN
        df["age"] = df_orig["age"].astype(dtype_pa_str).fillna(c.NODATA)
        df["gender"] = df_orig["gender"].astype(dtype_pa_str).fillna(c.NODATA)
        # These might not exist in older versions, so we fill them
        df["locale"] = (
            df_orig["locale"].astype(dtype_pa_str)
            if "locale" in df_orig.columns
            else lc
        )
        df["variant"] = (
            df_orig["variant"].astype(dtype_pa_str).fillna(c.NODATA)
            if "variant" in df_orig.columns
            else c.NODATA
        )
        df["segment"] = (
            df_orig["segment"].astype(dtype_pa_str).fillna(c.NODATA)
            if "segment" in df_orig.columns
            else c.NODATA
        )
        df["sentence_domain"] = (
            df_orig["sentence_domain"].astype(dtype_pa_str).fillna(c.NODATA)
            if "sentence_domain" in df_orig.columns
            else c.NODATA
        )
        # The "accent" column renamed to "accents" along the way
        if "accent" in df_orig.columns:
            df["accents"] = df_orig["accent"].astype(dtype_pa_str).fillna(c.NODATA)
        if "accents" in df_orig.columns:
            df["accents"] = df_orig["accents"].astype(dtype_pa_str).fillna(c.NODATA)
        # [TODO] this needs special consideration (back-lookup) but has quirks for now
        df["sentence_id"] = (
            df_orig["sentence_id"].astype(dtype_pa_str).fillna(c.NODATA)
            if "sentence_id" in df_orig.columns
            else c.NODATA
        )

        # backmap genders
        df = gender_backmapping(df)
        # add lowercase sentence column
        df["sentence_lower"] = df["sentence"].str.lower()

        # === DURATIONS: Calc duration agregate values
        # there must be records + v1 cannot be mapped
        ser: pd.Series
        arr: np.ndarray
        duration_freq = []
        # Assume no Duration data, set illegal defaults
        duration_total: float = -1
        duration_mean: float = -1
        duration_median: float = -1
        duration_std: float = -1
        if df_clip_durations.shape[0] > 0 and ver != "1":
            # Connect with duration table and convert to seconds
            df["duration"] = df["path"].map(
                df_clip_durations["duration[ms]"] / 1000, na_action="ignore"
            )
            ser = df["duration"].dropna()
            duration_total = ser.sum()
            duration_mean = ser.mean()
            duration_median = ser.median()
            duration_std = ser.std(ddof=0)
            # Calc duration distribution
            arr = np.fromiter(
                df["duration"].dropna().apply(int).reset_index(drop=True).to_list(), int
            )
            hist = np.histogram(arr, bins=c.BINS_DURATION)
            duration_freq = hist[0].tolist()

        # === VOICES (how many recordings per voice)
        voice_counts: pd.DataFrame = (
            df["client_id"].value_counts().dropna().to_frame().reset_index()
        )
        ser = voice_counts["count"]
        voice_mean: float = ser.mean()
        voice_median: float = ser.median()
        voice_std: float = ser.std(ddof=0)
        # Calc speaker recording distribution
        arr = np.fromiter(
            voice_counts["count"].dropna().apply(int).reset_index(drop=True).to_list(),
            int,
        )
        hist = np.histogram(arr, bins=c.BINS_VOICES)
        voice_freq = hist[0].tolist()[1:]

        # === SENTENCES (how many recordings per sentence)
        sentence_counts: pd.DataFrame = (
            df["sentence"].value_counts().dropna().to_frame().reset_index()
        )
        ser = sentence_counts["count"]
        sentence_mean: float = ser.mean()
        sentence_median: float = ser.median()
        sentence_std: float = ser.std(ddof=0)
        # Calc sentence recording distribution
        arr = np.fromiter(
            sentence_counts["count"]
            .dropna()
            .apply(int)
            .reset_index(drop=True)
            .to_list(),
            int,
        )
        hist = np.histogram(arr, bins=c.BINS_SENTENCES)
        sentence_freq = hist[0].tolist()[1:]

        # === VOTES
        bins: list[int] = c.BINS_VOTES_UP
        up_votes_sum: int = df["up_votes"].sum()
        vote_counts_df: pd.DataFrame = (
            df["up_votes"].value_counts().dropna().to_frame().astype(int).reset_index()
        )
        vote_counts_df.rename(columns={"up_votes": "votes"}, inplace=True)

        ser = vote_counts_df["count"]
        up_votes_mean: float = ser.mean()
        up_votes_median: float = ser.median()
        up_votes_std: float = ser.std(ddof=0)
        bin_val: int
        bin_next: int

        up_votes_freq: list[int] = []
        for i in range(0, len(bins) - 1):
            bin_val = bins[i]
            bin_next = bins[i + 1]
            up_votes_freq.append(
                int(
                    vote_counts_df.loc[
                        (vote_counts_df["votes"] >= bin_val)
                        & (vote_counts_df["votes"] < bin_next)
                    ]["count"].sum()
                )
            )

        bins = c.BINS_VOTES_DOWN
        down_votes_sum: int = df["down_votes"].sum()
        vote_counts_df = (
            df["down_votes"]
            .value_counts()
            .dropna()
            .to_frame()
            .astype(int)
            .reset_index()
        )
        vote_counts_df.rename(columns={"down_votes": "votes"}, inplace=True)

        ser = vote_counts_df["count"]
        down_votes_mean: float = ser.mean()
        down_votes_median: float = ser.median()
        down_votes_std: float = ser.std(ddof=0)

        down_votes_freq: list[int] = []
        for i in range(0, len(bins) - 1):
            bin_val = bins[i]
            bin_next = bins[i + 1]
            down_votes_freq.append(
                int(
                    vote_counts_df.loc[
                        (vote_counts_df["votes"] >= bin_val)
                        & (vote_counts_df["votes"] < bin_next)
                    ]["count"].sum()
                )
            )

        # === BASIC MEASURES
        clips_cnt: int = df.shape[0]
        unique_voices: int = df["client_id"].unique().shape[0]
        unique_sentences: int = df["sentence"].unique().shape[0]
        unique_sentences_lower: int = df["sentence_lower"].unique().shape[0]
        # Implement the following in the client:
        # duplicate_sentence_cnt: int = clips_cnt - unique_sentences
        # duplicate_sentence_cnt_lower: int = clips_cnt - unique_sentences_lower

        # === DEMOGRAPHICS

        # Add TOTAL to lists
        pt_ages: list[str] = c.CV_AGES.copy()
        pt_ages.append("TOTAL")
        pt_genders: list[str] = c.CV_GENDERS.copy()
        pt_genders.append("TOTAL")

        # get a pt for all demographics (based on recordings)
        _pt_dem: pd.DataFrame = pd.pivot_table(
            df,
            values="path",
            index=["age"],
            columns=["gender"],
            aggfunc="count",
            fill_value=0,
            dropna=False,
            margins=True,
            margins_name="TOTAL",
        )
        _pt_dem = (
            _pt_dem.reindex(pt_ages, axis=0)
            .reindex(pt_genders, axis=1)
            .fillna(value=0)
            .astype(int)
        )

        # get a pt for all demographics (based on unique voices)
        _df_uqdem: pd.DataFrame = df[["client_id", "age", "gender"]]
        _df_uqdem = _df_uqdem.drop_duplicates().reset_index(drop=True)
        _pt_uqdem: pd.DataFrame = pd.pivot_table(
            _df_uqdem,
            values="client_id",
            index=["age"],
            columns=["gender"],
            aggfunc="count",
            fill_value=0,
            dropna=False,
            margins=True,
            margins_name="TOTAL",
        )
        _pt_uqdem = (
            _pt_uqdem.reindex(pt_ages, axis=0)
            .reindex(pt_genders, axis=1)
            .fillna(value=0)
            .astype(int)
        )

        # Create a table for all demographic info corrections (based on recordings)
        # Correctable ones are: clients with both blank and a single gender (or age) specified
        # dem_fixes_list: list[list[int]] = find_fixes(df_orig)
        dem_fixes_list: list[list[int]] = [[], []]

        rec_ss: SplitStatsRec = SplitStatsRec(
            ver=ver,
            lc=lc,
            alg=algo,
            sp=split,
            clips=clips_cnt,
            uq_v=unique_voices,
            uq_s=unique_sentences,
            uq_sl=unique_sentences_lower,
            # Duration
            dur_total=dec3(duration_total),
            dur_avg=dec3(duration_mean),
            dur_med=dec3(duration_median),
            dur_std=dec3(duration_std),
            dur_freq=duration_freq,
            # Recordings per Voice
            v_avg=dec3(voice_mean),
            v_med=dec3(voice_median),
            v_std=dec3(voice_std),
            v_freq=voice_freq,
            # Recordings per Sentence
            s_avg=dec3(sentence_mean),
            s_med=dec3(sentence_median),
            s_std=dec3(sentence_std),
            s_freq=sentence_freq,
            # Votes
            uv_sum=up_votes_sum,
            uv_avg=dec3(up_votes_mean),
            uv_med=dec3(up_votes_median),
            uv_std=dec3(up_votes_std),
            uv_freq=up_votes_freq,
            dv_sum=down_votes_sum,
            dv_avg=dec3(down_votes_mean),
            dv_med=dec3(down_votes_median),
            dv_std=dec3(down_votes_std),
            dv_freq=down_votes_freq,
            # Demographics distribution for recordings
            dem_table=_pt_dem.to_numpy(int).tolist(),
            dem_uq=_pt_uqdem.to_numpy(int).tolist(),
            dem_fix_r=dem_fixes_list[0],
            dem_fix_v=dem_fixes_list[1],
        )

        # === AVERAGE AND PER USER CHAR SPEED

        rec_cs: CharSpeedRec = CharSpeedRec(
            ver=ver,
            lc=lc,
            alg=algo,
            sp=split,
        )

        if duration_total == -1 or clips_cnt == 0:
            df["char_speed"] = pd.NA
        else:
            df["s_len"] = (
                df["sentence"]
                .astype(str)
                .apply(lambda x: len(x) if pd.notna(x) else pd.NA)
            )
            avg_slen: float = df["s_len"].dropna().aggregate("mean")
            df = df.assign(
                char_speed=lambda x: round(1000 * (x["duration"] / x["s_len"]))
            )

            # calc general stats from real values
            ser = df["char_speed"]
            cs_mean: float = ser.mean()
            cs_median: float = ser.median()
            cs_std: float = ser.std(ddof=0)
            # decide which bin(s) should be used
            _cs_bins: list[int] = (
                c.BINS_CS_LOW if cs_mean < c.CS_BIN_THRESHOLD else c.BINS_CS_HIGH
            )
            _sl_bins: list[int] = (
                c.BINS_CHARS_SHORT
                if avg_slen < c.CHARS_BIN_THRESHOLD
                else c.BINS_CHARS_LONG
            )

            # add temp pre-calc bin cols
            df["bin_cs"] = pd.cut(
                df["char_speed"],
                bins=_cs_bins,
                right=False,
                labels=_cs_bins[:-1],
            )
            df["bin_slen"] = pd.cut(
                df["s_len"],
                bins=_sl_bins,
                right=False,
                labels=_sl_bins[:-1],
            )
            # simulated pivot
            df_char_speed: pd.DataFrame = (
                df["bin_cs"]
                .value_counts()
                .dropna()
                .to_frame()
                .astype(int)
                .reindex(index=_cs_bins, fill_value=0)
                .reset_index(drop=False)
                .sort_values("bin_cs")
            )
            cs_freq: list[int] = df_char_speed["count"].tolist()[:-1]

            #
            # Crosstab Calculations
            #

            # row_labels: list = c.BINS_CS[:-1]
            # row_labels.append("TOTAL")
            col_labels: list

            # char_speed versus sentence length (using bins)
            col_labels = _sl_bins[:-1]
            # col_labels.append("TOTAL")
            cs_vs_slen: pd.DataFrame = pd.crosstab(
                index=df["bin_cs"],
                columns=df["bin_slen"],
            )
            cs_row_labels: list[str] = cs_vs_slen.index.values.astype(str).tolist()
            cs_vs_slen_col_labels: list[str] = cs_vs_slen.columns.tolist()
            cs_vs_slen.reset_index(drop=True, inplace=True)
            cs_clips: int = cs_vs_slen.sum(skipna=True).sum()

            #
            # char_speed versus gender
            #
            col_labels = c.CV_GENDERS
            # col_labels.append("TOTAL")
            cs_vs_gender: pd.DataFrame = (
                pd.crosstab(
                    index=df["bin_cs"],
                    columns=df["gender"],
                )
                .reindex(columns=col_labels, fill_value=0)
                .reset_index(drop=True)
            )

            #
            # char_speed versus age group
            #
            col_labels = c.CV_AGES
            # col_labels.append("TOTAL")
            cs_vs_age: pd.DataFrame = (
                pd.crosstab(
                    index=df["bin_cs"],
                    columns=df["age"],
                )
                .reindex(columns=col_labels, fill_value=0)
                .reset_index(drop=True)
            )

            rec_cs = CharSpeedRec(
                ver=ver,
                lc=lc,
                alg=algo,
                sp=split,
                clips=cs_clips,
                # Character Speed
                cs_avg=dec3(cs_mean),
                cs_med=dec3(cs_median),
                cs_std=dec3(cs_std),
                cs_freq=cs_freq,
                cs_r=list2str(cs_row_labels),
                cs2s_c=list2str(cs_vs_slen_col_labels),
                cs2s=arr2str(cs_vs_slen.to_numpy(int).tolist()),
                cs2g=arr2str(cs_vs_gender.to_numpy(int).tolist()),
                cs2a=arr2str(cs_vs_age.to_numpy(int).tolist()),
            )

        #
        # Audio Specs Stats
        #
        df_aspecs_split: pd.DataFrame = pd.DataFrame()
        rec_as: AudioAnalysisStatsRec = AudioAnalysisStatsRec(
            ver=ver,
            lc=lc,
            alg=algo,
            sp=split,
        )

        # we cannot process v1
        if ver != "1":
            if df_aspecs_ds is not None:
                # get and pass a subset of audio specs for this split
                path_list = df_orig["path"].to_list()
                df_aspecs_split = df_aspecs_ds[
                    df_aspecs_ds["orig_path"].isin(path_list)
                ]

            rec_as = handle_split_audio_stats(
                ver=ver,
                lc=lc,
                algo=algo,
                split=split,
                df_aspecs_sub=df_aspecs_split,
            )

        return (rec_ss, rec_cs, rec_as)

    # END handle_split

    # --------------------------------------------
    # START main process for a single CV dataset
    # --------------------------------------------
    # we have input ds_path in format: # ...\data\voice-corpus\cv-corpus-12.0-2022-12-07\tr
    # <ver> <lc> [<algo>]

    # Source directories
    cv_dir_name: str = calc_dataset_prefix(params.ver)
    ds_ver_dir: str = os.path.join(conf.DATA_BASE_DIR, c.VC_DIRNAME, cv_dir_name)
    ds_meta_dir: str = os.path.join(conf.TBOX_META_DIR, "cv", cv_dir_name, params.lc)
    ds_ver_lc_dir: str = os.path.join(ds_ver_dir, params.lc)
    cd_dir: str = os.path.join(conf.DATA_BASE_DIR, c.CD_DIRNAME, params.lc)

    # Calc destinations if they do not exist
    tsv_path: str = os.path.join(
        conf.DATA_BASE_DIR, c.RES_DIRNAME, c.TSV_DIRNAME, params.lc
    )
    json_path: str = os.path.join(
        conf.DATA_BASE_DIR, c.RES_DIRNAME, c.JSON_DIRNAME, params.lc
    )

    # Any clip-error files from TBOX?
    df_clip_errors: pd.DataFrame = pd.DataFrame()
    if params.df_clip_errors is not None:
        df_clip_errors = params.df_clip_errors[
            (params.df_clip_errors["ds"] == "cv")
            & (params.df_clip_errors["lc"] == params.lc)
            & (params.df_clip_errors["ver"] == params.ver)
        ][["path", "source"]]

    # First Handle Splits in voice-corpus
    # Load general DF's if they exist, else initialize

    # === Clip Durations
    # cd_file: str = os.path.join(cd_dir, '$clip_durations.tsv')
    cd_file: str = os.path.join(cd_dir, "clip_durations.tsv")
    df_clip_durations: pd.DataFrame = pd.DataFrame(
        columns=c.FIELDS_CLIP_DURATIONS
    ).set_index("clip")
    if os.path.isfile(cd_file):
        df_clip_durations = df_read(cd_file).set_index("clip")
    elif conf.VERBOSE:
        print(f"WARNING: No duration data for {params.lc}\n")

    # === Clips Audio Specs Subset if any
    df_aspecs_ds: Optional[pd.DataFrame] = None
    if params.df_aspecs is not None:
        df_aspecs_ds = params.df_aspecs[
            (params.df_aspecs["ds"] == "cv")
            & (params.df_aspecs["lc"] == params.lc)
            & (params.df_aspecs["ver"] <= float(params.ver))  # compare as float
        ]
        # write-out audio_specs for to TBOX meta per ver-lc
        if df_aspecs_ds.shape[0] > 0:
            _dst_meta_as_file: str = os.path.join(ds_meta_dir, "audio_specs.tsv")
            if not os.path.isfile(_dst_meta_as_file):
                os.makedirs(ds_meta_dir, exist_ok=True)
                df_write(df_aspecs_ds, _dst_meta_as_file)

    # === MAIN BUCKETS (clips, validated, invalidated, other)
    ret_ss: SplitStatsRec
    ret_cs: CharSpeedRec
    ret_as: AudioAnalysisStatsRec
    res_ss: list[SplitStatsRec] = []  # Init the result list
    res_cs: list[CharSpeedRec] = []  # Init the result list
    res_as: list[AudioAnalysisStatsRec] = []  # Init the result list

    # Special case for temporary "clips.tsv"
    ret_ss, ret_cs, ret_as = handle_split(
        ver=params.ver,
        lc=params.lc,
        algo="",
        split="clips",
        src_ds_dir=ds_ver_lc_dir,
    )
    if ret_ss.clips > 0:
        res_ss.append(ret_ss)
        res_cs.append(ret_cs)
        res_as.append(ret_as)

    # Append to clips.tsv at the source, at the base of that version
    # (it will include all recording data for all locales to be used in CC & alternatives)
    # for sp in c.MAIN_BUCKETS:
    #     src: str = os.path.join(ds_ver_lc_dir, sp + ".tsv")
    #     dst: str = os.path.join(ds_ver_dir, "clips.tsv")
    #     df_write(df_read(src), fpath=dst, mode="a")
    validated_records: int = 0
    for sp in c.MAIN_BUCKETS:
        ret_ss, ret_cs, ret_as = handle_split(
            ver=params.ver,
            lc=params.lc,
            algo="",
            split=sp,
            src_ds_dir=ds_ver_lc_dir,
        )
        if sp == "validated":
            validated_records = ret_ss.clips
        if ret_ss.clips > 0:
            res_ss.append(ret_ss)
            res_cs.append(ret_cs)
            res_as.append(ret_as)

    # SPLITTING ALGO SPECIFIC (inc default splits)

    # If no record in validated, do not try further
    if validated_records > 0:
        for algo in c.ALGORITHMS:
            for sp in c.TRAINING_SPLITS:
                src_algo_dir: str = os.path.join(ds_ver_lc_dir, algo)
                if os.path.isdir(src_algo_dir):
                    ret_ss, ret_cs, ret_as = handle_split(
                        ver=params.ver,
                        lc=params.lc,
                        algo=algo,
                        split=sp,
                        src_ds_dir=src_algo_dir,
                    )
                    if ret_ss.clips > 0:
                        res_ss.append(ret_ss)
                        res_cs.append(ret_cs)
                        res_as.append(ret_as)

    # Create DataFrames
    df: pd.DataFrame = pd.DataFrame(res_ss)
    df_write(df, os.path.join(tsv_path, f"{params.lc}_{params.ver}_splits.tsv"))
    df.to_json(
        os.path.join(json_path, f"{params.lc}_{params.ver}_splits.json"),
        orient="table",
        index=False,
    )

    df = pd.DataFrame(res_cs)
    df_write(df, os.path.join(tsv_path, f"{params.lc}_{params.ver}_cs.tsv"))
    df.to_json(
        os.path.join(json_path, f"{params.lc}_{params.ver}_cs.json"),
        orient="table",
        index=False,
    )

    df = pd.DataFrame(res_as)
    df_write(df, os.path.join(tsv_path, f"{params.lc}_{params.ver}_aa.tsv"))
    df.to_json(
        os.path.join(json_path, f"{params.lc}_{params.ver}_aa.json"),
        orient="table",
        index=False,
    )

    gc.collect()

    return len(res_ss)


# END - Dataset Split Stats (MP Handler)
