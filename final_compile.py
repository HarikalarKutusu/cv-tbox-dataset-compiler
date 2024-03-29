#!/usr/bin/env python3
"""cv-tbox Dataset Compiler - Final Compilation Phase"""
###########################################################################
# final_compile.py
#
# From all data, compile result statistics data to be used in
# cv-tbox-dataset-analyzer
#
# Use:
# python final_compile.py
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-tbox-dataset-compiler
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################

# Standard Lib
import os
import sys
import glob
import multiprocessing as mp
from collections import Counter
from datetime import datetime

# External dependencies
from tqdm import tqdm
import numpy as np
import pandas as pd
import psutil
import cvutils as cvu

# Module
import const as c
import conf
from typedef import (
    Globals,
    ConfigRec,
    TextCorpusStatsRec,
    ReportedStatsRec,
    SplitStatsRec,
)
from lib import (
    df_read,
    df_write,
    init_directories,
    # list2str,
    # arr2str,
    dec3,
    calc_dataset_prefix,
    get_locales_from_cv_dataset,
    report_results,
)

# Globals

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

# PROC_COUNT: int = psutil.cpu_count(logical=False) - 1     # Limited usage
PROC_COUNT: int = psutil.cpu_count(logical=True)  # Full usage
MAX_BATCH_SIZE: int = 5

ALL_LOCALES: list[str] = get_locales_from_cv_dataset(c.CV_VERSIONS[-1])

cv: cvu.CV = cvu.CV()
VALIDATORS: list[str] = cv.validators()
PHONEMISERS: list[str] = cv.phonemisers()
# ALPHABETS: list[str] = [str(p).split(os.sep)[-2] for p in cv.alphabets()]
# SEGMENTERS: list[str] = [str(p).split(os.sep)[-2] for p in cv.segmenters()]

g: Globals = Globals(
    total_ver=len(c.CV_VERSIONS),
    total_algo=len(c.ALGORITHMS),
)
g_tc: Globals = Globals(total_ver=len(c.CV_VERSIONS))
g_rep: Globals = Globals(total_ver=len(c.CV_VERSIONS))
g_vc: Globals = Globals(
    total_ver=len(c.CV_VERSIONS),
    total_algo=len(c.ALGORITHMS),
)

########################################################
# Text-Corpus Stats (Multi Processing Handler)
########################################################


def handle_text_corpus(ver_lc: str) -> list[TextCorpusStatsRec]:
    """Multi-Process text-corpus for a single locale"""

    ver: str = ver_lc.split("|")[0]
    lc: str = ver_lc.split("|")[1]
    ver_dir: str = calc_dataset_prefix(ver)

    tc_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.TC_DIRNAME, ver_dir, lc)
    tc_file: str = os.path.join(tc_dir, f"{c.TEXT_CORPUS_FN}.tsv")

    # cvu - do we have them?
    validator: cvu.Validator | None = cvu.Validator(lc) if lc in VALIDATORS else None
    phonemiser: cvu.Phonemiser | None = (
        cvu.Phonemiser(lc) if lc in PHONEMISERS else None
    )
    tokeniser: cvu.Tokeniser = cvu.Tokeniser(lc)

    results: list[TextCorpusStatsRec] = []

    def handle_df(
        df: pd.DataFrame, algo: str = "", sp: str = ""
    ) -> TextCorpusStatsRec | None:
        """Calculate stats given a dataframe containing only sentences"""

        if df.shape[0] == 0:
            if conf.VERBOSE:
                print(f"WARN: Skipping empty data for: {ver} - {lc} - {algo} - {sp}")
            return None

        # prep result record with default
        res: TextCorpusStatsRec = TextCorpusStatsRec(
            ver=ver,
            lc=lc,
            algo=algo,
            sp=sp,
            has_val=lc in VALIDATORS,
            has_phon=lc in PHONEMISERS,
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

        # add columns
        _df: pd.DataFrame = (
            df.reindex(
                columns=[
                    "sentence",
                    "normalized",  # normalized sentence
                    "phonemised",  # phonemised sentence
                    "tokens",  # list of tokens
                    "char_cnt",  # number of characters (graphemes)
                    "word_cnt",  # number of words
                    "valid",  # is it a valid sentence according to commonvoice-utils? 1=valid
                ]
            )
            .copy()
            .reset_index(drop=True)
        )

        # pre-calc simpler values
        _df["char_cnt"] = [
            len(s) if isinstance(s, str) else 0 for s in _df["sentence"].to_list()
        ]

        # validator dependent
        if validator:
            _df["normalized"] = [
                validator.validate(s) if isinstance(s, str) else None
                for s in _df["sentence"].tolist()
            ]
            _df["valid"] = [0 if n is None else 1 for n in _df["normalized"].tolist()]
            _df["tokens"] = [
                None if s is None else tokeniser.tokenise(s)
                for s in _df["normalized"].tolist()
            ]
            _df["word_cnt"] = [
                None if ww is None else len(ww) for ww in _df["tokens"].tolist()
            ]
            _ = [token_counter.update(ww) for ww in _df["tokens"].dropna().tolist()]

            # word_cnt stats
            _ser = _df["word_cnt"].dropna()
            res.w_sum = _ser.sum()
            res.w_avg = dec3(_ser.mean())
            res.w_med = _ser.median()
            res.w_std = dec3(_ser.std(ddof=0))
            # Calc word count distribution
            _arr: np.ndarray = np.fromiter(
                _ser.apply(int).reset_index(drop=True).to_list(), int
            )
            _hist = np.histogram(_arr, bins=c.BINS_WORDS)
            res.w_freq = _hist[0].tolist()

            # token_cnt stats
            _df2 = pd.DataFrame(token_counter.most_common(), columns=c.COLS_TOKENS)
            # "token", "count"
            res.t_sum = _df2.shape[0]
            _ser = _df2["count"].dropna()
            res.t_avg = dec3(_ser.mean())
            res.t_med = _ser.median()
            res.t_std = dec3(_ser.std(ddof=0))
            # Token/word repeat distribution
            _arr: np.ndarray = np.fromiter(
                _df2["count"].dropna().apply(int).reset_index(drop=True).to_list(), int
            )
            _hist = np.histogram(_arr, bins=c.BINS_TOKENS)
            res.t_freq = _hist[0].tolist()
            if do_save:
                fn: str = os.path.join(
                    tc_dir,
                    f"{c.TOKENS_FN}_{algo}_{sp}.tsv".replace("__", "_").replace(
                        "_.", "."
                    ),
                )
                df_write(_df2, fn)

        # phonemiser dependent
        if phonemiser:
            _df["phonemised"] = [
                phonemiser.phonemise(s) if isinstance(s, str) else None
                for s in _df["sentence"].tolist()
                # for w in str(s).split(" ")
            ]
            _ = [phoneme_counter.update(p) for p in _df["phonemised"].dropna().tolist()]

            # PHONEMES
            _df2 = pd.DataFrame(phoneme_counter.most_common(), columns=c.COLS_PHONEMES)
            _values = _df2.values.tolist()
            res.p_cnt = len(_values)
            # res.p_freq = arr2str(_values)
            res.p_freq = _values
            if do_save:
                fn: str = os.path.join(
                    tc_dir,
                    f"{c.PHONEMES_FN}_{algo}_{sp}.tsv".replace("__", "_").replace(
                        "_.", "."
                    ),
                )
                df_write(_df2, fn)

        # simpler values which are independent
        res.s_cnt = _df.shape[0]
        res.val = _df["valid"].dropna().astype(int).sum()
        res.uq_s = _df["sentence"].dropna().unique().shape[0]
        res.uq_n = _df["normalized"].dropna().unique().shape[0]

        # char_cnt stats
        _ser = _df["char_cnt"].dropna()
        res.c_sum = _ser.sum()
        res.c_avg = dec3(_ser.mean())
        res.c_med = _ser.median()
        res.c_std = dec3(_ser.std(ddof=0))
        # Calc character length distribution
        _arr: np.ndarray = np.fromiter(
            _ser.apply(int).reset_index(drop=True).to_list(), int
        )
        _hist = np.histogram(_arr, bins=c.BINS_CHARS)
        res.c_freq = _hist[0].tolist()

        # GRAPHEMES
        _ = [grapheme_counter.update(s) for s in _df["sentence"].dropna().tolist()]
        _df2 = pd.DataFrame(grapheme_counter.most_common(), columns=c.COLS_GRAPHEMES)
        _values = _df2.values.tolist()
        res.g_cnt = len(_values)
        res.g_freq = _df2.values.tolist()
        if do_save:
            fn: str = os.path.join(
                tc_dir,
                f"{c.GRAPHEMES_FN}_{algo}_{sp}.tsv".replace("__", "_").replace(
                    "_.", "."
                ),
            )
            df_write(_df2, fn)
        # return result
        return res

    def handle_tc_global() -> None:
        """Calculate stats using the whole text corpus from server/data"""
        if not os.path.isfile(tc_file):
            if conf.VERBOSE:
                print(f"WARN: No text-corpus file for: {ver} - {lc}")
            return
        res: TextCorpusStatsRec | None = handle_df(
            df_read(tc_file).reset_index(drop=True)[["sentence"]]
        )
        if res is not None:
            results.append(res)

    def handle_tc_split(sp: str, algo: str = "") -> None:
        """Calculate stats using sentence data in a bucket/split"""
        fn: str = os.path.join(conf.SRC_BASE_DIR, algo, ver_dir, lc, f"{sp}.tsv")
        if not os.path.isfile(fn):
            if conf.VERBOSE:
                print(f"WARN: No such split file for: {ver} - {lc} - {algo} - {sp}")
            return
        res: TextCorpusStatsRec | None = handle_df(
            df_read(fn).reset_index(drop=True)[["sentence"]], algo=algo, sp=sp
        )
        if res is not None:
            results.append(res)

    # main
    handle_tc_global()
    for sp in ["validated", "invalidated", "other"]:
        handle_tc_split(sp, c.ALGORITHMS[0])

    for algo in c.ALGORITHMS:
        for sp in ["train", "dev", "test"]:
            handle_tc_split(sp, algo)
    # done
    return results


########################################################
# Reported Stats
########################################################


def handle_reported(ver_lc: str) -> ReportedStatsRec:
    """Process text-corpus for a single locale"""

    ver: str = ver_lc.split("|")[0]
    lc: str = ver_lc.split("|")[1]

    ver_dir: str = calc_dataset_prefix(ver)

    # Calc voice-corpus directory
    rep_file: str = os.path.join(
        HERE, c.DATA_DIRNAME, c.VC_DIRNAME, ver_dir, lc, "reported.tsv"
    )

    # skip process if no such file or there can be empty files :/
    if not os.path.isfile(rep_file) or os.path.getsize(rep_file) == 0:
        return ReportedStatsRec(ver=ver, lc=lc)
    # read file in - Columns: sentence sentence_id locale reason
    df: pd.DataFrame = df_read(rep_file)
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
    rep_freq = hist[0].tolist()

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


########################################################
# Dataset Split Stats (MP Handler)
########################################################


def handle_dataset_splits(ds_path: str) -> list[SplitStatsRec]:
    """Handle a single dataset (ver/lc)"""
    # Handle one split, this is where calculations happen
    # The default column structure of CV dataset splits is as follows [FIXME] variants?
    # client_id, path, sentence, up_votes, down_votes, age, gender, accents, locale, segment
    # we have as input:
    # 'version', 'locale', 'algorithm', 'split'

    # now, do calculate some statistics...
    def handle_split(
        ver: str, lc: str, algorithm: str, split: str, fpath: str
    ) -> SplitStatsRec:
        """Processes a single split and return calculated values"""

        nonlocal df_clip_durations

        # find_fixes
        def find_fixes(df_split: pd.DataFrame) -> list[list[int]]:
            """Finds fixable demographic info from the split and returns a string"""

            # df is local dataframe which will keep records
            # only necessary columns with some additional columns
            df: pd.DataFrame = df_split.copy().reset_index(drop=True)
            df["v_enum"], _ = pd.factorize(
                df["client_id"]
            )  # add an enumaration column for client_id's, more memory efficient
            df["p_enum"], _ = pd.factorize(
                df["path"]
            )  # add an enumaration column for recordings, more memory efficient
            df = (
                df[["v_enum", "age", "gender", "p_enum"]]
                .fillna(c.NODATA)
                .reset_index(drop=True)
            )

            # prepare empty results
            fixes: pd.DataFrame = pd.DataFrame(columns=df.columns).reset_index(
                drop=True
            )
            dem_fixes_recs: list[int] = []
            dem_fixes_voices: list[int] = []

            # get unique voices with multiple demographic values
            df_counts: pd.DataFrame = (
                df[["v_enum", "age", "gender"]]
                .drop_duplicates()
                .copy()
                .groupby("v_enum")
                .agg({"age": "count", "gender": "count"})
            )
            df_counts.reset_index(inplace=True)
            df_counts = df_counts[
                (df_counts["age"].astype(int) == 2)
                | (df_counts["gender"].astype(int) == 2)
            ]  # reduce that to only processible ones
            v_processable: list[int] = df_counts["v_enum"].unique().tolist()

            # now, work only on problem voices & records.
            # For each voice, get related records and decide
            for v in v_processable:
                recs: pd.DataFrame = df[df["v_enum"] == v].copy()
                recs_blanks: pd.DataFrame = recs[
                    (recs["gender"] == c.NODATA) | (recs["age"] == c.NODATA)
                ].copy()  # get full blanks
                # gender
                recs_w_gender: pd.DataFrame = recs[~(recs["gender"] == c.NODATA)].copy()
                if recs_w_gender.shape[0] > 0:
                    val: str = recs_w_gender["gender"].tolist()[0]
                    recs_blanks.loc[:, "gender"] = val
                # age
                recs_w_age: pd.DataFrame = recs[~(recs["age"] == c.NODATA)].copy()
                if recs_w_age.shape[0] > 0:
                    val: str = recs_w_age["age"].tolist()[0]
                    recs_blanks.loc[:, "age"] = val
                # now we can add them to the result fixed list
                fixes = pd.concat([fixes.loc[:], recs_blanks]).reset_index(drop=True)

            # Here, we have a df maybe with records of possible changes
            if fixes.shape[0] > 0:
                # records
                pt: pd.DataFrame = pd.pivot_table(
                    fixes,
                    values="p_enum",
                    index=["age"],
                    columns=["gender"],
                    aggfunc="count",
                    fill_value=0,
                    dropna=False,
                    margins=False,
                )
                # get only value parts : nodata is just negative sum of these, and TOTAL will be 0,
                # so we drop them for file size and leave computation to the client
                pt = (
                    pt.reindex(c.CV_AGES, axis=0)
                    .reindex(c.CV_GENDERS, axis=1)
                    .fillna(value=0)
                    .astype(int)
                    .drop(c.NODATA, axis=0)
                    .drop(c.NODATA, axis=1)
                )
                dem_fixes_recs = pt.to_numpy(int).tolist()

                # voices
                fixes = fixes.drop("p_enum", axis=1).drop_duplicates()
                pt: pd.DataFrame = pd.pivot_table(
                    fixes,
                    values="v_enum",
                    index=["age"],
                    columns=["gender"],
                    aggfunc="count",
                    fill_value=0,
                    dropna=False,
                    margins=False,
                )
                # get only value parts : nodata is just -sum of these, sum will be 0
                pt = (
                    pt.reindex(c.CV_AGES, axis=0)
                    .reindex(c.CV_GENDERS, axis=1)
                    .fillna(value=0)
                    .astype(int)
                    .drop(c.NODATA, axis=0)
                    .drop(c.NODATA, axis=1)
                )
                dem_fixes_voices = pt.to_numpy(int).tolist()

            return [dem_fixes_recs, dem_fixes_voices]

        # END - find_fixes

        #
        # === START ===
        #

        # Read in DataFrames
        if split != "clips":
            df_orig: pd.DataFrame = df_read(fpath)
        else:  # build "clips" from val+inval+other
            df_orig: pd.DataFrame = df_read(
                fpath
            )  # we passed validated here, first read it.
            df2: pd.DataFrame = df_read(
                fpath.replace("validated", "invalidated")
            )  # add invalidated
            df_orig = pd.concat([df_orig.loc[:], df2])
            df2: pd.DataFrame = df_read(
                fpath.replace("validated", "other")
            )  # add other
            df_orig = pd.concat([df_orig.loc[:], df2])

        # default result values
        res: SplitStatsRec = SplitStatsRec(ver=ver, lc=lc, alg=algorithm, sp=split)

        # Do nothing, if there is no data
        if df_orig.shape[0] == 0:
            return res

        # Replace NA with NODATA
        df: pd.DataFrame = df_orig.fillna(value=c.NODATA)
        # add lowercase sentence column
        df["sentence_lower"] = df["sentence"].str.lower()

        # === DURATIONS: Calc duration agregate values
        if (
            df_clip_durations.shape[0] > 0 and ver != "1"
        ):  # there must be records + v1 cannot be mapped
            # Connect with duration table
            df["duration"] = df["path"].map(
                df_clip_durations["duration[ms]"] / 1000, na_action="ignore"
            )  # convert to seconds
            ser: pd.Series = df["duration"].dropna()
            duration_total: float = ser.sum()
            duration_mean: float = ser.mean()
            duration_median: float = ser.median()
            duration_std: float = ser.std(ddof=0)
            # Calc duration distribution
            arr: np.ndarray = np.fromiter(
                df["duration"].dropna().apply(int).reset_index(drop=True).to_list(), int
            )
            hist = np.histogram(arr, bins=c.BINS_DURATION)
            duration_freq = hist[0].tolist()
        else:  # No Duration data, set illegal defaults and continue
            duration_total: float = -1
            duration_mean: float = -1
            duration_median: float = -1
            duration_std: float = -1
            duration_freq = []

        # === VOICES
        voice_counts: pd.DataFrame = (
            df["client_id"].value_counts().dropna().to_frame().reset_index()
        )
        ser = voice_counts["count"]
        voice_mean: float = ser.mean()
        voice_median: float = ser.median()
        voice_std: float = ser.std(ddof=0)
        # Calc speaker recording distribution
        arr: np.ndarray = np.fromiter(
            voice_counts["count"].dropna().apply(int).reset_index(drop=True).to_list(),
            int,
        )
        hist = np.histogram(arr, bins=c.BINS_VOICES)
        voice_freq = hist[0].tolist()

        # === SENTENCES
        sentence_counts: pd.DataFrame = (
            df["sentence"].value_counts().dropna().to_frame().reset_index()
        )
        ser = sentence_counts["count"]
        sentence_mean: float = ser.mean()
        sentence_median: float = ser.median()
        sentence_std: float = ser.std(ddof=0)
        # Calc speaker recording distribution
        arr: np.ndarray = np.fromiter(
            sentence_counts["count"]
            .dropna()
            .apply(int)
            .reset_index(drop=True)
            .to_list(),
            int,
        )
        hist = np.histogram(arr, bins=c.BINS_SENTENCES)
        sentence_freq = hist[0].tolist()

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

        up_votes_freq: list[int] = []
        for i in range(0, len(bins) - 1):
            bin_val: int = bins[i]
            bin_next: int = bins[i + 1]
            up_votes_freq.append(
                vote_counts_df.loc[
                    (vote_counts_df["votes"] >= bin_val)
                    & (vote_counts_df["votes"] < bin_next)
                ]["count"].sum()
            )

        bins: list[int] = c.BINS_VOTES_DOWN
        down_votes_sum: int = df["down_votes"].sum()
        vote_counts_df: pd.DataFrame = (
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
            bin_val: int = bins[i]
            bin_next: int = bins[i + 1]
            down_votes_freq.append(
                vote_counts_df.loc[
                    (vote_counts_df["votes"] >= bin_val)
                    & (vote_counts_df["votes"] < bin_next)
                ]["count"].sum()
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
        dem_fixes_list: list[list[int]] = find_fixes(df_orig)

        res: SplitStatsRec = SplitStatsRec(
            ver=ver,
            lc=lc,
            alg=algorithm,
            sp=split,
            clips=clips_cnt,
            uq_v=unique_voices,
            uq_s=unique_sentences,
            uq_sl=unique_sentences_lower,
            # Duration
            dur_total=dec3(duration_total),
            dur_avg=dec3(duration_mean),
            dur_med=duration_median,
            dur_std=dec3(duration_std),
            dur_freq=duration_freq,
            # Recordings per Voice
            v_avg=dec3(voice_mean),
            v_med=voice_median,
            v_std=dec3(voice_std),
            v_freq=voice_freq,
            # Recordings per Sentence
            s_avg=dec3(sentence_mean),
            s_med=sentence_median,
            s_std=dec3(sentence_std),
            s_freq=sentence_freq,
            # Votes
            uv_sum=up_votes_sum,
            uv_avg=dec3(up_votes_mean),
            uv_med=up_votes_median,
            uv_std=dec3(up_votes_std),
            uv_freq=up_votes_freq,
            dv_sum=down_votes_sum,
            dv_avg=dec3(down_votes_mean),
            dv_med=down_votes_median,
            dv_std=dec3(down_votes_std),
            dv_freq=down_votes_freq,
            # Demographics distribution for recordings
            dem_table=_pt_dem.to_numpy(int).tolist(),
            dem_uq=_pt_uqdem.to_numpy(int).tolist(),
            dem_fix_r=dem_fixes_list[0],
            dem_fix_v=dem_fixes_list[1],
        )

        return res

    # END handle_split

    # --------------------------------------------
    # START main process for a single CV dataset
    # --------------------------------------------
    # we have input ds_path in format: # ...\data\voice-corpus\cv-corpus-12.0-2022-12-07\tr
    # <ver> <lc> [<algo>]

    lc: str = os.path.split(ds_path)[1]
    cv_dir: str = os.path.split(ds_path)[0]
    cv_dir_name: str = os.path.split(cv_dir)[1]
    # extract version info
    ver: str = cv_dir_name.split("-")[2]

    # Source directories
    cd_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.CD_DIRNAME, lc)

    # Create destinations if thet do not exist
    tsv_path: str = os.path.join(HERE, c.DATA_DIRNAME, c.RES_DIRNAME, c.TSV_DIRNAME, lc)
    json_path: str = os.path.join(
        HERE, c.DATA_DIRNAME, c.RES_DIRNAME, c.JSON_DIRNAME, lc
    )

    # First Handle Splits in voice-corpus
    # Load general DF's if they exist, else initialize

    # === Clip Durations
    # cd_file: str = os.path.join(cd_dir, '$clip_durations.tsv')
    cd_file: str = os.path.join(cd_dir, "clip_durations.tsv")
    df_clip_durations: pd.DataFrame = pd.DataFrame(
        columns=c.COLS_CLIP_DURATIONS
    ).set_index("clip")
    if os.path.isfile(cd_file):
        df_clip_durations = df_read(cd_file).set_index("clip")
    elif conf.VERBOSE:
        print(f"WARNING: No duration data for {lc}\n")

    # === MAIN BUCKETS (clips, validated, invalidated, other)
    res: list[SplitStatsRec] = []  # Init the result list

    # Special case for temporary "clips.tsv"
    res.append(
        handle_split(
            ver=ver,
            lc=lc,
            algorithm="",
            split="clips",
            fpath=os.path.join(
                ds_path, "validated.tsv"
            ),  # to start with we set validated
        )
    )
    validated_result: SplitStatsRec = res[-1]
    validated_records: int = validated_result.clips
    # Append to clips.tsv at the source, at the base of that version
    # (it will include all recording data for all locales to be used in CC & alternatives)
    for sp in c.MAIN_BUCKETS:
        src: str = os.path.join(ds_path, sp + ".tsv")
        dst: str = os.path.join(cv_dir, "clips.tsv")
        df_write(df_read(src), fpath=dst, mode="a")

    for sp in c.MAIN_BUCKETS:
        res.append(
            handle_split(
                ver=ver,
                lc=lc,
                algorithm="",
                split=sp,
                fpath=os.path.join(ds_path, sp + ".tsv"),
            )
        )

    # If no record in validated, do not try further
    if validated_records == 0:
        return res

    # SPLITTING ALGO SPECIFIC (inc default splits)

    for algo in c.ALGORITHMS:
        for sp in c.TRAINING_SPLITS:
            if os.path.isfile(os.path.join(ds_path, algo, sp + ".tsv")):
                res.append(
                    handle_split(
                        ver=ver,
                        lc=lc,
                        algorithm=algo,
                        split=sp,
                        fpath=os.path.join(ds_path, algo, sp + ".tsv"),
                    )
                )

    # Create DataFrames
    df: pd.DataFrame = pd.DataFrame(res)
    df_write(df, os.path.join(tsv_path, f"{lc}_{ver}_splits.tsv"))
    df.to_json(
        os.path.join(json_path, f"{lc}_{ver}_splits.json"), orient="table", index=False
    )

    return res


########################################################
# MAIN PROCESS
########################################################


def main() -> None:
    """Compile all data by calculating stats"""

    dst_json_base: str = os.path.join(
        HERE, c.DATA_DIRNAME, c.RES_DIRNAME, c.JSON_DIRNAME
    )
    dst_tsv_base: str = os.path.join(HERE, c.DATA_DIRNAME, c.RES_DIRNAME, c.TSV_DIRNAME)

    def ver2vercol(ver: str) -> str:
        """Converts a data version in format '11.0' to column/variable name format 'v11_0'"""
        return "v" + ver.replace(".", "_")

    #
    # TEXT-CORPORA
    #
    def main_text_corpora() -> None:
        """Handle all text corpora"""
        nonlocal used_proc_count

        results: list[TextCorpusStatsRec] = []

        def save_results() -> pd.DataFrame:
            """Temporarily or finally save the returned results"""
            df: pd.DataFrame = pd.DataFrame(
                results, columns=c.COLS_TC_STATS
            ).reset_index(drop=True)
            df.sort_values(["lc", "ver"], inplace=True)
            # Write out combined (TSV only to use later for above existence checks)
            df_write(df, os.path.join(dst_tsv_base, f"${c.TEXT_CORPUS_STATS_FN}.tsv"))
            return df

        print("\n=== Start Text Corpora Analysis ===")

        tc_base: str = os.path.join(HERE, c.DATA_DIRNAME, c.TC_DIRNAME)
        combined_tsv_file: str = os.path.join(
            dst_tsv_base, f"${c.TEXT_CORPUS_STATS_FN}.tsv"
        )
        # Get joined TSV
        combined_df: pd.DataFrame = pd.DataFrame(columns=c.COLS_TC_STATS)
        if os.path.isfile(combined_tsv_file):
            combined_df = df_read(combined_tsv_file).reset_index(drop=True)
        combined_df = combined_df[["ver", "lc"]]
        combined_ver_lc: list[str] = [
            "|".join(row) for row in combined_df.values.tolist()
        ]
        del combined_df
        combined_df = pd.DataFrame()

        ver_lc_list: list[str] = []  # final
        # start with newer, thus larger / longer versions' data
        versions: list[str] = c.CV_VERSIONS.copy()
        versions.reverse()
        # For each version
        for ver in versions:
            ver_dir: str = calc_dataset_prefix(ver)

            # get all possible
            lc_list: list[str] = get_locales_from_cv_dataset(ver)
            g_tc.total_lc += len(lc_list)

            # remove already calculated ones
            if conf.FORCE_CREATE_TC_STATS:
                # if forced, use all
                ver_lc_list.extend([f"{ver}|{lc}" for lc in lc_list])
                g_tc.processed_lc += len(lc_list)
                g_tc.processed_ver += 1
            else:
                ver_lc_new: list[str] = []
                for lc in lc_list:
                    ver_lc: str = f"{ver}|{lc}"
                    tc_tsv: str = os.path.join(
                        tc_base,
                        ver_dir,
                        lc,
                        f"{c.TEXT_CORPUS_FN}.tsv",
                    )
                    if ver_lc in combined_ver_lc:
                        g_tc.skipped_exists += 1
                    elif not os.path.isfile(tc_tsv):
                        g_tc.skipped_nodata += 1
                    else:
                        ver_lc_new.append(ver_lc)
                num_to_process: int = len(ver_lc_new)
                ver_lc_list.extend(ver_lc_new)
                g_tc.processed_lc += num_to_process
                g_tc.processed_ver += 1 if num_to_process > 0 else 0

        # Now multi-process each record
        num_items: int = len(ver_lc_list)
        if num_items == 0:
            print("Nothing to process...")
            return

        chunk_size: int = min(
            MAX_BATCH_SIZE,
            num_items // used_proc_count
            + (0 if num_items % used_proc_count == 0 else 1),
        )
        print(
            f"Total: {g_tc.total_lc} Existing: {g_tc.skipped_exists} NoData: {g_tc.skipped_nodata} "
            + f"Remaining: {g_tc.processed_lc} Procs: {used_proc_count}  chunk_size: {chunk_size}..."
        )

        with mp.Pool(used_proc_count) as pool:
            with tqdm(total=num_items, desc="") as pbar:
                for res in pool.imap_unordered(
                    handle_text_corpus, ver_lc_list, chunksize=chunk_size
                ):
                    results.extend(res)
                    save_results()  # temporary saving, discard return
                    pbar.update()
                    for r in res:
                        if r.s_cnt == 0:
                            g_tc.skipped_nodata += 1

        # Create result DF
        print(">>> Finished... Now saving...")
        df: pd.DataFrame = save_results() # final save

        # Write out under locale dir (data/results/<lc>/<lc>_<ver>_tc_stats.json|tsv)
        df2: pd.DataFrame = pd.DataFrame()
        for ver in c.CV_VERSIONS:
            for lc in ALL_LOCALES:
                # pylint - false positive / fix not available yet: https://github.com/UCL/TLOmodel/pull/1193
                df2 = df[(df["ver"] == ver) & (df["lc"] == lc)]  # pylint: disable=E1136
                if df2.shape[0] > 0:
                    df_write(
                        df2,
                        os.path.join(
                            dst_tsv_base,
                            lc,
                            f"${lc}_{ver}_{c.TEXT_CORPUS_STATS_FN}.tsv",
                        ),
                    )
                    df2.to_json(
                        os.path.join(
                            dst_json_base,
                            lc,
                            f"${lc}_{ver}_{c.TEXT_CORPUS_STATS_FN}.json",
                        ),
                        orient="table",
                        index=False,
                    )
        # report
        report_results(g_tc)

    #
    # REPORTED SENTENCES
    #
    def main_reported() -> None:
        """Handle all reported sentences"""
        print("\n=== Start Reported Analysis ===")

        vc_base: str = os.path.join(HERE, c.DATA_DIRNAME, c.VC_DIRNAME)
        combined_tsv_file: str = os.path.join(
            dst_tsv_base, f"{c.REPORTED_STATS_FN}.tsv"
        )
        # Get joined TSV
        combined_df: pd.DataFrame = pd.DataFrame(columns=c.COLS_REPORTED_STATS)
        if os.path.isfile(combined_tsv_file):
            combined_df = df_read(combined_tsv_file).reset_index(drop=True)
        combined_df = combined_df[["ver", "lc"]]
        combined_ver_lc: list[str] = [
            "|".join([row[0], row[1]]) for row in combined_df.values.tolist()
        ]
        del combined_df
        combined_df = pd.DataFrame()

        ver_lc_list: list[str] = []  # final
        # For each version
        for ver in c.CV_VERSIONS:
            ver_dir: str = calc_dataset_prefix(ver)

            # get all possible
            lc_list: list[str] = get_locales_from_cv_dataset(ver)
            g_rep.total_lc += len(lc_list)

            # remove already calculated ones
            if conf.FORCE_CREATE_REPORTED_STATS:
                # if forced, use all
                ver_lc_list.extend([f"{ver}|{lc}" for lc in lc_list])
                g_rep.processed_lc += len(lc_list)
                g_rep.processed_ver += 1
            else:
                ver_lc_new: list[str] = []
                for lc in lc_list:
                    ver_lc: str = f"{ver}|{lc}"
                    rep_tsv: str = os.path.join(
                        vc_base,
                        ver_dir,
                        lc,
                        "reported.tsv",
                    )
                    if not ver_lc in combined_ver_lc and os.path.isfile(rep_tsv):
                        ver_lc_new.append(ver_lc)
                num_to_process: int = len(ver_lc_new)
                ver_lc_list.extend(ver_lc_new)
                g_rep.processed_lc += num_to_process
                g_rep.skipped_exists += len(lc_list) - num_to_process
                g_rep.processed_ver += 1 if num_to_process > 0 else 0

        # Now multi-process each record
        num_items: int = len(ver_lc_list)
        if num_items == 0:
            print("Nothing to process...")
            return

        chunk_size: int = min(
            MAX_BATCH_SIZE,
            num_items // used_proc_count
            + (0 if num_items % used_proc_count == 0 else 1),
        )
        print(
            f"Total: {g_rep.total_lc} Existing: {g_rep.skipped_exists} Remaining: {g_rep.processed_lc} "
            + f"Procs: {used_proc_count}  chunk_size: {chunk_size}..."
        )
        results: list[ReportedStatsRec] = []
        with mp.Pool(used_proc_count) as pool:
            with tqdm(total=num_items, desc="") as pbar:
                for res in pool.imap_unordered(
                    handle_reported, ver_lc_list, chunksize=chunk_size
                ):
                    results.append(res)
                    pbar.update()
                    if res.rep_sum == 0:
                        g.skipped_nodata += 1

        # Sort and write-out
        print(">>> Finished... Now saving...")
        df: pd.DataFrame = pd.DataFrame(results).reset_index(drop=True)
        df.sort_values(["lc", "ver"], inplace=True)

        # Write out combined (TSV only to use later)
        df_write(df, os.path.join(dst_tsv_base, f"{c.REPORTED_STATS_FN}.tsv"))
        # Write out per locale
        for lc in ALL_LOCALES:
            # pylint - false positive / fix not available yet: https://github.com/UCL/TLOmodel/pull/1193
            df_lc: pd.DataFrame = df[df["lc"] == lc]  # pylint: disable=E1136
            df_write(
                df_lc,
                os.path.join(
                    dst_tsv_base,
                    lc,
                    f"{c.REPORTED_STATS_FN}.tsv",
                ),
            )
            df_lc.to_json(
                os.path.join(
                    dst_json_base,
                    lc,
                    f"{c.REPORTED_STATS_FN}.json",
                ),
                orient="table",
                index=False,
            )
        # report
        report_results(g_rep)

    #
    # SPLITS
    #
    def main_splits() -> None:
        """Handle all splits"""
        print("\n=== Start Dataset/Split Analysis ===")

        # First get all source splits - a validated.tsv must exist if there is a dataset, even if it is empty
        vc_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.VC_DIRNAME)
        # get path part
        src_datasets: list[str] = [
            os.path.split(p)[0]
            for p in sorted(
                glob.glob(os.path.join(vc_dir, "**", "validated.tsv"), recursive=True)
            )
        ]

        # skip existing?
        ds_paths: list[str] = []
        if conf.FORCE_CREATE_VC_STATS:
            ds_paths = src_datasets
        else:
            tsv_path: str = os.path.join(
                HERE, c.DATA_DIRNAME, c.RES_DIRNAME, c.TSV_DIRNAME
            )
            json_path: str = os.path.join(
                HERE, c.DATA_DIRNAME, c.RES_DIRNAME, c.JSON_DIRNAME
            )

            for p in src_datasets:
                lc: str = os.path.split(p)[1]
                ver: str = os.path.split(os.path.split(p)[0])[1].split("-")[2]
                tsv_fn: str = os.path.join(tsv_path, lc, f"{lc}_{ver}_splits.tsv")
                json_fn: str = os.path.join(json_path, lc, f"{lc}_{ver}_splits.json")
                if not (os.path.isfile(tsv_fn) and os.path.isfile(json_fn)):
                    ds_paths.append(p)
        # finish filter out existing

        results: list[SplitStatsRec] = []
        cnt_to_process: int = len(ds_paths)

        if cnt_to_process == 0:
            print("Nothing to process")
            return

        chunk_size: int = min(
            MAX_BATCH_SIZE,
            cnt_to_process // used_proc_count
            + (0 if cnt_to_process % used_proc_count == 0 else 1),
        )
        print(
            f"Processing {cnt_to_process} locales in {used_proc_count} processes with chunk_size {chunk_size}..."
        )

        # now process each dataset
        with mp.Pool(used_proc_count) as pool:
            with tqdm(total=cnt_to_process, desc="") as pbar:
                for res in pool.imap_unordered(
                    handle_dataset_splits, ds_paths, chunksize=chunk_size
                ):
                    results.extend(res)
                    pbar.update()

        print(f">>> Processed {len(results)} splits...")

    #
    # SUPPORT MATRIX
    #
    def main_support_matrix() -> None:
        """Handle support matrix"""

        print("\n=== Build Support Matrix ===")

        # Scan files once again (we could have run it partial)
        all_tsv_paths: list[str] = sorted(
            glob.glob(
                os.path.join(
                    HERE, c.DATA_DIRNAME, c.RES_DIRNAME, c.TSV_DIRNAME, "**", "*.tsv"
                ),
                recursive=True,
            )
        )

        df: pd.DataFrame = pd.DataFrame().reset_index(drop=True)
        for tsv_path in all_tsv_paths:
            if (
                os.path.split(tsv_path)[1][0] != "$"
            ):  # ignore system files (starts with $)
                df = pd.concat([df.loc[:], df_read(tsv_path)]).reset_index(drop=True)

        g.total_splits = df.shape[0]
        g.total_lc = df[df["sp"] == "validated"].shape[0]

        df = df[["ver", "lc", "alg"]].drop_duplicates()
        df = (
            df[~df["alg"].isnull()]
            .sort_values(["lc", "ver", "alg"])
            .reset_index(drop=True)
        )
        g.total_algo = df.shape[0]

        # Prepare Support Matrix DataFrame
        rev_versions: list[str] = c.CV_VERSIONS.copy()  # versions in reverse order
        rev_versions.reverse()
        for inx, ver in enumerate(rev_versions):
            rev_versions[inx] = ver2vercol(ver)
        cols_support_matrix: list[str] = ["lc", "lang"]
        cols_support_matrix.extend(rev_versions)
        df_support_matrix: pd.DataFrame = pd.DataFrame(
            index=ALL_LOCALES,
            columns=cols_support_matrix,
        )
        df_support_matrix["lc"] = ALL_LOCALES

        # Now loop and put the results inside
        for lc in ALL_LOCALES:
            for ver in c.CV_VERSIONS:
                algo_list: list[str] = (
                    df[(df["lc"] == lc) & (df["ver"] == ver)]["alg"].unique().tolist()
                )
                algos: str = c.SEP_ALGO.join(algo_list)
                df_support_matrix.at[lc, ver2vercol(ver)] = algos

        # Write out
        print(">>> Saving Support Matrix...")
        df_write(
            df_support_matrix,
            os.path.join(
                HERE,
                c.DATA_DIRNAME,
                c.RES_DIRNAME,
                c.TSV_DIRNAME,
                f"{c.SUPPORT_MATRIX_FN}.tsv",
            ),
        )
        df_support_matrix.to_json(
            os.path.join(
                HERE,
                c.DATA_DIRNAME,
                c.RES_DIRNAME,
                c.JSON_DIRNAME,
                f"{c.SUPPORT_MATRIX_FN}.json",
            ),
            orient="table",
            index=False,
        )
        # report
        report_results(g)

    #
    # CONFIG
    #
    def main_config() -> None:
        """Save config"""
        config_data: ConfigRec = ConfigRec(
            date=datetime.now().strftime("%Y-%m-%d"),
            cv_versions=c.CV_VERSIONS,
            cv_dates=c.CV_DATES,
            cv_locales=ALL_LOCALES,
            algorithms=c.ALGORITHMS,
            # Drop the last huge values from bins
            bins_duration=c.BINS_DURATION[:-1],
            bins_voices=c.BINS_VOICES[:-1],
            bins_votes_up=c.BINS_VOTES_UP[:-1],
            bins_votes_down=c.BINS_VOTES_DOWN[:-1],
            bins_sentences=c.BINS_SENTENCES[:-1],
            bins_chars=c.BINS_CHARS[:-1],
            bins_words=c.BINS_WORDS[:-1],
            bins_tokens=c.BINS_TOKENS[:-1],
            bins_reported=c.BINS_REPORTED[:-1],
            bins_reasons=c.REPORTING_ALL,
        )
        df: pd.DataFrame = pd.DataFrame([config_data]).reset_index(drop=True)
        # Write out
        print("\n=== Save Configuration ===")
        df_write(
            df,
            os.path.join(
                HERE, c.DATA_DIRNAME, c.RES_DIRNAME, c.TSV_DIRNAME, "$config.tsv"
            ),
        )
        df.to_json(
            os.path.join(
                HERE, c.DATA_DIRNAME, c.RES_DIRNAME, c.JSON_DIRNAME, "$config.json"
            ),
            orient="table",
            index=False,
        )

    #
    # MAIN
    #
    used_proc_count: int = conf.DEBUG_PROC_COUNT if conf.DEBUG else PROC_COUNT
    start_time: datetime = datetime.now()

    # TEXT-CORPORA
    if not conf.SKIP_TEXT_CORPORA:
        main_text_corpora()
    # REPORTED SENTENCES
    if not conf.SKIP_REPORTED:
        main_reported()
    # SPLITS
    if not conf.SKIP_VOICE_CORPORA:
        main_splits()
    # SUPPORT MATRIX
    main_support_matrix()

    # [TODO] Fix DEM correction problem !!!
    # [TODO] Get CV-Wide Datasets => Measures / Totals
    # [TODO] Get global min/max/mean/median values for health measures
    # [TODO] Get some statistical plots as images (e.g. corrolation: age-char speed graph)

    # Save config
    main_config()

    # FINALIZE
    process_seconds: float = (datetime.now() - start_time).total_seconds()
    print("Finished compiling statistics!")
    print(
        f"Duration {dec3(process_seconds)} sec, avg={dec3(process_seconds/g.total_lc)} secs/dataset."
    )


if __name__ == "__main__":
    print("=== cv-tbox-dataset-analyzer - Final Statistics Compilation ===")
    init_directories(HERE)
    main()
