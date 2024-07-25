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
from collections import Counter
from datetime import datetime
from ast import literal_eval
import os
import sys
import glob
import multiprocessing as mp

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
    CharSpeedRec,
    dtype_pa_str,
    # dtype_pa_float64,
    # dtype_pa_uint16,
)
from lib import (
    dec1,
    df_concat,
    df_read,
    df_read_safe_reported,
    df_write,
    gender_backmapping,
    init_directories,
    dec3,
    calc_dataset_prefix,
    get_locales,
    arr2str,
    list2str,
    report_results,
    sort_by_largest_file,
)

# Globals

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

# PROC_COUNT: int = psutil.cpu_count(logical=False) - 1     # Limited usage
PROC_COUNT: int = psutil.cpu_count(logical=True)  # Full usage
MAX_BATCH_SIZE: int = 60

ALL_LOCALES: list[str] = get_locales(c.CV_VERSIONS[-1])

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
            alg=algo,
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
                _arr: np.ndarray = np.fromiter(
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
                _arr: np.ndarray = np.fromiter(
                    _df2["count"].dropna().apply(int).reset_index(drop=True).to_list(),
                    int,
                )
                _hist = np.histogram(_arr, bins=c.BINS_TOKENS)
                res.t_freq = _hist[0].tolist()[1:]
            if do_save:
                fn: str = os.path.join(
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
            res.p_items = list2str([x[0] for x in _values])
            res.p_freq = [x[1] for x in _values]
            if do_save:
                fn: str = os.path.join(
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
            _arr: np.ndarray = np.fromiter(
                _ser.apply(int).reset_index(drop=True).to_list(), int
            )
            _sl_bins: list[int] = (
                c.BINS_CHARS_SHORT
                if res.c_avg < c.CS_BIN_THRESHOLD
                else c.BINS_CHARS_LONG
            )
            _hist = np.histogram(_arr, bins=_sl_bins)
            res.c_freq = _hist[0].tolist()

        # GRAPHEMES
        _ = [grapheme_counter.update(s) for s in df["sentence"].dropna().tolist()]
        _df2 = pd.DataFrame(grapheme_counter.most_common(), columns=c.FIELDS_GRAPHEMES)
        _values = _df2.values.tolist()
        res.g_cnt = len(_values)
        res.g_items = list2str([x[0] for x in _values])
        res.g_freq = [x[1] for x in _values]
        if do_save:
            fn: str = os.path.join(
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
            fn: str = os.path.join(
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
        if float(ver) >= 17.0:
            # For newer versions, just use the sentence_id
            sentence_id_list: list[str] = (
                df_read(_fn)
                .reset_index(drop=True)["sentence_id"]
                .dropna()
                .drop_duplicates()
                .to_list()
            )
            df: pd.DataFrame = df_base_ver_tc[
                df_base_ver_tc["sentence_id"].isin(sentence_id_list)
            ]
            _res: TextCorpusStatsRec | None = handle_df(df, algo=algo, sp=sp)
        else:
            # For older versions, use the sentence
            sentence_list: list[str] = (
                df_read(_fn)
                .reset_index(drop=True)["sentence"]
                .dropna()
                .drop_duplicates()
                .to_list()
            )
            df: pd.DataFrame = df_base_ver_tc[
                df_base_ver_tc["sentence"].isin(sentence_list)
            ]
            _res: TextCorpusStatsRec | None = handle_df(df, algo=algo, sp=sp)
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
        HERE, c.DATA_DIRNAME, c.TC_DIRNAME, lc, f"{c.TEXT_CORPUS_FN}.tsv"
    )
    ver_tc_inx_file: str = os.path.join(
        HERE, c.DATA_DIRNAME, c.TC_DIRNAME, lc, f"{c.TEXT_CORPUS_FN}_{ver}.tsv"
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
        HERE, c.DATA_DIRNAME, c.TC_ANALYSIS_DIRNAME, ver_dir, lc
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
        HERE, c.DATA_DIRNAME, c.VC_DIRNAME, ver_dir, lc, "reported.tsv"
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
                os.path.join(
                    HERE, c.DATA_DIRNAME, ".debug", f"{lc}_{ver}_reported.tsv"
                ),
            )
            if len(problem_list) > 0:
                with open(
                    os.path.join(
                        HERE, c.DATA_DIRNAME, ".debug", f"{lc}_{ver}_problems.txt"
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
# Dataset Split Stats (MP Handler)
########################################################


def handle_dataset_splits(
    ds_path: str,
) -> tuple[list[SplitStatsRec], list[CharSpeedRec]]:
    """Handle a single dataset (ver/lc)"""
    # Handle one split, this is where calculations happen
    # The default column structure of CV dataset splits is as follows [FIXME] variants?
    # client_id, path, sentence, up_votes, down_votes, age, gender, accents, locale, segment
    # we have as input:
    # 'version', 'locale', 'algo', 'split'

    # now, do calculate some statistics...
    def handle_split(
        ver: str, lc: str, algo: str, split: str, fpath: str
    ) -> tuple[SplitStatsRec, CharSpeedRec]:
        """Processes a single split and return calculated values"""

        nonlocal df_clip_durations

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
        df_orig: pd.DataFrame = df_read(fpath)
        # .astype(c.FIELDS_BUCKETS_SPLITS) # WE CANNAOT USE THIS AS COLUMNS CHANGE WITH VERSIONS
        if split == "clips":  # build "clips" from val+inval+other
            # we already have validated (passed as param for clips)
            # add invalidated
            _df: pd.DataFrame = df_read(fpath.replace("validated", "invalidated"))
            if _df.shape[0] > 0:
                df_orig = pd.concat([df_orig, _df]) if df_orig.shape[0] > 0 else _df
            # add other
            _df = df_read(fpath.replace("validated", "other"))
            if _df.shape[0] > 0:
                df_orig = pd.concat([df_orig, _df]) if df_orig.shape[0] > 0 else _df

        # Do nothing, if there is no data
        if df_orig.shape[0] == 0:
            return (
                SplitStatsRec(ver=ver, lc=lc, alg=algo, sp=split),
                CharSpeedRec(ver=ver, lc=lc, alg=algo, sp=split),
            )

        # [TODO] Move these to split_compile: Make all confirm to current style?
        # Normalize data to the latest version's columns with typing
        # Replace NA with NODATA with some typing and conditionals
        # df: pd.DataFrame = df_orig.fillna(value=c.NODATA)
        df: pd.DataFrame = pd.DataFrame(columns=c.FIELDS_BUCKETS_SPLITS)
        # these should exist
        df["client_id"] = df_orig["client_id"]
        df["path"] = df_orig["path"]
        df["sentence"] = df_orig["sentence"]
        df["up_votes"] = df_orig["up_votes"]
        df["down_votes"] = df_orig["down_votes"]
        # these exist, but can be NaN
        df["age"] = df_orig["age"].fillna(c.NODATA)
        df["gender"] = df_orig["gender"].fillna(c.NODATA)
        # These might not exist in older versions, so we fill them
        df["locale"] = df_orig["locale"] if "locale" in df_orig.columns else lc
        df["variant"] = (
            df_orig["variant"].fillna(c.NODATA)
            if "variant" in df_orig.columns
            else c.NODATA
        )
        df["segment"] = (
            df_orig["segment"].fillna(c.NODATA)
            if "segment" in df_orig.columns
            else c.NODATA
        )
        df["sentence_domain"] = (
            df_orig["sentence_domain"].fillna(c.NODATA)
            if "sentence_domain" in df_orig.columns
            else c.NODATA
        )
        # The "accent" column renamed to "accents" along the way
        if "accent" in df_orig.columns:
            df["accents"] = df_orig["accent"].fillna(c.NODATA)
        if "accents" in df_orig.columns:
            df["accents"] = df_orig["accents"].fillna(c.NODATA)
        # [TODO] this needs special consideration (back-lookup) but has quirks for now
        df["sentence_id"] = (
            df_orig["sentence_id"].fillna(c.NODATA)
            if "sentence_id" in df_orig.columns
            else c.NODATA
        )

        # backmap genders
        df = gender_backmapping(df)
        # add lowercase sentence column
        df["sentence_lower"] = df["sentence"].str.lower()

        # === DURATIONS: Calc duration agregate values
        # there must be records + v1 cannot be mapped
        if df_clip_durations.shape[0] > 0 and ver != "1":
            # Connect with duration table and convert to seconds
            df["duration"] = df["path"].map(
                df_clip_durations["duration[ms]"] / 1000, na_action="ignore"
            )
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

        # === VOICES (how many recordings per voice)
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
        arr: np.ndarray = np.fromiter(
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

        up_votes_freq: list[int] = []
        for i in range(0, len(bins) - 1):
            bin_val: int = bins[i]
            bin_next: int = bins[i + 1]
            up_votes_freq.append(
                int(
                    vote_counts_df.loc[
                        (vote_counts_df["votes"] >= bin_val)
                        & (vote_counts_df["votes"] < bin_next)
                    ]["count"].sum()
                )
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

        res_ss: SplitStatsRec = SplitStatsRec(
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

        res_cs: CharSpeedRec = CharSpeedRec(
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
            df = df.assign(
                char_speed=lambda x: round(1000 * (x["duration"] / x["s_len"]))
            )

            # calc general stats from real values
            ser = df["char_speed"]
            cs_mean: float = ser.mean()
            cs_median: float = ser.median()
            cs_std: float = ser.std(ddof=0)
            # decide which bin should be used
            _cs_bins: list[int] = (
                c.BINS_CS_LOW if cs_mean < c.CS_BIN_THRESHOLD else c.BINS_CS_HIGH
            )
            _sl_bins: list[int] = (
                c.BINS_CHARS_SHORT
                if cs_mean < c.CS_BIN_THRESHOLD
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

            res_cs = CharSpeedRec(
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

        return (res_ss, res_cs)

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
        columns=c.FIELDS_CLIP_DURATIONS
    ).set_index("clip")
    if os.path.isfile(cd_file):
        df_clip_durations = df_read(cd_file).set_index("clip")
    elif conf.VERBOSE:
        print(f"WARNING: No duration data for {lc}\n")

    # === MAIN BUCKETS (clips, validated, invalidated, other)
    ret_ss: SplitStatsRec
    ret_cs: CharSpeedRec
    res_ss: list[SplitStatsRec] = []  # Init the result list
    res_cs: list[CharSpeedRec] = []  # Init the result list

    # Special case for temporary "clips.tsv"
    ret_ss, ret_cs = handle_split(
        ver=ver,
        lc=lc,
        algo="",
        split="clips",
        fpath=os.path.join(ds_path, "validated.tsv"),
    )
    res_ss.append(ret_ss)
    res_cs.append(ret_cs)

    validated_result: SplitStatsRec = res_ss[-1]
    validated_records: int = validated_result.clips
    # Append to clips.tsv at the source, at the base of that version
    # (it will include all recording data for all locales to be used in CC & alternatives)
    for sp in c.MAIN_BUCKETS:
        src: str = os.path.join(ds_path, sp + ".tsv")
        dst: str = os.path.join(cv_dir, "clips.tsv")
        df_write(df_read(src), fpath=dst, mode="a")

    for sp in c.MAIN_BUCKETS:
        ret_ss, ret_cs = handle_split(
            ver=ver,
            lc=lc,
            algo="",
            split=sp,
            fpath=os.path.join(ds_path, sp + ".tsv"),
        )
        res_ss.append(ret_ss)
        res_cs.append(ret_cs)

    # If no record in validated, do not try further
    if validated_records == 0:
        return (res_ss, res_cs)

    # SPLITTING ALGO SPECIFIC (inc default splits)

    for algo in c.ALGORITHMS:
        for sp in c.TRAINING_SPLITS:
            if os.path.isfile(os.path.join(ds_path, algo, sp + ".tsv")):
                ret_ss, ret_cs = handle_split(
                    ver=ver,
                    lc=lc,
                    algo=algo,
                    split=sp,
                    fpath=os.path.join(ds_path, algo, sp + ".tsv"),
                )
                res_ss.append(ret_ss)
                res_cs.append(ret_cs)

    # Create DataFrames
    df: pd.DataFrame = pd.DataFrame(res_ss)
    df_write(df, os.path.join(tsv_path, f"{lc}_{ver}_splits.tsv"))
    df.to_json(
        os.path.join(json_path, f"{lc}_{ver}_splits.json"), orient="table", index=False
    )

    df: pd.DataFrame = pd.DataFrame(res_cs)
    df_write(df, os.path.join(tsv_path, f"{lc}_{ver}_cs.tsv"))
    df.to_json(
        os.path.join(json_path, f"{lc}_{ver}_cs.json"), orient="table", index=False
    )

    return (res_ss, res_cs)


# END - Dataset Split Stats (MP Handler)


########################################################
# MAIN PROCESS
########################################################


def main() -> None:
    """Compile all data by calculating stats"""

    res_json_base_dir: str = os.path.join(
        HERE, c.DATA_DIRNAME, c.RES_DIRNAME, c.JSON_DIRNAME
    )
    res_tsv_base_dir: str = os.path.join(
        HERE, c.DATA_DIRNAME, c.RES_DIRNAME, c.TSV_DIRNAME
    )

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
            df: pd.DataFrame = df_concat(
                df_combined, pd.DataFrame(results, columns=c.FIELDS_TC_STATS)
            ).reset_index(drop=True)
            df.sort_values(by=["lc", "ver"], inplace=True)
            # Write out combined (TSV only to use later for above existence checks)
            df_write(
                df, os.path.join(res_tsv_base_dir, f"${c.TEXT_CORPUS_STATS_FN}.tsv")
            )
            return df

        print("\n=== Start Text Corpora Analysis ===")

        tc_base_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.TC_DIRNAME)
        combined_tsv_fpath: str = os.path.join(
            res_tsv_base_dir, f"${c.TEXT_CORPUS_STATS_FN}.tsv"
        )
        # Get joined TSV
        combined_ver_lc: list[str] = []
        df_combined: pd.DataFrame = pd.DataFrame()

        if os.path.isfile(combined_tsv_fpath):
            df_combined = df_read(combined_tsv_fpath).reset_index(drop=True)
            combined_ver_lc: list[str] = [
                "|".join(row)
                for row in df_combined[["ver", "lc"]].astype(str).values.tolist()
            ]

            # try:
            #     combined_ver_lc = [
            #         "|".join(row)
            #         for row in df_read(combined_tsv_fpath, use_cols=["ver", "lc"])
            #         .reset_index(drop=True)
            #         .dropna()
            #         .drop_duplicates()
            #         .astype(str)
            #         .values.tolist()
            #     ]
            # except ValueError as e:
            #     print(e)

        ver_lc_list: list[str] = []  # final
        # start with newer, thus larger / longer versions' data
        versions: list[str] = c.CV_VERSIONS.copy()
        versions.reverse()
        # For each version
        for ver in versions:
            # ver_dir: str = calc_dataset_prefix(ver)

            # get all possible
            lc_list: list[str] = get_locales(ver)
            g_tc.total_lc += len(lc_list)

            # Get list of existing processed text corpus files, in reverse size order
            # then get a list of language codes in that order
            # This assumes that the larger the latest TC, the larger data we will have in previous versions,
            # so that multiprocessing is maximized
            pp: list[str] = glob.glob(
                os.path.join(
                    HERE, c.DATA_DIRNAME, c.TC_DIRNAME, "**", f"{c.TEXT_CORPUS_FN}.tsv"
                )
            )
            lc_complete_list: list[str] = [
                p.split(os.sep)[-2] for p in sort_by_largest_file(pp)
            ]
            lc_list = (
                [lc for lc in lc_complete_list if lc in lc_list]
                if not conf.DEBUG
                else conf.DEBUG_CV_LC
            )

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
                        tc_base_dir,
                        lc,
                        f"{c.TEXT_CORPUS_FN}_{ver}.tsv",
                    )
                    if ver_lc in combined_ver_lc:
                        g_tc.skipped_exists += 1
                    elif not os.path.isfile(tc_tsv):
                        g_tc.skipped_nodata += 1
                    else:
                        ver_lc_new.append(ver_lc)
                new_num_to_process: int = len(ver_lc_new)
                ver_lc_list.extend(ver_lc_new)
                g_tc.processed_lc += new_num_to_process
                g_tc.processed_ver += 1 if new_num_to_process > 0 else 0

        # Now multi-process each record
        num_items: int = len(ver_lc_list)
        if num_items == 0:
            report_results(g_tc)
            print("Nothing to process...")
            return

        chunk_size: int = min(
            MAX_BATCH_SIZE,
            num_items // 100 + 1,
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
                    save_results()  # temporary saving: it takes a long time which might end, discard return
                    pbar.update()
                    for r in res:
                        if r.s_cnt == 0:
                            g_tc.skipped_nodata += 1

        # Create result DF
        print(">>> Finished... Now saving...")
        df: pd.DataFrame = save_results()  # final save

        # Write out under locale dir (data/results/<lc>/<lc>_<ver>_tc_stats.json|tsv)
        df2: pd.DataFrame = pd.DataFrame()
        for ver in c.CV_VERSIONS:
            for lc in ALL_LOCALES:
                df2 = df[(df["ver"] == ver) & (df["lc"] == lc)]
                if df2.shape[0] > 0:
                    df_write(
                        df2,
                        os.path.join(
                            res_tsv_base_dir,
                            lc,
                            f"${lc}_{ver}_{c.TEXT_CORPUS_STATS_FN}.tsv",
                        ),
                    )
                    df2.to_json(
                        os.path.join(
                            res_json_base_dir,
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

        vc_base_dir: str = os.path.join(HERE, c.DATA_DIRNAME, c.VC_DIRNAME)
        combined_tsv_fpath: str = os.path.join(
            res_tsv_base_dir, f"{c.REPORTED_STATS_FN}.tsv"
        )
        # Get joined TSV, get ver-lc list for all previously
        combined_ver_lc: list[str] = []
        df_combined: pd.DataFrame = pd.DataFrame()
        if os.path.isfile(combined_tsv_fpath):
            df_combined = df_read(combined_tsv_fpath).reset_index(drop=True)
            combined_ver_lc: list[str] = [
                "|".join(row)
                for row in df_combined[["ver", "lc"]].astype(str).values.tolist()
            ]

        # For each version
        ver_lc_list: list[str] = []  # final
        ver_to_process: list[str] = conf.DEBUG_CV_VER if conf.DEBUG else c.CV_VERSIONS
        for ver in ver_to_process:
            ver_dir: str = calc_dataset_prefix(ver)

            # get all possible or use DEBUG list
            lc_list: list[str] = conf.DEBUG_CV_LC if conf.DEBUG else get_locales(ver)
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
                    if not ver_lc in combined_ver_lc and os.path.isfile(
                        os.path.join(
                            vc_base_dir,
                            ver_dir,
                            lc,
                            "reported.tsv",
                        )
                    ):
                        ver_lc_new.append(ver_lc)
                num_to_process: int = len(ver_lc_new)
                ver_lc_list.extend(ver_lc_new)
                g_rep.processed_lc += num_to_process
                g_rep.skipped_nodata += len(lc_list) - num_to_process
                g_rep.processed_ver += 1 if num_to_process > 0 else 0

        # Now multi-process each record
        num_items: int = len(ver_lc_list)
        if num_items == 0:
            print("Nothing to process...")
            return

        used_proc_count = 1

        chunk_size: int = min(
            MAX_BATCH_SIZE,
            num_items // 100 + 1,
            num_items // used_proc_count
            + (0 if num_items % used_proc_count == 0 else 1),
        )
        print(
            f"Total: {g_rep.total_lc} Missing: {g_rep.skipped_nodata} Remaining: {g_rep.processed_lc} "
            + f"Procs: {used_proc_count}  chunk_size: {chunk_size}..."
        )
        results: list[ReportedStatsRec] = []
        with mp.Pool(used_proc_count) as pool:
            with tqdm(total=num_items, desc="") as pbar:
                for res in pool.imap_unordered(
                    handle_reported, ver_lc_list, chunksize=chunk_size
                ):
                    # pbar.write(f"Finished {res.ver} - {res.lc}")
                    results.append(res)
                    pbar.update()
                    if res.rep_sum == 0:
                        g.skipped_nodata += 1

        # Sort and write-out
        print(">>> Finished... Now saving...")
        df: pd.DataFrame = df_concat(
            df_combined, pd.DataFrame(results).reset_index(drop=True)
        )
        df.sort_values(by=["lc", "ver"], inplace=True)

        # Write out combined (TSV only to use later)
        df_write(df, os.path.join(res_tsv_base_dir, f"{c.REPORTED_STATS_FN}.tsv"))
        # Write out per locale
        for lc in ALL_LOCALES:
            df_lc: pd.DataFrame = df[df["lc"] == lc]
            df_write(
                df_lc,
                os.path.join(
                    res_tsv_base_dir,
                    lc,
                    f"{c.REPORTED_STATS_FN}.tsv",
                ),
            )
            df_lc.to_json(
                os.path.join(
                    res_json_base_dir,
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
        pp: list[str] = [
            os.path.split(p)[0]
            for p in sorted(
                glob.glob(os.path.join(vc_dir, "**", "validated.tsv"), recursive=True)
            )
        ]
        # sort by largest first
        pp = sort_by_largest_file(pp)

        tsv_path: str = os.path.join(HERE, c.DATA_DIRNAME, c.RES_DIRNAME, c.TSV_DIRNAME)
        json_path: str = os.path.join(
            HERE, c.DATA_DIRNAME, c.RES_DIRNAME, c.JSON_DIRNAME
        )
        ds_paths: list[str] = []
        # handle debug
        if conf.DEBUG:
            for p in pp:
                lc: str = os.path.split(p)[1]
                ver: str = os.path.split(os.path.split(p)[0])[1].split("-")[2]
                if lc in conf.DEBUG_CV_LC and ver in conf.DEBUG_CV_VER:
                    ds_paths.append(p)
        else:
            # skip existing?
            if conf.FORCE_CREATE_VC_STATS:
                ds_paths = pp
            else:
                for p in pp:
                    lc: str = os.path.split(p)[1]
                    ver: str = os.path.split(os.path.split(p)[0])[1].split("-")[2]
                    tsv_fn: str = os.path.join(tsv_path, lc, f"{lc}_{ver}_splits.tsv")
                    json_fn: str = os.path.join(
                        json_path, lc, f"{lc}_{ver}_splits.json"
                    )
                    if not (os.path.isfile(tsv_fn) and os.path.isfile(json_fn)):
                        ds_paths.append(p)
        # finish filter out existing

        ret_ss: list[SplitStatsRec]
        ret_cs: list[CharSpeedRec]
        results_ss: list[SplitStatsRec] = []
        results_cs: list[CharSpeedRec] = []
        cnt_to_process: int = len(ds_paths)

        if cnt_to_process == 0:
            print("Nothing to process")
            return

        chunk_size: int = min(
            MAX_BATCH_SIZE,
            cnt_to_process // 100 + 1,
            cnt_to_process // used_proc_count
            + (0 if cnt_to_process % used_proc_count == 0 else 1),
        )
        print(
            f"Processing {cnt_to_process} locales in {used_proc_count} processes with chunk_size {chunk_size}..."
        )

        # now process each dataset
        with mp.Pool(used_proc_count) as pool:
            with tqdm(total=cnt_to_process, desc="") as pbar:
                for ret_ss, ret_cs in pool.imap_unordered(
                    handle_dataset_splits, ds_paths, chunksize=chunk_size
                ):
                    results_ss.extend(ret_ss)
                    results_cs.extend(ret_cs)
                    pbar.update()

        print(f">>> Processed {len(results_ss)} splits...")

    #
    # SUPPORT MATRIX
    #
    def main_support_matrix() -> None:
        """Handle support matrix"""

        print("\n=== Build Support Matrix ===")

        # Scan files once again (we could have run it partial)
        # "df" will contain combined split stats (which we will save and only use "validated" from it)
        # df: pd.DataFrame = pd.DataFrame(
        #     columns=list(c.FIELDS_SPLIT_STATS.keys())
        # ).astype(c.FIELDS_SPLIT_STATS)
        df: pd.DataFrame = pd.DataFrame()
        all_tsv_paths: list[str] = sorted(
            glob.glob(
                os.path.join(
                    HERE,
                    c.DATA_DIRNAME,
                    c.RES_DIRNAME,
                    c.TSV_DIRNAME,
                    "**",
                    "*_splits.tsv",
                ),
                recursive=True,
            )
        )
        # preload all TSV to concat later
        df_list: list[pd.DataFrame] = []
        for tsv_path in all_tsv_paths:
            # prevent "ver" col to be converted to float
            df_list.append(df_read(tsv_path, dtypes={"ver": dtype_pa_str}))
        # concat
        df = pd.concat(df_list, copy=False).reset_index(drop=True)
        # save to root
        print(">>> Saving combined split stats...")
        dst: str = os.path.join(
            HERE,
            c.DATA_DIRNAME,
            c.RES_DIRNAME,
            c.TSV_DIRNAME,
            "$combined_splits.tsv",
        )
        df_write(df, dst)

        # clean
        df = df.drop(
            columns=list(set(df.columns) - set(["ver", "lc", "alg", "sp", "dur_total"]))
        )

        # get some stats
        g.total_splits = df.shape[0]
        g.total_lc = df[df["sp"] == ""].shape[0]

        # get algo view
        df_algo: pd.DataFrame = df[["ver", "lc", "alg"]].drop_duplicates()
        df_algo = (
            df_algo[~df_algo["alg"].isnull()].sort_values(["lc", "ver", "alg"])
            # .astype(dtype_pa_str)
            .reset_index(drop=True)
        )
        g.total_algo = df_algo.shape[0]

        # Prepare Support Matrix DataFrame
        rev_versions: list[str] = c.CV_VERSIONS.copy()  # versions in reverse order
        rev_versions.reverse()

        cols_support_matrix: list[str] = ["lc", "lang"] + [
            ver2vercol(v) for v in rev_versions
        ]

        df_support_matrix: pd.DataFrame = pd.DataFrame(
            columns=cols_support_matrix,
            dtype=dtype_pa_str,
            index=ALL_LOCALES,
        )
        df_support_matrix["lc"] = ALL_LOCALES

        # Now loop and put the results inside
        for lc in ALL_LOCALES:
            for ver in c.CV_VERSIONS:
                algo_list: list[str] = (
                    df_algo[(df_algo["lc"] == lc) & (df_algo["ver"] == ver)]["alg"]
                    .unique()
                    .tolist()
                )
                hours: str = "0.0"
                if algo_list:
                    dur: float = df[
                        (df["lc"] == lc)
                        & (df["ver"] == ver)
                        & (df["sp"] == "validated")
                    ]["dur_total"].to_list()[0]
                    hours = str(dec1(dur / 3600)) if dur >= 0 else "0"

                df_support_matrix.at[lc, ver2vercol(ver)] = (
                    f"{hours}{c.SEP_ALGO}{c.SEP_ALGO.join(algo_list)}"
                    if algo_list
                    else pd.NA
                )

        # Write out
        print(">>> Saving Support Matrix...")
        dst = os.path.join(
            HERE,
            c.DATA_DIRNAME,
            c.RES_DIRNAME,
            c.TSV_DIRNAME,
            f"{c.SUPPORT_MATRIX_FN}.tsv",
        )
        df_write(df_support_matrix, dst)
        df_support_matrix.to_json(
            dst.replace("tsv", "json"),
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
            bins_voices=c.BINS_VOICES[1:-1],
            bins_votes_up=c.BINS_VOTES_UP[:-1],
            bins_votes_down=c.BINS_VOTES_DOWN[:-1],
            bins_sentences=c.BINS_SENTENCES[1:-1],
            cs_threshold=c.CS_BIN_THRESHOLD,
            bins_cs_low=c.BINS_CS_LOW[:-1],
            bins_cs_high=c.BINS_CS_HIGH[:-1],
            ch_threshold=c.CHARS_BIN_THRESHOLD,
            bins_chars_short=c.BINS_CHARS_SHORT[:-1],
            bins_chars_long=c.BINS_CHARS_LONG[:-1],
            bins_words=c.BINS_WORDS[1:-1],
            bins_tokens=c.BINS_TOKENS[1:-1],
            bins_reported=c.BINS_REPORTED[1:-1],
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
    if not conf.SKIP_SUPPORT_MATRIX:
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
