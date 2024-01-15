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
from datetime import datetime, timedelta
import multiprocessing as mp

# External dependencies
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

# Module
import const as c
import conf
from typedef import ConfigRec, TextCorpusStatsRec, ReportedStatsRec, SplitStatsRec
from lib import (
    df_read,
    df_write,
    list2str,
    arr2str,
    dec3,
    calc_dataset_prefix,
    get_locales,
)

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

#
# Constants
#

# PROC_COUNT: int = psutil.cpu_count(logical=False) - 1     # Limited usage
PROC_COUNT: int = psutil.cpu_count(logical=True)  # Full usage
ALL_LOCALES: list[str] = get_locales(c.CV_VERSIONS[-1])


########################################################
# Text-Corpus Stats (Multi Processing Handler)
########################################################


def handle_text_corpus(lc: str) -> TextCorpusStatsRec:
    """Multi-Process text-corpus for a single locale"""

    tc_dir: str = os.path.join(HERE, "data", "text-corpus", lc)
    tc_file: str = os.path.join(tc_dir, "$text_corpus.tsv")

    if not os.path.isfile(tc_file):
        print(f"WARN: Skipping no-text-corpus-file locale: {lc}")
        return TextCorpusStatsRec(lc=lc)

    tokens_file: str = os.path.join(tc_dir, "$tokens.tsv")

    if conf.VERBOSE:
        print(f"Processing text-corpus for locale: {lc}")
    else:
        print("\033[F" + " " * 80)
        print(f"\033[FProcessing text-corpus for locale: {lc}")

    # TEXT CORPUS

    # "file", "sentence", "lower", "normalized", "chars", "words", 'valid'
    df: pd.DataFrame = df_read(tc_file)

    # basic measures
    sentence_cnt: int = df.shape[0]
    unique_sentences: int = df["sentence"].dropna().unique().shape[0]
    unique_normalized: int = df["normalized"].dropna().unique().shape[0]
    valid: int = df[df["valid"].dropna().astype(int) == 1].shape[0]

    # CHARS
    ser: pd.Series[int] = df["chars"].dropna()  # pylint: disable=unsubscriptable-object
    chars_total: int = ser.sum()
    chars_mean: float = ser.mean()
    chars_median: float = ser.median()
    chars_std: float = ser.std(ddof=0)
    # Calc character length distribution
    arr: np.ndarray = np.fromiter(ser.apply(int).reset_index(drop=True).to_list(), int)
    hist = np.histogram(arr, bins=c.BINS_CHARS)
    character_freq = hist[0].tolist()

    has_val: int = 0

    # WORDS
    ser: pd.Series[int] = df["words"].dropna()  # pylint: disable=unsubscriptable-object
    words_total: int = ser.sum()
    words_mean: float = 0.0
    words_median: float = 0.0
    words_std: float = 0.0
    word_freq: list[int] = []
    if words_total != 0:
        has_val = 1
        words_mean = ser.mean()
        words_median = ser.median()
        words_std = ser.std(ddof=0)
    # Calc word count distribution
    if has_val == 1:
        arr: np.ndarray = np.fromiter(
            ser.apply(int).reset_index(drop=True).to_list(), int
        )
        hist = np.histogram(arr, bins=c.BINS_WORDS)
        word_freq = hist[0].tolist()

    # TOKENS
    tokens_total: int = 0
    tokens_mean: float = 0.0
    tokens_median: float = 0.0
    tokens_std: float = 0.0
    token_freq: list[int] = []
    if has_val == 1:
        df: pd.DataFrame = df_read(tokens_file)
        # "token", "count"
        tokens_total = df.shape[0]
        ser = df["count"].dropna()
        tokens_mean = ser.mean()
        tokens_median = ser.median()
        tokens_std = ser.std(ddof=0)
        # Token/word repeat distribution
        arr: np.ndarray = np.fromiter(
            df["count"].dropna().apply(int).reset_index(drop=True).to_list(), int
        )
        hist = np.histogram(arr, bins=c.BINS_TOKENS)
        token_freq = hist[0].tolist()

    res: TextCorpusStatsRec = TextCorpusStatsRec(
        lc=lc,
        s_cnt=sentence_cnt,
        uq_s=unique_sentences,
        uq_n=unique_normalized,
        has_val=has_val,
        val=valid,
        c_sum=chars_total,
        c_avg=dec3(chars_mean),
        c_med=dec3(chars_median),
        c_std=dec3(chars_std),
        c_freq=list2str(character_freq),
        w_sum=words_total,
        w_avg=dec3(words_mean),
        w_med=dec3(words_median),
        w_std=dec3(words_std),
        w_freq=list2str(word_freq),
        t_sum=tokens_total,
        t_avg=dec3(tokens_mean),
        t_med=dec3(tokens_median),
        t_std=dec3(tokens_std),
        t_freq=list2str(token_freq),
    )
    return res


########################################################
# Reported Stats
########################################################


def handle_reported(cv_ver: str) -> "list[ReportedStatsRec]":
    """Process text-corpus for a single locale"""

    # Fins idx
    cv_idx: int = c.CV_VERSIONS.index(cv_ver)
    # Calc CV_DIR
    ver: str = c.CV_VERSIONS[cv_idx]
    cv_dir_name: str = calc_dataset_prefix(ver)
    # Calc voice-corpus directory
    cv_dir: str = os.path.join(HERE, "data", "voice-corpus", cv_dir_name)

    # Get a list of available language codes
    lc_paths: list[str] = sorted(glob.glob(os.path.join(cv_dir, "*"), recursive=False))
    lc_list: list[str] = []
    for lc_path in lc_paths:
        if os.path.isdir(lc_path):  # ignore files
            lc_list.append(os.path.split(lc_path)[1])

    lc_to_process: list[str] = conf.DEBUG_CV_LC if conf.DEBUG else lc_list
    res_all: list[ReportedStatsRec] = []
    for lc in lc_to_process:
        # Source
        vc_dir: str = os.path.join(cv_dir, lc)
        rep_file: str = os.path.join(vc_dir, "reported.tsv")
        if conf.VERBOSE:
            print(f"Handling reported.tsv in v{ver} - {lc}")
        else:
            print("\033[F" + " " * 80)
            print(f"\033[FHandling reported.tsv in v{ver} - {lc}")
        if not os.path.isfile(rep_file):  # skip process if no such file
            continue
        if os.path.getsize(rep_file) == 0:  # there can be empty files :/
            continue
        # read file in - Columns: sentence sentence_id locale reason
        df: pd.DataFrame = df_read(rep_file)
        if df.shape[0] == 0:  # skip those without records
            continue

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
            rep_freq=list2str(rep_freq),
            rea_freq=list2str(reason_freq),
        )

        res_all.append(res)
        # end of a single version-locale
    # end of a single version / all locales

    return res_all  # Return combined results for one version


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
        def find_fixes(df_split: pd.DataFrame) -> list[str]:
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
            dem_fixes_recs: str = ""
            dem_fixes_voices: str = ""

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
                dem_fixes_recs = arr2str(pt.to_numpy(int).tolist())

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
                dem_fixes_voices = arr2str(pt.to_numpy(int).tolist())

            return [dem_fixes_recs, dem_fixes_voices]

        # END - find_fixes

        #
        # === START ===
        #
        if conf.DEBUG:
            print("ver=", ver, "lc=", lc, "algorithm=", algorithm, "split=", split)

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
        dem_fixes_list: list[str] = find_fixes(df_orig)

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
            dur_med=dec3(duration_median),
            dur_std=dec3(duration_std),
            dur_freq=list2str(duration_freq),
            # Recordings per Voice
            v_avg=dec3(voice_mean),
            v_med=dec3(voice_median),
            v_std=dec3(voice_std),
            v_freq=list2str(voice_freq),
            # Recordings per Sentence
            s_avg=dec3(sentence_mean),
            s_med=dec3(sentence_median),
            s_std=dec3(sentence_std),
            s_freq=list2str(sentence_freq),
            # Votes
            uv_sum=up_votes_sum,
            uv_avg=dec3(up_votes_mean),
            uv_med=dec3(up_votes_median),
            uv_std=dec3(up_votes_std),
            uv_freq=list2str(up_votes_freq),
            dv_sum=down_votes_sum,
            dv_avg=dec3(down_votes_mean),
            dv_med=dec3(down_votes_median),
            dv_std=dec3(down_votes_std),
            dv_freq=list2str(down_votes_freq),
            # Demographics distribution for recordings
            dem_table=arr2str(_pt_dem.to_numpy(int).tolist()),
            dem_uq=arr2str(_pt_uqdem.to_numpy(int).tolist()),
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

    if conf.VERBOSE:
        print(f"Processing version {ver} - {lc}")
    else:
        print("\033[F" + " " * 80)
        print(f"\033[FProcessing version {ver} - {lc}")

    # Source directories
    cd_dir: str = os.path.join(HERE, "data", "clip-durations", lc)

    # Create destinations if thet do not exist
    tsv_path: str = os.path.join(HERE, "data", "results", "tsv", lc)
    json_path: str = os.path.join(HERE, "data", "results", "json", lc)

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
    else:
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

    num_datasets: int = 0
    num_algorithms: int = 0
    num_splits: int = 0

    def ver2vercol(ver: str) -> str:
        """Converts a data version in format '11.0' to column/variable name format 'v11_0'"""
        return "v" + ver.replace(".", "_")

    #
    # TEXT-CORPORA
    #
    def main_text_corpora() -> None:
        """Handle all text corpora"""
        print("\n=== Start Text Corpora Analysis ===\n")

        # First get list of languages with text corpora
        tc_base: str = os.path.join(HERE, "data", "text-corpus")
        lc_paths: list[str] = sorted(
            glob.glob(os.path.join(tc_base, "*"), recursive=False)
        )
        lc_list: list[str] = []
        for lc_path in lc_paths:
            if os.path.isdir(lc_path):  # ignore files
                lc_list.append(os.path.split(lc_path)[1])

        lc_to_process: list[str] = conf.DEBUG_CV_LC if conf.DEBUG else lc_list

        print(f">>> Processing {len(lc_to_process)} text-corpora...\n")
        # Now multi-process each lc
        # num_locales: int = len(lc_to_process)
        # chunk_size: int = min(10, num_locales // used_proc_count + 0 if num_locales % used_proc_count == 0 else 1)
        # print(
        #     f"Processing {num_locales} locales in {used_proc_count} processes with chunk_size {chunk_size}...\n"
        # )
        # results: list[TextCorpusStatsRec] = []
        # with mp.Pool(used_proc_count) as pool:
        #     with tqdm(total=num_locales, desc="") as pbar:
        #         for res in pool.imap_unordered(handle_text_corpus, lc_to_process, chunksize=chunk_size):
        #             results.append(res)
        #             pbar.update()

        with mp.Pool(used_proc_count) as pool:
            results = pool.map(handle_text_corpus, lc_to_process)

        # Create result DF
        # df: pd.DataFrame = pd.DataFrame(tc_stats, columns=c.COLS_TEXT_CORPUS)
        print(">>> Finished... Now saving...")
        df: pd.DataFrame = pd.DataFrame(results)
        df_write(
            df, os.path.join(HERE, "data", "results", "tsv", f"{c.TEXT_CORPUS_STATS_FN}.tsv")
        )
        df.to_json(
            os.path.join(HERE, "data", "results", "json", f"{c.TEXT_CORPUS_STATS_FN}.json"),
            orient="table",
            index=False,
        )

    #
    # REPORTED SENTENCES
    #
    def main_reported() -> None:
        """Handle all reported sentences"""
        vers_to_process: list[str] = (
            conf.DEBUG_CV_VER if conf.DEBUG else c.CV_VERSIONS.copy()
        )
        vers_to_process.reverse()  # start with largest (latest) to maximize core usage
        print("\n=== Start Reported Analysis ===\n")
        print(f">>> Processing {len(vers_to_process)} versions...\n")
        results: list[list[ReportedStatsRec]] = []
        with mp.Pool(processes=used_proc_count, maxtasksperchild=1) as pool:
            results = pool.map(handle_reported, vers_to_process)
        # done, first flatten them
        flattened: list[ReportedStatsRec] = []
        for res in results:
            flattened += res
        # Sort and write-out
        df: pd.DataFrame = pd.DataFrame(flattened).reset_index(drop=True)
        df.sort_values(["ver", "lc"], inplace=True)
        print(">>> Finished... Now saving...")
        # Write out per locale
        for lc in ALL_LOCALES:
            # pylint - false positive / fix not available yet: https://github.com/UCL/TLOmodel/pull/1193
            df_lc: pd.DataFrame = df[df["lc"] == lc]  # pylint: disable=E1136
            df_write(
                df_lc, os.path.join(HERE, "data", "results", "tsv", lc, f"{c.REPORTED_STATS_FN}.tsv")
            )
            df_lc.to_json(
                os.path.join(HERE, "data", "results", "json", lc, f"{c.REPORTED_STATS_FN}.json"),
                orient="table",
                index=False,
            )

    #
    # SPLITS
    #
    def main_splits() -> None:
        """Handle all splits"""
        print("\n=== Start Dataset/Split Analysis ===\n")

        # First get all source splits - a validated.tsv must exist if there is a dataset, even if it is empty
        vc_dir: str = os.path.join(HERE, "data", "voice-corpus")
        all_validated: list[str] = sorted(
            glob.glob(os.path.join(vc_dir, "**", "validated.tsv"), recursive=True)
        )
        # get path part
        source_datasets: list[str] = [os.path.split(p)[0] for p in all_validated]
        cnt_datasets: int = len(source_datasets)
        print(f">>> We have {cnt_datasets} datasets total...\n")

        # skip existing?
        ds_paths: list[str] = []
        if conf.FORCE_CREATE_SPLIT_STATS:
            ds_paths = source_datasets
        else:
            print(">>> Check existing dataset statistics to not re-create...\n")
            tsv_path: str = os.path.join(HERE, "data", "results", "tsv")
            json_path: str = os.path.join(HERE, "data", "results", "json")

            for p in source_datasets:
                lc: str = os.path.split(p)[1]
                ver: str = os.path.split(os.path.split(p)[0])[1].split("-")[2]
                tsv_fn: str = os.path.join(tsv_path, lc, f"{lc}_{ver}_splits.tsv")
                json_fn: str = os.path.join(json_path, lc, f"{lc}_{ver}_splits.json")
                if not (os.path.isfile(tsv_fn) and os.path.isfile(json_fn)):
                    ds_paths.append(p)
        # finish filter out existing

        cnt_to_process: int = len(ds_paths)
        print(f">>> We have {cnt_to_process} datasets queued to be processed...\n")

        # now process each dataset
        results: list[list[SplitStatsRec]]
        with mp.Pool(used_proc_count) as pool:
            results = pool.map(handle_dataset_splits, ds_paths)

        # done, first flatten them
        flattened: list[SplitStatsRec] = []
        for res in results:
            flattened += res

        print(f">>> Processed {len(flattened)} splits...\n")

    #
    # SUPPORT MATRIX
    #
    def main_support_matrix() -> None:
        """Handle support matrix"""
        nonlocal num_datasets, num_splits, num_algorithms

        print("\n=== Build Support Matrix ===\n")

        # Scan files once again (we could have run it partial)
        all_tsv_paths: list[str] = sorted(
            glob.glob(
                os.path.join(HERE, "data", "results", "tsv", "**", "*.tsv"),
                recursive=True,
            )
        )

        df: pd.DataFrame = pd.DataFrame().reset_index(drop=True)
        for tsv_path in all_tsv_paths:
            if (
                os.path.split(tsv_path)[1][0] != "$"
            ):  # ignore system files (starts with $)
                df = pd.concat([df.loc[:], df_read(tsv_path)]).reset_index(drop=True)

        num_splits = df.shape[0]
        num_datasets = df[df["sp"] == "validated"].shape[0]

        df = df[["ver", "lc", "alg"]].drop_duplicates()
        df = (
            df[~df["alg"].isnull()]
            .sort_values(["lc", "ver", "alg"])
            .reset_index(drop=True)
        )
        num_algorithms = df.shape[0]

        print(
            f"Read in - Datasets: {num_datasets}, Algorithms: {num_algorithms}, Splits: {num_splits}"
        )

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
        print(">>> Save Support Matrix")
        df_write(
            df_support_matrix,
            os.path.join(HERE, "data", "results", "tsv", f"{c.SUPPORT_MATRIX_FN}.tsv"),
        )
        df_support_matrix.to_json(
            os.path.join(HERE, "data", "results", "json", f"{c.SUPPORT_MATRIX_FN}.json"),
            orient="table",
            index=False,
        )

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
        print("\n=== Save Configuration ===\n")
        df_write(df, os.path.join(HERE, "data", "results", "tsv", "$config.tsv"))
        df.to_json(
            os.path.join(HERE, "data", "results", "json", "$config.json"),
            orient="table",
            index=False,
        )

    #
    # MAIN
    #
    used_proc_count: int = conf.DEBUG_PROC_COUNT if conf.DEBUG else PROC_COUNT
    print(
        f"=== Statistics Compilation Process for cv-tbox-dataset-analyzer ({used_proc_count} processes)==="
    )
    start_time: datetime = datetime.now()

    # PREPARE DIRECTORY STRUCTURES
    for lc in ALL_LOCALES:
        os.makedirs(os.path.join(HERE, "data", "results", "tsv", lc), exist_ok=True)
        os.makedirs(os.path.join(HERE, "data", "results", "json", lc), exist_ok=True)

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
    finish_time: datetime = datetime.now()
    process_timedelta: timedelta = finish_time - start_time
    process_seconds: float = process_timedelta.total_seconds()
    print(
        f"Finished compiling statistics for {num_datasets} datasets, {num_algorithms} algorithms, {num_splits} splits"
    )
    print(
        f"Duration {str(process_timedelta)} sec, avg={process_seconds/num_datasets} secs/dataset."
    )


if __name__ == "__main__":
    main()
