#!/usr/bin/env python3

###########################################################################
# final_compile.py
#
# From all data, compile result statistics data to be used in
# cv-tbox-dataset-analyzer
#
# Use:
# python final_compile.py
#
#
#
# This script is part of Common Voice ToolBox Package
#
# [github]
# [copyright]
###########################################################################

import sys
import os
import glob
import csv
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

# MULTIPROCESSING
import multiprocessing as mp
import psutil

# This package
import const

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

#
# Constants
#

# Program parameters
VERBOSE: bool = False
FAIL_ON_NOT_FOUND: bool = True
PROC_COUNT: int = psutil.cpu_count(logical=False) - 1

COMPILE_THESE: "list[str]" = const.CV_VERSIONS
# COMPILE_THESE: "list[str]" = ['6.1']              # A samaller one for debugging purposes

cnt_datasets: int = 0


########################################################
# Tooling
########################################################

def df_read(fpath: str) -> pd.DataFrame:
    """Read a tsv file into a dataframe"""
    if not os.path.isfile(fpath):
        print(f'FATAL: File {fpath} cannot be located!')
        if FAIL_ON_NOT_FOUND:
            sys.exit(1)

    df: pd.DataFrame = pd.read_csv(
        fpath,
        sep="\t",
        parse_dates=False,
        engine="python",
        encoding="utf-8",
        on_bad_lines='skip',
        quotechar='"',
        quoting=csv.QUOTE_NONE,
    )
    return df


def df_write(df: pd.DataFrame, fpath: str) -> bool:
    """
    Writes out a dataframe to a file.
    """
    # Create/override the file
    df.to_csv(
        fpath,
        header=True,
        index=False,
        encoding="utf-8",
        sep='\t',
        escapechar='\\',
        quoting=csv.QUOTE_NONE,
        # float_format="%.4f"
    )
    if VERBOSE:
        print(f'Generated: {fpath} Records={df.shape[0]}')
    return True


def df_int_convert(x: pd.Series) -> Any:
    try:
        return x.astype(int)
    except:
        return x


def list2str(lst: "list[Any]") -> str:
    return const.SEP_COL.join(str(x) for x in lst)


def arr2str(arr: "list[list[Any]]") -> str:
    return const.SEP_ROW.join(list2str(x) for x in arr)

########################################################
# Split Stats
########################################################


def split_stats(cv_idx: int) -> "list[dict[str,Any]]":
    global cnt_datasets
    #
    # Handle one split, this is where calculations happen
    #
    # The default column structure of CV dataset splits is as follows
    # client_id, path, sentence, up_votes, down_votes, age, gender, accents, locale, segment

    # we have as input:
    # 'version', 'locale', 'algorithm', 'split'

    # now, do calculate some statistics. We need:
    # 'clips', 'unique_voices', 'unique_sentences', 'duplicate_sentences',
    # 'genders_nodata', 'genders_male', 'genders_female', 'genders_other',
    # 'ages_nodata', 'ages_teens', 'ages_twenties', 'ages_thirties', 'ages_fourties', 'ages_fifties', 'ages_sixties', 'ages_seventies', 'ages_eighties', 'ages_nineties'

    def handle_split(ver: str, lc: str, algorithm: str, split: str, fpath: str) -> "dict[str, Any]":
        """Processes a single split and return calculated values"""

        nonlocal df_clip_durations, df_text_corpus

        # def _get_row_total(pt: pd.DataFrame, lbl: str) -> int:
        #     # return int(pd.to_numeric(pt.loc[lbl, 'TOTAL'])) if lbl in list(pt.index.values) else int(0)
        #     return int(pd.to_numeric(pt.loc[lbl, 'TOTAL']))

        # def _get_col_total(pt: pd.DataFrame, lbl: str) -> int:
        #     # return int(pd.to_numeric(pt.loc['TOTAL', lbl])) if lbl in list(pt.columns.values) else int(0)
        #     return pt.at['TOTAL', lbl]

        # === START ===
        # print("ver=", ver, "lc=", lc, "algorithm=", algorithm, "split=", split)

        # Read in DataFrames
        if split != 'clips':
            df: pd.DataFrame = df_read(fpath)
        else: # build "clips" from val+inval+other
            df: pd.DataFrame = df_read(fpath) # we passed validated here, first read it.
            df2: pd.DataFrame = df_read(fpath.replace('validated', 'invalidated'))  # add invalidated
            df = pd.concat( [df.loc[:], df2] )
            df2: pd.DataFrame = df_read(fpath.replace('validated', 'other'))  # add other
            df = pd.concat( [df.loc[:], df2] )

        # Do nothing, if there is no data
        if df.shape[0] == 0:
            results = {
                'ver':  ver,
                'lc':   lc,
                'alg':  algorithm,
                'sp':   split,
            }
            return results

        # Replace NA with NODATA
        df.fillna(value=const.NODATA, inplace=True)
        # add lowercase sentence column
        df['sentence_lower'] = df['sentence'].str.lower()

        #
        # DURATIONS
        #

        # Connect with duration table
        df["duration"] = df["path"].map(df_clip_durations["duration"])
        # Calc duration agregate values
        ser: pd.Series = df["duration"]
        if df_clip_durations.shape[0] > 0:
            duration_total: float = ser.sum()
            duration_mean: float = ser.mean()
            duration_median: float = ser.median()
            # Calc duration distribution
            arr: np.ndarray = np.fromiter(df["duration"].dropna().apply(
                int).reset_index(drop=True).to_list(), int)
            hist: list[list[int]] = np.histogram(arr, bins=const.BINS_DURATION)
            duration_freq: "list[int]" = hist[0]
            # duration_bins: "list[int]" = hist[1]
        else:  # No Duration data, set illegal defaults and continue
            duration_total: float = -1
            duration_mean: float = -1
            duration_median: float = -1
            duration_freq: "list[int]" = []
            # duration_bins: "list[int]" = []

        #
        # VOICES
        #
        voice_counts: pd.DataFrame = df["client_id"].value_counts(
        ).to_frame().reset_index()
        voice_counts.rename(
            columns={"index": "voice", "client_id": "recordings"}, inplace=True)
        voice_mean: float = voice_counts["recordings"].mean()
        voice_median: float = voice_counts["recordings"].median()
        # Calc speaker recording distribution
        arr: np.ndarray = np.fromiter(voice_counts["recordings"].dropna().apply(
            int).reset_index(drop=True).to_list(), int)
        hist: list[list[int]] = np.histogram(arr, bins=const.BINS_VOICES)
        voice_freq: "list[int]" = hist[0]
        # voice_bins: "list[int]" = hist[1]

        #
        # SENTENCES
        #
        sentence_counts: pd.DataFrame = df["sentence"].value_counts(
        ).to_frame().reset_index()
        # print(sentence_counts)
        # sys.exit()

        sentence_counts.rename(
            columns={"index": "sentence", "sentence": "recordings"}, inplace=True)
        sentence_mean: float = sentence_counts["recordings"].mean()
        sentence_median: float = sentence_counts["recordings"].median()
        # Calc speaker recording distribution
        arr: np.ndarray = np.fromiter(sentence_counts["recordings"].dropna().apply(
            int).reset_index(drop=True).to_list(), int)
        hist: list[list[int]] = np.histogram(arr, bins=const.BINS_SENTENCES)
        sentence_freq: "list[int]" = hist[0]
        sentence_bins: "list[int]" = hist[1]

        # print(duration_freq)
        # print(voice_freq)
        # print(sentence_freq)
        # sys.exit()

        # basic measures
        clips_cnt: int = df.shape[0]
        unique_voices: int = df['client_id'].unique().shape[0]
        unique_sentences: int = df['sentence'].unique().shape[0]
        unique_sentences_lower: int = df['sentence_lower'].unique().shape[0]
        duplicate_sentence_cnt: int = clips_cnt - unique_sentences
        duplicate_sentence_cnt_lower: int = clips_cnt - unique_sentences_lower

        # get a pt for all demographics
        _pt: pd.DataFrame = pd.pivot_table(
            df, values='path', index=['age'], columns=['gender'], aggfunc='count',
            fill_value=0, dropna=False, margins=True, margins_name='TOTAL'
        )
        # print(_pt)
        pt_ages: list[str] = const.CV_AGES.copy()
        pt_ages.append("TOTAL")
        pt_genders: list[str] = const.CV_GENDERS.copy()
        pt_genders.append("TOTAL")
        _pt = _pt.reindex(pt_ages, axis=0)
        _pt = _pt.reindex(pt_genders, axis=1)
        _pt = _pt.fillna(value=0).astype(int)
        # print(_pt)

        # _get_col_total(_pt, 'male')
        _males: int = _pt.at['TOTAL', 'male'].item()
        _females: int = _pt.at['TOTAL', 'female'].item()

        results: dict[str, Any] = {
            'ver':          ver,
            'lc':           lc,
            'alg':          algorithm,
            'sp':           split,

            'clips':        clips_cnt,
            'uq_v':         unique_voices,
            'uq_s':         unique_sentences,
            'uq_sl':        unique_sentences_lower,
            'dup_s':        duplicate_sentence_cnt,
            'dup_sl':       duplicate_sentence_cnt_lower,
            # 'dup_r':        duplicate_sentence_cnt / unique_sentences,
            # 'dup_rl':       duplicate_sentence_cnt_lower / unique_sentences_lower,

            # 'g_m':          _males,
            # 'g_f':          _females,
            # 'g_o':          _pt.at['TOTAL', 'other'].item(),
            # 'g_nd':         _pt.at['TOTAL', const.NODATA].item(),

            # 'a_t':          _pt.at['teens', 'TOTAL'].item(),
            # 'a_20':         _pt.at['twenties', 'TOTAL'].item(),
            # 'a_30':         _pt.at['thirties', 'TOTAL'].item(),
            # 'a_40':         _pt.at['fourties', 'TOTAL'].item(),
            # 'a_50':         _pt.at['fifties', 'TOTAL'].item(),
            # 'a_60':         _pt.at['sixties', 'TOTAL'].item(),
            # 'a_70':         _pt.at['seventies', 'TOTAL'].item(),
            # 'a_80':         _pt.at['eighties', 'TOTAL'].item(),
            # 'a_90':         _pt.at['nineties', 'TOTAL'].item(),
            # 'a_nd':         _pt.at[const.NODATA, 'TOTAL'].item(),

            # 'g_f_m_ratio':        _females / _males if _males > 0 else -1,

            # TODO Add More columns

            # Duration
            'dur_total':    duration_total,
            'dur_mean':     duration_mean,
            'dur_median':   duration_median,
            'dur_freq':     list2str(duration_freq),

            # Recordings per Voice
            'v_mean':       voice_mean,
            'v_median':     voice_median,
            'v_freq':       list2str(voice_freq),

            # Recordings per Sentence
            's_mean':       sentence_mean,
            's_median':     sentence_median,
            's_freq':       list2str(sentence_freq),

            # Demographics distribution
            'dem_table':    arr2str(_pt.to_numpy(int).tolist()),

            # Tables & lists
            # 'dur_bins':     list2str(duration_bins),
            # 'v_bins':       list2str(voice_bins),
            # 's_bins':       list2str(sentence_bins),

        }

        # print(results)
        # sys.exit()

        return results
    # END handle_split

    # --------------------------------------------
    # START main process for a single CV version
    # --------------------------------------------

    # Calc CV_DIR
    ver: str = const.CV_VERSIONS[cv_idx]
    cv_dir_name: str = "cv-corpus-" + ver + "-" + \
        const.CV_DATES[cv_idx]  # TODO Handle older version naming
    # Calc voice-corpus directory
    cv_dir: str = os.path.join(HERE, "data", "voice-corpus", cv_dir_name)

    # Get a list of available language codes
    lc_paths: "list[str]" = glob.glob(
        os.path.join(cv_dir, '*'), recursive=False)
    lc_list: "list[str]" = []
    for lc_path in lc_paths:
        if os.path.isdir(lc_path): # ignore files
            lc_list.append(os.path.split(lc_path)[1])

    cnt_datasets += len(lc_list)

    # Loop all locales
    cnt: int = 0
    res_all: "list[dict[str,Any]]" = []
    for lc in lc_list:
        cnt += 1
        res: "list[dict[str,Any]]" = []
        # print('\033[F' + ' ' * 80)
        # print(f'\033[FProcessing locale {cnt}/{len(lc_list)} : {lc}')
        print(f'Processing {ver} locale {cnt}/{len(lc_list)} : {lc}')

        # Source directories
        vc_dir: str = os.path.join(cv_dir, lc)
        tc_dir: str = os.path.join(HERE, "data", "text-corpus", lc)
        cd_dir: str = os.path.join(HERE, "data", "clip-durations", lc)

        # Create destinations if thet do not exist
        tsv_path: str = os.path.join(HERE, "data", 'results', 'tsv', lc)
        json_path: str = os.path.join(HERE, "data", 'results', 'json', lc)
        os.makedirs(tsv_path, exist_ok=True)
        os.makedirs(json_path, exist_ok=True)

        #
        # First Handle Splits in voice-corpus
        #

        # Load general DF's if they exist, else initialize

        # Clip Durations
        cd_file: str = os.path.join(cd_dir, '$clip_durations.tsv')
        if os.path.isfile(cd_file):
            df_clip_durations: pd.DataFrame = df_read(
                cd_file).set_index("clip")
        else:
            print(f'WARNING: No duration data for {lc}\n')
            df_clip_durations: pd.DataFrame = pd.DataFrame(
                columns=const.COLS_CLIP_DURATIONS).set_index("clip")

        # Text Corpus
        tc_file: str = os.path.join(tc_dir, '$text_corpus.tsv')
        if os.path.isfile(tc_file):
            df_text_corpus: pd.DataFrame = df_read(
                tc_file)  # .set_index("sentence")
        else:
            print(f'WARNING: No text-corpus for {lc}\n')
            df_text_corpus: pd.DataFrame = pd.DataFrame(
                columns=const.COLS_TEXT_CORPUS)  # .set_index("sentence")

        # MAIN SPLITS

        # Special case for temporary "clips.tsv"
        res.append(handle_split(
            ver=ver,
            lc=lc,
            algorithm="",
            split="clips",
            fpath=os.path.join(vc_dir, 'validated.tsv') # to start with we set validated
        ))

        # def handle_split(ver: str, lc: str, algorithm: str, split: str, fpath: str) -> "dict[str, Any]":
        for sp in const.MAIN_SPLITS:
            res.append(handle_split(
                ver=ver,
                lc=lc,
                algorithm="",
                split=sp,
                fpath=os.path.join(vc_dir, sp + '.tsv')
            ))

        # SPLITTING ALGO SPECIFIC (inc default splits)

        for algo in const.ALGORITHMS:
            if os.path.isdir(os.path.join(vc_dir, algo)):
                for sp in const.TRAINING_SPLITS:
                    res.append(handle_split(
                        ver=ver,
                        lc=lc,
                        algorithm=algo,
                        split=sp,
                        fpath=os.path.join(vc_dir, algo, sp + '.tsv')
                    ))

        # Create DataFrames
        df: pd.DataFrame = pd.DataFrame(res)
        df_write(df, os.path.join(tsv_path, f"{lc}_{ver}_splits.tsv"))
        df.to_json(os.path.join(json_path, f"{lc}_{ver}_splits.json"),
                   orient='table', index=False)
        res_all.extend(res)

    return res_all


########################################################
# Reported Stats
########################################################


# def reported_stats(lc_list: "list[str]") -> None:
#     pass

########################################################
# Text-Corpus Stats (Multi Processing Handler)
########################################################

def handle_text_corpus(lc: str) -> "dict[str,Any]":

    print(f'Processing text-corpus for locale: {lc}')

    tc_dir: str = os.path.join(HERE, "data", "text-corpus", lc)
 
    tc_file: str = os.path.join(tc_dir, '$text_corpus.tsv')
    tokens_file: str = os.path.join(tc_dir, '$tokens.tsv')

    # TEXT CORPUS

    # "file", "sentence", "lower", "normalized", "chars", "words", 'valid'
    df: pd.DataFrame = df_read(tc_file)

    # basic measures
    sentence_cnt: int = df.shape[0]
    unique_sentences: int = df['sentence'].dropna().unique().shape[0]
    unique_normalized: int = df['normalized'].dropna().unique().shape[0]
    valid: int = df[df['valid'].dropna().astype(int) == 1].shape[0]

    chars_total: int = df['chars'].dropna().sum()
    chars_mean: float = df['chars'].dropna().mean()
    chars_median: int = int(df['chars'].dropna().median())

    has_val: int = 0
    words_total: int = df['words'].dropna().sum()
    words_mean: float = 0
    words_median: int = 0
    if words_total != 0:
        has_val = 1
        words_mean: float = df['words'].dropna().mean()
        words_median: int = int(df['words'].dropna().median())

    # freq distributions

    # Calc character length distribution
    arr: np.ndarray = np.fromiter(df["chars"].dropna().apply(
        int).reset_index(drop=True).to_list(), int)
    hist: list[list[int]] = np.histogram(arr, bins=const.BINS_CHARS)
    character_freq: "list[int]" = hist[0]
    character_bins: "list[int]" = hist[1]

    # Calc word count distribution
    word_freq: "list[int]" = []
    if has_val == 1:
        arr: np.ndarray = np.fromiter(df["words"].dropna().apply(
            int).reset_index(drop=True).to_list(), int)
        hist: list[list[int]] = np.histogram(arr, bins=const.BINS_WORDS)
        word_freq: "list[int]" = hist[0]
        word_bins: "list[int]" = hist[1]

    # TOKENS

    tokens_total: int = 0
    tokens_mean: float = 0
    tokens_median: int = 0
    token_freq: "list[int]" = []
    if has_val == 1:
        df: pd.DataFrame = df_read(tokens_file)
        # "token", "count"
        tokens_total: int = df.shape[0]
        tokens_mean: float = df['count'].mean()
        tokens_median: int = int(df['count'].median())
        # Token/word repeat distribution
        arr: np.ndarray = np.fromiter(df["count"].dropna().apply(
            int).reset_index(drop=True).to_list(), int)
        hist: list[list[int]] = np.histogram(arr, bins=const.BINS_TOKENS)
        token_freq: "list[int]" = hist[0]
        token_bins: "list[int]" = hist[1]


    results: "dict[str, Any]" = {
        'lc': lc,
        's_cnt': sentence_cnt,
        'uq_s': unique_sentences,
        'uq_n': unique_normalized,
        'has_val': has_val,
        'val': valid,
        'c_total': chars_total,
        'c_mean': chars_mean,
        'c_median': chars_median,
        'c_freq': list2str(character_freq),
        'w_total': words_total,
        'w_mean': words_mean,
        'w_median': words_median,
        'w_freq': list2str(word_freq),
        't_total': tokens_total,
        't_mean': tokens_mean,
        't_median': tokens_median,
        't_freq': list2str(token_freq),
    }

    return results

########################################################
# Split Stats (Multi Processing Handler)
########################################################

def handle_version(cv_ver: str) -> "list[dict[str,Any]]":
    # Fins idx
    cv_idx: int = const.CV_VERSIONS.index(cv_ver)
    # Start Process
    return split_stats(cv_idx)

########################################################
# MAIN PROCESS
########################################################

def main() -> None:

    def ver2vercol(ver: str) -> str:
        """Converts a data version in format '11.0' to column/variable name format 'v11_0' """
        return 'v' + ver.replace('.', '_')


    print(f'=== Statistics Compilation Process for cv-tbox-dataset-analyzer ({PROC_COUNT} processes)===')
    start_time: datetime = datetime.now()

    #
    # MultiProcessing on text-corpora: Loop all locales, each locale in one process
    #
    print('\n=== Start Text Corpora Analysis ===\n')
    
    # First get list of languages with text corpora
    tc_base: str = os.path.join(HERE, "data", "text-corpus")
    lc_paths: "list[str]" = glob.glob(
        os.path.join(tc_base, '*'), recursive=False)
    lc_list: "list[str]" = []
    for lc_path in lc_paths:
        if os.path.isdir(lc_path): # ignore files
            lc_list.append(os.path.split(lc_path)[1])

    # Now multi-process each lc
    with mp.Pool(PROC_COUNT) as pool:
        tc_stats: list[dict[str, Any]] = pool.map(
            handle_text_corpus, lc_list)

    # done, first flatten it
    # all_tc: list[dict[str, Any]] = []
    # for res in res_tc:
    #     all_tc.append(res)

    # Create result DF
    # df: pd.DataFrame = pd.DataFrame(tc_stats, columns=const.COLS_TEXT_CORPUS)
    df: pd.DataFrame = pd.DataFrame(tc_stats)
    df_write(df, os.path.join(
        HERE, "data", "results", 'tsv', '$text_corpus_stats.tsv'))
    df.to_json(os.path.join(
        HERE, 'data', 'results', 'json', '$text_corpus_stats.json'),
        orient='table', index=False)

    #
    # MultiProcessing on versions to handle splits: Loop all versions/languages/splits, each version in one process (TODO not ideal, should be refactored)
    #
    print('\n=== Start Dataset/Split Analysis ===\n')

    with mp.Pool(PROC_COUNT) as pool:
        results: list[list[dict[str, Any]]] = pool.map(handle_version, COMPILE_THESE)

    # done, first flatten it

    all_splits: list[dict[str, Any]] = []
    for res in results:
        all_splits.extend(res)

    # print(all_splits)
    # print(len(all_splits))

    # next form the support matrix
    df: pd.DataFrame = pd.DataFrame(all_splits).reset_index(drop=True)
    df = df[['ver', 'lc', 'alg']]
    df.drop_duplicates(['ver', 'lc', 'alg'], inplace=True)
    df.sort_values(['lc', 'ver', 'alg'], inplace=True)
    df = df[ df['alg'] != ""].reset_index(drop=True)
    # print(df)
    # sys.exit()

    # Prepare Support Matrix DataFrame
    rev_versions: "list[str]" = const.CV_VERSIONS.copy()
    rev_versions.reverse()
    for inx, ver in enumerate(rev_versions):
        rev_versions[inx] = ver2vercol(ver)
    COLS_SUPPORT_MATRIX: "list[str]" = ['lc', 'lang']
    COLS_SUPPORT_MATRIX.extend(rev_versions)
    df_support_matrix: pd.DataFrame = pd.DataFrame(
        index=const.ALL_LOCALES,
        columns=COLS_SUPPORT_MATRIX,
    )
    df_support_matrix['lc'] = const.ALL_LOCALES

    # Now loop and put the results inside
    for lc in const.ALL_LOCALES:
        for ver in const.CV_VERSIONS:
            subset: pd.DataFrame = df[(df['lc'] == lc ) & (df['ver'] == ver)]
            algo: "list[str]" = subset['alg'].unique().tolist()
            algos: str = const.SEP_ALGO.join(algo)
            df_support_matrix.at[lc, ver2vercol(ver)] = algos

    # Write out
    df_write(df_support_matrix, os.path.join(
        HERE, 'data', 'results', 'tsv', '$support_matrix.tsv'))
    df_support_matrix.to_json(os.path.join(
        HERE, 'data', 'results', 'json', '$support_matrix.json'),
        orient='table', index=False)

    finish_time: datetime = datetime.now()
    process_timedelta: timedelta = finish_time - start_time
    process_seconds: float = process_timedelta.total_seconds()
    print(
        f'Finished compiling statistics for {df.shape[0]} datasets in {str(process_timedelta)} secs, avg={process_seconds/df.shape[0]} secs.')


if __name__ == '__main__':
    main()
