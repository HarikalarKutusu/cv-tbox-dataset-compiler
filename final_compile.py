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
from datetime import datetime
import os
import sys
import glob
import multiprocessing as mp

# External dependencies
from tqdm import tqdm
import pandas as pd

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
)
from lib import (
    dec1,
    df_concat,
    df_read,
    df_write,
    init_directories,
    dec3,
    calc_dataset_prefix,
    get_locales,
    report_results,
    sort_by_largest_file,
    mp_schedular,
)
from lib_final import handle_text_corpus, handle_reported, handle_dataset_splits

# Globals

HERE: str = os.path.dirname(os.path.realpath(__file__))
if not HERE in sys.path:
    sys.path.append(HERE)

ALL_LOCALES: list[str] = get_locales(c.CV_VERSIONS[-1])

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
# MAIN PROCESS
########################################################


def main() -> None:
    """Compile all data by calculating stats"""

    res_json_base_dir: str = os.path.join(
        conf.DATA_BASE_DIR, c.RES_DIRNAME, c.JSON_DIRNAME
    )
    res_tsv_base_dir: str = os.path.join(
        conf.DATA_BASE_DIR, c.RES_DIRNAME, c.TSV_DIRNAME
    )

    def ver2vercol(ver: str) -> str:
        """Converts a data version in format '11.0' to column/variable name format 'v11_0'"""
        return "v" + ver.replace(".", "_")

    #
    # TEXT-CORPORA
    #
    def main_text_corpora() -> None:
        """Handle all text corpora"""
        # nonlocal proc_count

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

        tc_base_dir: str = os.path.join(conf.DATA_BASE_DIR, c.TC_DIRNAME)
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
        versions: list[str] = (
            c.CV_VERSIONS.copy() if not conf.DEBUG else conf.DEBUG_CV_VER.copy()
        )
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
                    conf.DATA_BASE_DIR, c.TC_DIRNAME, "**", f"{c.TEXT_CORPUS_FN}.tsv"
                )
            )
            avg_size: int
            max_size: int
            pp, avg_size, max_size = sort_by_largest_file(pp)
            lc_complete_list: list[str] = [p.split(os.sep)[-2] for p in pp]
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

        proc_count: int
        chunk_size: int
        proc_count, chunk_size = mp_schedular(num_items, max_size, avg_size)
        print(
            f"Total: {g_tc.total_lc} Existing: {g_tc.skipped_exists} NoData: {g_tc.skipped_nodata} "
            + f"Remaining: {g_tc.processed_lc} Procs: {proc_count}  chunk_size: {chunk_size}..."
        )

        with mp.Pool(proc_count, maxtasksperchild=conf.CHUNKS_HARD_MAX) as pool:
            with tqdm(total=num_items, desc="") as pbar:
                for res in pool.imap_unordered(
                    handle_text_corpus, ver_lc_list, chunksize=chunk_size
                ):
                    results.extend(res)
                    # save_results()  # temporary saving: it takes a long time which might end, discard return
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

        vc_base_dir: str = os.path.join(conf.DATA_BASE_DIR, c.VC_DIRNAME)
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

        proc_count: int
        chunk_size: int
        proc_count, chunk_size = mp_schedular(num_items, 1, 1)
        print(
            f"Total: {g_rep.total_lc} Missing: {g_rep.skipped_nodata} Remaining: {g_rep.processed_lc} "
            + f"Procs: {proc_count}  chunk_size: {chunk_size}..."
        )
        results: list[ReportedStatsRec] = []
        with mp.Pool(proc_count, maxtasksperchild=conf.CHUNKS_HARD_MAX) as pool:
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
        vc_dir: str = os.path.join(conf.DATA_BASE_DIR, c.VC_DIRNAME)
        # get path part
        pp: list[str] = [
            os.path.split(p)[0]
            for p in sorted(
                glob.glob(os.path.join(vc_dir, "**", "validated.tsv"), recursive=True)
            )
        ]
        # sort by largest first
        avg_size: int
        max_size: int
        pp, avg_size, max_size = sort_by_largest_file(pp)

        tsv_path: str = os.path.join(conf.DATA_BASE_DIR, c.RES_DIRNAME, c.TSV_DIRNAME)
        json_path: str = os.path.join(conf.DATA_BASE_DIR, c.RES_DIRNAME, c.JSON_DIRNAME)
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
        num_items: int = len(ds_paths)

        if num_items == 0:
            print("Nothing to process")
            return

        proc_count: int
        chunk_size: int
        proc_count, chunk_size = mp_schedular(num_items, max_size, avg_size)
        print(
            f"Processing {num_items} locales in {proc_count} processes with chunk_size {chunk_size}..."
        )

        # now process each dataset
        with mp.Pool(proc_count, maxtasksperchild=conf.CHUNKS_HARD_MAX) as pool:
            with tqdm(total=num_items, desc="") as pbar:
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
                    conf.DATA_BASE_DIR,
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
            conf.DATA_BASE_DIR,
            c.RES_DIRNAME,
            c.TSV_DIRNAME,
            "$vc_stats.tsv",
        )
        df_write(df, dst)

        # clean
        df = df.drop(
            columns=list(set(df.columns) - set(["ver", "lc", "alg", "sp", "dur_total"]))
        )

        # get some stats
        g.total_splits = df.shape[0]
        g.total_lc = df["lc"].unique().shape[0]

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
            conf.DATA_BASE_DIR,
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
                conf.DATA_BASE_DIR, c.RES_DIRNAME, c.TSV_DIRNAME, "$config.tsv"
            ),
        )
        df.to_json(
            os.path.join(
                conf.DATA_BASE_DIR, c.RES_DIRNAME, c.JSON_DIRNAME, "$config.json"
            ),
            orient="table",
            index=False,
        )

    #
    # MAIN
    #
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


def mp_test():

    def test(patt: str):
        pp: list[str] = glob.glob(patt, recursive=True)
        print(patt, len(pp))
        num_items: int = len(pp)
        avg_size: int
        max_size: int
        pp, avg_size, max_size = sort_by_largest_file(pp)
        proc_count: int
        chunk_size: int
        proc_count, chunk_size = mp_schedular(num_items, max_size, avg_size)
        print(
            num_items, max_size // 1000000, avg_size // 1000000, proc_count, chunk_size
        )

    patt: str = os.path.join(conf.DATA_BASE_DIR, c.VC_DIRNAME, "**", "validated.tsv")
    test(patt)
    patt: str = os.path.join(conf.DATA_BASE_DIR, c.VC_DIRNAME, "**", "reported.tsv")
    test(patt)
    patt: str = os.path.join(conf.DATA_BASE_DIR, c.TC_DIRNAME, "**", "$text_corpus.tsv")
    test(patt)
    sys.exit()


if __name__ == "__main__":
    # mp_test()
    print("=== cv-tbox-dataset-analyzer - Final Statistics Compilation ===")
    init_directories(HERE)
    main()
