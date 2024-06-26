"""cv-tbox dataset analyzer - Simple library for helper functions"""

# Standard Lib
# from ast import literal_eval
from datetime import datetime
from typing import Literal, Any, Tuple
from urllib.request import urlopen
import os
import sys
import csv
import json
import multiprocessing as mp

# External dependencies
from git import Repo
import pandas as pd
import psutil

# Module
from typedef import GitRec, Globals
import const as c
import conf

#
# Application Specific
#


def init_directories(basedir: str) -> None:
    """Creates data directory structures"""
    data_dir: str = os.path.join(basedir, c.DATA_DIRNAME)
    # if os.path.isfile(os.path.join(data_dir, ".gitkeep")):
    #     return

    print("Preparing directory structures...")
    # get these either from cv-dataset repo clone or API data (cv_datasets)
    all_locales: list[str] = get_locales(c.CV_VERSIONS[-1])
    for lc in all_locales:
        os.makedirs(os.path.join(data_dir, c.CD_DIRNAME, lc), exist_ok=True)
        os.makedirs(os.path.join(data_dir, c.TC_DIRNAME, lc), exist_ok=True)
        os.makedirs(
            os.path.join(data_dir, c.RES_DIRNAME, c.TSV_DIRNAME, lc), exist_ok=True
        )
        os.makedirs(
            os.path.join(data_dir, c.RES_DIRNAME, c.JSON_DIRNAME, lc), exist_ok=True
        )
    for ver in c.CV_VERSIONS:
        ver_lc: list[str]
        ds_prefix: str = calc_dataset_prefix(ver)
        ver_lc: list[str] = get_locales(ver)
        os.makedirs(
            os.path.join(data_dir, c.TC_ANALYSIS_DIRNAME, ds_prefix),
            exist_ok=True,
        )
        for lc in ver_lc:
            os.makedirs(
                os.path.join(data_dir, c.VC_DIRNAME, ds_prefix, lc),
                exist_ok=True,
            )
    # create .gitkeep
    open(os.path.join(data_dir, ".gitkeep"), "a", encoding="utf8").close()
    open(os.path.join(data_dir, c.CD_DIRNAME, ".gitkeep"), "a", encoding="utf8").close()
    open(
        os.path.join(data_dir, c.RES_DIRNAME, ".gitkeep"), "a", encoding="utf8"
    ).close()
    open(
        os.path.join(data_dir, c.RES_DIRNAME, c.TSV_DIRNAME, ".gitkeep"),
        "a",
        encoding="utf8",
    ).close()
    open(
        os.path.join(data_dir, c.RES_DIRNAME, c.JSON_DIRNAME, ".gitkeep"),
        "a",
        encoding="utf8",
    ).close()
    open(os.path.join(data_dir, c.TC_DIRNAME, ".gitkeep"), "a", encoding="utf8").close()
    open(
        os.path.join(data_dir, c.TC_ANALYSIS_DIRNAME, ".gitkeep"), "a", encoding="utf8"
    ).close()
    open(os.path.join(data_dir, c.VC_DIRNAME, ".gitkeep"), "a", encoding="utf8").close()

    # outside common cache
    os.makedirs(conf.CV_TBOX_CACHE, exist_ok=True)
    os.makedirs(os.path.join(conf.CV_TBOX_CACHE, c.CLONES_DIRNAME), exist_ok=True)
    os.makedirs(os.path.join(conf.CV_TBOX_CACHE, c.API_DIRNAME), exist_ok=True)
    open(os.path.join(conf.CV_TBOX_CACHE, ".gitkeep"), "a", encoding="utf8").close()
    open(
        os.path.join(conf.CV_TBOX_CACHE, c.CLONES_DIRNAME, ".gitkeep"),
        "a",
        encoding="utf8",
    ).close()
    open(
        os.path.join(conf.CV_TBOX_CACHE, c.API_DIRNAME, ".gitkeep"),
        "a",
        encoding="utf8",
    ).close()


def report_results(g: Globals) -> None:
    """Prints out simple report from global counters"""
    process_seconds: float = (datetime.now() - g.start_time).total_seconds()
    print("=" * 80)
    print(
        f"Total\t\t: Ver: {g.total_ver} LC: {g.total_lc} Algo: {g.total_algo} Splits: {g.total_splits}"
    )
    print(
        f"Processed\t: Ver: {g.processed_ver} LC: {g.processed_lc} Algo: {g.processed_algo}"
    )
    print(f"Skipped\t\t: Exists: {g.skipped_exists} No Data: {g.skipped_nodata}")
    print(
        f"Duration(s)\t: Total: {dec3(process_seconds)} Avg: {dec3(process_seconds/ g.processed_lc) if g.processed_lc > 0 else '-'}"
    )


#
# DataFrames
#


def df_read(
    fpath: str, dtypes: dict | None = None, use_cols: list[str] | None = None
) -> pd.DataFrame:
    """Read a tsv file into a dataframe"""
    _df: pd.DataFrame = pd.DataFrame()
    if not os.path.isfile(fpath):
        print(f"FATAL: File {fpath} cannot be located!")
        if conf.FAIL_ON_NOT_FOUND:
            sys.exit(1)
        return _df

    _df = pd.read_csv(
        fpath,
        sep="\t",
        parse_dates=False,
        encoding="utf-8",
        # on_bad_lines="skip",
        on_bad_lines="warn",
        quotechar='"',
        quoting=csv.QUOTE_NONE,
        skip_blank_lines=True,
        engine="python",  # "pyarrow"
        usecols=use_cols,
        dtype_backend="pyarrow",
        dtype=dtypes,
    )
    return _df


def df_write(df: pd.DataFrame, fpath: Any, mode: Any = "w") -> bool:
    """
    Writes out a dataframe to a file.
    """

    _head: bool = mode != "a"
    # Create/override the file
    df.to_csv(
        fpath,
        mode=mode,
        header=_head,
        index=False,
        encoding="utf-8",
        sep="\t",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )
    # float_format="%.4f"
    if conf.DEBUG:
        print(f"Generated: {fpath} Records={df.shape[0]}")
    return True


def df_int_convert(x: pd.Series) -> Any:
    """Convert columns to int if possible"""
    try:
        return x.astype(int)
    except ValueError as e:  # pylint: disable=W0612
        return x


def df_concat(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Controlled concat of two dataframes"""
    return (
        df1
        if df2.shape[0] == 0
        else df2 if df1.shape[0] == 0 else pd.concat([df1, df2])
    )


#
# Safe DataFrame readers for Common Voice to handle CRLF/LF/TAB cases in text
#


def df_read_safe_tc_validated(fpath: str) -> Tuple[pd.DataFrame, list[str]]:
    """Read in possibly malformed validated_sentences.tsv file"""

    def has_valid_columns(ss: list[str]) -> bool:
        """Check if we have an acceptable row"""
        return (
            len(ss) == col_count_needed
            and len(ss[0]) == 64
            and ss[-2].isdigit()
            and ss[-1].isdigit()
        )

    lines_read: list[str] = []
    final_arr: list[list[str]] = []
    problem_arr: list[str] = []
    col_count_needed: int = len(c.FIELDS_TC_VALIDATED)

    # read all data and split to lines
    with open(fpath, encoding="utf8") as fd:
        lines_read = fd.read().splitlines()

    total_lines: int = len(lines_read)

    # we expect these:
    # sentence_id	sentence	sentence_domain	source	is_used	clips_count
    # first line is column names, so we skip it because we predefine them
    cur_source_line: int = 1
    while cur_source_line < len(lines_read):
        line: str = lines_read[cur_source_line].replace("\r\n", "\n").replace("\n", "")
        ss1: list[str] = line.split("\t")

        # No problem: We have good data (most common)
        # Action: Get it
        if has_valid_columns(ss1):
            final_arr.append(ss1)
            cur_source_line += 1
            continue

        # Problem-1: More columns than needed, most probably caused by having tab character(s) inside "sentence"
        # Solution: Squieze sentence field(s) into one
        if len(ss1) > col_count_needed:
            while not has_valid_columns(ss1) and len(ss1) > col_count_needed:
                ss1[1] = (ss1[1] + ss1[2]).replace("\t", " ").replace("  ", " ")
                ss1.pop(2)
            if has_valid_columns(ss1):
                final_arr.append(ss1)
                cur_source_line += 1
                continue

        # Problem-2: Fewer columns than needed, most probably caused by having newline character(s) inside "sentence"
        # Solution: Look ahead more lines until corrected
        # Check if we are at the last line!
        look_ahead: int = 0
        if len(ss1) < col_count_needed and cur_source_line < total_lines:
            ss2: list[str] = ss1
            while not has_valid_columns(ss2) and len(ss1) < col_count_needed:
                if cur_source_line + look_ahead >= total_lines - 1:
                    break
                look_ahead += 1
                next_line: str = (
                    lines_read[cur_source_line + look_ahead]
                    .replace("\r\n", "\n")
                    .replace("\n", "")
                )
                line = (line + " " + next_line).replace("  ", " ")
                ss2 = line.split("\t")
            if has_valid_columns(ss2):
                final_arr.append(ss2)
                cur_source_line += look_ahead + 1
                continue

        # FATAL: If we reached here, we have an unhandled case
        # We should skip these lines and report problem lines
        for line in lines_read[cur_source_line : cur_source_line + look_ahead]:
            problem_arr.append(line)
        cur_source_line += look_ahead + 1

        # EOF check
        if cur_source_line >= total_lines - 1:
            break
    # end of loop

    df_final: pd.DataFrame = (
        pd.DataFrame(final_arr, columns=c.FIELDS_TC_VALIDATED)
        .astype(c.FIELDS_TC_VALIDATED)
        .drop_duplicates()
    )
    return df_final, problem_arr


def df_read_safe_reported(fpath: str) -> Tuple[pd.DataFrame, list[str]]:
    """Read in possibly malformed reported.tsv file"""

    def has_valid_columns(ss: list[str]) -> bool:
        """Check if we have an acceptable row"""
        return len(ss) == col_count_needed and len(ss[sentence_id_inx]) == 64

    ver: str = fpath.split(os.sep)[-3].split("-")[2]
    fields: dict[str, pd.ArrowDtype] = (
        c.FIELDS_REPORTED if float(ver) >= 17.0 else c.FIELDS_REPORTED_OLD
    )

    # look_ahead: int = 0
    lines_read: list[str] = []
    final_arr: list[list[str]] = []
    problem_arr: list[str] = []
    col_count_needed: int = len(fields)

    # read all data and split to lines
    with open(fpath, encoding="utf8") as fd:
        lines_read = fd.read().splitlines()

    # get first line to find the column indexes (they changed in versions)
    sentence_inx: int = lines_read[0].split("\t").index("sentence")
    sentence_id_inx: int = lines_read[0].split("\t").index("sentence_id")

    # we expect these:
    # sentence_id	sentence	locale	reason
    # first line is column names, so we skip it because we predefine them

    while lines_read:
        line: str = lines_read.pop(0).replace("\r\n", "\n").replace("\n", "")
        ss1: list[str] = line.split("\t")

        next_line: str = (
            lines_read[0].replace("\r\n", "\n").replace("\n", "") if lines_read else ""
        )
        ss2: list[str] = next_line.split("\t")

        # No problem: We have good data in this line and next (most common)
        # Action: Get it
        if (
            has_valid_columns(ss1)
            and next_line
            and len(ss2) == col_count_needed
            and has_valid_columns(ss2)
        ) or (has_valid_columns(ss1) and not next_line):
            final_arr.append(ss1)
            continue

        # Problem-1: More columns than needed, most probably caused by having tab character(s) inside fields
        # Solution: Squieze sentence field(s) into one
        if len(ss1) > col_count_needed:
            while not has_valid_columns(ss1):
                ss1[sentence_inx] = (
                    (ss1[sentence_inx] + ss1[sentence_inx + 1])
                    .replace("\t", " ")
                    .replace("  ", " ")
                )
                ss1.pop(sentence_inx + 1)
            if has_valid_columns(ss1):
                final_arr.append(ss1)
            else:
                problem_arr.append(f"P1\t{line}")
            continue

        # Problem-2: We have good data in this line but not in the next, it should be overflow in "reason"
        # Action: Append it to current until next line is OK

        # Try to merge with the next line,
        # Else, just get rid of bad lines until we find a good one
        if not has_valid_columns(ss1) and len(ss2) < col_count_needed:
            line_to_try: str = line + " " + next_line
            ss3: list[str] = line_to_try.split("\t")
            if has_valid_columns(ss3):
                final_arr.append(ss3)
                if lines_read:
                    lines_read.pop(0)
            elif len(ss3) < col_count_needed:
                inx_to_add: int = 0
                while (
                    inx_to_add < len(lines_read) - 1
                    and len(ss3) < col_count_needed
                    and not has_valid_columns(ss3)
                ):
                    inx_to_add += 1
                    line_to_try += lines_read[inx_to_add]
                    ss3 = line_to_try.split("\t")
                if has_valid_columns(ss3):
                    final_arr.append(ss3)
                    for _ in range(inx_to_add - 1):
                        if lines_read:
                            lines_read.pop(0)
            else:
                while lines_read and not has_valid_columns(lines_read[0].split("\t")):
                    problem_arr.append(f"P2\t{lines_read.pop(0)}")
    # end of loop

    # Finalise
    df_final: pd.DataFrame = pd.DataFrame(final_arr, columns=fields).astype(fields)
    return df_final, problem_arr


#
# GIT
#


def _git_clone_or_pull(gitrec: GitRec) -> None:
    """Local multiprocessing sub to clone a single repo or update it by pulling if it exist"""
    local_repo_path: str = os.path.join(
        conf.CV_TBOX_CACHE, c.CLONES_DIRNAME, gitrec.repo
    )
    git_path: str = f"{c.GITHUB_BASE}{gitrec.user}/{gitrec.repo}.git"
    if os.path.isdir(local_repo_path):
        # repo exists, so pull only
        if conf.VERBOSE:
            print(f"GIT PULL: {git_path} => {local_repo_path}")
        repo: Repo = Repo(path=local_repo_path)
        repo.remotes.origin.pull()
        if conf.VERBOSE:
            print(f"FINISHED PULL: {gitrec.repo}")
    else:
        # no local repo, so clone
        if conf.VERBOSE:
            print(f"GIT CLONE: {git_path} => {local_repo_path}")
        repo: Repo = Repo.clone_from(
            url=git_path,
            to_path=local_repo_path,
            multi_options=["--single-branch", "--branch", gitrec.branch],
        )
        if conf.VERBOSE:
            print(f"FINISHED CLONING: {gitrec.repo}")


def git_clone_or_pull_all() -> None:
    """Clones all repos or updates them by pulling if they exist - in multiprocessing"""
    with mp.Pool(psutil.cpu_count(logical=True)) as pool:
        pool.map(_git_clone_or_pull, c.CLONES)


def git_checkout(gitrec: GitRec, checkout_to: str = "main") -> None:
    """Checkouts a cloned repo at the given date or main if not given"""
    local_repo_path: str = os.path.join(
        conf.CV_TBOX_CACHE, c.CLONES_DIRNAME, gitrec.repo
    )
    if os.path.isdir(local_repo_path):
        # repo exists, so we can checkout
        if conf.VERBOSE:
            print(f"CHECKOUT: {local_repo_path} @ {checkout_to}")
        repo: Repo = Repo(path=local_repo_path)
        if checkout_to == "main":
            repo.git.execute(command="git checkout main")
        else:
            commit_hash = repo.git.execute(
                command=f"git rev-list -n 1 --before='{checkout_to}' origin/{gitrec.branch}"
            )
            repo.git.execute(command=f"git checkout {commit_hash}")
    else:
        print(f"WARNING: Could not find {gitrec.repo}")


#
# Common Voice Dataset & API Related
#


def is_version_valid(ver: str) -> Literal[True]:
    """Check a ver string in valid"""

    if not ver in c.CV_VERSIONS:
        print(f"FATAL: {ver} is not a supported Common Voice version.")

        sys.exit(1)
    return True


def get_cutoff_date(ver: str) -> str:
    """Given version, get the cutoff-date of that version"""

    if is_version_valid(ver):
        inx: int = c.CV_VERSIONS.index(ver)
        return c.CV_DATES[inx]
    return ""


def calc_dataset_prefix(ver: str) -> str:
    """Build the dataset string from version (valid for > v4)"""

    if is_version_valid(ver):
        inx: int = c.CV_VERSIONS.index(ver)
        # if ver in ["1", "2", "3", "4"]:
        if ver in ["1", "2", "3"]:
            return f"cv-corpus-{ver}"
        return f"cv-corpus-{ver}-{c.CV_DATES[inx]}"
    return ""


def get_from_cv_api(url: str) -> Any:
    """Get data from CV API"""
    try:
        res: Any = urlopen(url)
    except RuntimeError as e:
        print(f"Metadata at {url} could not be located!")
        print(f"Error: {e}")
        sys.exit(-1)
    return json.loads(res.read())


def get_locales_from_api(ver: str) -> list[str]:
    """Get data from API 'datasets' endpoint"""
    jdict: Any = get_from_cv_api(
        f"{c.CV_DATASET_BASE_URL}/{calc_dataset_prefix(ver)}.json"
    )
    jlocales: Any = jdict["locales"]
    locales: list[str] = []
    for loc, _data in jlocales.items():
        locales.append(loc)
    return sorted(locales)


def get_from_cv_dataset_clone(p: str) -> Any:
    """Get data from cloned CV DATASET"""
    with open(p, mode="r", encoding="utf8") as fd:
        s: str = fd.read()
    return json.loads(s)


def get_locales_from_cv_dataset_clone(ver: str) -> list[str]:
    """Get data from ds-datasets json file"""
    p: str = os.path.join(
        conf.CV_TBOX_CACHE,
        c.CLONES_DIRNAME,
        c.CV_DATASET_GITREC.repo,
        "datasets",
        f"{calc_dataset_prefix(ver)}.json",
    )
    if not os.path.isfile(p):
        raise FileNotFoundError
    jdict: Any = get_from_cv_dataset_clone(p)
    locales: list[str] = [item[0] for item in jdict["locales"].items()]
    return sorted(locales)


def get_locales_from_cv_dataset_api(ver: str) -> list[str]:
    """Get data from API 'datasets' endpoint"""
    ds_dir: str = calc_dataset_prefix(ver)
    df_ds_languages: pd.DataFrame = df_read(
        os.path.join(conf.TBOX_TSV_CACHE_DIR, "$cv_dataset_languages.tsv")
    ).astype({"name": str, "id": int})
    df_ds: pd.DataFrame = df_read(
        os.path.join(conf.TBOX_TSV_CACHE_DIR, "$cv_datasets.tsv")
    )
    lc_id_list: list[int] = (
        df_ds[df_ds["release_dir"] == ds_dir]["locale_id"].astype(int).to_list()
    )
    lc_list: list[str] = df_ds_languages[df_ds_languages["id"].isin(lc_id_list)][
        "name"
    ].to_list()
    # [FIXME] Temporary fix for missing "ka" locale (Georgian) in early v18.0 release
    lc_list.append("ka")
    return sorted(lc_list)


def get_locales(ver: str) -> list[str]:
    """Get version locale list"""
    all_locales: list[str] = []
    try:
        all_locales = get_locales_from_cv_dataset_clone(ver)
    except FileNotFoundError:
        all_locales = get_locales_from_cv_dataset_api(ver)
    return all_locales


#
# Conversion
#


def list2str(lst: list[Any]) -> str:
    """Convert a list into a string"""
    return c.SEP_COL.join(str(x) for x in lst)


def arr2str(arr: list[list[Any]]) -> str:
    """Convert an array (list of lists) into a string"""
    return c.SEP_ROW.join(list2str(x) for x in arr)


# def flatten(arr: list[list[Any]]) -> list[Any]:
#     """Flattens a list of lists to a single list"""
#     res: list[Any] = []
#     for row in arr:
#         if isinstance(row,list):
#             res.extend(flatten(row))
#         else: res.append(row)
#     return res

#
# Numbers
#


def dec3(x: float) -> float:
    """Make to 3 decimals"""
    return round(1000 * x) / 1000


#
# FS
#
def sort_by_largest_file(fpaths: list[str]) -> list[str]:
    """Given a list of file paths, this gets the files sizes, sonts on them decending and returns the sorted file paths"""
    recs: list[list[str | int]] = []
    for p in fpaths:
        recs.append([p, os.path.getsize(p)])
    recs = sorted(recs, key=(lambda x: x[1]), reverse=True)
    return [str(row[0]) for row in recs]


#
# Gender back-mapping
#
def gender_backmapping(df: pd.DataFrame) -> pd.DataFrame:
    """Backmap new genders back to older ones for backward compatibility"""
    df["gender"] = df["gender"].replace(c.CV_GENDERS_MAPPING)
    return df
