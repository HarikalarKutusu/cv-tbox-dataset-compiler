"""cv-tbox dataset analyzer - Simple library for helper functions"""

# Standard Lib
import os
import sys
import csv
import json
import multiprocessing as mp
from datetime import datetime
from typing import Literal, Any
from urllib.request import urlopen

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
    all_locales: list[str] = get_locales_from_cv_dataset(c.CV_VERSIONS[-1])
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
        ver_lc: list[str] = get_locales_from_cv_dataset(ver)
        ds_prefix: str = calc_dataset_prefix(ver)
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
    """Prints out simpğle report from global counters"""
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


def df_read(fpath: str) -> pd.DataFrame:
    """Read a tsv file into a dataframe"""
    if not os.path.isfile(fpath):
        print(f"FATAL: File {fpath} cannot be located!")
        if conf.FAIL_ON_NOT_FOUND:
            sys.exit(1)

    df: pd.DataFrame = pd.read_csv(
        fpath,
        sep="\t",
        parse_dates=False,
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip",
        quotechar='"',
        quoting=csv.QUOTE_NONE,
        dtype={"ver": str},
    )
    return df


def df_write(df: pd.DataFrame, fpath: Any, mode: Any = "w") -> bool:
    """
    Writes out a dataframe to a file.
    """

    _head: bool = False if mode == "a" else True
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
    if conf.VERBOSE:
        print(f"Generated: {fpath} Records={df.shape[0]}")
    return True


def df_int_convert(x: pd.Series) -> Any:
    """Convert columns to int if possible"""
    try:
        return x.astype(int)
    except ValueError as e:  # pylint: disable=W0612
        return x


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


def get_locales_from_cv_dataset(ver: str) -> list[str]:
    """Get data from API 'datasets' endpoint"""
    p: str = os.path.join(
        conf.CV_TBOX_CACHE,
        c.CLONES_DIRNAME,
        c.CV_DATASET_GITREC.repo,
        "datasets",
        f"{calc_dataset_prefix(ver)}.json",
    )
    jdict: Any = get_from_cv_dataset_clone(p)
    locales: list[str] = [item[0] for item in jdict["locales"].items()]
    return sorted(locales)


#
# Conversion
#


def list2str(lst: list[Any]) -> str:
    """Convert a list into a string"""
    return c.SEP_COL.join(str(x) for x in lst)


def arr2str(arr: list[list[Any]]) -> str:
    """Convert an array (list of lists) into a string"""
    return c.SEP_ROW.join(list2str(x) for x in arr)


#
# Numbers
#


def dec3(x: float) -> float:
    """Make to 3 decimals"""
    return round(1000 * x) / 1000


#
# Gender back-mapping
#
def gender_backmapping(df: pd.DataFrame) -> pd.DataFrame:
    """Backmap new genders back to older ones for backward compatibility"""
    df["gender"] = df["gender"].replace(c.CV_GENDERS_MAPPING)
    return df
