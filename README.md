# Common Voice Dataset Compiler - Common Voice ToolBox

This repository contains python command line utilities for pre-calculating the data for [Common Voice Dataset Analyzer](https://github.com/HarikalarKutusu/cv-tbox-dataset-analyzer) (which is actually a viewer) in an offline process.
To be able to use this, you have to download the datasets yourselves and provide the tsv files and other data (such as clip durations and text-corpora).
It is a lengthy process using huge amounts of data (in our case it is 274 GB uncompressed after Common Voice v16.1 - i.e. without the audio files) and you are not expected to use this utility.
We already use it, and you can view the results in the [Common Voice Dataset Analyzer](https://github.com/HarikalarKutusu/cv-tbox-dataset-analyzer).

This code is provided as part of being open-source and as a reference for calculations.

## Scripts

The scripts should be run in the listed order below.
Location of files among other settings are set in `conf.py` file.

### split_compile.py

Assumes:

- You have downloaded all datasets from all Common Voice versions
- Extracted the tsv files
- Ran different splitting algorithms on them and kept the results in different directories (i.e. using the ones in [Common Voice Diversity Check](https://github.com/HarikalarKutusu/common-voice-diversity-check)).

This script reorganizes (compiles) them under this repo to be processed.

This script skips already processed versions/locales, but should be run from scratch if new script algorithm is added.

### text_corpus_compile.py

Assumes:

- You executed split_compile.py
- [TODO] common-voice-cache repo

With Common Voice v17.0, text corpora is part of the releases. For each locale, this script takes the `validated_sentences.tsv` file; then -whenever possible - pre-calculates a normalized version, tokens in it, char count, word count and validity; it than saves it under `./text-corpus/<lc>/$text_corpus.tsv` file. On consequiteive versions, only the newly sentences get preprocessed and added to it.

For normalization and validation it depends on [commonvoice-utils](https://github.com/ftyers/commonvoice-utils) (please see that repo for installation). Some of the languages there do not have this support, so they will not have the whole data.

The script uses multi-processing. Should be run from time to time to update the data with new strings, e.g. monthly.

Known issues:

1. String length calculation with unicode locales might be wrong, to be addressed shortly.
2. Between v14.0 - v16.1, the sentences which are entered though the new write page are NOT in the text-corpora.
3. The sentences in text corpora might be different in splits, as the CorporaCreator cleans the recorded ones in a multiprocessing (global and some language specific ones). We did not incorporate those cleanings in our code - yet. Some statistics prior to v17.0 should be done from sentences, thus the results might not reflect the correct values because of these changes.

### final_compile.py

Assumes:

- You ran the two scripts above and have all the data here as specified on the next section.

This script processes all data and calculates statistics and distributions, and saves them in tsv & json formats under the results directory, grouped by languages. Each version's data is kept in a different file, keeping the file sizes small, so that we can limit the memory needed in the client and probably cache them whenever the client becomes a PWA. We do not include easily calculatable values into these results and leave simple divisions or substractions to the client.

Uses multi-processing.
The json results will be copied to the [Common Voice Dataset Analyzer](https://github.com/HarikalarKutusu/cv-tbox-dataset-analyzer).

### pack_splits.py

Simple script to create `tar.xz` files from training splits (train/dev/test) to make them downloadable. It usually points out of the git area, where you could upload it to a server (we currently use Google Drive, after it got big). The files are names in the format `<lc>-<ver>_<algorithm>.tar.xz`

## Data Structure

The following structure is under the `data` directory

```txt
STRUCTURE AT SOURCE

clip-durations
  <lc>
    clip_durations.tsv            # Durations of all clips, previousşy calculated using external process during download, with v14.0 it is generated from times.txt file provided in the datasets
text-analysis                     # Compiled by "text_corpus_compile.py" from fresh Common Voice repository clone
  <cvver>                         # eg: "cv-corpus-11.0-2022-09-21"
    <lc>                          # eg: "tr"
      tokens*.tsv                 # Results of token frequencies, global and per split - if supported
      graphemes*.tsv              # Results of grapheme frequencies, global and per split
      phonemes*.tsv               # Results of phonme frequencies, global and per split - if supported
      domains*.tsv                # Results of sentence domain frequencies, global and per split - if supported
text-corpus                       # Compiled and pre-processed by "text_corpus_compile.py"
  <lc>
    $text_corpus.tsv              # Combined text corpus with additional info
voice-corpus
  <cvver>                         # eg: "cv-corpus-11.0-2022-09-21"
    <lc>                          # eg: "tr"
      validated.tsv               # These are splitting algorithm independent files
      invalidated.tsv
      other.tsv
      reported.tsv
      unvalidated_sentences.tsv
      validated_sentences.tsv
      <splitdir>                  # Splitting algorithms: s1, s99, v1, ...
        train.tsv                 # These are splitting algorithm dependent files
        dev.tsv
        test.tsv
results
  tsv                               # Created for internal use / further python pandas processing if needed
    $config.tsv                     # Configuration used cv-tobox-dataset-compiler to be used in upper level applications, such as cv-tbox-dataset-analyzer
    $support_matrix.tsv             # Combined support matrix, keeping what languages/versions/splitting algorithms are supported by the system
    <lc>
      <lc>_<ver>_splits.tsv         # eg: "tr_v11.0_splits.tsv", keeps all split statistics of this version of locale dataset.
      <lc>_<ver>_tc_stats.tsv       # Combined text-corpus statistics global, per bucket and per split - one per version.
      $reported.tsv                 # Statistics of reported sentences - all versions combined
  json                              # Same as above, for being copied into webapp's public/assets/data directory 
    $config.json
    $support_matrix.json
    <lc>
      <lc>_<ver>_splits.json
      <lc>_<ver>_tc_stats.tsv
      $reported.json
```

## Setup and Run

Initially developed and tested on Python 3.8.x, but continued on v3.12.x with new type definition possibilities. It is preferred to use a virtual environment.

1. Create venv and activate it
2. Clone the repo and cd into it
3. Install main dependencies using `pip install -r requirements.txt`
4. For normalization and validation of text-corpora install [commonvoice-utils](https://github.com/ftyers/commonvoice-utils) as defined in the link.
5. Prepare your data as described above, edit config.py to point to correct directories in your system
6. Run the script you want

## Other

### License

AGPL v3.0

### TO-DO/Project plan, issues and feature requests

We did not list the statistics we compiled yet, as this is actively developed.
You can look at the results [Common Voice Dataset Analyzer](https://github.com/HarikalarKutusu/cv-tbox-dataset-analyzer).
This will eventually be part of the Common Voice Toolbox's Core, but it will be developed here...

The project status can be found on the [project page](https://github.com/users/HarikalarKutusu/projects/10). Please post [issues and feature requests](https://github.com/HarikalarKutusu/cv-tbox-dataset-compiler/issues), or [Pull Requests](https://github.com/HarikalarKutusu/cv-tbox-dataset-compiler/pulls) to enhance.

### Hardware & Performance

We used a development notebook for the timing with the following specs:

- Intel i7 8700K 6 core / 12 tread @3.7/4.3 GHz, 48 GB DDR4 RAM @3000 GHz (>32 GB empty)
- Compressed Data: On an external 8TB Seagate Backup+ HUB w. USB3 (100-110 MB/s, ~60-65 MB/s continuous read)
- Expanded Data: On an internal Western Digital WD100EMAZ SATA 3 HDD (~90-120 MB/s R/W)
- Working Data: On system drive 2TB Samsung 990 Pro (w. PCIe v3.0), thus ~3500/3500 MB/s R/W speed

With multiple huge changes in Common Voice v17.0, we needed to recalculate everthing after substantial code changes.
We worked on 16 versions (leaving v2 out), all languages in them -total 1336 releases-, 5 splitting algorithms (s1, s99, v1, vw, vx) if they are allowed (vw & vx algorithms is not applied to all), resulting in analysis of 20,234 combinations (version/language/algorithm/all-or-bucket-or-split).

Here are results:

```text
$ python .\split_compile.py
=== cv-tbox-dataset-compiler: Data Algorithms/Splits Collection Process ===
Preparing directory structures...
   v1: 100%|█████████████████████████████████████████████████████| 19/19 [00:02<00:00,  7.80 Locale/s]
   v3: 100%|█████████████████████████████████████████████████████| 29/29 [00:02<00:00, 10.88 Locale/s]
   v4: 100%|█████████████████████████████████████████████████████| 40/40 [00:04<00:00,  9.55 Locale/s]
 v5.1: 100%|█████████████████████████████████████████████████████| 54/54 [00:06<00:00,  7.90 Locale/s]
 v6.1: 100%|█████████████████████████████████████████████████████| 60/60 [00:08<00:00,  7.34 Locale/s]
 v7.0: 100%|█████████████████████████████████████████████████████| 76/76 [00:11<00:00,  6.45 Locale/s]
 v8.0: 100%|█████████████████████████████████████████████████████| 87/87 [00:15<00:00,  5.78 Locale/s]
 v9.0: 100%|█████████████████████████████████████████████████████| 93/93 [00:15<00:00,  5.88 Locale/s]
v10.0: 100%|█████████████████████████████████████████████████████| 96/96 [00:16<00:00,  5.69 Locale/s]
v11.0: 100%|████████████████████████████████████████████████████| 100/100 [00:18<00:00,  5.55 Locale/s]
v12.0: 100%|████████████████████████████████████████████████████| 104/104 [00:18<00:00,  5.61 Locale/s]
v13.0: 100%|████████████████████████████████████████████████████| 108/108 [00:20<00:00,  5.39 Locale/s]
v14.0: 100%|████████████████████████████████████████████████████| 112/112 [00:25<00:00,  4.32 Locale/s]
v15.0: 100%|████████████████████████████████████████████████████| 114/114 [00:27<00:00,  4.07 Locale/s]
v16.1: 100%|████████████████████████████████████████████████████| 120/120 [00:29<00:00,  4.12 Locale/s]
v17.0: 100%|████████████████████████████████████████████████████| 124/124 [00:46<00:00,  2.65 Locale/s]
================================================================================
Total           : Ver: 16 LC: 1336 Algo: 5 Splits: 0
Processed       : Ver: 16 LC: 1336 Algo: 6680
Skipped         : Exists: 0 No Data: 0
Duration(s)     : Total: 270.86 Avg: 0.203


$ python .\text_corpus_compile.py
=== cv-tbox-dataset-compiler: Text-Corpora Compilation Process ===
Preparing directory structures...
=== HANDLE: v17.0 @ 2024-03-15 ===
Total: 124 Existing: 0 Remaining: 124 Procs: 12  chunk_size: 1...
Processing: ['de', 'fr', 'en', 'rw', 'ab', 'ca', 'es', 'bn', 'it', 'gl', 'be', 'tr', 'cs', 'mhr', 'hy-AM', 'hu', 'pl', 'nl', 'ka', 'uk', 'ta', 'eu', 'lg', 'ba', 'eo', 'kab', 'sw', 'uz', 'lt', 'cy', 'az', 'th', 'ru', 'ar', 'mrj', 'fa', 'zh-CN', 'bg', 'hi', 'tw', 'pt', 'ja', 'ug', 'lv', 'sv-SE', 'nan-tw', 'ur', 'da', 'ckb', 'tt', 'pa-IN', 'zh-HK', 'zh-TW', 'kmr', 'ig', 'et', 'ro', 'id', 'mr', 'fy-NL', 'dv', 'or', 'lo', 'ltg', 'sah', 'yue', 'mn', 'rm-sursilv', 'ml', 'hsb', 'ia', 'as', 'lij', 'ko', 'sk', 'el', 'kk', 'myv', 'mk', 'mdf', 'br', 'sat', 'yo', 'sq', 'tok', 'rm-vallader', 'skr', 'tig', 'sr', 'ky', 'fi', 'is', 'he', 'ti', 'cv', 'af', 'vi', 'mt', 'gn', 'sc', 'ha', 'nn-NO', 'dyu', 'oc', 'vot', 'cnh', 'bas', 'zza', 'sl', 'am', 'zgh', 'tk', 'ps', 'nso', 'os', 'ast', 'ne-NP', 'ga-IE', 'zu', 'yi', 'nhi', 'quy', 'te', 'ht']
Locales: 100%|██████████████████████████████████████████████████████| 124/124 [40:42<00:00, 19.70s/it] 
=== HANDLE INDEXING: v17.0 @ 2024-03-15 ===
Total: 124 Existing: 124 Remaining: 0 Procs: 12  chunk_size: 0...
=== HANDLE INDEXING: v16.1 @ 2023-12-06 ===
Total: 120 Existing: 0 Remaining: 120 Procs: 12  chunk_size: 1...
Locales: 100%|██████████████████████████████████████████████████████| 120/120 [00:26<00:00,  4.50it/s] 
=== HANDLE INDEXING: v15.0 @ 2023-09-08 ===
Total: 114 Existing: 0 Remaining: 114 Procs: 12  chunk_size: 1...
Locales: 100%|██████████████████████████████████████████████████████| 114/114 [00:22<00:00,  5.09it/s] 
=== HANDLE INDEXING: v14.0 @ 2023-06-23 ===
Total: 112 Existing: 0 Remaining: 112 Procs: 12  chunk_size: 1...
Locales: 100%|██████████████████████████████████████████████████████| 112/112 [00:22<00:00,  4.89it/s] 
=== HANDLE INDEXING: v13.0 @ 2023-03-09 ===
Total: 108 Existing: 0 Remaining: 108 Procs: 12  chunk_size: 1...
Locales: 100%|██████████████████████████████████████████████████████| 108/108 [00:22<00:00,  4.82it/s] 
=== HANDLE INDEXING: v12.0 @ 2022-12-07 ===
Total: 104 Existing: 0 Remaining: 104 Procs: 12  chunk_size: 1...
Locales: 100%|██████████████████████████████████████████████████████| 104/104 [00:22<00:00,  4.66it/s] 
=== HANDLE INDEXING: v11.0 @ 2022-09-21 ===
Total: 100 Existing: 0 Remaining: 100 Procs: 12  chunk_size: 1...
Locales: 100%|██████████████████████████████████████████████████████| 100/100 [00:22<00:00,  4.42it/s] 
=== HANDLE INDEXING: v10.0 @ 2022-07-04 ===
Total: 96 Existing: 0 Remaining: 96 Procs: 12  chunk_size: 1...
Locales: 100%|██████████████████████████████████████████████████████| 96/96 [00:22<00:00,  4.34it/s]
=== HANDLE INDEXING: v9.0 @ 2022-04-27 ===
Total: 93 Existing: 0 Remaining: 93 Procs: 12  chunk_size: 1...
Locales: 100%|██████████████████████████████████████████████████████| 93/93 [00:21<00:00,  4.31it/s]
=== HANDLE INDEXING: v8.0 @ 2022-01-19 ===
Total: 87 Existing: 0 Remaining: 87 Procs: 12  chunk_size: 1...
Locales: 100%|██████████████████████████████████████████████████████| 87/87 [00:20<00:00,  4.17it/s]
=== HANDLE INDEXING: v7.0 @ 2021-07-21 ===
Total: 76 Existing: 0 Remaining: 76 Procs: 12  chunk_size: 1...
Locales: 100%|██████████████████████████████████████████████████████| 76/76 [00:20<00:00,  3.76it/s]
=== HANDLE INDEXING: v6.1 @ 2020-12-11 ===
Total: 60 Existing: 0 Remaining: 60 Procs: 12  chunk_size: 1...
Locales: 100%|██████████████████████████████████████████████████████| 60/60 [00:17<00:00,  3.34it/s]
=== HANDLE INDEXING: v5.1 @ 2020-06-22 ===
Total: 54 Existing: 0 Remaining: 54 Procs: 12  chunk_size: 1...
Locales: 100%|██████████████████████████████████████████████████████| 54/54 [00:17<00:00,  3.12it/s]
=== HANDLE INDEXING: v4 @ 2019-12-10 ===
Total: 40 Existing: 0 Remaining: 40 Procs: 12  chunk_size: 1...
Locales: 100%|██████████████████████████████████████████████████████| 40/40 [00:15<00:00,  2.54it/s]
=== HANDLE INDEXING: v3 @ 2019-06-24 ===
Total: 29 Existing: 0 Remaining: 29 Procs: 12  chunk_size: 1...
Locales: 100%|██████████████████████████████████████████████████████| 29/29 [00:12<00:00,  2.26it/s]
=== HANDLE INDEXING: v1 @ 2019-02-25 ===
Total: 19 Existing: 0 Remaining: 19 Procs: 12  chunk_size: 1...
Locales: 100%|██████████████████████████████████████████████████████| 19/19 [00:04<00:00,  3.93it/s]
================================================================================
Total           : Ver: 16 LC: 1460 Algo: 5 Splits: 0
Processed       : Ver: 17 LC: 1336 Algo: 0
Skipped         : Exists: 124 No Data: 0
Duration(s)     : Total: 2778.214 Avg: 2.08


```

The process took about 3.5 hours for the whole set (1336 datasets). Before the latest changes, in v16.1, the process was taking about 24 hours.
The results are cached. Thus, when calculating after a new CV release, the code will only calculate the new ones.
On the other hand, the cached data totals 264 GB uncompressed for v17.0 and prior.
But, the `.json` files passed to the Common Voice Dataset Analyzer is only 76.6 MB.
