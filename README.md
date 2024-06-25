# Common Voice Dataset Compiler - Common Voice ToolBox

IMPORTANT NOTICE: We are moving our Common Voice Toolbox related packages (and more) into a combined monorepo, so this repo will be archived in the future.

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
- Ran different splitting algorithms on them and kept the results in different directories (i.e. using the ones in [Common Voice Split Maker](https://github.com/HarikalarKutusu/cv-tbox-split-maker)).

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

The details can be found in the [PERFORMANCE.md](PERFORMANCE.md) file.
