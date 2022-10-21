# Common Voice Dataset Compiler - Common Voice ToolBox

This repository contains python commmand line utilities for pre-calculating the data for [Common Voice Dataset Analyzer](https://github.com/HarikalarKutusu/cv-tbox-dataset-analyzer) (which is actually a viewer) in an offline process.
To be able to use this, you have to download the datasets yourselves and provide the tsv files and other data (such as clip durations and text-corpora).
It is a lengthy process using huge amounts of data (in our case it was 42GB uncompressed, without audio files) and you are not expected to use this utility.
We already use it, and you can view the results in the [Common Voice Dataset Analyzer](https://github.com/HarikalarKutusu/cv-tbox-dataset-analyzer).

This code is provided as part of being open-source and as a reference for calculations.

## Scripts

The first two scripts should be run before the final one.

### split_compile.py

Assumes:

- You have downloaded all datasets from many Common Voice versions
- Extracted the tsv files
- Ran different splitting algorithms on them and kept the results in different directories (e.g. using the ones in [Common Voice Diversity Check](https://github.com/HarikalarKutusu/common-voice-diversity-check)).
- You ran a script to measure the duration of each mp3 file and saved it into a dataframe (.tsv) as a look-up-table.

This script reorganizes (compiles) them under this repo to be processed.

This script currently runs on a single process to prevent disk overload, takes 5-6 mins on a 6 core (12 logical) development notebook, nVME SSD.
Should be run with new versions/additions and/or new splitting algorithms.

### text_corpus_compile.py

Assumes:

- You have a current Common Voice repo clone (to reach the text-corpora)

For each locale, this script combines existing files under that language into a single pandas dataframe compatible tsv file, also calculating a normalized version, char count, word count and validity. For normalization and validation it depends on [commonvoice-utils](https://github.com/ftyers/commonvoice-utils) (please see that repo for installation). Some of the languages there do not have this support, so they will not have the whole data.

The script uses multi-processing, running in 5 processes on a 6 real core notebook, execution takes 5-6 mins.
Should be ran from time to time to update the data with new strings, e.g. monthly.

Known issue: String length calculation with unicode locales might be wrong, to be addressed shortly.

### final_compile.py

Assumes:

- You ran the two scripts above and have all the data here as specified on the next section.

This script processes all data and calculates some statistics and distributions and saves them in tsv & json formats under the results directory, grouped by languages. Each version's data is kept in a different file, keeping the file sizes at 10k max, so that we can limit the memory needed in the client and probably cache them whenever the client becomes a PWA. We do not include easily calculatable values into these results and leave simple divisions or substractions to the client.

Uses multi-processing, running in 5 processes on a 6 real core notebook, takes about 25 mins.
Should be run whenever previous scripts run.
The json results will be copied to the [Common Voice Dataset Analyzer](https://github.com/HarikalarKutusu/cv-tbox-dataset-analyzer).

## Data Structure

.The following structure is under the web/data directory

```txt
STRUCTURE AT SOURCE

clip-durations
  <lc>
    $clip_durations.tsv           # Durations of all clips, calculated using external process during download (not provided yet)
text-corpus                       # Compiled by "text_corpus_compile.py" from fresh Common Voice repository clone
  <lc>
    $text_corpus.tsv              # Combined text corpus with additional info
    $tokens.tsv                   # Results of tokenisation (if supported) with frequencies
voice-corpus
  <cvver>                         # eg: "cv-corpus-11.0-2022-09-21"
    <lc>                          # eg: "tr"
      validated.tsv               # These are splitting algorithm independent files
      invalidated.tsv
      other.tsv
      reported.tsv
      <splitdir>                  # Splitting algorithms: s1, s99, v1, ...
        train.tsv                 # These are splitting algorithm dependent files
        dev.tsv
        test.tsv

FINAL STRUCTURE AT DESTINATION

results
  tsv                             # Created for internal use / further python pandas processing if needed
    $support_matrix.tsv           # Combined support matrix, keeping what languages/versions/splitting algorithms are supported by the system
    $text_corpus_stats.tsv        # Combined text-corpus statistics
    <lc>
      <lc>_<ver>_splits.tsv       # eg: "tr_v11.0_splits.tsv", keeps all split statistics of this version of locale dataset.
  json                            # For being copied into webapp's public/assets/data directory 
    $support_matrix.json          # Combined support matrix, keeping what languages/versions/splitting algorithms are supported by the system
    $text_corpus_stats.json       # Combined text-corpus statistics
    <lc>
      <lc>_<ver>_splits.json      # eg: "tr_v11.0_splits.json", keeps all split statistics of this version of locale dataset.
```

## Setup and Run

Developed and tested on Python 3.8.x but should work on later versions. It is preferred to use a virtual environment.

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
