# Common Voice Dataset Compiler - Common Voice ToolBox

This repository contains a python commmand line utility for offline pre-calculating the data for Common Voice Dataset Analyzer, which is actually a viewer.
To be able to use this, you have to download the datasets yourselves and provide the tsv files and other data (such as clip durations and text-corpora).
It is a lengthy process using huge amounts of data (42GB uncompressed, without audio files) and you are not expected to use this utility.
We already use it, and you can view the results in the Common Voice Dataset Analyzer.

This code is provided as part of being open-source and as a reference for calculations.

## Scripts

These should be run in the given order.

**split_compile.py**

Assumes:

- You have downloaded all datasets from many Common Voice versions
- Extracted the tsv files
- Ran different splitting algorithms on them and kept the results in different directories.
- You ran a script to measure the duration of each mp3 file and saved it into a dataframe (.tsv) as a look-up-table.

This script reorganizes (compiles) them under this repo to be processed.

Currently runs on a single process, takes 5-6 mins on a 6 core (12 logical) development notebook, nVME SSD.
Should be run with new versions/additions and/or new splitting algorithms.

**text_corpus_compile.py**

Assumes:

- You have a current Common Voice repo clone (to reach the text-corpora)

This script combines them into a single dataframe (.tsv), also calculating a normalized version, char count, word count and validity. For normalization and validation it depends on commonvoice-utils. Some of the languages there do not have this support, so they will not have the whole data.

Multi-process, runs on 5 processes on a 6 real core notebook, takes 5-6 mins.
Should be ran from time to time to update the data with new strings, e.g. monthly.

Known issue: String length calculation with unicode, to be addressed shortly.

**final_compile.py**

Assumes:

- You ran the two scripts above and have all the data here as specified on the next section.

This script processes all data and calculates some statistics and distributions and saves them .

Multi-process, runs on 5 processes on a 6 real core notebook, takes about 25 mins.
Should be run whenever previous scripts run.
The json results are copied to the webapp.


## Data Structure

.The following structure is under the web/data directory

```txt
STRUCTURE AT SOURCE

clip-durations
  <lc>
      $clip_durations.tsv           # Durations of all clips, calculated using external process during download (not provided yet)
text-corpus                         # Compiled by "text_corpus_compile.py" from fresh Common Voice repository clone
  <lc>
      $text_corpus.tsv              # Combined text corpus with additional info
      $tokens.tsv                   # Results of tokenisation (if supported) with frequencies
voice-corpus
  <cvver>                           # eg: "cv-corpus-11.0-2022-09-21"
      <lc>                          # eg: "tr"
          validated.tsv             # These are splitting algorithm independent files
          invalidated.tsv
          other.tsv
          reported.tsv
          <splitdir>                # Splitting algorithms: s1, s99, v1, ...
              train.tsv             # These are splitting algorithm dependent files
              dev.tsv
              test.tsv

FINAL STRUCTURE AT DESTINATION

results
  tsv                               # Created for internal use
    $support_matrix.tsv             # Combined support matrix, keeping what languages/versions/splitting algorithms are supported by the system
    $text_corpus_stats.tsv          # Combined text-corpus statistics
    <lc>
      <lc>_<ver>_splits.tsv         # eg: "tr_v11.0_splits.tsv", keeps all split statistics of this version of locale dataset.
  json                              # For being copied into webapp's public/assets/data directory 
    $support_matrix.json            # Combined support matrix, keeping what languages/versions/splitting algorithms are supported by the system
    $text_corpus_stats.json         # Combined text-corpus statistics
    <lc>
      <lc>_<ver>_splits.json        # eg: "tr_v11.0_splits.json", keeps all split statistics of this version of locale dataset.
```

## Future Work

This will evantually be part of the Common Voice Toolbox's Core...
