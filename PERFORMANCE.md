# Hardware & Performance

We used a development desktop for the timing, which has the following specs:

- Intel i7 8700K 6 core / 12 tread @3.7/4.3 GHz, 48 GB DDR4 RAM @3000 GHz (>32 GB empty)
- Compressed Data: On an external 8TB Seagate Backup+ HUB w. USB3 (100-110 MB/s, ~60-65 MB/s continuous read)
- Expanded Data: On an internal Western Digital WD100EMAZ SATA 3 HDD (~90-120 MB/s R/W)
- Working Data: On system drive 2TB Samsung 990 Pro (w. PCIe v3.0), thus ~3500/3500 MB/s R/W speed

With multiple huge changes in Common Voice v17.0, we needed to recalculate everything after substantial code changes.
We worked on 16 versions (leaving v2 out), all languages in them -total 1336 releases-, 5 splitting algorithms (s1, s99, v1, vw, vx) if they are allowed (vw & vx algorithms is not applied to all), resulting in analysis of 20,234 combinations (version/language/algorithm/all-or-bucket-or-split).

Here are results for:

- 16 Common Voice releases
- Which have 1336 datasets
- Which has 6680 splitting algorithm results (some do not have all)

The process took about 4 hours for the whole set (1336 datasets). Before the latest changes, in v16.1, the process was taking about 24 hours.
The results are cached. Thus, when calculating after a new CV release, the code will only calculate the new ones.
On the other hand, the cached data totals 294 GB uncompressed + 16.5 GB compressed for v17.0 and prior.
But, the `.json` files passed to the Common Voice Dataset Analyzer is only 90.6 MB.

## split_compile.py

```text
$ python split_compile.py
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
```

## text_corpus_compile.py

```text
$ python text_corpus_compile.py
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

## final_compile.py

```text
$ python final_compile.py
=== cv-tbox-dataset-analyzer - Final Statistics Compilation ===
Preparing directory structures...

=== Start Text Corpora Analysis ===
Total: 1336 Existing: 0 NoData: 0 Remaining: 1336 Procs: 12  chunk_size: 1...
100%|████████████████████████████████████████████████████████████████| 1336/1336 [2:15:33<00:00,  6.09s/it]
>>> Finished... Now saving...
================================================================================
Total           : Ver: 16 LC: 1336 Algo: 0 Splits: 0
Processed       : Ver: 16 LC: 1336 Algo: 0
Skipped         : Exists: 0 No Data: 0
Duration(s)     : Total: 8149.147 Avg: 6.1

=== Start Reported Analysis ===
Total: 1336 Missing: 139 Remaining: 1197 Procs: 12  chunk_size: 1...
100%|████████████████████████████████████████████████████████████████| 1197/1197 [00:04<00:00, 267.28it/s] 
>>> Finished... Now saving...
================================================================================
Total           : Ver: 16 LC: 1336 Algo: 0 Splits: 0
Processed       : Ver: 13 LC: 1197 Algo: 0
Skipped         : Exists: 139 No Data: 0
Duration(s)     : Total: 8154.527 Avg: 6.812

=== Start Dataset/Split Analysis ===
Processing 1336 locales in 12 processes with chunk_size 1...
100%|████████████████████████████████████████████████████████████████| 1336/1336 [56:31<00:00,  2.54s/it]
>>> Processed 24723 splits...

=== Build Support Matrix ===
>>> Saving Support Matrix...
================================================================================
Total           : Ver: 16 LC: 1336 Algo: 6684 Splits: 24723
Processed       : Ver: 0 LC: 0 Algo: 0
Skipped         : Exists: 0 No Data: 24
Duration(s)     : Total: 11598.727 Avg: -

=== Save Configuration ===
Finished compiling statistics!
Duration 11598.341 sec, avg=8.681 secs/dataset.

```

## pack_splits.py

```text
$ python pack_splits.py
...
Finished compressing for 1336 datasets in 1:57:56.424048 avg=5.296724586826347 sec/locale
```
