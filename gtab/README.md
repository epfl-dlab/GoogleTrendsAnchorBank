# Source code

This folder contains the Python source code for G-TAB.

In addition to the code, there are 4 subfolders:

## `config`

The `config` folder contains two config files, one for the Python interface, and one for the command line interface.
The config files are in JSON format and are structured as described below.

Note that you should not touch the config files in this folder.
Rather, when creating a `GTAB` object with `gtab.GTAB(my_path)`, the config files will be automatically copied to `my_path`, and you may then edit the copies in `my_path`.
Instead of editing the (copied) config files directly, you may also set the config options programmatically by calling `set_options()` on your `GTAB` object in Python,
or by calling `gtab-set-options` via the command line interface.

- `BLACKLIST`: A list of Freebase IDs that are disallowed as anchor queries (because they were found to cause trouble)

- `CONN`
    - `backoff_factor`: see https://urllib3.readthedocs.io/en/latest/
    - `proxies`: a list of proxy addresses
    - `retries`: the maximum number of connection retries
    - `timeout`: see https://urllib3.readthedocs.io/en/latest/

- `GTAB`
    - `num_anchor_candidates`: number _N_ of anchor candidates (see [paper](https://arxiv.org/abs/2007.13861)), i.e., the number of top entries from `data/anchor_candidate_list.txt` to consider as anchor candidates
    - `num_anchors`: number _n_ of anchor queries to select from the set of anchor candidates (see [paper](https://arxiv.org/abs/2007.13861))
    - `thresh_offline`: threshold 0 < τ < 100 below which to discard Google Trends queries (see [paper](https://arxiv.org/abs/2007.13861))
    - `seed`: random seed
    - `sleep`: number of seconds to wait between PyTrends API queries

- `PYTRENDS` (for more details, see [pytrends documentation](https://github.com/GeneralMills/pytrends))
    - `timeframe`: time period to use for Google Trends requests
    - `geo`: location to use for Google Trends requests - uses ISO two-letter country format, or "" (empty string) for global

## `data`

The `data` folder contains the list of anchor candidates in the file `anchor_candidate_list.txt`.
It additionally contains a subfolder `internal` that is used internally during anchor bank construction. You shouldn't need to tinker with the `internal` folder.

## `logs`

Log files produced by G-TAB.

## `output`

The `output` folder contains a single subfolder `google_anchorbanks`.
This is where the anchor banks are placed once they've been constructed (one file per anchor bank).
Each file contains two header lines (starting with `#`) that specify the `PYTRENDS` and `GTAB` config options (see above) that were used when constructing the anchor bank.
The remaining lines specify the anchor bank:

- `google_query`: the anchor query, specified via a language-agnostic Freebase ID (e.g., /m/0dm32) or via a language-specific plain-text query string (e.g., "sweet potato")
- `max_ratio`: the max ratio of this query and the reference query (the reference query is the query with max ratio 1.0)
- `max_ratio_hi`: the highest possible true max ratio (of which `max_ratio` is an estimate, due to Google Trend's rounding, see [paper](https://arxiv.org/abs/2007.13861))
- `max_ratio_lo`: analogously, the lowest possible true max ratio
