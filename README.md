# Google Trends Anchor Bank (G-TAB)

[Google Trends](https://trends.google.com/) is a tool that allows users to analyze the popularity of Google search queries across time and space.
In a single request, users can obtain time series for up to 5 Google queries on a common scale, normalized to the range from 0 
to 100 and rounded to integer precision.
Despite the overall value of Google Trends, rounding causes major problems, to the extent that entirely uninformative, 
all-zero time series may be returned for unpopular Google queries when requested together with more popular queries.

**Google Trends Anchor Bank (G-TAB)**
addresses this issue by offering an efficient solution for the calibration of Google Trends data.
G-TAB expresses the popularity of an arbitrary number of Google queries on a common scale without being compromised by 
rounding errors.

The method proceeds in two phases:

1. In the *offline pre-processing phase*, an "anchor bank" is constructed, a set of Google queries spanning the full spectrum 
of popularity, all calibrated against a common reference query by carefully chaining multiple Google Trends requests.

2. In the *online deployment phase*, any given search query is calibrated by performing an efficient binary search in the anchor bank.
Each search step requires one Google Trends request (via [pytrends](https://github.com/GeneralMills/pytrends)), but few
 steps suffice (see [empirical evaluation](https://arxiv.org/abs/2007.13861)).

A full description of the G-TAB method is available in the following paper:

> Robert West. **Calibration of Google Trends Time Series.** In *Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM)*. 2020. [**[PDF](https://arxiv.org/abs/2007.13861)**]

Please cite this paper when using G-TAB in your own work.

Code and data for reproducing the results of the paper are available in the directory [`_cikm2020_paper`](_cikm2020_paper).



# Repository structure

The repository contains two folders:

- `gtab` contains the Python code and some required data
- `example` contains a Jupyter notebook with an easy-to-follow example


# Installation

The package is available on pip, so you just need to call
~~~python
python -m pip install gtab
~~~

The explicit list of requirements can be found in [`requirements.txt`](requirements.txt).

Note: G-TAB was developed and tested in Python 3.8.1.


# Example usage

(See `[example/example.ipynb](example/example.ipynb)` for an interactive example in a Jupyter notebook.)

Since Google Trends requires users to specify a time period and location for which search interest is to be returned, G-TAB has the same requirement:
every anchor bank is specific to a time period and location.

To get you started quickly, this repo comes with 3 example anchor banks, all for the 8-month time period from 2019-11-28 to 2020-07-28, but for 3 different locations (countries): United States, Italy, Sweden.
The first example below shows you how to use the 3 pre-existing anchor banks in order to calibrate any Google query.

The 3 included anchor banks are great for getting to know G-TAB and starting to play around, but if you have more specific needs, you might need to construct a new anchor bank for your specific time period and location. But don't worry, it's easy!
The second example walks you through the process of constructing your own anchor bank.

## Example with a pre-existing anchor bank

After installing G-TAB with pip, first import the module in your script:
~~~python
import gtab
~~~

Then, create a `GTAB` object with the path to a working directory specified by you:
~~~python
t = gtab.GTAB(dir_path = my_path)
~~~
If the directory `my_path` already exists, it will be used as is.
Otherwise, it will be created and initialized with the subdirectories of the [`gtab`](gtab) directory.

In order to list the available anchor banks, call
~~~python
t.list_gtabs()
~~~
This will produce a list of already-existing anchor banks from which you can select. Three defaults are included in this repository, and the output should look like this:
~~~
Existing GTABs:
        google_anchorbank_geo=IT_timeframe=2019-11-28 2020-07-28.tsv
        google_anchorbank_geo=SE_timeframe=2019-11-28 2020-07-28.tsv
        google_anchorbank_geo=US_timeframe=2019-11-28 2020-07-28.tsv
Active anchorbank: None selected.
~~~

To select which anchor bank to use, call
~~~python
t.set_active_gtab("google_anchorbank_geo=IT_timeframe=2019-11-28 2020-07-28.tsv")
~~~
You should obtain the following confirmation:
~~~
Active anchorbank changed to: google_anchorbank_geo=IT_timeframe=2019-11-28 2020-07-28.tsv
~~~

This will automatically set the corresponding config options. If you want to change them explicitly for the Python interface you can do so by calling `set_options()` or manually editing your config file at `my_path/config/config_py.json`.
If you want to use the command line interface you can use the command `gtab-set-options`, or manually edit the corresponding config file at `my_path/config/config_cl.json`.

Now we can request a calibrated time series for a new query:
~~~python
mid = "/m/0jg7r"    # Freebase ID for EPFL
nq_res= t.new_query(mid) 

print(f'Ratio: {nq_res[mid]["ratio"]}')
print(f'Ratio low: {nq_res[mid]["ratio_lo"]}')
print(f'Ratio high: {nq_res[mid]["ratio_hi"]}')
print(f'Ratio high: {nq_res[mid]["ts"]}')
~~~

## Example with a newly created anchor bank

As in the previous example, the module is imported and then the object is created with the path to a working directory:
~~~python
import gtab
t = gtab.GTAB(my_path)
~~~

The desired config options can be set through the object method `set_options()`. For example, if we want to construct an anchor bank with data from France between 5 March 2020 and 5 May 2020, use
~~~python
t.set_options(pytrends_config = {"geo": "FR", "timeframe": "2019-01-01 2020-07-05"})
~~~

We also need to specify the file that contains a list of candidate queries from which anchor queries will be selected when constructing the anchor bank:
~~~python
t.set_options(gtab_config = {"anchor_candidates_file": "anchor_candidate_list.txt"})
~~~
This file must be located at `my_path/data/anchor_candidate_list.txt` and contain one query per line.
Note that, as described in the [paper](https://arxiv.org/abs/2007.13861), we recommend using language-agnostic Freebase IDs (e.g., /m/0dm32) as queries, rather than language-specific plain-text queries (e.g., "sweet potato").
A good [list of candidate queries](gtab/data/anchor_candidate_list.txt) is shipped with G-TAB by default, so only advanced users should need to tinker with the list.

We then need to set the size of the anchor bank to be constructed,
as well as the number of candidate queries to be used in constructing the anchor bank
(these are called *n* and *N*, respectively, in the [paper](https://arxiv.org/abs/2007.13861); see the description in Sec. 2.1 of the paper for details).
For example, if you want to construct an anchor bank consisting of *n*=100 queries selected from *N*=3000 candidates, call:
~~~python
t.set_options(gtab_config = {"num_anchors": 100, "num_anchor_candidates": 3000})
~~~
(Note that the specified `num_anchors` is used only to construct an initial anchor bank, which is then automatically optimized in order to produce the smaller, more precise final anchor bank, typically containing no more than 20 anchor queries. See Appendix B of the [paper](https://arxiv.org/abs/2007.13861).)

Note that all config options can be directly edited in the config file found at `my_path/config/config.json`.

Finally, we construct the anchor bank by calling
~~~python
t.create_anchorbank()
~~~
This will start sending requests to Google Trends and calibrate the data.
It may take some time, depending on the specified size `num_anchors` of the anchor bank and the sleep interval between two Google Trends requests.
Once the anchor bank has been constructed, it can be listed and selected as described in the previous example.  
