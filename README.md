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

Code and data for reproducing the results of the paper are available in the directory [`_cikm2020_paper`](_cikm2020_paper).



# Repository structure

**!!! THIS IS OUTDATED AND NEEDS TO BE BROUGHT UP TO DATE AND MADE COMPLETE !!!**

The src/python project structure is as follows:
- config - contains four necessary config files:
- data - contains the input data set as well as the outputs of G-TAB.
- logs - contains logs that are written while constructing new G-TABs.


## Config files 
The config file is a single JSON file with four dictionaries:

### BLACKLIST:
A list with FreeBase IDs that are disallowed when sampling.

### CONN
Contains the following parameters:
- "proxies": a list of proxy addresses
- "retries": the maximum number of connection retries.
- "backoff_factor": see https://urllib3.readthedocs.io/en/latest/
- "timeout": see https://urllib3.readthedocs.io/en/latest/

### GTAB
Contains the following parameters:
- "num_anchors": the number of anchors that are sampled from the anchor candidate data set.
- "num_anchor_candidates": how many entries in the anchor candidate data set to use.
- "thresh_offline": threshold below which to discard Google Trends queries (see paper below)
- "seed": random seed.
- "sleep": how many secons to wait between PyTrends API queries.
For more details see https://arxiv.org/pdf/2007.13861.pdf.

### PTRENDS
Contains the following parameters:
- "timeframe": in which timeframe to collect data.
- "geo": which location to query.
For more details see https://pypi.org/project/pytrends/.




# Installation

First you need to set up a Python virtual environment and install the required packages:

1. Activate your virtual environment ([instructions](https://docs.python.org/3/tutorial/venv.html)). (Note that G-TAB was written and tested using Python 3.8.1.)

2. Install the packages listed in [`requirements.txt`](requirements.txt) using your preferred package manager. For instance, if you use pip, simply run this command:

~~~
pip install -r requirements.txt
~~~


# Example usage

## Example with a pre-existing anchor bank

After installing G-TAB with pip, first import the module in your script:
~~~python
import gtab
~~~

Then, create a `GTAB` object with a working path specified by you:
~~~python
t = gtab.GTAB(my_path)
~~~
If the directory `my_path` already exists, it will be used as is.
Otherwise, it will be created and initialized with the subdirectories contained in [`gtab`](gtab).

In order to list the available anchor banks, call:
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

To select which anchor bank to use, we call
~~~python
t.set_active_gtab("google_anchorbank_geo=IT_timeframe=2019-11-28 2020-07-28.tsv")
~~~
obtaining the following confirmation:
~~~
Active anchorbank changed to: google_anchorbank_geo=IT_timeframe=2019-11-28 2020-07-28.tsv
~~~

Next we need ensure we have the correct corresponding config options (for new queries only the *ptrends_config* is relevant). To set them, we call
~~~python
t.set_options(ptrends_config = {"geo": "IT", "timeframe": "2019-11-28 2020-07-28" })
~~~
This can also be done by manually editing the config file found at: *my_path/config/config.json*

Now we can request a calibrated time series for a new query:
~~~python
nq_res= t.new_query("Sweet potato")
~~~


## Example with a newly created anchor bank

As in the previous example, the module is imported and then the object is created with the desired path:
~~~python
import gtab
t = gtab.GTAB(my_path)
~~~

The desired config options can be set through the object method *set_options*. For example, if we want to construct an anchor bank with data from France between 5 March 2020 and 5 May 2020, we use
~~~python
t.set_options(ptrends_config = {"geo": "FR", "timeframe": "2020-03-05 2020-05-05"})
~~~

We also need to specify the file that contains a list of candidate queries from which anchor queries will be selected when constructing the anchor bank:
~~~python
t.set_options(gtab_config = {"anchor_candidates_file": "my_data_file.txt"})
~~~
This file must be located at *my_path/data/my_data_file.txt* and contain one query per line.
Note that, as described in the [paper](https://arxiv.org/abs/2007.13861), we recommend using language-agnostic Freebase IDs (e.g., /m/0dm32) as queries, rather than language-specific plain-text queries (e.g., "sweet potato").
A good [list of candidate queries](gtab/data/anchor_candidate_list.txt) is shipped with G-TAB by default, so only advanced users should need to tinker with the list.

We then need to set the size of the anchor bank to be constructed,
as well as the number of candidate queries to be used in constructing the anchor bank
(these are called *n* and *N*, respectively, in the [paper](https://arxiv.org/abs/2007.13861)).
For example, if we want to construct an anchor bank consisting of *n*=100 queries selected from *N*=3000 candidates, we call
~~~python
t.set_options(gtab_config = {"num_anchors": 100, "num_anchor_candidates": 3000})
~~~
(Note that the specified `num_anchors` is used only to construct an initial, intermediate anchor bank, which is then automatically optimized in order to produce a smaller, more precise anchor bank, typically containing no more than 20 anchor queries. See Appendix B of the [paper](https://arxiv.org/abs/2007.13861).)

Note that all config options can be directly edited in the config file found at *my_path/config/config.json*.

Finally, we construct the anchor bank by calling
~~~python
t.create_anchorbank()
~~~
This will start sending requests to Google Trends and calibrate the data.
It may take some time, depending on the specified size `num_anchors` of the anchor bank and the sleep interval between two Google Trends requests.
Once the anchor bank has been constructed, it can be listed and selected as described in the previous example.  
