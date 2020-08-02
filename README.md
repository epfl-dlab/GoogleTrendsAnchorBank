# Google Trends Anchor Bank (G-TAB)

Google Trends is a tool that allows users to analyze the popularity of Google search queries across time and space.
In a single request, users can obtain time series for up to 5 queries on a common scale, normalized to the range from 0 to 100 and rounded to integer precision.
Despite the overall value of Google Trends, rounding causes major problems, to the extent that entirely uninformative, all-zero time series may be returned for unpopular queries when requested together with more popular queries.
We address this issue by proposing
*Google Trends Anchor Bank (G-TAB),*
an efficient solution for the calibration of Google Trends data.
Our method expresses the popularity of an arbitrary number of queries on a common scale without being affected by rounding errors.
The method proceeds in two phases.
In the offline preprocessing phase, an "anchor bank" is constructed, a set of queries spanning the full spectrum of popularity, all calibrated against a common reference query by carefully chaining together multiple Google Trends requests.
In the online deployment phase, any given search query is calibrated by performing an efficient binary search in the anchor bank.
Each search step requires one Google Trends request, but few steps suffice, as we demonstrate in an empirical evaluation.

# Using G-TAB

First you need to setup a Python (written and tested using 3.8.1) virtual environment and installing the packages listed in python/requirements.txt. Activate your environment (https://docs.python.org/3/tutorial/venv.html) and then install the packages with your preferred package manager. For pip, run:
~~~
pip install -r requirements.txt
~~~

The src/python project structure is as follows:
- config - contains four necessary config files:
- data - contains the input data set as well as the outputs of G-TAB.
- logs - contains logs that are written while constructing new G-TABs.

## Config files 
Each config file needs to contain a single line that is evaluable in Python (i.e. eval()) and contains some parameters: 
### blacklist.config:
Contains a Python set with FreeBase IDs that are disallowed when sampling.
### conn.config
Contains the following evaluable dictionary:
- "proxies": a list of proxy addresses
- "retries": the maximum number of connection retries.
- "backoff_factor": see https://urllib3.readthedocs.io/en/latest/
- "timeout": see https://urllib3.readthedocs.io/en/latest/
### gtab.config
Contains the following evaluable dictionary:
- "num_anchors": the number of anchors that are sampled from the anchor candidate data set.
- "num_anchor_candidates": how many entries in the anchor candidate data set to use.
- "thresh_offline": threshold below which to discard Google Trends queries (see paper below)
- "seed": random seed.
- "sleep": how many secons to wait between PyTrends API queries.
For more details see https://arxiv.org/pdf/2007.13861.pdf.
### ptrends.config
Contains the following evaluable dictionary:
- "timeframe": in which timeframe to collect data.
- "geo": which location to query.
For more details see https://pypi.org/project/pytrends/.


## How to use the GTAB class:
First input your preferred settings in the config files and then initialize the GTAB class by calling:
~~~python
    from gtab import GTAB
    t = GTAB()
    t.init()
~~~
This will first start querying Google Trends and then constructing the GoogleTrends Anchor Bank as described in the paper. Once done, to query a new keyword call:
~~~python
    t.new_query(keyword)
~~~

Once the initalization is done and the GTAB is constructed, it can be found as a .tsv file in the folder "python/data/google_anchorbanks". If there already exists a GTAB with the same parameters, it loads is from the aforementioned folder instead of constructing a new one. If for some reason the initialization is interrupted, but the data has been collected, and then started again with the same configs, it will load the query data from "data/google_results" and/or "data/google_pairs" instead of re-querying. 

To use proxies, you have to set *use_proxies = True* in the object's constructor, i.e.:
~~~python
    t = GTAB(use_proxies = True)
~~~

To keep all the computed values calculated while initializing the anchorbank you can set *keep_diagnostics = True* the method *init*, i.e.:

~~~python
    t.init(keep_diagnostics = True)
~~~
