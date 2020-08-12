# Google Trends Anchor Bank (G-TAB)

Google Trends is a tool that allows users to analyze the popularity of Google search queries across time and space.
In a single request, users can obtain time series for up to 5 queries on a common scale, normalized to the range from 0 
to 100 and rounded to integer precision.
Despite the overall value of Google Trends, rounding causes major problems, to the extent that entirely uninformative, 
all-zero time series may be returned for unpopular queries when requested together with more popular queries.
We address this issue by proposing
*Google Trends Anchor Bank (G-TAB),*
an efficient solution for the calibration of Google Trends data.

Our method expresses the popularity of an arbitrary number of queries on a common scale without being affected by 
rounding errors.
The method proceeds in two phases:

1.  In the *offline pre-processing phase*, an "anchor bank" is constructed, a set of queries spanning the full spectrum 
of popularity, all calibrated against a common reference query by carefully chaining multiple Google Trends requests.

2. In the *online deployment phase*, any given search query is calibrated through a binary search in the anchor bank.
Each search step requires a Trends request (done with [pytrends](https://github.com/GeneralMills/pytrends)), but few
 steps suffice, as we demonstrate in an [empirical evaluation](https://arxiv.org/abs/2007.13861).

# Using G-TAB

First you need to setup a Python (written and tested using 3.8.1) virtual environment and installing the packages listed
in python/requirements.txt. Activate your environment (https://docs.python.org/3/tutorial/venv.html) and then install 
the packages with your preferred package manager. 
For pip, run:

~~~
pip install -r requirements.txt
~~~

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


## Example usage:

### Example with a pre-existing Google Trends AnchorBank
After installing it with pip, first import the module in your script:
~~~python
import gtab
~~~

Then, create a GTAB object with the desired path:
~~~python
t = gtab.GTAB(my_path)
~~~
This will create and initialize the directory structure of *my_path*.

To list the available G-TABs call:
~~~python
t.list_gtabs()
~~~
This will produce a list of the selectable anchorbanks. There are three included defaults, and the output should look like:
~~~
Existing GTABs:
        google_anchorbank_geo=IT_timeframe=2019-11-28 2020-07-28.tsv
        google_anchorbank_geo=SE_timeframe=2019-11-28 2020-07-28.tsv
        google_anchorbank_geo=US_timeframe=2019-11-28 2020-07-28.tsv
Active anchorbank: None selected.
~~~

To select which G-TAB to use, call:
~~~python
t.set_active_gtab("google_anchorbank_geo=IT_timeframe=2019-11-28 2020-07-28.tsv")
~~~
This will confirm:
~~~
Active anchorbank changed to: google_anchorbank_geo=IT_timeframe=2019-11-28 2020-07-28.tsv
~~~

Next we need ensure we have the correct corresponding config options (for new queries only the *ptrends_config* is relevant). To set them, we call:
~~~python
t.set_options(ptrends_config = {"geo": "IT", "timeframe": "2019-11-28 2020-07-28" })
~~~
This can also be done by manually editing the config file found at: *my_path/config/config.json*

Now we can request calibrated data for a new query:
~~~python
nq_res= t.new_query("Sweet potato")
~~~


### Example creation of an anchorbank:
As in the previous example, the module is imported and then the object is created with the desired pat.
~~~python
import gtab
t = gtab.GTAB(my_path)
~~~

The desired config options can be set through the object method *set_options*. For example, if we want to construct a G-TAB with data from Germany between March 5th 2020 and May 5th 2020, we use:

~~~python
t.set_options(ptrends_config = {"geo": "DE", "timeframe": "2020-03-05 2020-05-05"})
~~~

We also need to specify which file to use for sampling the keywords:

~~~python
t.set_options(gtab_config = {"anchor_candidates_file": "my_data_file.txt"})
~~~
This file needs to be located at *my_path/data/my_data_file.txt* and contains one keyword per line.

We then need to set *N* and *K*, as described in the paper. For example, if we want to set *N=3000* and *K=500*, we call:
~~~python
t.set_options(gtab_config = {"num_anchor_candidates": 3000, "num_anchors": 500})
~~~

All of the config options can be directly edited in the config file found at *my_path/config/config.json*,

Finally, we construct the G-TAB:
~~~python
t.create_anchorbank()
~~~
This will start querying Google Trends and calibrate the data and will take some time, depending on *K*. After it is constructed it can be listed and selected as described in the previous example.  


