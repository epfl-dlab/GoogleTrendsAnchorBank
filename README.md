![Repository logo](./logo.png)


[Google Trends](https://trends.google.com/) allows users to analyze the popularity of Google search
queries across time and space.
Despite the overall value of Google Trends, results are rounded to integer-level precision
which may causes major problems.

For example, lets say you want to compare the popularity of searches of the term "Switzerland,"
 to searches of the term "Facebook"!

![Image portraying rounding issues with Google Trends](./example/imgs/lead.png)

We find that the comparison is highly non-informative!
Since the popularity of Switzerland is always "<1%", we simply can't compare the two!

## G-TAB to the rescue!

Fortunately, this library solves this problem! You can simply do:

~~~python
import gtab
t = gtab.GTAB()
# Make the queries which will return precise values!
query_facebook = t.new_query("Facebook")
query_switzerland = t.new_query("Switzerland")
~~~

And you will have the two queries in comparable units! In fact, any subsequent query you make will also be comparable!
You could then plot those (for example in log-scale to make things simpler and get something like):

~~~python
import matplotlib.pyplot as plt 
plt.plot(query_switzerland.max_ratio )
plt.plot(query_facebook.max_ratio)
plt.show()
~~~

![Image portraying output of the library, where issues are fixed](./example/imgs/lead2.png)

## How does it work?

**TL;DR:
G-TABS constructs a series of pre-computed queries,
and is able to calibrate any query by cleverly inspecting those.**

More formally, the method proceeds in two phases:

1. In the *offline pre-processing phase*, an "anchor bank" is constructed, a set of Google queries spanning the full spectrum 
of popularity, all calibrated against a common reference query by carefully chaining multiple Google Trends requests.

2. In the *online deployment phase*, any given search query is calibrated by performing an efficient binary search in the anchor bank.
Each search step requires one Google Trends request (via [pytrends](https://github.com/GeneralMills/pytrends)), but few
 steps suffice (see [empirical evaluation](https://arxiv.org/abs/2007.13861)).

A full description of the G-TAB method is available in the following paper:

> Robert West. **Calibration of Google Trends Time Series.** In *Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM)*. 2020. [**[PDF](https://arxiv.org/abs/2007.13861)**]

Please cite this paper when using G-TAB in your own work.

Code and data for reproducing the results of the paper are available in the directory [`cikm2020_paper`](_cikm2020_paper).

# Installation

The package is available on pip, so you just need to call
~~~python
python -m pip install gtab
~~~

The explicit list of requirements can be found in [`requirements.txt`](requirements.txt).
We developed and tested it in Python 3.8.1.

# Example usage

Want to use python? See [`example/example.ipynb`](example/example.ipynb).

Want to use command line? See TODO;

## F.A.Q.

**Q: Where can I understand more on the maths behind G-TAB?**

R: Your best bet is to read the CIKM paper, pointers to that can be found [here](cikm2020_paper/README.md).
Additionally, [this](cikm2020_paper/README.md) appendix explains how to calculate the error margins for the method

**Q: Do I need a new anchorbank for each different location and time I wanna query google trends with?**

R: Yes! But building those is easy! Be sure to check our examples, we teach how to do this [there](example/example.ipynb).

**Q: Okay, so you always build the anchorbanks with the same candidates (those in `/gtab/data/anchor_candidate_list.txt`), can I change that?**

R: Yes, you can! You can provide your own candidate list (a file with one word per line). 
Place it over the `./data` folder for whatever path you created and enforce its usage with:

~~~python
import gtab
t = gtab.GTAB()
t.set_options(gtab_config = {"anchor_candidates_file": "your_file_name_here.txt"})
~~~

We then need to set N and K, as described in the paper. 
Choosing N and K depends on the anchor candidate data set we are using.
 N specifies how deep to go in the data set, i.e. take the first N keywords from the file for sampling. 
 K specifies how many stratified samples we want to get. N has to be smaller than the total number of keywords in the
  anchor candidate data set, while it is good practice to set K to be in the range [0.1N, 0.2N]. 
  For example, if we want to set N=3000 and K=500, we call:
  
~~~python
t.set_options(gtab_config = {"num_anchor_candidates": 3000, "num_anchors": 500})
~~~

Confused? Don't worry! The default candidate works pretty well!
