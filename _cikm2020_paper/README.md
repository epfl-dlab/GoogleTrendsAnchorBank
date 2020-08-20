# Calibration of Google Trends Time Series

This directory contains all code and data to reproduce the results published in this paper:

> Robert West. **Calibration of Google Trends Time Series.** In *Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM)*. 2020. [**[PDF](https://arxiv.org/abs/2007.13861)**]

Note that, whereas the results in the paper were produced with an R implementation, we have since switched over to Python.
The R implementation that was used for the paper and that is contained in this directory is not updated anymore.
If you are interested in using Google Trends Anchor Bank (G-TAB) in your own research, please use the [Python implementation](https://github.com/epfl-dlab/GoogleTrendsAnchorBank) instead.

Directory contents:

- The master script for reproducing the paper is `src/R/code_for_reproducing_cikm2020_paper.Rmd`.
- Data can be found in `data/`; in particular, `data/calibration` contains the cached results of Google Trends requests for constructing the anchor bank in the offline preprocessing stage (Sec. 2.1 of the paper); `data/binsearch` contains cached results of Google Trends requests from the online binary search phase (Sec. 2.2 of the paper).
- All plots from the paper can be found in `plots/`.
