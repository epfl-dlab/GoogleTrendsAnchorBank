import ast
import copy
import datetime
import glob
import json
import os
import pickle
import random
import shutil
import time
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from pytrends.request import TrendReq
from tqdm import tqdm


class GTAB:

    def __delete_all_internal_files(self):
        """  Deletes all saved caches (keywords, results and pairs). Be careful! """
        dirs = ("google_keywords", "google_pairs", "google_results")

        files = list()
        for d in dirs:
            files += glob.glob(os.path.join(self.dir_path, "data", "internal", d, "*"))
            files += glob.glob(os.path.join(self.dir_path, "logs", "*"))

        nl = '\n'
        tb = '\t'
        print(f"This will delete the following files:\n\t{(nl + tb).join([os.path.basename(f) for f in files])}")
        c = input("Are you sure? (y/n): ")
        if c[0].lower() == 'y':
            print("Deleting...")
            for f in files:
                os.remove(f)
            print("Internal files deleted!")
        else:
            print("Delete cancelled.")

    def __init__(self, dir_path=None, from_cli=False):
        """
        Initializes a GTAB instance with the desired directory.

        :param dir_path:  path where to create a directory. If left to None, uses default package directory;
        :param from_cli: Is from command line interface?
        :param high_traffic: If true, adds high traffic keywords.
        """

        self.from_cli = from_cli
        if dir_path is None:
            self.dir_path = os.path.dirname(os.path.abspath(__file__))
        else:
            self.dir_path = dir_path
            if not os.path.exists(dir_path):
                default_path = os.path.dirname(os.path.abspath(__file__))

                # creating directory structure
                os.makedirs(os.path.join(self.dir_path, "logs"))
                os.makedirs(os.path.join(self.dir_path, "config"))
                os.makedirs(os.path.join(self.dir_path, "data", "internal", "google_keywords"))
                os.makedirs(os.path.join(self.dir_path, "data", "internal", "google_pairs"))
                os.makedirs(os.path.join(self.dir_path, "data", "internal", "google_results"))
                os.makedirs(os.path.join(self.dir_path, "output", "google_anchorbanks"))

                # copying defaults
                shutil.copyfile(os.path.join(default_path, "data", "anchor_candidate_list.txt"),
                                os.path.join(self.dir_path, "data", "anchor_candidate_list.txt"))
                shutil.copyfile(os.path.join(default_path, "config", "config_py.json"),
                                os.path.join(self.dir_path, "config", "config_py.json"))
                shutil.copyfile(os.path.join(default_path, "config", "config_cl.json"),
                                os.path.join(self.dir_path, "config", "config_cl.json"))
                for f in glob.glob(os.path.join(default_path, "output", "google_anchorbanks", "*.tsv")):
                    shutil.copyfile(f, os.path.join(self.dir_path, "output", "google_anchorbanks", os.path.basename(f)))
            else:
                print("Directory already exists, loading data from it.")

        print(f"Using directory '{self.dir_path}'")
        if from_cli:
            with open(os.path.join(self.dir_path, "config", "config_cl.json"), 'r') as fp:
                self.CONFIG = json.load(fp)
        else:
            with open(os.path.join(self.dir_path, "config", "config_py.json"), 'r') as fp:
                self.CONFIG = json.load(fp)

        self.CONFIG['CONN']['timeout'] = tuple(self.CONFIG['CONN']['timeout'])
        self.ANCHOR_CANDIDATES = [el.strip() for el in open(
            os.path.join(self.dir_path, "data", self.CONFIG['GTAB']['anchor_candidates_file']), "r")]

        if self.CONFIG['GTAB']['num_anchor_candidates'] >= len(self.ANCHOR_CANDIDATES):
            self.CONFIG['GTAB']['num_anchor_candidates'] = len(self.ANCHOR_CANDIDATES)

        self.HITRAFFIC = self.CONFIG["HITRAFFIC"]

        self.active_gtab = None
        self.pytrends = TrendReq(hl='en-US', **self.CONFIG['CONN'])

        # sets default anchorbank
        if not self.from_cli:
            default_anchorbank = "google_anchorbank_geo=_timeframe=2019-01-01 2020-08-01.tsv"
            self.set_active_gtab(default_anchorbank)

    # --- UTILITY METHODS --- 

    def _print_and_log(self, text, verbose=True):
        if verbose:
            print(text)
        self._log_con.write(text + '\n')

    def _make_file_suffix(self):
        return "_".join([f"{k}={v}" for k, v in self.CONFIG['PYTRENDS'].items()])

    def _query_google(self, keywords=["Keywords"]):
        time.sleep(self.CONFIG['GTAB']['sleep'])
        if type(keywords) == str:
            keywords = [keywords]

        if len(keywords) > 5:
            raise ValueError("Number of keywords must be at most than 5.")

        # avoids duplicate query
        if len(keywords) == 2 and keywords[0] == keywords[1]:
            self.pytrends.build_payload(kw_list=[keywords[0]], **self.CONFIG['PYTRENDS'])
            ret = self.pytrends.interest_over_time()
            ret.insert(loc=1, column="tmp", value=ret.iloc[:, 0])
        else:
            self.pytrends.build_payload(kw_list=keywords, **self.CONFIG['PYTRENDS'])
            ret = self.pytrends.interest_over_time()
        return ret

    def _is_not_blacklisted(self, keyword):
        return keyword not in self.CONFIG['BLACKLIST']

    def _check_keyword(self, keyword):
        time.sleep(self.CONFIG['GTAB']['sleep'])
        try:
            rez = self._query_google(keywords=keyword)
            return self._is_not_blacklisted(keyword) and not rez.empty
        except ValueError as e:
            raise e
        except Exception as e:
            self._print_and_log(f"\nBad keyword '{keyword}', because {str(e)}")
            if "response" in dir(e):
                if e.response.status_code == 429:
                    raise ConnectionError("Code 429: Query limit reached on this IP!")

            return False

    def _check_ts(self, ts):
        return ts.max() >= self.CONFIG['GTAB']['thresh_offline']

    def _find_nans(self, W0):
        nans = set()
        idxs = set()
        for row in range(W0.shape[0]):
            for col in range(W0.shape[1]):
                if np.isnan(W0.iloc[row, col]):
                    nans.add(W0.index[row])
                    idxs.add(row)

        ret = list(zip(nans, idxs))
        return sorted(ret, key=lambda x: x[1])

    def _diagnose_keywords(self, rez_dict):
        ret_keys = []
        for gres in rez_dict.values():
            ret_keys.append(list(gres.columns))
        for i in range(len(ret_keys) - 1):
            for t_idx in range(1, 5):
                if ret_keys[i][t_idx] != ret_keys[i + 1][t_idx - 1]:
                    self._print_and_log("Non-continuous groups at: ")
                    self._print_and_log(f"{str(ret_keys[i])}")
                    self._print_and_log(f"{str(ret_keys[i + 1])}")
                    self._error_flag = True
        return True

    def _check_groups(self, gres):

        ret = []
        for val in gres.values():
            if sum(val.max() < self.CONFIG['GTAB']['thresh_offline']) == 4:  # try with 3 as well, less restrictive
                ret.append(val.max().idxmax())

        return ret

    def _find_bads(self, gres, keywords):

        ret = {k[1]: [k[0], 0] for k in enumerate(keywords)}
        for val in gres.values():
            bads = list(val.columns[val.max() < self.CONFIG['GTAB']['thresh_offline']])
            for b in bads:
                ret[b][1] += 1

        return ret

    def _diagnose_bads(self, gres, keywords):

        ret = []
        # new heuristic
        # count each bad occurence per keyword
        bad_dict = self._find_bads(gres, keywords)
        # which are always bad?
        bad_kws1 = [k for k, v in bad_dict.items() if v[1] >= 5]
        # find their index in the original list
        bad_idxs1 = [bad_dict[kw][0] for kw in bad_kws1]

        # old heuristic
        bad_kws2 = self._check_groups(gres)
        bad_idxs2 = [bad_dict[kw][0] for kw in bad_kws2 if kw in bad_dict and bad_dict[kw][0] < len(gres) - 4]

        ret = sorted(bad_idxs1 + bad_idxs2)

        print(f"Total bad: {len(ret)}")
        # self._print_and_log(f"Total bad: {len(bad_idxs)}")

        return ret

    #  --- ANCHOR BANK METHODS ---
    def _get_google_results(self):

        fpath = os.path.join(self.dir_path, "data", "internal", "google_results",
                             f"google_results_{self._make_file_suffix()}.pkl")
        fpath_keywords = os.path.join(self.dir_path, "data", "internal", "google_keywords",
                                      f"google_keywords_{self._make_file_suffix()}.pkl")

        if os.path.exists(fpath):
            self._print_and_log(f"Loading google results from {fpath}...")
            with open(fpath, 'rb') as f_in:
                ret = pickle.load(f_in)
            self._diagnose_keywords(ret)
            return ret
        else:

            def get_kws(req_rng, new_kws):

                ret = []
                for j in range(req_rng[0], req_rng[1] + 1):
                    if not new_kws[j][1]:
                        continue
                    t = []
                    cnt = 0
                    idx = 0
                    while cnt < 5:
                        if new_kws[j + idx][1]:
                            t.append(new_kws[j + idx][0])
                            cnt += 1
                        idx += 1
                    ret.append(t)
                return ret

            def compute_requery_ranges(bad_idxs):

                ret = []
                ret_cnts = []
                i = 0
                while i < len(bad_idxs):
                    L = bad_idxs[i] - 4
                    R = bad_idxs[i]
                    next_i = i + 1

                    counter = 1
                    while True:
                        if next_i >= len(bad_idxs):
                            ret.append((L, R))
                            ret_cnts.append(counter)
                            break

                        if R >= bad_idxs[next_i] - 4:
                            R = bad_idxs[next_i]
                            next_i += 1
                            counter += 1
                        else:
                            ret.append((L, R))
                            ret_cnts.append(counter)
                            break
                    i = next_i

                return ret, ret_cnts

            N = self.CONFIG['GTAB']['num_anchor_candidates']
            K = self.CONFIG['GTAB']['num_anchors']

            # --- Get stratified samples, 1 per stratum. ---
            if not os.path.exists(fpath_keywords):

                self._print_and_log("Sampling keywords...")
                random.seed(self.CONFIG['GTAB']['seed'])
                np.random.seed(self.CONFIG['GTAB']['seed'])

                samples = []
                for j in range(K):
                    start_idx = (j * N) // K
                    end_idx = (((j + 1) * N) // K) - 1
                    s1 = random.randint(start_idx, end_idx)
                    samples.append(self.ANCHOR_CANDIDATES[s1])

                keywords = self.HITRAFFIC + samples
                keywords = [k for k in tqdm(keywords, total=len(keywords)) if self._check_keyword(k)]

            else:
                print(f"Getting keywords from {fpath_keywords}")
                with open(fpath_keywords, 'rb') as f_kw_in:
                    keywords = pickle.load(f_kw_in)
                keywords = [k for k in tqdm(keywords, total=len(keywords)) if self._is_not_blacklisted(k)]

            with open(fpath_keywords, 'wb') as f_out_keywords:
                pickle.dump(keywords, f_out_keywords, protocol=4)

            keywords = np.array(keywords)

            # --- Query google on keywords and dump the file. ---
            self._print_and_log("Querying google...")

            t_ret = dict()
            for i in tqdm(range(len(keywords) - 4)):

                try:
                    df_query = self._query_google(keywords=keywords[i:i + 5]).iloc[:, 0:5]
                except ValueError as e:
                    raise e
                except Exception as e:
                    if "response" in dir(e):
                        if e.response.status_code == 429:
                            input("Quota reached! Please change IP and press any key to continue.")
                            df_query = self._query_google(keywords=keywords[i:i + 5]).iloc[:, 0:5]
                        else:
                            self._print_and_log(str(e))
                t_ret[i] = df_query

            self._print_and_log("Removing bad queries...")
            no_passes = 0
            bad_queries_total = 0
            while True:
                ret = dict()
                new_kws = [[kw, True] for kw in keywords]

                bad_idxs = self._diagnose_bads(t_ret, keywords)
                bad_queries_total += len(bad_idxs)

                # no bads? done.
                if not bad_idxs:
                    ret = dict()
                    for copy_idx in range(0, len(t_ret) - 4):
                        ret[copy_idx] = copy.deepcopy(t_ret[copy_idx])
                    # ret = copy.deepcopy(t_ret[:, len(t_ret) - 4])
                    break
                no_passes += 1

                # flag the keywords that are bad
                for b in bad_idxs:
                    new_kws[b][1] = False

                # update the keywords
                keywords = [el[0] for el in new_kws if el[1]]

                # compute where we should re-query
                requery_ranges, requery_counts = compute_requery_ranges(bad_idxs=bad_idxs)

                idx_new = 0
                start_copy = 0
                for req_rng, _ in zip(requery_ranges, requery_counts):

                    # copy the query groups that don't change... assuming the first 5 don't break
                    for j in range(start_copy, req_rng[0]):
                        ret[idx_new] = copy.deepcopy(t_ret[j])
                        idx_new += 1
                    start_copy = req_rng[1] + 1

                    # ...re-query the ones that do
                    kw_groups = get_kws(req_rng, new_kws)
                    for kw_group in tqdm(kw_groups):

                        try:
                            df_query = self._query_google(keywords=kw_group).iloc[:, 0:5]
                        except ValueError as e:
                            raise e
                        except Exception as e:
                            if "response" in dir(e):
                                if e.response.status_code == 429:
                                    input("Quota reached! Please change IP and press any key to continue.")
                                    df_query = self._query_google(keywords=kw_group).iloc[:, 0:5]
                                else:
                                    self._print_and_log(str(e))

                        ret[idx_new] = df_query
                        idx_new += 1

                if start_copy < len(t_ret):
                    for j in range(start_copy, len(t_ret)):
                        ret[idx_new] = copy.deepcopy(t_ret[j])
                        idx_new += 1

                self._diagnose_keywords(ret)
                t_ret = copy.deepcopy(ret)

            self._print_and_log("Diagnostics done!")

            # log how many keywords are removed
            self._print_and_log(f"A total of {bad_queries_total} bad keywords removed.")
            # object id sanity check
            assert id(ret[0]) != id(ret[1])

            with open(fpath, 'wb') as f_out:
                self._print_and_log(f"Saving google results as '{fpath}'...")
                pickle.dump(ret, f_out, protocol=4)

            return (ret)

    def _compute_hi_and_lo(self, max1, max2):

        if max1 < 100:
            hi1 = max1 + 0.5
            lo1 = max1 - 0.5
        else:
            hi1 = 100
            lo1 = 100

        if max2 < 100:
            hi2 = max2 + 0.5
            lo2 = max2 - 0.5
        else:
            hi2 = 100
            lo2 = 100

        if max1 == 100 and max2 == 100:
            lo1 = 99.5
            lo2 = 99.5

        if lo1 < 0:
            lo1 = 0.0

        if lo2 < 0:
            lo2 = 0.0
            
        return lo1, hi1, lo2, hi2

    def _compute_max_ratios(self, google_results):

        anchors = []
        v1, v2 = [], []
        ratios, ratios_hi, ratios_lo = [], [], []
        errors, log_errors = [], []

        for _, val in google_results.items():
            for j in range(val.shape[1]):
                for k in range(val.shape[1]):
                    if j == k:
                        continue

                    if self._check_ts(val.iloc[:, j]) and self._check_ts(val.iloc[:, k]):
                        anchors.append(val.columns[0])  # first element of the group
                        v1.append(val.columns[j])
                        v2.append(val.columns[k])
                        max1 = val.iloc[:, j].max()
                        max2 = val.iloc[:, k].max()
                        lo1, hi1, lo2, hi2 = self._compute_hi_and_lo(max1, max2)
                        ratios.append(max2 / max1)
                        ratios_hi.append(hi2 / lo1)
                        ratios_lo.append(lo2 / hi1)
                        errors.append(hi2 / lo2 * hi1 / lo1)
                        log_errors.append(np.log(hi2 / lo2 * hi1 / lo1))

        ret = pd.DataFrame(
            data={'v1': v1, 'v2': v2, 'anchor': anchors, 'ratio': ratios, 'ratio_hi': ratios_hi, 'ratio_lo': ratios_lo,
                  'error': errors, 'weight': log_errors})

        # Removing excess edges for MultiGraph -> Graph conversion.
        ret = ret.sort_values('weight', ascending=True).drop_duplicates(["v1", "v2"]).sort_index()

        return ret

    def _infer_all_ratios(self, ratios):

        def compute_path_attribs(path, G, attribs):
            ret = {attr: 1.0 for attr in attribs}
            if len(path) > 1:
                for i in range(len(path) - 1):
                    for attr in attribs:
                        ret[attr] *= G[path[i]][path[i + 1]][attr]

            return ret

        G = nx.convert_matrix.from_pandas_edgelist(ratios, 'v1', 'v2',
                                                   edge_attr=['weight', "ratio", "ratio_hi", "ratio_lo", 'error'],
                                                   create_using=nx.DiGraph)

        assert len(G.edges()) % 2 == 0

        # strongly?
        if not nx.is_strongly_connected(G) or not nx.is_weakly_connected(G) or nx.number_connected_components(
                nx.Graph(G)) > 1:
            for comp in nx.connected_components(nx.Graph(G)):
                self._print_and_log(f"Component with length {len(comp)}: {str(comp)}")
                self._error_flag = True

            self._print_and_log(
                f"Num. connected components: {nx.number_strongly_connected_components(G)}. "
                f"Directed graph is not strongly connected!")
            warnings.warn("Directed graph is not connected!")

            # ratios.to_csv("ratios_test.tsv", sep = '\t')

        paths = list(nx.all_pairs_dijkstra_path(G))
        self.paths = paths
        node_names = [el[0] for el in paths]
        W = pd.DataFrame(columns=node_names, index=node_names, dtype=float)
        W_lo = pd.DataFrame(columns=node_names, index=node_names, dtype=float)
        W_hi = pd.DataFrame(columns=node_names, index=node_names, dtype=float)

        self._print_and_log("Finding paths...")
        for p_obj in tqdm(paths):
            for p in p_obj[1].values():
                path_attrib_dict = compute_path_attribs(p, G, ['weight', "ratio", "ratio_hi", "ratio_lo", 'error'])
                v1 = p[0]
                v2 = p[-1]
                W.loc[v1][v2] = path_attrib_dict['ratio']
                W_lo.loc[v1][v2] = path_attrib_dict['ratio_lo']
                W_hi.loc[v1][v2] = path_attrib_dict['ratio_hi']
        self._print_and_log("Paths done!")

        return W, W_lo, W_hi

    def _find_optimal_query_set(self, W0):

        def get_extreme(which):
            if which not in ['top', 'bottom']:
                raise ValueError()

            sign = 1 if which == 'top' else -1
            t_W0 = (sign * W0 <= sign).sum(axis=1)
            t_max = t_W0.max()

            ext = t_W0.index[t_W0 == t_max]
            if len(ext) > 1:
                ext = [W0[ext].max(axis=0).idxmax()]
            return ext[0]

        top = get_extreme('top')
        bot = get_extreme('bottom')
        D = nx.from_pandas_adjacency(np.abs(np.log(W0) + 1), create_using=nx.DiGraph)
        path = nx.bellman_ford_path(D, source=top, target=bot, weight='weight')

        return path

    def _build_optimal_anchor_bank(self, mids):

        N = len(mids)
        fpath = os.path.join(self.dir_path, "data", "internal", "google_pairs",
                             f"google_pairs_{self._make_file_suffix()}.pkl")

        if os.path.exists(fpath):
            with open(fpath, 'rb') as f_in:
                pairwise_dict = pickle.load(f_in)
        else:
            pairwise_dict = dict()
            self._print_and_log("Querying pairs...")
            for i in tqdm(range(0, N - 1)):
                pairwise_dict[i] = self._query_google(keywords=[mids[i], mids[i + 1]])
            with open(fpath, 'wb') as f_out:
                pickle.dump(pairwise_dict, f_out)

        pairs_max_ratios = [1.0] + [np.max(ts.iloc[:, 1] / 100) for ts in pairwise_dict.values()]
        pairs_max_ratios_hi = [1.0] + [np.max((ts.iloc[:, 1] + 0.5) / 100) for ts in pairwise_dict.values()]
        pairs_max_ratios_lo = [1.0] + [np.max((ts.iloc[:, 1] - 0.5) / 100) for ts in pairwise_dict.values()]

        W = pd.DataFrame(0, index=mids, columns=mids)
        W_hi = pd.DataFrame(0, index=mids, columns=mids)
        W_lo = pd.DataFrame(0, index=mids, columns=mids)

        for i in range(0, N):
            W.iloc[i] = np.cumprod(pairs_max_ratios) / np.prod(pairs_max_ratios[0:(i + 1)])
            W_hi.iloc[i] = np.cumprod(pairs_max_ratios_hi) / np.prod(pairs_max_ratios_hi[0:(i + 1)])
            W_lo.iloc[i] = np.cumprod(pairs_max_ratios_lo) / np.prod(pairs_max_ratios_lo[0:(i + 1)])

        W_hi = pd.DataFrame(np.triu(W_hi) + np.tril(1 / W_lo.transpose(), k=-1), index=mids, columns=mids)
        W_lo = pd.DataFrame(np.triu(W_lo) + np.tril(1 / W_hi.transpose(), k=-1), index=mids, columns=mids)

        return W, W_lo, W_hi

    # --- "EXPOSED" METHODS ---
    def print_options(self):
        """ Prints the current config options in the active directory. """
        if self.from_cli:
            print(f"Config file: {os.path.join(self.dir_path, 'config', 'config_cl.json')}")
        else:
            print(f"Config file: {os.path.join(self.dir_path, 'config', 'config_py.json')}")
        print(self.CONFIG)

    def set_options(self, pytrends_config=None, gtab_config=None, conn_config=None, overwite_file: bool = False):
        """
            Overwrites specified options. This can also be done manually by editing 'config_py.json' in the active directory.

                pytrends_config - a dictionary containing values to overrwite some of the configuration parameters for the pytrends library when 
                building the payload. It consists of two parameters:
                    - geo (str) - containing the two-letter ISO code of the country, e.g. "US", or "" empty string for global;
                    - timeframe (str) - containing the timeframe in which to query, e.g. "2019-11-28 2020-07-28".

                gtab_config - a dictionary containing values to overrwite some of the configuration parameters for the GTAB methodology. It contains the following parameters:
                    - num_anchor_candidates (int) - how many candidates to consider from the anchor candidate list;
                    - num_anchors (int) - how many to sample;
                    - seed (int) - the random seed;
                    - sleep (float) - sleep  e.g. 0.5,
                    - thresh_offline - the threshold below which the Google Trends query is discarded in the offline phase, e.g. 10.

                conn_config - a dictionary containing values to overrwite some of the configuration parameters for the connection. It contains:
                    - backoff_factor (float) - e.g. 0.1;
                    - proxies (list of strings) - which proxies to use, e.g. ["https://50.2.15.109:8800", "https://50.2.15.1:8800"];
                    - retries (int) - how many times to retry connection;
                    - timeout (list of two values) - e.g. [25, 25]

                overwrite_file - whether to overwrite the config_py.json file in the active directory.
            """

        if pytrends_config is not None:
            if type(pytrends_config) != dict:
                raise TypeError("The pytrends_config argument must be a dictionary with valid parameters!")
            for k in pytrends_config:
                if k not in self.CONFIG['PYTRENDS']:
                    raise ValueError(f"Invalid parameter: {k}")
                self.CONFIG["PYTRENDS"][k] = pytrends_config[k]

        if gtab_config is not None:
            if type(gtab_config) != dict:
                raise TypeError("The gtab_config argument must be a dictionary with valid parameters!")
            for k in gtab_config:
                if k not in self.CONFIG['GTAB']:
                    raise ValueError(f"Invalid parameter: {k}")
                self.CONFIG["GTAB"][k] = gtab_config[k]

        if conn_config is not None:
            if type(conn_config) != dict:
                raise TypeError("The conn_config argument must be a dictionary with valid parameters!")
            for k in conn_config:
                if k not in self.CONFIG['CONN']:
                    raise ValueError(f"Invalid parameter: {k}")
                self.CONFIG["CONN"][k] = conn_config[k]

        # update objects whose state depends on config jsons
        self.CONFIG['CONN']['timeout'] = tuple(self.CONFIG['CONN']['timeout'])
        self.pytrends = TrendReq(hl='en-US', **self.CONFIG['CONN'])

        if overwite_file:
            if self.from_cli:
                config_path = os.path.join(self.dir_path, "config", "config_cl.json")
            else:
                config_path = os.path.join(self.dir_path, "config", "config_py.json")
            print(f"Overwriting config at: {config_path}\n")
            with open(config_path, 'w') as fp:
                json.dump(self.CONFIG, fp, indent=4, sort_keys=True)

    def set_blacklist(self, keywords, overwrite_file: bool = False):
        """
            Sets a new blacklist. This can also be done manually by editing 'config_py.json' in the active directory.

            Input parameters:
                keywords - list of str keywords;
                overwrite_file - whether to overwrite the config_py.json file in the active directory.

        """
        if type(keywords) != list:
            raise TypeError("Keywords paremeter must be a list with elements of type str.")
        keywords = [str(kw) for kw in keywords]
        self.CONFIG["BLACKLIST"] = keywords

        if overwrite_file:
            if self.from_cli:
                config_path = os.path.join(self.dir_path, "config", "config_cl.json")
            else:
                config_path = os.path.join(self.dir_path, "config", "config_py.json")
            print(f"Overwriting config at: {config_path}\n")
            with open(config_path, 'w') as fp:
                json.dump(self.CONFIG, fp, indent=4, sort_keys=True)

    def set_hitraffic(self, keywords, overwrite_file: bool = False):
        """
            Sets a new hitraffic list. This can also be done manually by editing 'config_py.json' in the active directory.

            Input parameters:
                keywords - list of str keywords;
                overwrite_file - whether to overwrite the config_py.json file in the active directory.

        """
        if type(keywords) != list:
            raise TypeError("Keywords paremeter must be a list with elements of type str.")
        keywords = [str(kw) for kw in keywords]
        self.CONFIG["HITRAFFIC"] = keywords

        if overwrite_file:
            if self.from_cli:
                config_path = os.path.join(self.dir_path, "config", "config_cl.json")
            else:
                config_path = os.path.join(self.dir_path, "config", "config_py.json")
            print(f"Overwriting config at: {config_path}\n")
            with open(config_path, 'w') as fp:
                json.dump(self.CONFIG, fp, indent=4, sort_keys=True)

    def list_gtabs(self):
        """
            Lists filenames of constructed gtabs.
        """
        print("Existing GTABs:")
        for f in glob.glob(os.path.join(self.dir_path, "output", 'google_anchorbanks', '*')):
            print(f"\t{os.path.basename(f)}")

        print(
            f"Active anchorbank: {'None selected.' if self.active_gtab == None else os.path.basename(self.active_gtab)}")

    def rename_gtab(self, src, dst):
        """
            Renames a gtab.
            
            Input parameters:
                src - filename of source;
                dst - filename of destination.

        """
        src_path = os.path.join(os.path.join(self.dir_path, "output", "google_anchorbanks", src))
        if not os.path.exists(src_path):
            raise FileNotFoundError(src_path)
        os.rename(src_path, os.path.join(self.dir_path, "output", "google_anchorbanks", dst))
        print(f"Renamed '{src}' -> '{dst}'\n")
        if self.active_gtab is not None:
            if src.strip() == os.path.basename(self.active_gtab).strip():
                self.set_active_gtab(dst)

    def delete_gtab(self, src, require_confirmation=True):
        """
            Deletes a gtab.

            Input parameters:
                src - filename of source;
                require_confirmation - whether to prompt user for confirmation.

        """
        src_path = os.path.join(os.path.join(self.dir_path, "output", "google_anchorbanks", src))
        if not os.path.exists(src_path):
            raise FileNotFoundError(src_path)

        if require_confirmation:
            c = input(f"Are you sure you want to delete {src}? (y/n) ")
            if c[0].lower() == 'y':
                os.remove(src_path)
        else:
            os.remove(src_path)

        if src.strip() == os.path.basename(self.active_gtab).strip():
            print("The deleted gtab was set to active.")
            self.active_gtab = None

    def set_active_gtab(self, def_gtab):
        """
            Sets the active gtab for querying in the online phase.

            Input parameters:
                def_gtab - filename of the desired gtab. N.B. must exist in 'output/google_anchorbanks'
        """
        def_gtab_fpath = os.path.join(self.dir_path, "output", 'google_anchorbanks', def_gtab)
        if not os.path.exists(def_gtab_fpath):
            raise FileNotFoundError(def_gtab_fpath)

        self.active_gtab = def_gtab_fpath
        self.anchor_bank_full = pd.read_csv(self.active_gtab, sep='\t', comment='#', index_col=1).drop('Unnamed: 0',
                                                                                                       axis=1)
        self.anchor_bank_lo = self.anchor_bank_full.loc[:, 'max_ratio_lo']
        self.anchor_bank_hi = self.anchor_bank_full.loc[:, 'max_ratio_hi']
        self.anchor_bank = self.anchor_bank_full.loc[:, 'max_ratio']
        self.top_anchor = self.anchor_bank_full.index[0]
        self.top_anchor = self.anchor_bank_full.index[0]
        self.ref_anchor = self.anchor_bank_full.index[self.anchor_bank_full.loc[:, 'max_ratio'] == 1.0][0]

        # Set the options that are in a commented header in the GTAB file
        with open(self.active_gtab, "r") as f_in:
            t_gtab_config = ast.literal_eval(f_in.readline()[1:].strip())['GTAB']
            t_pytrends_config = ast.literal_eval(f_in.readline()[1:].strip())['PYTRENDS']
            self.set_options(pytrends_config=t_pytrends_config, gtab_config=t_gtab_config)

        print(f"Active anchorbank changed to: {os.path.basename(self.active_gtab)}\n")

    def create_anchorbank(self, verbose=False, keep_diagnostics=False):

        """
        Creates a gtab according to the config files found in the directory "./config/" and saves it.
        """
        self._error_flag = False
        self._log_con = open(os.path.join(self.dir_path, "logs", f"log_{self._make_file_suffix()}.txt"),
                             'a')  # append vs write
        self._log_con.write(f"\n{datetime.datetime.now()}\n")
        self._print_and_log(
            f"Start AnchorBank init for region {self.CONFIG['PYTRENDS']['geo']} in timeframe {self.CONFIG['PYTRENDS']['timeframe']}...")

        if verbose:
            self._print_and_log(f"Full option parameters are: {self.CONFIG}")

        fname_base = os.path.join(self.dir_path, "output", "google_anchorbanks",
                                  f"google_anchorbank_{self._make_file_suffix()}.tsv")
        if not os.path.exists(fname_base):
            google_results = self._get_google_results()
            if keep_diagnostics:
                self.google_results = google_results

            # # write to file for testing
            # os.makedirs("test_data", exist_ok=True)
            # idx = 0
            # for gres in google_results.values():
            #     gres.to_csv(f"test_data/{idx}.tsv", sep = '\t')
            #     idx+=1

            self._print_and_log(f"Total queries (groups of 5 keywords): {len(google_results)}")
            time_series = pd.concat(google_results, axis=1)
            if keep_diagnostics:
                self.time_series = time_series

            ratios = self._compute_max_ratios(google_results)
            if keep_diagnostics:
                self.ratios = ratios

            W0, W0_lo, W0_hi = self._infer_all_ratios(ratios)
            if keep_diagnostics:
                self.W0, self.W0_lo, self.W0_hi = W0, W0_lo, W0_hi

            # Rounding produces epsilon differences after transposing.
            err = np.abs(1 - W0 * W0.transpose()).to_numpy().max()
            self._print_and_log(f"Err: {err}")
            if err > 1e-12:
                warnings.warn("W0 doesn't seem to be multiplicatively symmetric: W0[i,j] != 1/W0[j,i].")
                self._log_con.write("W0 doesn't seem to be multiplicatively symmetric: W0[i,j] != 1/W0[j,i].\n")
            if np.isnan(err):
                warnings.warn("Err is NaN.")

            opt_query_set = self._find_optimal_query_set(W0)
            if keep_diagnostics:
                self.err = err
                self.opt_query_set = opt_query_set

            W, W_lo, W_hi = self._build_optimal_anchor_bank(opt_query_set)
            if keep_diagnostics:
                self.W, self.W_lo, self.W_hi = W, W_lo, W_hi

            top_anchor = opt_query_set[0]
            ref_anchor = np.abs(W.loc[top_anchor, :] - np.median(W0.loc[top_anchor, :])).idxmin()
            # anchor_bank = W.loc[ref_anchor, :]
            # anchor_bank_hi = W_hi.loc[ref_anchor, :]
            # anchor_bank_lo = W_lo.loc[ref_anchor, :]
            anchor_bank_full = pd.DataFrame({"max_ratio": W.loc[ref_anchor, :], "max_ratio_lo": W_lo.loc[ref_anchor, :],
                                             "max_ratio_hi": W_hi.loc[ref_anchor, :]}).reset_index().rename(
                {"index": "google_query"}, axis=1)
            self._print_and_log(f"Total range: {anchor_bank_full.iloc[-1, 1] / anchor_bank_full.iloc[0, 1]}")

            self._print_and_log(f"Saving anchorbank as '{fname_base}'...")

            with open(fname_base, 'w', newline='') as f_ab_out:  # 'w' vs 'a'?!
                t_gtab_config = {"GTAB": self.CONFIG['GTAB']}
                t_pytrends_config = {"PYTRENDS": self.CONFIG['PYTRENDS']}
                f_ab_out.write(f"# {t_gtab_config}\n")
                f_ab_out.write(f"# {t_pytrends_config}\n")
                anchor_bank_full.to_csv(f_ab_out, sep='\t', header=True)

            self._print_and_log("AnchorBank init done.")
            if self._error_flag:
                self._print_and_log("There was an error. Please check the log file.")
        else:
            print(
                "GTAB with such parameters already exists! Load it with 'set_active_gtab(filename)' or rename/delete it"
                " to create another one with this name.")

        self._log_con.close()

    def new_query(self, query, first_comparison=None, thresh=10, verbose=False, complete=False):
        """ Request a new Google Trends query and calibrate it with the active gtab.  The 'PYTRENDS' key in
        config_py.json is used. Make sure to set it according to your chosen anchorbank!

        :param query: string containing a single query.
        :param first_comparison: first pivot for the binary search.
        :param thresh: threshold for comparison.
        :param verbose: want to hear what we're doing?
        :param complete: if True, returns params related to the execution.
        :return: The return is not deterministic
            (1) If not able to calibrate, returns -1;

            (1) you are calling gtab from the CLI
            (2)
                    If not able to calibrate, returns -1.

        """

        if self.active_gtab is None:
            raise ValueError("Must use 'set_active_gtab()' to select anchorbank before querying!")
        self._log_con = open(os.path.join(self.dir_path, "logs", f"log_{self._make_file_suffix()}.txt"), 'a')
        self._log_con.write(f"\n{datetime.datetime.now()}\n")
        self._print_and_log(f"Using {self.active_gtab}")
        self._print_and_log(f"New query '{query}'")
        mid = list(self.anchor_bank.index).index(self.ref_anchor) if first_comparison is None \
            else list(self.anchor_bank.index).index(first_comparison)
        anchors = tuple(self.anchor_bank.index)

        # if query in anchors:
        #     self._print_and_log(f"The query is already present in the active gtab!")
        #     self._log_con.close()
        #     return -2

        if not self._check_keyword(query):
            self._print_and_log(f"Keyword {query} is bad!")
            return -1

        lo = 0
        hi = len(self.anchor_bank) - 1
        n_iter = 0

        while hi >= lo:

            anchor = anchors[mid]
            if verbose:
                self._print_and_log(f"\tQuery: '{query}'\tAnchor:'{anchor}'")
            try:
                ts = self._query_google(keywords=[anchor, query]).iloc[:, 0:2]
            except ValueError as e:
                raise e
            except Exception as e:
                if "response" in dir(e):
                    if e.response.status_code == 429:
                        input("Quota reached! Please change IP and press any key to continue.")
                        ts = self._query_google(keywords=[anchor, query]).iloc[:, 0:2]
                else:
                    self._print_and_log(f"Google query '{query}' failed because: {str(e)}")
                    break

            if anchor == query:
                query = "tmp"

            timestamps = ts.index
            max_anchor = ts.loc[:, anchor].max()
            max_query = ts.loc[:, query].max()

            if max_query >= thresh and max_anchor >= thresh:

                max_query_lo, max_query_hi, max_anchor_lo, max_anchor_hi = self._compute_hi_and_lo(max_query,
                                                                                                   max_anchor)
                ts_query_lo, ts_query_hi = map(list,
                                               zip(*[self._compute_hi_and_lo(100, el)[2:] for el in ts.loc[:, query]]))

                if np.sum(ts.loc[:, query] == 100) == 1:
                    ts_query_lo[list(ts.loc[:, query]).index(100)] = 100

                ratio_anchor = self.anchor_bank[anchor]
                ratio_anchor_lo = self.anchor_bank_lo[anchor]
                ratio_anchor_hi = self.anchor_bank_hi[anchor]

                #NEW 
                ts_query = ts.loc[:, query] / max_anchor * ratio_anchor
                ts_query_hi = np.array(ts_query_hi) / max_anchor_lo * ratio_anchor_hi
                ts_query_lo = np.array(ts_query_lo) / max_anchor_hi * ratio_anchor_lo

                # OLD
                # ratio = ratio_anchor * (max_query / max_anchor)
                # ratio_hi = ratio_anchor_hi * (max_query_hi / max_anchor_lo)
                # ratio_lo = ratio_anchor_lo * (max_query_lo / max_anchor_hi)

                # ts_query = ts.loc[:, query] / max_query * ratio
                # ts_query_hi = np.array(ts_query_hi) / max_query_lo * ratio_hi
                # ts_query_lo = np.array(ts_query_lo) / max_query_hi * ratio_lo

                self._print_and_log("New query calibrated!")
                self._log_con.close()
                if self.from_cli:
                    if complete:
                        return {"max_ratio": float(ratio),
                                "max_ratio_hi": float(ratio_hi),
                                "max_ratio_lo": float(ratio_lo),
                                "ts_timestamp": [str(el) for el in timestamps],
                                "ts_max_ratio": list(ts_query),
                                "ts_max_ratio_hi": list(ts_query_hi),
                                "ts_max_ratio_lo": list(ts_query_lo),
                                "no_iters": n_iter
                                }
                    else:
                        return {"ts_timestamp": [str(el) for el in timestamps],
                                "ts_max_ratio": list(ts_query),
                                "ts_max_ratio_hi": list(ts_query_hi),
                                "ts_max_ratio_lo": list(ts_query_lo)
                                }
                else:
                    ts_df = pd.DataFrame(
                        {"max_ratio": list(ts_query), "max_ratio_hi": ts_query_hi, "max_ratio_lo": ts_query_lo},
                        index=timestamps)
                    if complete:
                        return {"max_ratio": float(ratio),
                                "max_ratio_hi": float(ratio_hi),
                                "max_ratio_lo": float(ratio_lo),
                                "ts": ts_df,
                                "no_iters": n_iter,
                                "query": query}
                    else:
                        return ts_df

            elif max_query < thresh:
                lo = mid + 1
            else:
                hi = mid - 1

            mid = (hi + lo) // 2
            n_iter += 1

        if hi <= 0:
            self._print_and_log('Could not calibrate. Time series for query too high everywhere.')
        else:
            self._print_and_log('Could not calibrate. Time series for query too low everywhere.')
        self._log_con.close()

        # Unable to be calibrated.
        return None
