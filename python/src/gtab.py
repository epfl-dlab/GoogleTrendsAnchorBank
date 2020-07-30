import ast
import codecs
import copy
import datetime
import glob
import math
import os
import pickle
import random
import sys
import time
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from pytrends.request import TrendReq
from tqdm import tqdm


class GTAB:

    # Static methods
    
    @staticmethod
    def __delete_all_files():

        """
            Deletes all saved anchorbanks, keywords, results and pairs. Be very careful!
        """
        dirs = ("google_anchorbanks", "google_keywords", "google_pairs", "google_results")

        files = []
        for d in dirs:
            files += glob.glob(os.path.join("..", "data", d, "*"))
            files += glob.glob(os.path.join("..", "logs", "*"))
        
        print(f"This will delete the following files: {files}")
        c = input("Are you sure? (y/n): ")
        if c[0].lower() == 'y':
            print("Deleting...")
            for f in files:
                os.remove(f)

    def __init__(self, ptrends_config = None, gtab_config = None, conn_config = None, blacklist = None, use_proxies = False):
        """
            Initializes the GTAB instance.

            Input parameters:
                ptrends_config - a dictionary containing the configuration parameters for the pytrends library when building the payload (i.e. timeframe and geo).
                gtab_config - a dictionary containing the configuration parameters for the GTAB methodology when making the connection.
                conn_config - a dictionary containing the configuration parameters for the connection (e.g. proxies, timeout, retries...).

            If any parameter is None, the corresponding file in the directory "./config/" is taken. It should contain a single line with the corresponding config dictionary.
        """

        if ptrends_config == None:
            with open(os.path.join("..", "config", "ptrends.config"), "r") as f_conf:
                self.PTRENDS_CONFIG = ast.literal_eval(f_conf.readline())
        else: 
            self.PTRENDS_CONFIG = ptrends_config

        if gtab_config == None:
            with open(os.path.join("..", "config", "gtab.config"), "r") as f_conf:
                self.GTAB_CONFIG = ast.literal_eval(f_conf.readline())
        else: 
            self.GTAB_CONFIG = gtab_config

        if conn_config == None:
            with open(os.path.join("..", "config", "conn.config"), "r") as f_conf:
                self.CONN_CONFIG = ast.literal_eval(f_conf.readline())
        else:
            self.CONN_CONFIG = conn_config
            
        if blacklist == None:
            with open(os.path.join("..", "config", "blacklist.config"), "r") as f_conf:
                self.BLACKLIST = ast.literal_eval(f_conf.readline()) 
        else:
            self.BLACKLIST = blacklist

        with open(os.path.join("..", "data", "freebase_foods_all.tsv"), 'r', encoding = 'utf-8') as f_anchor_set:
            self.ANCHOR_CANDIDATE_SET = pd.read_csv(f_anchor_set, sep = '\t')

        self.ANCHOR_CANDIDATE_SET.rename(index = {k: v for k, v in enumerate(list(self.ANCHOR_CANDIDATE_SET['mid']))}, inplace = True)
        mask = (np.array(self.ANCHOR_CANDIDATE_SET['is_dish'] == 1) | np.array(self.ANCHOR_CANDIDATE_SET['is_food'] == 1))
        self.ANCHOR_CANDIDATE_SET = self.ANCHOR_CANDIDATE_SET[mask]
        if self.GTAB_CONFIG['num_anchor_candidates'] >= len(self.ANCHOR_CANDIDATE_SET):
            self.GTAB_CONFIG['num_anchor_candidates'] = len(self.ANCHOR_CANDIDATE_SET)

        self.HITRAFFIC = {'Facebook': '/m/02y1vz', 'YouTube': '/m/09jcvs', 'Instagram': '/m/0glpjll', 'Amazon.com': '/m/0mgkg', 'Netflix': '/m/017rf_', 'Yahoo!': '/m/019rl6', 'Twitter': '/m/0289n8t', 'Wikipedia': '/m/0d07ph', 'Reddit': '/m/0b2334', 'LinkedIn': '/m/03vgrr' , 'Airbnb': '/m/010qmszp'}#, 'Coca-Cola': '/m/01yvs'}
        self._init_done = False
        
        if use_proxies:
            self.ptrends = TrendReq(hl='en-US', **self.CONN_CONFIG)
        else:
            self.ptrends = TrendReq(hl='en-US', timeout = (20, 20))

    ## --- UTILITY METHODS --- #

    def _print_and_log(self, text):
        print(text)
        self._log_con.write(text + '\n')


    def _make_file_suffix(self):
        return "_".join([f"{k}={v}" for k, v in self.PTRENDS_CONFIG.items()])
        # return "_".join([f"{k}={v}" for k, v in self.GTAB_CONFIG.items()]) + "_" + "_".join([f"{k}={v}" for k, v in self.PTRENDS_CONFIG.items()])

    def _query_google(self, keywords = ["Keywords"]):
        time.sleep(self.GTAB_CONFIG['sleep'])
        if type(keywords) == str:
            keywords = [keywords]
    
        if len(keywords) > 5:
            raise ValueError("Query keywords must be at most than 5.")

        self.ptrends.build_payload(kw_list = keywords, **self.PTRENDS_CONFIG)
        return(self.ptrends.interest_over_time())

    def _is_blacklisted(self, keyword):
        return not keyword in self.BLACKLIST

    def _check_keyword(self, keyword):
        try:
            self._query_google(keywords = keyword)
            return self._is_blacklisted(keyword)
        except Exception as e:
            self._print_and_log(f"Bad keyword '{keyword}', because {str(e)}")
            if "response" in dir(e):
                if e.response.status_code == 429:
                    raise ConnectionError("Code 429: Query limit reached on this IP!")

            return False
    
    def _check_ts(self, ts):
        return ts.max().max() >= self.GTAB_CONFIG['thresh_offline']

    def _find_nans(self, W0):
        nans = set()
        idxs = set()
        for row in range(W0.shape[0]):
            for col in range(W0.shape[1]):
                if np.isnan(W0.iloc[row, col]):
                    # print(W0.index[row], W0.index[col])
                    nans.add(W0.index[row])
                    idxs.add(row)
        
        ret = list(zip(nans, idxs))
        return sorted(ret, key = lambda x: x[1])



    def _diagnose_keywords(self, rez_dict):

        ret_keys = []
        for gres in rez_dict.values():
            ret_keys.append(list(gres.columns))
        for i in range(len(ret_keys)-1):
            for t_idx in range(1, 5):
                if ret_keys[i][t_idx] != ret_keys[i+1][t_idx-1]:
                    self._print_and_log("Non-continuous groups at: ")
                    self._print_and_log(f"{str(ret_keys[i])}")
                    self._print_and_log(f"{str(ret_keys[i+1])}")
                    self._error_flag = True


        return True

    def _check_groups(self, gres):

        ret = []
        for val in gres.values():
            if sum(val.max() < self.GTAB_CONFIG['thresh_offline']) == 4: # try with 3 as well, less restrictive
                ret.append(val.max().idxmax())
                
        return ret


    def _find_bads(self, gres, keywords):

        ret = {k[1]: [k[0], 0] for k in enumerate(keywords)}
        for val in gres.values():
            bads = list(val.columns[val.max() < self.GTAB_CONFIG['thresh_offline']])
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
        bad_idxs2 = [bad_dict[kw][0] for kw in bad_kws2 if bad_dict[kw][0] < len(gres) - 4]

        ret = sorted(bad_idxs1 + bad_idxs2)

        print(f"Total bad: {len(ret)}")
        # self._print_and_log(f"Total bad: {len(bad_idxs)}")


        return ret

        

    ## --- ANCHOR BANK METHODS --- ##
    def _get_google_results(self):

        fpath = os.path.join("..", "data", "google_results", f"google_results_{self._make_file_suffix()}.pkl")
        fpath_keywords = os.path.join("..", "data", "google_keywords", f"google_keywords_{self._make_file_suffix()}.pkl")

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

            N = self.GTAB_CONFIG['num_anchor_candidates']
            K = self.GTAB_CONFIG['num_anchors']

            ## --- Get stratified samples, 1 per stratum. --- ##
            if not os.path.exists(fpath_keywords):

                self._print_and_log("Sampling keywords...")
                random.seed(self.GTAB_CONFIG['seed'])
                np.random.seed(self.GTAB_CONFIG['seed'])

                samples = []
                for j in range(K):
                    start_idx = (j * N) // K
                    end_idx = (((j+1) * N) // K) - 1
                    s1 = random.randint(start_idx, end_idx)
                    samples.append(self.ANCHOR_CANDIDATE_SET.index[s1])

                keywords = list(self.HITRAFFIC.values()) + samples
                keywords = [k for k in tqdm(keywords, total = len(keywords)) if self._check_keyword(k)]

            else:
                print(f"Getting keywords from {fpath_keywords}")
                with open(fpath_keywords, 'rb') as f_kw_in:
                    keywords = pickle.load(f_kw_in)         
                keywords = [k for k in tqdm(keywords, total = len(keywords)) if self._is_blacklisted(k)]

            with open(fpath_keywords, 'wb') as f_out_keywords:
                pickle.dump(keywords, f_out_keywords, protocol=4)
            
            keywords = np.array(keywords)
            
            ## --- Query google on keywords and dump the file. --- ##
            self._print_and_log("Querying google...")
    
            t_ret = dict()                
            for i in tqdm(range(len(keywords) - 4)):
                
                try:
                    df_query = self._query_google(keywords = keywords[i:i+5]).iloc[:, 0:5]
                except Exception as e:
                    if "response" in dir(e):
                        if e.response.status_code == 429:
                            c = input("Quota reached! Please change IP and press any key to continue.") 
                            df_query = self._query_google(keywords = keywords[i:i+5]).iloc[:, 0:5]
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
                    ret = copy.deepcopy(t_ret)
                    break
                no_passes +=1 
                
                # flag the keywords that are bad
                for b in bad_idxs:
                    new_kws[b][1] = False

                # update the keywords
                keywords = [el[0] for el in new_kws if el[1]]
                
                # compute where we should requerry                
                requery_ranges, requery_counts = compute_requery_ranges(bad_idxs = bad_idxs)

                idx_new = 0
                start_copy = 0
                for req_rng, req_cnt in zip(requery_ranges, requery_counts):
            
                    # copy the query groups that don't change... assuming the first 5 don't break
                    for j in range(start_copy, req_rng[0]):
                        ret[idx_new] = copy.deepcopy(t_ret[j])
                        idx_new += 1
                    start_copy = req_rng[1] + 1 

                    # ...requery the ones that do
                    kw_groups = get_kws(req_rng, new_kws)
                    for kw_group in tqdm(kw_groups):

                        try:
                            df_query = self._query_google(keywords = kw_group).iloc[:, 0:5]
                        except Exception as e:
                            if "response" in dir(e):
                                if e.response.status_code == 429:
                                    c = input("Quota reached! Please change IP and press any key to continue.") 
                                    df_query = self._query_google(keywords = kw_group).iloc[:, 0:5]
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

            return(ret)

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
                    
                    if(self._check_ts(val.iloc[:, j]) and self._check_ts(val.iloc[:, k])):
                        
                        anchors.append(val.columns[0]) # first element of the group
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

        ret = pd.DataFrame(data = {'v1': v1, 'v2': v2, 'anchor': anchors, 'ratio': ratios, 'ratio_hi': ratios_hi, 'ratio_lo': ratios_lo, 'error': errors, 'weight': log_errors})

        # Removing excess edges for MultiGraph -> Graph conversion.
        ret = ret.sort_values('weight', ascending = True).drop_duplicates(["v1", "v2"]).sort_index()        

        return ret


    def _infer_all_ratios(self, ratios):

        def compute_path_attribs( path, G, attribs):
            ret = {attr: 1.0 for attr in attribs}
            if len(path) > 1:
                for i in range(len(path) - 1):
                    for attr in attribs:
                        ret[attr] *= G[path[i]][path[i+1]][attr]

            return ret
                

        G = nx.convert_matrix.from_pandas_edgelist(ratios, 'v1', 'v2', edge_attr = ['weight', "ratio", "ratio_hi", "ratio_lo", 'error'], create_using = nx.DiGraph)

        assert len(G.edges()) % 2 == 0

        # strongly?
        if not nx.is_strongly_connected(G) or not nx.is_weakly_connected(G) or nx.number_connected_components(nx.Graph(G)) > 1:
            for comp in nx.connected_components(nx.Graph(G)):
                self._print_and_log(f"Component with length {len(comp)}: {str(comp)}")
                self._error_flag = True

            self._print_and_log(f"Num. connected components: {nx.number_strongly_connected_components(nx.Graph(G))}. Directed graph is not strongly connected!")
            warnings.warn("Directed graph is not connected!")

            # ratios.to_csv("ratios_test.tsv", sep = '\t')
        

        # log_lens = list(nx.all_pairs_bellman_ford_path_length(G))
        paths = list(nx.all_pairs_dijkstra_path(G))
        self.paths = paths
        node_names = [el[0] for el in paths]
        W = pd.DataFrame(columns = node_names, index = node_names, dtype = float)
        W_lo = pd.DataFrame(columns = node_names, index = node_names, dtype = float)
        W_hi = pd.DataFrame(columns = node_names, index = node_names, dtype = float)

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
        
    # colSums equivalent is df.sum(axis = 0)
    def _find_optimal_query_set(self, W0):

        def get_extreme(which):
            if which not in ['top', 'bottom']:
                raise ValueError()

            sign = 1 if which == 'top' else -1
            ext = list(W0.index[(sign * W0 <= sign).apply(all, axis = 1)])
            if len(ext) > 1 and isinstance(ext, list):
                ext = [ext[W0[ext].max().argmax()]]
            return ext[0]
                
        top = get_extreme('top')
        bot = get_extreme('bottom')
        D = nx.from_pandas_adjacency(np.abs(np.log(W0) + 1), create_using=nx.DiGraph)
        path = nx.bellman_ford_path(D, source = top, target = bot, weight = 'weight')
        
        return path

    def _build_optimal_anchor_bank(self, mids):

        N = len(mids)
        fpath = os.path.join("..", "data", "google_pairs", f"google_pairs_{self._make_file_suffix()}.pkl") 

        if os.path.exists(fpath):
            with open(fpath, 'rb') as f_in:
                pairwise_dict = pickle.load(f_in)
        else:
            pairwise_dict = dict()
            self._print_and_log("Querying pairs...")
            for i in tqdm(range(0, N-1)):
                pairwise_dict[i] = self._query_google(keywords = [mids[i], mids[i+1]])
            with open(fpath, 'wb') as f_out:
                pickle.dump(pairwise_dict, f_out)

        pairs_max_ratios = [1.0] + [np.max(ts.iloc[:, 1] / 100) for ts in pairwise_dict.values()] 
        pairs_max_ratios_hi = [1.0] + [np.max((ts.iloc[:, 1] + 0.5) / 100) for ts in pairwise_dict.values()] 
        pairs_max_ratios_lo = [1.0] + [np.max((ts.iloc[:, 1] - 0.5) / 100) for ts in pairwise_dict.values()] 

        W = pd.DataFrame(0, index = mids, columns = mids)
        W_hi = pd.DataFrame(0, index = mids, columns = mids)
        W_lo = pd.DataFrame(0, index = mids, columns = mids)
        
        for i in range(0, N):
            W.iloc[i] = np.cumprod(pairs_max_ratios) / np.prod(pairs_max_ratios[0:(i+1)])
            W_hi.iloc[i] = np.cumprod(pairs_max_ratios_hi) / np.prod(pairs_max_ratios_hi[0:(i+1)])
            W_lo.iloc[i] = np.cumprod(pairs_max_ratios_lo) / np.prod(pairs_max_ratios_lo[0:(i+1)])

        W_hi = pd.DataFrame(np.triu(W_hi) + np.tril(1/W_lo.transpose(), k = -1), index = mids, columns = mids)
        W_lo = pd.DataFrame(np.triu(W_lo) + np.tril(1/W_hi.transpose(), k = -1), index = mids, columns = mids)
            
        return W, W_lo, W_hi

    ## --- "EXPOSED" METHODS --- ##
    def init(self, verbose = False, keep_diagnostics = False):
        """
        Initializes the GTAB instance according to the config files found in the directory "./config/".
        """
        self._error_flag = False
        self._log_con = open(os.path.join("..", "logs", f"log_{self._make_file_suffix()}.txt"), 'a') # append vs write
        self._log_con.write(f"\n{datetime.datetime.now()}\n")
        self._print_and_log("Start AnchorBank init...")

        if verbose:
            print(self.GTAB_CONFIG)
            print(self.PTRENDS_CONFIG)

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
            self.opt_query_set = opt_query_set

        W, W_lo, W_hi = self._build_optimal_anchor_bank(opt_query_set)
        if keep_diagnostics:
            self.W, self.W_lo, self.W_hi = W, W_lo, W_hi
        
        top_anchor = opt_query_set[0]
        ref_anchor = np.abs(W.loc[top_anchor, :] - np.median(W0.loc[top_anchor, :])).idxmin()
        anchor_bank = W.loc[ref_anchor, :]
        anchor_bank_hi = W_hi.loc[ref_anchor, :]
        anchor_bank_lo = W_lo.loc[ref_anchor, :]
        anchor_bank_full = pd.DataFrame({"base": W.loc[ref_anchor, :], "lo": W_lo.loc[ref_anchor, :], "hi": W_hi.loc[ref_anchor, :]})

        self.err = err
        self.top_anchor = top_anchor
        self.ref_anchor = ref_anchor
        self.anchor_bank = anchor_bank
        self.anchor_bank_hi = anchor_bank_hi
        self.anchor_bank_lo = anchor_bank_lo
        self.anchor_bank_full = anchor_bank_full
        self._init_done = True

        fname_base = os.path.join("..", "data", "google_anchorbanks", f"google_anchorbank_{self._make_file_suffix()}")

        self._print_and_log(f"Saving anchorbanks as '{fname_base}'...")
        if os.path.exists(fname_base + ".tsv"):

            save_counter = 1
            while True:
                new_fname = fname_base = os.path.join("..", "data", "google_anchorbanks", f"google_anchorbank_{self._make_file_suffix()}({save_counter})")
                if os.path.exists(new_fname + ".tsv"):
                    save_counter += 1
                else:
                    fname_base = new_fname
                    break
            self._print_and_log(f"File already exists! Saving as {fname_base}.")
                

        with open(fname_base + ".tsv", 'a') as f_ab_out:
            f_ab_out.write(f"#{self.GTAB_CONFIG}\n")
            f_ab_out.write(f"#{self.PTRENDS_CONFIG}\n")
            self.anchor_bank_full.to_csv(f_ab_out, sep = '\t', header = True)

        self._print_and_log("AnchorBank init done.")
        if self._error_flag:
            self._print_and_log("There was an error. Please check the log file.")
        self._log_con.close()


    def new_query(self, query, first_comparison = None, thresh = 100 / np.e, verbose = False):

        """
        Request a new GTrends query and calibrate it with the GTAB instance.
        Input parameters:
            keyword - string containing a single query.
            first_comparison - first pivot for the binary search.
            thresh - threshold for comparison.
        Returns a dictionary containing:
            ratio - the computed ratio.
            ts - the calibrated time series for the keyword.
            iter - the number of iterations needed by the binary search.

            If not able to calibrate, returns -1.
            
        """

        if not self._init_done:
            print("Must use GTAB.init() to initialize first!")
            return None

        self._log_con = open(os.path.join("..", "logs", f"log_{self._make_file_suffix()}.txt"), 'a')
        self._log_con.write(f"\n{datetime.datetime.now()}\n")
        self._print_and_log(f"New query '{query}'")
        mid = list(self.anchor_bank.index).index(self.ref_anchor) if first_comparison == None else list(self.anchor_bank.index).index(first_comparison)
        anchors = tuple(self.anchor_bank.index)

        lo = 0
        hi = len(self.anchor_bank)
        n_iter = 0

        while hi >= lo:
            
            anchor = anchors[mid]
            if verbose:
                self._print_and_log(f"\tQuery: '{query}'\tAnchor:'{anchor}'")
            try:
                ts = self._query_google(keywords = [anchor, query]).iloc[:, 0:2]
            except Exception as e:
                self._print_and_log(f"Google query '{query}' failed because: {str(e)}")
                break

            max_anchor = ts.loc[:, anchor].max()
            max_query = ts.loc[:, query].max()

            if max_query >= thresh and max_anchor >= thresh:

                max_query_lo, max_query_hi, max_anchor_lo, max_anchor_hi = self._compute_hi_and_lo(max_query, max_anchor)
                ts_query_lo, ts_query_hi = map(list, zip(*[self._compute_hi_and_lo(100, el)[2:] for el in ts.loc[:, query]]))

                if np.sum(ts.loc[:, query] == 100) == 1: 
                    ts_query_lo[list(ts.loc[:, query]).index(100)] = 100
                 
                ratio_anchor = self.anchor_bank[anchor]
                ratio_anchor_lo = self.anchor_bank_lo[anchor]
                ratio_anchor_hi = self.anchor_bank_hi[anchor]

                ratio = ratio_anchor * (max_query / max_anchor)
                ratio_hi = ratio_anchor_hi * (max_query_hi / max_anchor_lo)
                ratio_lo = ratio_anchor_lo * (max_query_lo / max_anchor_hi)

                ts_query = ts.loc[:, query] / max_query * ratio
                ts_query_hi = np.array(ts_query_hi) / max_query_lo * ratio_hi
                ts_query_lo = np.array(ts_query_lo) / max_query_hi * ratio_lo

                self._print_and_log("New query calibrated!")                
                return {query:{"ratio": ratio, "ratio_hi": ratio_hi, "ratio_lo": ratio_lo, "ts": list(ts_query), "ts_hi": list(ts_query_hi), "ts_lo": list(ts_query_lo), "iter": n_iter}}        
                
            elif max_query < thresh:
                lo = mid + 1 
            else: 
                hi = mid - 1

            mid = (hi + lo) // 2
            n_iter += 1

        if (hi <= 0):
            self._print_and_log('Could not calibrate. Time series for query too low everywhere.')
        else:
            self._print_and_log('Could not calibrate. Time series for query too high everywhere.')
        self._log_con.close()

        return None
