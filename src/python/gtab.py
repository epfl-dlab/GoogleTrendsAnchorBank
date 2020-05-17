# TODO:
#   add plots

import copy
import ast
import codecs
import os
import pickle
import random
import time

import networkx as nx
import numpy as np
import pandas as pd
from pytrends.request import TrendReq
from tqdm import tqdm

log_con = open("log.txt", "w")

class GTAB:

    def __init__(self, ptrends_config = None, gtab_config = None, conn_config = None, blacklist = None, use_proxies = False):

        if ptrends_config == None:
            with open(os.path.join("config", "ptrends.config"), "r") as f_conf:
                self.PTRENDS_CONFIG = ast.literal_eval(f_conf.readline())
        else: 
            self.PTRENDS_CONFIG = ptrends_config

        if gtab_config == None:
            with open(os.path.join("config", "gtab.config"), "r") as f_conf:
                self.GTAB_CONFIG = ast.literal_eval(f_conf.readline())
        else: 
            self.GTAB_CONFIG = gtab_config

        if conn_config == None:
            with open(os.path.join("config", "conn.config"), "r") as f_conf:
                self.CONN_CONFIG = ast.literal_eval(f_conf.readline())
        else:
            self.CONN_CONFIG = conn_config
            
        if blacklist == None:
            with open(os.path.join("config", "blacklist.config"), "r") as f_conf:
                self.BLACKLIST = ast.literal_eval(f_conf.readline()) 
        else:
            self.BLACKLIST = blacklist

        with open(os.path.join("data", "anchor_candidates.tsv"), 'r', encoding = 'utf-8') as f_anchor_set:
            self.ANCHOR_CANDIDATE_SET = pd.read_csv(f_anchor_set, sep = '\t')

        self.ANCHOR_CANDIDATE_SET.rename(index = {k: v for k, v in enumerate(list(self.ANCHOR_CANDIDATE_SET['mid']))}, inplace = True)
        mask = (np.array(self.ANCHOR_CANDIDATE_SET['is_dish'] == 1) | np.array(self.ANCHOR_CANDIDATE_SET['is_food'] == 1))
        self.ANCHOR_CANDIDATE_SET = self.ANCHOR_CANDIDATE_SET[mask]
        self.HITRAFFIC = {'LinkedIn': '/m/03vgrr', 'Twitter': '/m/0289n8t', 'Facebook': '/m/02y1vz', 'Yahoo!': '/m/019rl6', 'Reddit': '/m/0b2334', 'Airbnb': '/m/010qmszp' , 'Coca-Cola': '/m/01yvs'}

        if use_proxies:
            self.ptrends = TrendReq(hl='en-US', **self.CONN_CONFIG)
        else:
            self.ptrends = TrendReq(hl='en-US', retries = 2)

    ## --- UTILITY METHODS --- #
    def _query_google(self, keywords = ["Keywords"]):
        time.sleep(self.GTAB_CONFIG['sleep'])
        if type(keywords) == str:
            keywords = [keywords]
    
        if len(keywords) > 5:
            raise ValueError("Query keywords must be less than 5.")

        self.ptrends.build_payload(kw_list = keywords, **self.PTRENDS_CONFIG)
        return(self.ptrends.interest_over_time())

    def _check_keyword(self, keyword):
        try:
            self._query_google(keywords = keyword)
            return not keyword in self.BLACKLIST
        except:
            log_con.write(f"Bad keyword '{keyword}'\n")
            return False
    
    def _check_ts(self, ts):
        return ts.max().max() >= self.GTAB_CONFIG['thresh_offline']


    ## --- ANCHOR BANK METHODS --- ##
    def _get_google_results(self):
        fpath = os.path.join("data", "google_results_test.pkl")
        if os.path.exists(fpath):
            print(f"Loading google results from {fpath}...")
            with open(fpath, 'rb') as f_in:
                ret = pickle.load(f_in)
            return ret
        else:
            print("Sampling keywords...")
            N = self.GTAB_CONFIG['num_anchor_candidates']
            K = self.GTAB_CONFIG['num_anchors']

            ## --- Get stratified samples, 1 per stratum. --- ##
            random.seed(self.GTAB_CONFIG['seed'])
            samples = []
            for j in range(K):
                start_idx = (j * N) // K
                end_idx = (((j+1) * N) // K) - 1
                s1 = random.randint(start_idx, end_idx)
                samples.append(self.ANCHOR_CANDIDATE_SET.index[s1])
                    
            ## --- Query google on keywords and dump the file. --- ##
            keywords = samples[0:len(samples)//2] + list(self.HITRAFFIC.values()) + samples[len(samples)//2:]
            assert len(keywords) == (len(samples) + len(self.HITRAFFIC.values()))
            keywords = [k for k in tqdm(keywords, total = len(keywords)) if self._check_keyword(k)]
            
            print("Querying google...")
            ret = dict()
            for i in tqdm(range(0, len(keywords) - 5 + 1)):
                df_query = self._query_google(keywords = keywords[i:i+5]).iloc[:, 0:5]
                ret[i] = df_query
        
            # object id sanity check
            assert id(ret[0]) != id(ret[1])

            with open(fpath, 'wb') as f_out:
                pickle.dump(ret, f_out, protocol=4)

            return(ret)

    # TODO: Fix bug
    def _compute_max_ratios(self, google_results):
        anchors = []
        v1 = []
        v2 = []
        ratios_vec = []

        for _, val in google_results.items():
            for j in range(val.shape[1]-1):
                for k in range(j + 1, val.shape[1]):
                    if(self._check_ts(val.iloc[:, j]) and self._check_ts(val.iloc[:, k])):
                        anchors.append(val.columns[0]) # first element of group
                        v1.append(val.columns[j])
                        v2.append(val.columns[k])
                        ratios_vec.append(val.iloc[:, j].max().max() / val.iloc[:, k].max().max())
        

        # If ratio > 1, flip positions and change ratio value to 1/ratio.
        for j in range(len(ratios_vec)):
            if ratios_vec[j] > 1: # if ratio >= 1?
                ratios_vec[j] = 1./ratios_vec[j]
                v1[j], v2[j] = v2[j], v1[j]

        # TODO: ratio == 1

        assert len(ratios_vec) == len(v1) == len(v2) == len(anchors)
        df_ratios = pd.DataFrame({"anchor": anchors, "v1": v1, "v2": v2, "ratio": ratios_vec})
        df_ratios_aggr = df_ratios.groupby(['v1', 'v2']).agg({"ratio": ['mean', 'std']})
        df_ratios_aggr.columns = df_ratios_aggr.columns.droplevel(0)
        df_ratios_aggr.reset_index(inplace = True)
        df_ratios_aggr.set_index(df_ratios_aggr.apply(lambda x: " ".join([x[0], x[1]]), axis = 1), inplace=True)

        return df_ratios_aggr

    def _infer_all_ratios(self, ratios_aggr):
        
        vcore = list(set().union(ratios_aggr['v1'], ratios_aggr['v2']))
        vcore.sort()
        W0 = pd.DataFrame(np.zeros((len(vcore), len(vcore))), index = vcore, columns = vcore, dtype=float)
        A0 = pd.DataFrame(np.zeros((len(vcore), len(vcore))), index = vcore, columns = vcore, dtype=float)
        
        for u in vcore:
            for v in vcore:
                ratio = np.nan
                if u == v:
                    ratio = 1.0
                else:
                    key = " ".join([u, v])
                    if key in ratios_aggr.index:
                        ratio = ratios_aggr.loc[key, 'mean']
                if np.isfinite(ratio):
                    W0.loc[u, v] = ratio
                    W0.loc[v, u] = 1./ratio
                    A0.loc[u, v] = 1
                    A0.loc[v, u] = 1

        W = copy.deepcopy(W0)
        A = copy.deepcopy(A0)

        while True:
            AA = A.dot(A)
            WW = A * W + (1-A) * ((W.dot(W)) / AA)
            WW.fillna(0.0, inplace = True)
            W = copy.deepcopy(WW)
            A_old = copy.deepcopy(A)

            # object id sanity check
            assert id(W) != id(WW)
            assert (W == WW).all().all()

            A = (AA > 0).astype(int)

            if (A == A_old).all().all():
                break

        return W
        
    ## --- "EXPOSED" METHODS --- ##
    def init(self):

        google_results = self._get_google_results() 
        time_series = pd.concat([google_results[0], google_results[1]], axis=1)
        ratios = self._compute_max_ratios(google_results)   
        G = nx.convert_matrix.from_pandas_edgelist(ratios, 'v1', 'v2', create_using = nx.Graph)

        ## checks don't work?
        # print(nx.number_connected_components(nx.Graph(G)))
        # assert nx.is_directed_acyclic_graph(G)  
        # assert nx.is_weakly_connected(G)

                # ratios = pd.read_csv("test3.tsv", sep = '\t')
                
                # ratios.rename(mapper = {"mean_ratio": "mean"}, axis = 1, inplace = True)
    
        W = self._infer_all_ratios(ratios)
        err = np.nanmax(np.abs(1 - W * W.transpose()))

        # colSums equivalent is df.sum(axis = 0)
        ref_anchor = W[(W > 1).sum(axis = 0) == 0].index[0]
        
        if len(ref_anchor) > 1 and type(ref_anchor) is not str: 
            ref_anchor = (W.loc[:, ref_anchor] > 0).sum(axis = 0).idxmax()[0]

        self.ref_anchor = ref_anchor
        self.err = err
        self.W = W
        self.calib_max_vol = W.loc[:, ref_anchor]
        self.ratios = ratios
        self.time_series = time_series

    def new_query(self):
        #TODO
        return None
        

if __name__ == '__main__':

    t = GTAB()
    t.init()
    log_con.close()
    

    # test = pd.DataFrame()

    # tokens = ['A', 'B', 'C', 'D', 'E']
    # v1 = []
    # v2 = []
    # vals = []
    # idxs = []
    # for i in range(len(tokens) -1):
    #     for j in range(i + 1, len(tokens)):
    #         v1.append(tokens[i])
    #         v2.append(tokens[j])
    #         vals.append(0.5)
    #         idxs.append(" ".join([tokens[i], tokens[j]]))

    # test = pd.DataFrame({"v1": v1, "v2": v2, "mean":vals}, index = idxs)
    # test.to_csv("test_data.tsv", sep = '\t')
    # t._infer_all_ratios(test)
