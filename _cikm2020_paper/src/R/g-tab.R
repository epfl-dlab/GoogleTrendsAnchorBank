library(gtrendsR)
library(igraph)
library(Matrix)

options(stringsAsFactors=FALSE)

###############################################################################
# Constants
###############################################################################

# Root directory of the GitHub repo.
BASE_DIR <- sprintf('%s/github/GoogleTrendsAnchorBank', Sys.getenv('HOME'))

DATA_DIR <- sprintf('%s/_cikm2020_paper/data', BASE_DIR)

# num_anchors should be even; num_anchor_candidates should be a multiple of num_anchors/2.
DEFAULT_CONFIG <- list(num_anchors=100,
                       num_anchor_candidates=2000,
                       thresh_offline=10,
                       timespan='2019-01-01 2019-12-31',
                       geo='',
                       seed=1,
                       sleep=0.5)

# Food queries from which anchor queries will be selected.
FOODS <- read.table(sprintf('%s/freebase_foods.tsv', DATA_DIR), sep='\t', quote='',
                    comment.char='', header=TRUE, stringsAsFactors=FALSE)
FOODS <- FOODS[FOODS$is_dish | FOODS$is_food,]
rownames(FOODS) <- FOODS$mid

# A set of high-traffic navigational queries, which will populate the upper end of the anchor bank,
# as food queries, which are used for the rest of the anchor bank, are not sufficiently frequently
# searched for on Google in order to cover the whole range of search frequencies.
HI_TRAFFIC <- c('/m/03vgrr', '/m/0289n8t', '/m/02y1vz', '/m/019rl6', '/m/0b2334', '/m/010qmszp', '/m/01yvs')
names(HI_TRAFFIC) <- c('LinkedIn', 'Twitter', 'Facebook', 'Yahoo!', 'Reddit', 'Airbnb', 'Coca-Cola')

# Manually constructed blacklist of queries that should not be used as anchor queries as they would
# break the chain that transitively connects the most to the least frequently searched-for anchor queries).
# In the up-to-date Python library, we have implemented a method for detecting and removing such
# "rogue queries" on the fly, but we did not add the feature to this outdated, proof-of-concept R code.
BLACKLIST <- c('/m/0bcgdv') # Irvingia gabonensis

###############################################################################
# Functions
###############################################################################

build_anchor_bank <- function(config) {
  # Load ring graph.
  # Note: Arranging queries in a ring is not strictly necessary; it would actually suffice to arrange
  # the queries in a simple linear array (as also visualized in Fig. 2 of the CIKM'20 paper).
  # In the Python library, the simpler setup is used, whereas in this outdated, proof-of-concept
  # R code, we didn't make the change.
  G <- load_anchor_ring_graph(config)
  
  # Load Google results.
  google_results <- load_google_results(G, config)
  time_series <- do.call(cbind, google_results)
  
  # Compute max ratios.
  ratios <- compute_max_ratios(google_results, plot=FALSE)
  
  # Estimate max ratios for all query pairs.
  Ws0 <- infer_all_ratios(ratios)
  W0 <- Ws0$W
  W0_hi <- Ws0$W_hi
  W0_lo <- Ws0$W_lo
  W0_paths <- Ws0$paths

  # Sanity check: Is W0 multiplicatively symmetric?
  err <- max(abs(1-W0*t(W0)), na.rm=TRUE)
  if (err > 1e-12) warning("W0 doesn't seem to be multiplicatively symmetric: W0[i,j] != 1/W0[j,i].")

  # Find an ordered subset of queries (from highest to lowest query volume, according to W0),
  # such that the ratio of neighboring queries is approximately 1/e (cf. Appendix A of CIKM'20 paper).
  opt_query_set <- find_optimal_query_set(W0)

  # Query Google Trends to get max ratios for neighboring queries in the optimal subset.
  Ws <- build_optimal_anchor_bank(opt_query_set$mids)
  W <- Ws$W
  W_hi <- Ws$W_hi
  W_lo <- Ws$W_lo

  # The top most frequent query.
  top_anchor <- opt_query_set$mids[1]

  # The reference query.
  ref_anchor <- top_anchor
  
  # The anchor from the optimal set that is closest to the median of the larger set.
  median_anchor <- names(which.min(abs(W[ref_anchor,] - median(W0[ref_anchor,]))))

  # The final anchor bank consists of the optimal query set calibrated against the reference anchor.
  # We also provide the upper and lower bounds for each max ratio.
  anchor_bank <- W[ref_anchor,]
  anchor_bank_hi <- W_hi[ref_anchor,]
  anchor_bank_lo <- W_lo[ref_anchor,]
  
  list(anchor_bank=anchor_bank, anchor_bank_hi=anchor_bank_hi, anchor_bank_lo=anchor_bank_lo,
       ref_anchor=ref_anchor, median_anchor=median_anchor,
       W=W, W_hi=W_hi, W_lo=W_lo,
       W0=W0, W0_hi=W0_hi, W0_lo=W0_lo, W0_paths=W0_paths,
       G=G, time_series=time_series, ratios=ratios)
}

mid2name <- function(mid) {
  lookup <- c(FOODS$name, names(HI_TRAFFIC))
  names(lookup) <- c(FOODS$mid, HI_TRAFFIC)
  lookup[mid]
}

make_RData_file_suffix <- function(config) {
  conf <- config
  # config$sleep doesn't matter.
  conf$sleep <- NULL
  paste(sapply(1:length(conf), function(i) paste(names(conf)[i], conf[[i]], sep='=')), collapse='.')
}

google <- function(keywords, config) {
  Sys.sleep(config$sleep)
  gtrends(keywords, geo=config$geo, time=config$timespan, onlyInterest=TRUE)
}

keyword_ok <- function(keyword, config) {
  tryCatch({
    google(keyword, config)
    return(!(keyword %in% BLACKLIST))
  }, error = function(e) {
    return(FALSE)
  })
}

query_google <- function(keywords, config) {
  trends <- google(keywords, config)
  interest <- trends$interest_over_time[c('date', 'hits', 'keyword')]
  ts <- reshape(interest, timevar='date', idvar='keyword', direction='wide')
  rownames(ts) <- ts$keyword
  ts <- ts[,-1]
  ts[ts=='<1'] <- '0'
  ts <- t(ts)
  ts <- apply(ts, 2, as.numeric)
  list(full=trends, ts=ts)
}

time_series_ok <- function(ts, config) {
  max(ts) >= config$thresh_offline
}

load_anchor_ring_graph <- function(config) {
  file <- sprintf('%s/calibration/G.%s.RData', DATA_DIR, make_RData_file_suffix(config))
  if (file.exists(file)) {
    load(file)
  } else {
    N <- config$num_anchor_candidates
    K <- config$num_anchors/2
    # Stratified sampling: 2 per stratum, so we can build a ring without "discontinuities".
    mat <- matrix(FOODS$mid[1:N], nrow=N/K, ncol=K)
    set.seed(config$seed)
    samples2 <- apply(mat, 2, function(r) sample(r, 2))
    keywords <- c(rev(samples2[1,]), HI_TRAFFIC, samples2[2,])
    V <- NULL
    i <- 1
    for (k in keywords) {
      if (keyword_ok(k, config)) {
        print(sprintf('%d: Checking %s: ok', i, k))
        V <- c(V,k)
      } else {
        print(sprintf('%d: Checking %s: error', i, k))
      }
      i <- i + 1
    }
    n <- length(V)
    G <- make_ring(n)
    vertex_attr(G) <- list(name=V)
    for (i in 0:(n-1)) {
      left <- ((i-2) %% n) + 1
      right <- ((i+2) %% n) + 1
      if (i+1 < left) G <- G + edge(i+1, left)
      if (i+1 < right) G <- G + edge(i+1, right)
    }
    save(G, file=file)
  }
  return(G)
}

load_google_results <- function(G, config) {
  file <- sprintf('%s/calibration/google_results.%s.RData', DATA_DIR, make_RData_file_suffix(config))
  if (file.exists(file)) {
    load(file)
  } else {
    google_results <- list()
    i <- 0
    for (v in V(G)$name) {
      i <- i + 1
      keywords <- c(v, neighbors(G, v)$name)
      print(sprintf('%d: %s (%s), keywords: %s (%s)', i, v, mid2name(v),
                    paste(keywords, collapse=', '),
                    paste(mid2name(keywords), collapse=', ')))
      trends <- query_google(keywords, config)
      google_results[[v]] <- trends$ts
    }
    save(google_results, file=file)
  }
  return(google_results)
}

compute_hi_and_lo <- function(max1, max2) {
  if (max1 < 100) {
    hi1 <- max1 + 0.5
    lo1 <- max1 - 0.5
  } else {
    hi1 <- lo1 <- 100
  }
  if (max2 < 100) {
    hi2 <- max2 + 0.5
    lo2 <- max2 - 0.5
  } else {
    hi2 <- lo2 <- 100
  }
  # The case where both queries are tied for the max needs to be handled separately,
  # since it is unclear which one is the larger one before rounding.
  if (max1 == 100 && max2 == 100) {
    lo1 <- lo2 <- 99.5
  }
  result <- c(lo1, hi1, lo2, hi2)
  names(result) <- c('lo1', 'hi1', 'lo2', 'hi2')
  return(result)
}

compute_max_ratios <- function(google_results, plot=FALSE) {
  anchors <- NULL
  v1 <- NULL
  v2 <- NULL
  ratios <- NULL
  ratios_hi <- NULL
  ratios_lo <- NULL
  errors <- NULL
  log_errors <- NULL
  
  for (v in names(google_results)) {
    ts <- google_results[[v]]
    if (plot) {
      matplot(ts, type='l', lty=1, main=v)
      legend('topright', colnames(ts), lty=1, col=1:ncol(ts))
    }
    for (j in 1:ncol(ts)) {
      for (k in 1:ncol(ts)) {
        if (j != k) {
          if (time_series_ok(ts[,j], config) && time_series_ok(ts[,k], config)) {
            anchors <- c(anchors, v)
            v1 <- c(v1, colnames(ts)[j])
            v2 <- c(v2, colnames(ts)[k])
            max1 <- max(ts[,j])
            max2 <- max(ts[,k])
            hilo <- compute_hi_and_lo(max1, max2)
            hi1 <- hilo['hi1']
            lo1 <- hilo['lo1']
            hi2 <- hilo['hi2']
            lo2 <- hilo['lo2']
            ratios <- c(ratios, max2/max1)
            ratios_hi <- c(ratios_hi, hi2/lo1)
            ratios_lo <- c(ratios_lo, lo2/hi1)
            errors <- c(errors, hi2/lo2 * hi1/lo1)
            log_errors <- c(log_errors, log(hi2/lo2 * hi1/lo1))
          }
        }
      }
    }
  }
  
  return(data.frame(v1=v1, v2=v2, anchor=anchors,
                    ratio=ratios, ratio_hi=ratios_hi, ratio_lo=ratios_lo,
                    error=errors, weight=log_errors))
}

infer_all_ratios <- function(ratios) {
  graph <- graph_from_data_frame(ratios)

  # Sanity check: Is graph connected?
  if (!is.connected(graph)) warning('Graph is not connected!')

  # Compute shortest paths between all pairs of nodes, where log errors serve as edge weights.
  paths <- lapply(as.list(V(graph)$name), function(v) shortest_paths(graph, from=v, mode='out', output='epath')$epath)
  names(paths) <- V(graph)$name

  # Helper function for computing the product of edge attributes along paths and storing the results in a matrix.
  multiply <- function(attr) {
    mat <- t(sapply(paths, function(paths_for_source)
      sapply(paths_for_source, function(es) prod(edge_attr(graph, attr, es)))))
    colnames(mat) <- V(graph)$name
    return(mat)
  }
  W <- multiply('ratio')
  W_hi <- multiply('ratio_hi')
  W_lo <- multiply('ratio_lo')

  return(list(W=W, W_hi=W_hi, W_lo=W_lo, paths=paths))
}

find_optimal_query_set <- function(W0) {
  # Helper function for getting the most and least frequent queries from the matrix W0.
  get_extreme <- function(type=c('top', 'bottom')) {
    sign <- if (type == 'top') 1 else if (type == 'bottom') -1 else stop("type must be 'top' or 'bottom'")
    ext <- names(which(apply(sign*W0 <= sign, 1, all)))
    if (length(ext) > 1) {
      warning(sprintf('There are multiple %s keywords! Choosing the one from the largest component.', type))
      ext <- ext[which.max(colSums(W0[,ext] > 0))]
    }
    return(ext)
  }
  top <- get_extreme('top')
  bot <- get_extreme('bottom')

  # We want the chosen edges to have weights close to 1/e, or log weights close to -1.
  D <- graph_from_adjacency_matrix(abs(log(W0)+1), weighted=TRUE)
  sp <- shortest_paths(D, from=top, to=bot, mode='out', output='both')
  vpath <- sp$vpath[[1]]
  epath <- sp$epath[[1]]
  ratios <- apply(ends(D, epath), 1, function(vs) W0[vs[2],vs[1]])
  return(list(mids=vpath$name, ratios=ratios))
}

build_optimal_anchor_bank <- function(mids) {
  N <- length(mids)
  file <- sprintf('%s/calibration/pairwise_ts.%s.RData', DATA_DIR, make_RData_file_suffix(config))
  if (file.exists(file)) {
    load(file)
  } else {
    pairwise_ts <- list()
    for (i in 2:N) {
      keywords <- mids[(i-1):i]
      mid_pair <- paste(keywords, collapse=' ')
      name_pair <- paste(mid2name(keywords), collapse=', ')
      print(sprintf('%d: keywords: %s (%s)', i, mid_pair, name_pair))
      trends <- query_google(keywords, config)
      pairwise_ts[[mid_pair]] <- trends$ts
    }
    save(pairwise_ts, file=file)
  }

  # Add ratio 1 for reference query itself.
  pairwise_max_ratios <- c(1, sapply(pairwise_ts, function(ts) max(ts[,2])/100))
  pairwise_max_ratios_hi <- c(1, sapply(pairwise_ts, function(ts) (max(ts[,2])+0.5)/100))
  pairwise_max_ratios_lo <- c(1, sapply(pairwise_ts, function(ts) (max(ts[,2])-0.5)/100))
  names(pairwise_max_ratios) <- names(pairwise_max_ratios_hi) <- names(pairwise_max_ratios_lo) <- mids
  
  # Matrix entry (i,j) has prod(...[1:j])/prod(...[1:i]), ...
  W <- W_hi <- W_lo <- matrix(nrow=N, ncol=N, dimnames=list(mids, mids))
  for (i in 1:N)    W[i,] <- cumprod(pairwise_max_ratios[1:N])/prod(pairwise_max_ratios[1:i])
  for (i in 1:N) W_hi[i,] <- cumprod(pairwise_max_ratios_hi[1:N])/prod(pairwise_max_ratios_hi[1:i])
  for (i in 1:N) W_lo[i,] <- cumprod(pairwise_max_ratios_lo[1:N])/prod(pairwise_max_ratios_lo[1:i])
  
  # ... so the lower triangles need to be swapped (since for the true hi and lo, we have hi[i,j]=1/lo[j,i]).
  W_hi <- as.matrix(triu(W_hi) + tril(1/t(W_lo), k=-1))
  W_lo <- as.matrix(triu(W_lo) + tril(1/t(W_hi), k=-1))
  
  return(list(W=W, W_hi=W_hi, W_lo=W_lo))
}

# Online binary search.
binsearch <- function(query, anchor_bank, anchor_bank_hi, anchor_bank_lo, config, first_comparison=NULL,
                      # 100/e is the optimal threshold as derived in Appendix A of the CIKM'20 paper.
                      thresh=100/exp(1), plot=FALSE, quiet=TRUE, silent=FALSE) {
  left <- 1
  right <- length(anchor_bank)
  anchors <- names(anchor_bank)
  iter <- 0
  tryCatch({
    while (right > left) {
      iter <- iter + 1
      if (iter == 1) {
        pivot <- if (is.null(first_comparison)) ceiling(length(anchor_bank)/2) else which(names(anchor_bank)==first_comparison)
      } else {
        pivot <- left + floor((right-left)/2)
      }
      anchor <- anchors[pivot]
      if (!quiet) message(sprintf('   Comparing to %s (%s)', anchor, mid2name(anchor)))
      ts <- query_google(c(query, anchor), config)$ts
      max_query <- max(ts[,query])
      max_anchor <- max(ts[,anchor])
      if (plot) {
        matplot(ts, type='l', lty=1, ylim=c(0,100))
        legend('topright', c(query, mid2name(anchor)), lty=1, col=1:2, bty='n')
      }
      if (max_query >= thresh && max_anchor >= thresh) {
        max_hilo <- compute_hi_and_lo(max_query, max_anchor)
        # as.numeric in order to remove item names.
        max_query_hi <- as.numeric(max_hilo['hi1'])
        max_query_lo <- as.numeric(max_hilo['lo1'])
        max_anchor_hi <- as.numeric(max_hilo['hi2'])
        max_anchor_lo <- as.numeric(max_hilo['lo2'])
        # Compute hi and lo for the query time series.
        ts_hilo <- sapply(ts[,query], function(x) compute_hi_and_lo(100, x))
        ts_query_hi <- ts_hilo['hi2',]
        ts_query_lo <- ts_hilo['lo2',]
        # The above call to compute_hi_and_lo will change 100 to 99.5. Undo this if there is exactly
        # one value of 100, since then there is no uncertainty that that value is the true maximum.
        if (sum(ts[,query]==100) == 1) ts_query_lo[ts[,query]==100] <- 100

        ratio_anchor <- as.numeric(anchor_bank[anchor])
        ratio_anchor_hi <- as.numeric(anchor_bank_hi[anchor])
        ratio_anchor_lo <- as.numeric(anchor_bank_lo[anchor])
        ratio <- ratio_anchor * (max_query / max_anchor)
        ratio_hi <- ratio_anchor_hi * (max_query_hi / max_anchor_lo)
        ratio_lo <- ratio_anchor_lo * (max_query_lo / max_anchor_hi)
        ts_query <- ts[,query] / max_query * ratio
        ts_query_hi <- ts_query_hi / max_query_lo * ratio_hi
        ts_query_lo <- ts_query_lo / max_query_hi * ratio_lo
        if (!silent) message(sprintf('Done with %s after %d steps', query, iter))
        return(list(ratio=ratio, ratio_hi=ratio_hi, ratio_lo=ratio_lo,
                    ts=ts_query, ts_hi=ts_query_hi, ts_lo=ts_query_lo,
                    iter=iter))
      } else if (max_query < thresh) {
        if (!quiet) message('   Going lower')
        left <- pivot + 1
      } else {
        if (!quiet) message('   Going higher')
        right <- pivot - 1
      }
    }
    if (right <= 1) {
      message('Could not calibrate. Time series for query too low everywhere.')
    } else {
      message('Could not calibrate. Time series for query too high everywhere.')
    }
    return(NULL)
  }, error = function(e) {
    message(sprintf('Google gives error for %s', query))
    message(e)
    return(NULL)
  })
}
