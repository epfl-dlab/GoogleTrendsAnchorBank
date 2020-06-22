library(gtrendsR)
library(igraph)

options(stringsAsFactors=FALSE)

###############################################################################
# Constants
###############################################################################
#DATA_DIR <- "../../data/"
DATA_DIR <- sprintf('%s/github/GoogleTrendsAnchorBank/data', Sys.getenv('HOME'))

# num_anchors should be even; num_anchor_candidates should be a multiple of num_anchors/2.
DEFAULT_CONFIG <- list(num_anchors=100,
                       num_anchor_candidates=2000,
                       thresh_offline=10,
                       timespan='2019-01-01 2019-12-31',
                       geo='',
                       seed=1,
                       sleep=0.5,
                       randomize_ring_order=FALSE)

HI_TRAFFIC <- c('/m/03vgrr', '/m/0289n8t', '/m/02y1vz', '/m/019rl6', '/m/0b2334', '/m/010qmszp', '/m/01yvs')
names(HI_TRAFFIC) <- c('LinkedIn', 'Twitter', 'Facebook', 'Yahoo!', 'Reddit', 'Airbnb', 'Coca-Cola')

# LO_TRAFFIC <- c('/m/09xr_3', '/m/0cp4q_')
# names(LO_TRAFFIC) <- c('FC Wacker Innsbruck', 'Northwood F.C.')

FOODS <- read.table(sprintf('%s/freebase_foods.tsv', DATA_DIR), sep='\t', quote='',
                    comment.char='', header=TRUE, stringsAsFactors=FALSE)
FOODS <- FOODS[FOODS$is_dish | FOODS$is_food,]
rownames(FOODS) <- FOODS$mid

BLACKLIST <- c('/m/0bcgdv') # Irvingia gabonensis

###############################################################################
# Functions
###############################################################################

build_anchor_bank <- function(config) {
  # Load ring graph.
  G <- load_anchor_ring_graph(config)
  
  # Load Google results.
  google_results <- load_google_results(G, config)
  time_series <- do.call(cbind, google_results)
  
  # Compute max ratios.
  ratios <- compute_max_ratios(google_results, plot=FALSE)
  
  # Make DAG induced by ratios.
  # D <- graph_from_data_frame(ratios)
  
  # Sanity check: Is D indeed acyclic as it should be?
  # if (!is.dag(D)) warning('Directed graph is not acyclic!')
  
  # Sanity check: Is D connected?
  # if (!is.connected(D)) warning('Directed graph is not connected!')
  
  # Propagate ratios through the graph via matrix multiplication.
  W <- infer_all_ratios(ratios)
  err <- max(abs(1-W*t(W)), na.rm=TRUE)

  opt_query_set <- find_optimal_query_set(W)

  # The top keyword is the one with which all other keywords have a ratio < 1.
  # top_anchor <- opt_query_set$mids[1]
  
  # Calibrate all other anchors against the top anchor.
  Ws <- build_optimal_anchor_bank(opt_query_set$mids)
  W <- Ws$W
  W_hi <- Ws$W_hi
  W_lo <- Ws$W_lo

  ordered_mids <- colnames(W)[order(W[1,])]
  median_mid <- ordered_mids[floor(length(ordered_mids)/2)]

  ...........................
  
  list(G=G, time_series=time_series, ratios=ratios, D=D, W=W, err=err, ref_anchor=ref_anchor,
       calib_max_vol=calib_max_vol)
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
    # keywords <- c(rev(samples2[1,]), HI_TRAFFIC, samples2[2,], LO_TRAFFIC)
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
    # Randomized baseline to see effect of popularity ordering in ring.
    if (config$randomize_ring_order) V <- V[sample(n,n)]
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
  ratios_vec <- NULL
  ratios_lo_vec <- NULL
  ratios_hi_vec <- NULL
  errors_vec <- NULL
  log_errors_vec <- NULL
  
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
            hi_and_lo <- compute_hi_and_lo(max1, max2)
            hi1 <- hi_and_lo['hi1']
            lo1 <- hi_and_lo['lo1']
            hi2 <- hi_and_lo['hi2']
            lo2 <- hi_and_lo['lo2']
            ratios_vec <- c(ratios_vec, max1/max2)
            ratios_lo_vec <- c(ratios_lo_vec, lo1/hi2)
            ratios_hi_vec <- c(ratios_hi_vec, hi1/lo2)
            errors_vec <- c(errors_vec, hi1/lo1 * hi2/lo2)
            log_errors_vec <- c(log_errors_vec, log(hi1/lo1*hi2/lo2))
          }
        }
      }
    }
  }
  
  ratios <- data.frame(v1=v1, v2=v2, anchor=anchors,
                       ratio=ratios_vec, ratio_lo=ratios_lo_vec, ratio_hi=ratios_hi_vec,
                       error=errors_vec, weight=log_errors_vec)
  
  return(ratios)
}

infer_all_ratios <- function(ratios) {
  graph <- graph_from_data_frame(ratios)
  if (!is.connected(graph)) warning('Graph is not connected!')
  epaths <- lapply(as.list(V(graph)$name), function(v) shortest_paths(graph, from=v, mode='out', output='epath')$epath)
  names(epaths) <- V(graph)$name

  # d <- distances(graph, mode='out')
  # idx <- sort(rownames(d))
  # d <- d[idx,idx]
  
  aggr <- function(graph, epaths, attr) {
    mat <- t(sapply(epaths, function(epaths_for_source)
      sapply(epaths_for_source, function(es) prod(edge_attr(graph, attr, es)))))
    colnames(mat) <- V(graph)$name
    return(mat)
  }
  
  W     <- aggr(graph, epaths, 'ratio')
  W_hi  <- aggr(graph, epaths, 'ratio_hi')
  W_lo  <- aggr(graph, epaths, 'ratio_lo')
  W_err <- aggr(graph, epaths, 'error')
  
  # Sanity check: W_hi/W_lo == W_err
  # max(W_hi/W_lo - W_err)
  
  return(W)
}

find_optimal_query_set <- function(W) {
  get_extreme <- function(sign) {
    ext <- names(which(apply(sign*W <= sign, 2, all)))
    if (length(ext) > 1) {
      warning(sprintf('There are multiple extreme keywords of sign %d! Choosing the one from the largest component.', sign))
      ext <- ext[which.max(colSums(W[,ext] > 0))]
    }
    return(ext)
  }
  top <- get_extreme(1)
  bot <- get_extreme(-1)

  D <- graph_from_adjacency_matrix(abs(log(W)-1), weighted=TRUE)
  sp <- shortest_paths(D, from=top, to=bot, mode='out', output='both')
  vpath <- sp$vpath[[1]]
  epath <- sp$epath[[1]]
  ratios <- apply(ends(D, epath), 1, function(vs) W[vs[2],vs[1]])
  return(list(mids=vpath$name, ratios=ratios))
}

#########################################
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
  
  W <- W_hi <- W_lo <- matrix(nrow=N, ncol=N, dimnames=list(mids, mids))
  for (i in 1:N)    W[i,] <- cumprod(pairwise_max_ratios[1:N])/prod(pairwise_max_ratios[1:i])
  for (i in 1:N) W_hi[i,] <- cumprod(pairwise_max_ratios_hi[1:N])/prod(pairwise_max_ratios_hi[1:i])
  for (i in 1:N) W_lo[i,] <- cumprod(pairwise_max_ratios_lo[1:N])/prod(pairwise_max_ratios_lo[1:i])
  
  return(list(W=W, W_hi=W_hi, W_lo=W_lo))
}

# Visualize step sizes.
plot(log(sort(W[fb,])), panel.first=grid())
es <- epaths[[fb]][[which(names(epaths)==names(which.max(W[fb,])))]]
abline(h=cumsum(c(0, log(es$ratio))), col='green')
abline(h=log(1/calib_max_vol), col='red')
#########################################


# infer_all_ratios__OLD <- function(ratios_aggr) {
#   Vcore <- unique(c(ratios_aggr$v1, ratios_aggr$v2))
#   W0 <- matrix(data=0, nrow=length(Vcore), ncol=length(Vcore), dimnames=list(Vcore,Vcore))
#   A0 <- W0
#   
#   for (u in Vcore) {
#     for (v in Vcore) {
#       # Set diagonal to 1.
#       if (u == v) {
#         ratio <- 1
#       } else {
#         uv <- paste(c(u,v), collapse=' ')
#         ratio <- ratios_aggr[uv, 'mean_ratio']
#       }
#       if (is.finite(ratio)) {
#         W0[u,v] <- ratio
#         W0[v,u] <- 1/ratio
#         A0[u,v] <- 1
#         A0[v,u] <- 1
#       }
#     }
#   }
#   
#   A <- A0
#   W <- W0
#   
#   repeat {
#     AA <- A %*% A
#     # If we already have the ratio, don't recompute it.
#     WW <- A * W + (1-A) * ((W %*% W) / AA)
#     # Print the maximum error.
#     # print(max(abs(1-WW*t(WW)), na.rm=TRUE))
#     WW[is.nan(WW)] <- 0
#     W <- WW
#     A_old <- A
#     # Update the binary adjacency matrix.
#     A <- (AA > 0) * 1
#     if (all(A==A_old)) break
#   }
#   
#   return(W)
# }

binsearch <- function(keyword, calib_max_vol, thresh, config, plot=FALSE, quiet=TRUE, silent=FALSE) {
  lo <- 1
  hi <- length(calib_max_vol)
  anchors <- names(calib_max_vol)
  iter <- 0
  tryCatch({
    while (hi > lo) {
      iter <- iter + 1
      pivot <- lo + floor((hi-lo)/2)
      anchor <- anchors[pivot]
      if (!quiet) message(sprintf('   Comparing to %s (%s)', anchor, mid2name(anchor)))
      ts <- query_google(c(keyword, anchor), config)$ts
      max_keyword <- max(ts[,keyword])
      max_anchor <- max(ts[,anchor])
      if (plot) {
        matplot(ts, type='l', lty=1, ylim=c(0,100))
        legend('topright', c(keyword, mid2name(anchor)), lty=1, col=1:2, bty='n')
      }
      if (max_keyword >= thresh && max_anchor >= thresh) {
        # as.numeric in order to remove anchor name.
        ratio <- as.numeric(calib_max_vol[anchor]) * (max_keyword / max_anchor)
        ts_keyword <- ts[,keyword] / max_keyword * ratio
        if (!silent) message(sprintf('Done with %s after %d steps', keyword, iter))
        return(list(ratio=ratio, iter=iter, ts=ts_keyword))
      } else if (max_keyword < thresh) {
        if (!quiet) message('   Going lower')
        hi <- pivot - 1
      } else {
        if (!quiet) message('   Going higher')
        lo <- pivot + 1
      }
    }
    if (hi <= 1) {
      message('Could not calibrate. Time series for keyword too low everywhere.')
    } else {
      message('Could not calibrate. Time series for keyword too high everywhere.')
    }
    return(NULL)
  }, error = function(e) {
    message(sprintf('Google gives error for %s', keyword))
    message(e)
    return(NULL)
  })
}
