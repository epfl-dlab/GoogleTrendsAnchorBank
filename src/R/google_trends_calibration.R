library(gtrendsR)
library(igraph)

options(stringsAsFactors=FALSE)

###############################################################################
# Constants
###############################################################################
DATA_DIR <- "../../data/"
#DATA_DIR <- sprintf('%s/github/GoogleTrendsAnchorBank/data', Sys.getenv('HOME'))

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
  D <- graph_from_data_frame(ratios)
  
  # Sanity check: Is D indeed acyclic as it should be?
  if (!is.dag(D)) warning('Directed graph is not acyclic!')
  
  # Sanity check: Is D connected?
  if (!is.connected(D)) warning('Directed graph is not connected!')
  
  # Propagate ratios through the graph via matrix multiplication.
  W <- infer_all_ratios(ratios)
  err <- max(abs(1-W*t(W)), na.rm=TRUE)
  
  # The top keyword is the one with which all other keywords have a ratio < 1.
  ref_anchor <- names(which(colSums(W > 1) == 0))
  
  if (length(ref_anchor) > 1) {
    warning('There are multiple top keywords! Choosing the one from the largest component.')
    ref_anchor <- ref_anchor[which.max(colSums(W[,ref_anchor] > 0))]
  }
  
  # Calibrate all other anchors against the top anchor.
  calib_max_vol <- sort(W[,ref_anchor])
  
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

compute_max_ratios <- function(google_results, plot=FALSE) {
  anchors <- NULL
  v1 <- NULL
  v2 <- NULL
  ratios_vec <- NULL
  
  for (v in names(google_results)) {
    ts <- google_results[[v]]
    if (plot) {
      matplot(ts, type='l', lty=1, main=v)
      legend('topright', colnames(ts), lty=1, col=1:ncol(ts))
    }
    for (j in 1:ncol(ts)) {
      for (k in 1:ncol(ts)) {
        if (j < k) {
          if (time_series_ok(ts[,j], config) && time_series_ok(ts[,k], config)) {
            anchors <- c(anchors, v)
            v1 <- c(v1, colnames(ts)[j])
            v2 <- c(v2, colnames(ts)[k])
            ratios_vec <- c(ratios_vec, max(ts[,j])/max(ts[,k]))
          }
        }
      }
    }
  }
  
  ratios <- data.frame(anchor=anchors, v1=v1, v2=v2, ratio=ratios_vec)
  
  # Flip edges so they all point from higher- to lower-volume keywords.
  for (r in 1:nrow(ratios)) {
    if (ratios$ratio[r] > 1) {
      tmp <- ratios$v1[r]
      ratios$v1[r] <- ratios$v2[r]
      ratios$v2[r] <- tmp
      ratios$ratio[r] <- 1 / ratios$ratio[r]
    }
  }
  
  # Take care of the special case of ratio == 1.
  idx0 <- which(ratios$ratio < 1)
  pair_names <- unique(paste(ratios$v1[idx0], ratios$v2[idx0], sep=' '))
  for (i in 1:nrow(ratios)) {
    if (ratios$ratio[i] == 1) {
      if ((paste(ratios$v1[i], ratios$v2[i], sep=' ') %in% pair_names)) {
        # Nothing to do.
      } else if ((paste(ratios$v2[i], ratios$v1[i], sep=' ') %in% pair_names)) {
        tmp <- ratios$v1[i]
        ratios$v1[i] <- ratios$v2[i]
        ratios$v2[i] <- tmp
      } else {
        tmp <- sort(c(ratios$v1[i], ratios$v2[i]))
        ratios$v1[i] <- tmp[1]
        ratios$v2[i] <- tmp[2]
      }
    }
  }
  
  ratios_aggr <- as.data.frame(t(simplify2array(by(ratios[,c('v1', 'v2', 'ratio')],
                                                   paste(ratios$v1, ratios$v2),
                                                   function(r) c(mean(r$ratio), sd(r$ratio))))))
  ratios_aggr <- cbind(do.call(rbind, strsplit(rownames(ratios_aggr), ' ')), ratios_aggr)
  colnames(ratios_aggr) <- c('v1', 'v2', 'mean_ratio', 'sd_ratio')
  return(ratios_aggr)
}

infer_all_ratios <- function(ratios_aggr) {
  Vcore <- unique(c(ratios_aggr$v1, ratios_aggr$v2))
  W0 <- matrix(data=0, nrow=length(Vcore), ncol=length(Vcore), dimnames=list(Vcore,Vcore))
  A0 <- W0
  
  for (u in Vcore) {
    for (v in Vcore) {
      # Set diagonal to 1.
      if (u == v) {
        ratio <- 1
      } else {
        uv <- paste(c(u,v), collapse=' ')
        ratio <- ratios_aggr[uv, 'mean_ratio']
      }
      if (is.finite(ratio)) {
        W0[u,v] <- ratio
        W0[v,u] <- 1/ratio
        A0[u,v] <- 1
        A0[v,u] <- 1
      }
    }
  }
  
  A <- A0
  W <- W0
  
  repeat {
    AA <- A %*% A
    # If we already have the ratio, don't recompute it.
    WW <- A * W + (1-A) * ((W %*% W) / AA)
    # Print the maximum error.
    # print(max(abs(1-WW*t(WW)), na.rm=TRUE))
    WW[is.nan(WW)] <- 0
    W <- WW
    A_old <- A
    # Update the binary adjacency matrix.
    A <- (AA > 0) * 1
    if (all(A==A_old)) break
  }
  
  return(W)
}

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
