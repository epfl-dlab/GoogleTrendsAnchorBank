library(gtrendsR)
library(igraph)

options(stringsAsFactors=FALSE)

DATA_DIR <- sprintf('%s/github/corona-food-trends/data', Sys.getenv('HOME'))

# NUM_TOP_ENT should be a multiple of NUM_VERTICES; NUM_VERTICES should be even.
NUM_TOP_ENT <- 1500
NUM_VERTICES <- 40
MAX_VOL_THRESH_OFFLINE <- 10
MAX_VOL_THRESH_ONLINE <- 10
TIME <- '2019-08-01 2020-04-08'
GEO <- ''

HI_TRAFFIC <- c('/m/03vgrr', '/m/0289n8t', '/m/02y1vz', '/m/019rl6', '/m/0b2334', '/m/010qmszp', '/m/01yvs')
names(HI_TRAFFIC) <- c('LinkedIn', 'Twitter', 'Facebook', 'Yahoo!', 'Reddit', 'Airbnb', 'Coca-Cola')

FOODS <- read.table(sprintf('%s/freebase_foods.tsv', DATA_DIR), sep='\t', quote='',
                    comment.char='', header=TRUE, stringsAsFactors=FALSE)
FOODS <- FOODS[FOODS$is_dish | FOODS$is_food,]
rownames(FOODS) <- FOODS$mid

BLACKLIST <- c('/m/0bcgdv') # Irvingia gabonensis

mid2name <- function(mid) {
  lookup <- c(FOODS$name, names(HI_TRAFFIC))
  names(lookup) <- c(FOODS$mid, HI_TRAFFIC)
  lookup[mid]
}

google <- function(keywords) {
  gtrends(keywords, geo=GEO, time=TIME, onlyInterest=TRUE)
}

keyword_ok <- function(keyword) {
  tryCatch({
    google(keyword)
  }, error = function(e) {
    FALSE
  })
  !(keyword %in% BLACKLIST)
}

query_google <- function(keywords) {
  trends <- google(keywords)
  interest <- trends$interest_over_time[c('date', 'hits', 'keyword')]
  ts <- reshape(interest, timevar='date', idvar='keyword', direction='wide')
  rownames(ts) <- ts$keyword
  ts <- ts[,-1]
  ts[ts=='<1'] <- '0'
  ts <- t(ts)
  ts <- apply(ts, 2, as.numeric)
  list(full=trends, ts=ts)
}

time_series_ok <- function(ts) {
  max(ts) >= MAX_VOL_THRESH_OFFLINE
}

load_keyword_ring_graph <- function() {
  file <- sprintf('%s/calibration/G.RData', DATA_DIR)
  if (file.exists(file)) {
    load(file)
  } else {
    # Stratified sampling: 2 per stratum, so we can build a ring without "discontinuities".
    mat <- matrix(FOODS$mid[1:NUM_TOP_ENT], nrow=NUM_TOP_ENT/(NUM_VERTICES/2), ncol=NUM_VERTICES/2)
    samples2 <- apply(mat, 2, function(r) sample(r, 2))
    keywords <- c(rev(samples2[1,]), HI_TRAFFIC, samples2[2,])
    V <- NULL
    i <- 1
    for (k in keywords) {
      if (keyword_ok(k)) {
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

load_google_results <- function(G) {
  file <- sprintf('%s/calibration/google_results.RData', DATA_DIR)
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
      trends <- query_google(keywords)
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
          if (time_series_ok(ts[,j]) && time_series_ok(ts[,k])) {
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
    print(max(abs(1-WW*t(WW)), na.rm=TRUE))
    WW[is.nan(WW)] <- 0
    W <- WW
    A_old <- A
    # Update the binary adjacency matrix.
    A <- (AA > 0) * 1
    if (all(A==A_old)) break
  }
  
  return(W)
}

binsearch <- function(keyword, calibration, plot=FALSE) {
  lo <- 1
  hi <- length(calibration)
  anchors <- names(calibration)
  iter <- 0
  while (hi > lo) {
    iter <- iter + 1
    pivot <- lo + floor((hi-lo)/2)
    anchor <- anchors[pivot]
    print(sprintf('Comparing to %s (%s)', anchor, mid2name(anchor)))
    ts <- query_google(c(keyword, anchor))$ts
    max_keyword <- max(ts[,keyword])
    max_anchor <- max(ts[,anchor])
    if (plot) {
      matplot(ts, type='l', lty=1)
      legend('topright', c(keyword, mid2name(anchor)), lty=1, col=1:2, bty='n')
    }
    if (max_keyword >= MAX_VOL_THRESH_ONLINE && max_anchor >= MAX_VOL_THRESH_ONLINE) {
      ratio <- calibration[anchor] * (max_keyword / max_anchor)
      ts_keyword <- ts[,keyword] / max_keyword * ratio
      return(list(ratio=ratio, iter=iter, ts=ts_keyword))
    } else if (max_keyword < MAX_VOL_THRESH_ONLINE) {
      print('Going lower')
      hi <- pivot
    } else {
      print('Going higher')
      lo <- pivot
    }
  }
  if (hi == 1) {
    warning('Could not calibrate. Time series for keyword too low everywhere.')
  } else {
    warning('Could not calibrate. Time series for keyword too high everywhere.')
  }
}
