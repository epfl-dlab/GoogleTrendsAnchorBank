import os, os.path, sys, re, gzip

reload(sys)  
sys.setdefaultencoding('utf8')

DATA_DIR = os.environ['HOME'] + '/github/corona-food-trends/data/'
FREEBASE_DIR = '/dlabdata1/dlab_common_datasets/freebase-easy/freebase-easy-14-04-14/'

with gzip.open(FREEBASE_DIR + 'facts.txt.gz', 'rt') as f:
  for line in f:
    print('got line', line)
