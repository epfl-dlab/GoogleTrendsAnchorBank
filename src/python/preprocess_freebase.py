import os, os.path, sys, re, gzip

DATA_DIR = os.environ['HOME'] + '/github/corona-food-trends/data/'
FREEBASE_DIR = '/dlabdata1/dlab_common_datasets/freebase-easy/freebase-easy-14-04-14/'

FOOD_TOPICS = {'Food', 'Beverage', 'Dish', 'Ingredient'}

with gzip.open(FREEBASE_DIR + 'facts.txt.gz', 'rt', encoding='utf-8') as f:
  for line in f:
    (s, p, o, dot) = tuple(line.split('\t'))
    if p == 'is-a' and o in FOOD_TOPICS:
      print(s)
