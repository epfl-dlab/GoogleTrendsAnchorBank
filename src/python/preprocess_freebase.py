import os, os.path, sys, re, gzip
from collections import defaultdict

DATA_DIR = os.environ['HOME'] + '/github/corona-food-trends/data/'
FREEBASE_DIR = '/dlabdata1/dlab_common_datasets/freebase-easy/freebase-easy-14-04-14/'

FOOD_TYPES = {'Food', 'Beverage', 'Dish', 'Ingredient'}

mids = dict()
types = defaultdict(set)

print('Collecting food entities...')

i = 0
with gzip.open(FREEBASE_DIR + 'facts.txt.gz', 'rt', encoding='utf-8') as f:
  for line in f:
    i += 1
    # if i > 1200000: break
    # Some lines are malformatted.
    try:
      (s, p, o, dot) = tuple(line.split('\t'))
      if p == 'is-a' and o in FOOD_TYPES:
        types[s].add(o)
    except ValueError:
      continue

print('Collecting Freebase links for food entities...')

with gzip.open(FREEBASE_DIR + 'freebase-links.txt.gz', 'rt', encoding='utf-8') as f:
  for line in f:
    # Some lines are malformatted.
    try:
      (s, p, o, dot) = tuple(line.split('\t'))
      if s in types:
        mids[s] = re.sub(r'<http://rdf.freebase.com/ns/(m/.*)>', r'\1', o)
    except ValueError:
      continue

print('Collecting and outputting scores for food entities...')

with open(DATA_DIR + 'freebase_foods.tsv', 'w') as out:
  with gzip.open(FREEBASE_DIR + 'scores.txt.gz', 'rt', encoding='utf-8') as f:
    for line in f:
      try:
        (s, p, o, dot) = tuple(line.split('\t'))
        if s in types and s in mids:
          type_string = ','.join(sorted(list(types[s])))
          print('\t'.join([mids[s], s, o, type_string]), file=out)
      except ValueError:
        continue

