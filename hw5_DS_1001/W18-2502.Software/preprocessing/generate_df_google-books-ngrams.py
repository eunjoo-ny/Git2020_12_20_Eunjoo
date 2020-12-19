# The Google Books ngrams can be downloaded with
#  aws s3 sync s3://stop-words-nlposs/google-books-eng-all-1gram.parq/ google-books-eng-all-1gram.parq/
# This will download and store on disk ~5 GB
# Alternatively you can run  download_google-books-ngrams.py
# Requirements: dask, pandas, pyarrow

import pandas as pd

from dask import delayed
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

df = dd.read_parquet('google-books-eng-all-1gram.parq')

with ProgressBar():

    mask = df.year > 2000
    df_g = df[mask].groupby('ngram').sum()

    del df_g['year']
    del df_g['volume_count']

    (df_g.compute()
         .sort_values('match_count', ascending=False)
         .to_csv('df_google-books-eng-all-1gram.csv.gz', compression='gzip'))
