# Download the google books ngram dataset in Parquet format
# This will download and store on disk ~5GB of data
# Requirements: dask, pandas, pyarrow and google-ngram-downloader

import pandas as pd

from dask import delayed
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

from google_ngram_downloader import readline_google_store


urls = (url for _, url, _ in readline_google_store(ngram_len=1))


# dd.read_csv fails to parse files for some reason here
df_delayed = (delayed(pd.read_csv)(url, sep='\t', encoding='latin1',
                                   names=['ngram', 'year', 'match_count', 'page_count', 'volume_count'])
              for url in urls)


with ProgressBar():
    df = dd.from_delayed(df_delayed)
    df.to_parquet('google-books-eng-all-1gram.parq', compression='snappy')
