#
# Generate a CSV with document frequency and term frequency

from collections import Counter
import sys
import argparse

import pandas as pd


def get_corpora():
    import corpus
    out = {}
    for name, func in vars(corpus).items():
        if name.startswith('generate_') and name.endswith('_texts'):
            name = name.partition('_')[-1].rpartition('_')[0]
            out[name] = func
    return out


def get_analyzers():
    import analyzers
    out = {}
    for name, func in vars(analyzers).items():
        if name.endswith('_analyzer'):
            out[name.rpartition('_')[0]] = func
    return out


def main():
    ap = argparse.ArgumentParser()
    corpora = get_corpora()
    ap.add_argument('corpus', choices=sorted(corpora))
    ap.add_argument('--case-sensitive', action='store_true', default=False)
    ap.add_argument('--out', type=argparse.FileType('w'), default=sys.stdout)
    args = ap.parse_args()
    analyzers = sorted(get_analyzers().items())

    out = {}
    for analyzer_name, _ in analyzers:
        out[analyzer_name + '_tf'] = Counter()
        out[analyzer_name + '_df'] = Counter()

    for doc in corpora[args.corpus]():
        for analyzer_name, analyzer in analyzers:
            toks = analyzer(doc)
            if not args.case_sensitive:
                toks = (tok.lower() for tok in toks)
            tf = Counter(toks)
            out[analyzer_name + '_tf'].update(tf)
            out[analyzer_name + '_df'].update(tf.keys())

    out = pd.DataFrame(out)
    out.index.names = ['word']
    """  # TODO: this post-processing somewhere else
    out.fillna(0, inplace=True)
    for col in out.columns:
        out[col + '_pct'] = out[col].rank(pct=True)

    for suf in ['_df', '_df_pct', '_tf', '_tf_pct']:
        out['max' + suf] = out[[col for col in out.columns if col.endswith(suf)]].max(axis=1)
    out.sort_values('max_df_pct', ascending=False, inplace=True)
    """
    out.to_csv(args.out)


if __name__ == '__main__':
    main()
