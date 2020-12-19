import glob
import os

import seaborn
import numpy as np
import pandas as pd

DIR = 'stopwords/en'


def read_sets(directory):
    out = pd.DataFrame()
    for path in glob.glob(os.path.join(directory, '[a-z0-9]*.txt')):
        if 'galago_forumstop' in path or 'galago_structured' in path or "gilner_morales" in path:
            continue
        for l in open(path):
            out.loc[os.path.basename(path)[:-4], l.strip()] = 1
    assert len(out)
    return out.fillna(0)


df = pd.DataFrame(read_sets(DIR))

# remove non-lowercase and non-unigram
df = df.loc[:, [w for w in df.columns if ('a' + w).islower()]]
df = df.loc[:, [w for w in df.columns if ' ' not in w]]

# include totals in row labels
df.index = ['%s: %d' % (name, n)
                   for name, n in zip(df.index.values, df.sum(axis=1).values)]
df = df.reindex(index=df.sum(axis=1).sort_values().index)

# reorder by (#words, word length, lexicographic)
#df = df.loc[:, sorted(df.columns, key=lambda s: (s.count(' '), len(s), s))]

# order columns by descending frequency in a big corpus
gigaword_nyt_freq = pd.read_csv('document-frequency/out/gigaword_nyt.csv.gz',
                                index_col=0)
gigaword_nyt_freq['max_df'] = gigaword_nyt_freq[[c for c in gigaword_nyt_freq.columns
                                                 if c.endswith('_df')]].max(axis=1)


def get_df(w):
    try:
        return gigaword_nyt_freq.max_df[w]
    except KeyError:
        return 0

df = df.loc[:, sorted(df.columns, key=get_df, reverse=True)]

# cluster and plot

cm = seaborn.clustermap(df, metric='jaccard', method='single',
                        col_cluster=False,
###                        row_cluster=False
                        )
cm.ax_heatmap.get_yaxis().set_tick_params(labelrotation=0)
cm.ax_heatmap.get_xaxis().set_tick_params(labelrotation=90)
labels = cm.ax_heatmap.get_xaxis().get_ticklabels()
step = len(labels) // 30
cm.ax_heatmap.get_xaxis().set_ticklabels([l if i % step == 2 else ''
                                          for i, l in enumerate(labels)])
if cm.dendrogram_row is not None:
    # show ticks for max jaccard similarity between any pair of lists in merge
    cm.dendrogram_row.xticks = np.linspace(0, 1, 6)[1:]
    cm.dendrogram_row.xticklabels = ['JD=%0.1f' % t
                                     for t in cm.dendrogram_row.xticks]
    cm.ax_row_dendrogram.grid(axis='x', color='lightgrey')
    cm.ax_row_dendrogram.get_xaxis().set_tick_params(labelrotation=90)
    cm.dendrogram_row.plot(cm.ax_row_dendrogram)

# hide colorbar
cm.cax.set_visible(False)

cm.savefig('overleaf-paper/figures/clustermap.pdf')



from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import pyplot as plt
from matplotlib import colors
seaborn.set_style("whitegrid")
fig = plt.figure(figsize=(15, 12))
gs = plt.GridSpec(11, 24)


Zjaccard = linkage(df.values, metric='jaccard', optimal_ordering=True)

shading_ax = fig.add_subplot(gs[1:, 5:])
dendro_ax = fig.add_subplot(gs[1:, :5], sharey=shading_ax)
heat_ax = fig.add_subplot(gs[1:, 5:-6])#, sharey=shading_ax)
nlist_ax = fig.add_subplot(gs[:1, 5:-6], sharex=heat_ax)
bar_ax = fig.add_subplot(gs[1:, -3:], sharey=shading_ax)

from matplotlib.style import context

with context({'lines.linewidth': 1}):
    dendro = dendrogram(Zjaccard, ax=dendro_ax, labels=[x.rpartition(':')[0] for x in df.index.values],
                        orientation='left', link_color_func=lambda *a, **kw: 'k')

reordered_df = df.iloc[dendro['leaves']]

dendro_ax.set_xlabel("Jaccard distance")

def hide_yticks(ax):
    ax.yaxis.set_tick_params(
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        right='off',         # ticks along the top edge are off
        labelleft='off', labelright='off')


def hide_xticks(ax):
    ax.xaxis.set_tick_params(
        which='both',
        top='off',
        bottom='off',
        labeltop='off', labelbottom='off')


dendro_ax.grid(False, axis='y')
dendro_ax.patch.set_visible(False)
hide_yticks(dendro_ax)
bar_ax.barh(np.arange(len(df)) * 10, reordered_df.sum(axis=1), .8 * 10, align='edge', color='black',)
bar_ax.grid(False, axis='y')
bar_ax.set_xlabel('# words')
bar_ax.yaxis.set_tick_params(labelsize=8)
bar_ax.patch.set_visible(False)

nlist_ax.bar(np.arange(df.shape[1]), reordered_df.sum(axis=0), .8, align='edge', color='black',)
nlist_ax.grid(False)
nlist_ax.set_ylabel('# lists')
hide_xticks(nlist_ax)
#bar_ax.yaxis.set_tick_params(labelsize=8)
nlist_ax.patch.set_visible(False)

shading_ax.xaxis.set_visible(False)
shading_ax.yaxis.set_visible(False)
shading_ax.barh(np.arange(1, len(df), 2) * 10, 1, 1. * 10, color='#eeeeff', align='edge')
hide_yticks(shading_ax)
shading_ax.patch.set_visible(False)
shading_ax.axis('off')

hide_yticks(heat_ax)
heat_ax.patch.set_visible(False)
heat_ax.grid(False)
black_only = colors.LinearSegmentedColormap('black', {'red': [(0.0,  0.0, 0.0), (1.0,  0.0, 0.0)],
                                                      'green': [(0.0,  0.0, 0.0), (1.0,  0.0, 0.0)],
                                                      'blue': [(0.0,  0.0, 0.0), (1.0,  0.0, 0.0)],
                                                      })
heat_ax.pcolormesh(np.ma.masked_where(reordered_df == 0, reordered_df), cmap=black_only)#, vmin=self.vmin, vmax=self.vmax,
                      #       cmap=self.cmap, **kws)
ticks = np.arange(1, df.shape[1], df.shape[1] // 30)
heat_ax.set_xticklabels(df.columns[ticks])
heat_ax.xaxis.set_tick_params(rotation=90)
heat_ax.set_xticks(ticks)

heat_ax.set_xlabel('Words in descending Gigaword frequency')

#fig.tight_layout()
fig
fig.savefig('overleaf-paper/figures/clusterplus.pdf')





from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import pyplot as plt
seaborn.set_style("whitegrid")
fig = plt.figure(figsize=(6, 6))
gs = plt.GridSpec(1, 15)


Zjaccard = linkage(df.values, metric='jaccard')

shading_ax = fig.add_subplot(gs[:, 6:])
dendro_ax = fig.add_subplot(gs[0, :6], sharey=shading_ax)
bar_ax = fig.add_subplot(gs[0, 11:], sharey=shading_ax)

from matplotlib.style import context

with context({'lines.linewidth': 1}):
    # TODO: flip data top-bottom to match seaborn
    dendro = dendrogram(Zjaccard, ax=dendro_ax, labels=[x.rpartition(':')[0] for x in df.index.values],
                        orientation='left', link_color_func=lambda *a, **kw: 'k')
dendro_ax.set_xlabel("Jaccard distance")

dendro_ax.grid(False, axis='y')
dendro_ax.patch.set_visible(False)
bar_ax.barh(np.arange(len(df)) * 10, df.iloc[dendro['leaves']].sum(axis=1), .8 * 10, align='edge', color='black',)
bar_ax.grid(False, axis='y')
bar_ax.set_xlabel('Number of words')
bar_ax.yaxis.set_tick_params(labelsize=8)
dendro_ax.yaxis.set_tick_params(
    which='both',      # both major and minor ticks are affected
    left='off',      # ticks along the bottom edge are off
    right='off',         # ticks along the top edge are off
    labelleft='off', labelright='off')
bar_ax.patch.set_visible(False)

shading_ax.xaxis.set_visible(False)
shading_ax.yaxis.set_visible(False)
shading_ax.barh(np.arange(1, len(df), 2) * 10, 1, 1. * 10, color='#eeeeff', align='edge')
shading_ax.yaxis.set_tick_params(
    which='both',      # both major and minor ticks are affected
    left='off',      # ticks along the bottom edge are off
    right='off',         # ticks along the top edge are off
    labelleft='off', labelright='off')
shading_ax.patch.set_visible(False)
shading_ax.axis('off')

#fig.tight_layout()
fig.savefig('overleaf-paper/figures/clusterbar.pdf')
fig
