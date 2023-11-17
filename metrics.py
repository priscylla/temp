import numpy as np
import pandas as pd

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

from itertools import combinations


def get_top_k(data, k):
    '''
    Return dict with a sorted series where the name of feature is the index and value with sign
    '''
    dict_topk = {}
    for index, row in data.iterrows():
        features = row.abs().nlargest(k, keep='first')
        dict_topk[index] = features
        for feature_name, value in features.items():
            features[feature_name] = row[feature_name]
    return dict_topk

def feature_agreement(topk_1, topk_2):
    '''
    Return a value indicating the feature agreement between the top k feature importance of two explanation methods
    Parameters
    ----------
    topk_1 : pandas.core.series.Series
      Sort Serie with the topk features from an explanation method.
    topk_2 : pandas.core.series.Series
      Sort Serie with the topk features from an explanation method.
    Returns
    -------
    metric : int
      Value of the metric feature agreement (between 0 and 1)
    '''
    if len(topk_1) != len(topk_2):
        raise ValueError('The two parameters have different sizes: ', len(topk_1), ' and ', len(topk_2), '. They must be the same size.')
    else:
        #calcula a intersecção e divite por k
        return len(set(topk_1.index) & set(topk_2.index)) / len(topk_1)
    
def rank_agreement(topk_1, topk_2):
    '''
    Return a value indicating the rank agreement between the top k feature importance of two explanation methods
    Parameters
    ----------
    topk_1 : pandas.core.series.Series
      Sort Serie with the topk features from an explanation method.
    topk_2 : pandas.core.series.Series
      Sort Serie with the topk features from an explanation method.
    Returns
    -------
    metric : int
      Value of the metric rank agreement (between 0 and 1)
    '''
    if len(topk_1) != len(topk_2):
        raise ValueError('The two parameters have different sizes: ', len(topk_1), ' and ', len(topk_2), '. They must be the same size.')
    else:
        #calcula a quantidade de features na mesma posição em ambos os topk e divite por k
        list1 = topk_1.index.to_list()
        list2 = topk_2.index.to_list()
        return sum(first == second for (first, second) in zip(list1, list2)) / len(topk_1)
    
def sign_agreement(topk_1, topk_2):
    '''
    Return a value indicating the sign agreement between the top k feature importance of two explanation methods
    Parameters
    ----------
    topk_1 : pandas.core.series.Series
      Sort Serie with the topk features from an explanation method.
    topk_2 : pandas.core.series.Series
      Sort Serie with the topk features from an explanation method.
    Returns
    -------
    metric : int
      Value of the metric sign agreement (between 0 and 1)
    '''
    if len(topk_1) != len(topk_2):
        raise ValueError('The two parameters have different sizes: ', len(topk_1), ' and ', len(topk_2), '. They must be the same size.')
    else:
#         count_same_sign = (topk_1 * topk_2 >= 0).sum() - (topk_1 * topk_2 == 0).sum()
#         count_same_sign = (topk_1 * topk_2 >= 0).sum() - (topk_1 * topk_2 == 0).sum() + ((topk_1 ** 2 + topk_2 ** 2) == 0).sum()
#         count_same_sign = abs(count_same_sign)
        count_same_sign = (topk_1 * topk_2 > 0).sum() + ((topk_1 ** 2 + topk_2 ** 2) == 0).sum()
        return count_same_sign / len(topk_1)              
    
    
def sign_rank_agreement(topk_1, topk_2):
    '''
    Return a value indicating the signed rank agreement between the top k feature importance of two explanation methods
    Parameters
    ----------
    topk_1 : pandas.core.series.Series
      Sort Serie with the topk features from an explanation method.
    topk_2 : pandas.core.series.Series
      Sort Serie with the topk features from an explanation method.
    Returns
    -------
    metric : int
      Value of the metric signed rank agreement (between 0 and 1)
    '''
    if len(topk_1) != len(topk_2):
        raise ValueError('The two parameters have different sizes: ', len(topk_1), ' and ', len(topk_2), '. They must be the same size.')
    else:
        sameSign = (topk_1 * topk_2 > 0) | ((topk_1 ** 2 + topk_2 ** 2) == 0)
        sameSign = sameSign[sameSign == True]
        count = 0
        for feature_name, value in sameSign.items():
            pos1 = topk_1.index.to_list().index(feature_name)
            pos2 = topk_2.index.to_list().index(feature_name)
            if pos1 == pos2:
                count+=1
        return count / len(topk_1)
    
    

def heatmap(data, row_labels, col_labels, title, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    
    ax.set_title(title)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def create_matrix_combination_methdos_by_metric(dict_topk, func_metric, num_instancias, methods):
    src = methods
    combinations_methods = []
    for s in combinations(src, 2):
        combinations_methods.append(s)
    list_combinations_methods = [str(t) for t in combinations_methods]
    
    N = len(combinations_methods)
    matrix_points = {}
    matrix_points = np.zeros((num_instancias,N))
    for instance in range(0,num_instancias):
        num_combination = 0
        for method_names in combinations_methods:
            method1 = dict_topk[method_names[0]][instance]
            method2 = dict_topk[method_names[1]][instance]
            metric = func_metric(method1, method2)
            matrix_points[instance][num_combination] = metric
            num_combination += 1
        
    return matrix_points, list_combinations_methods