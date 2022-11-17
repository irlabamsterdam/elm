import json
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List
import matplotlib.pyplot as plt
from collections import defaultdict
from IPython.display import display, Markdown

def get_ground_truth_from_dataframe(dataframe: pd.DataFrame, col: str) -> Dict[str, list]:
    """
    This function takes as input the test dataframe, and return a dictionary
    with stream names as keys and the gold standard streams in
    binary vector format.
    """
    out = {}
    for doc_id, content in dataframe.groupby('name'):
        out[doc_id] = [item for item in content[col].tolist()]
    return out


def read_json(filename: str):
    with open(filename, 'r') as file:
        return json.load(file)

def length_list_to_bin(array_of_lengths: np.array) -> np.array:
    """
    @param array_of_lengths:  containing the lengths of the individual documents
    in a stream as integers.
    @return: numpy array representing the stream in binary format.
    """

    # Set up the output array
    out = np.zeros(shape=(sum(array_of_lengths)))

    # First document is always a boundary
    out[0] = 1

    # if only one document return the current representation
    if len(array_of_lengths) == 1:
        return out

    # Boundaries are at the cumulative sums of the number of pages
    # >>> doc_list = [2, 4, 3, 1]
    # >>> np.cumsum(doc_list) -> [2 6 9]

    # [:-1] because last document has boundary at end of array
    out[np.cumsum(array_of_lengths[:-1])] = 1
    return out


def bin_to_length_list(binary_vector: np.array) -> np.array:
    """
    @param binary_vector: np array containing the stream of pages
    in the binary format.
    @return: A numpy array representing the stream as a list of
    document lengths.
    """

    # We retrieve the indices of the ones with np.nonzero
    bounds = binary_vector.nonzero()[0]

    # We add the length of the array so that it works
    # with ediff1d, as this get the differences between
    # consecutive elements, and otherwise we would miss
    # the list document.
    bounds = np.append(bounds, len(binary_vector))
    
    # get consecutive indices
    return np.ediff1d(bounds)


def make_index(binary_vec: np.array) -> Dict[int, set]:
    # make index variant for binary vectors. we First get a split of elements by
    # using np.split and using the indices of the boundary pages.
    splits = np.split(np.arange(len(binary_vec)), binary_vec.nonzero()[0][1:])
    repeated_splits = np.repeat(np.array(splits, dtype=object),
                                [len(split) for split in splits], axis=0)
    # Now we have a list of splits, we repeat this split n times, where n
    # is the length of the split
    out = { i: set(item) for i, item in enumerate(repeated_splits)}
    
    return out


def bcubed(truth, pred, aggfunc=np.mean):
    assert len(truth) == len(pred)  # same amount of pages
    truth, pred = make_index(truth), make_index(pred)
    
    df = {i: {'size': len(truth[i]), 'P': 0, 'R': 0, 'F1': 0} for i in truth}
    for i in truth:
        df[i]['P'] = len(truth[i] & pred[i])/len(pred[i])
        df[i]['R'] = len(truth[i] & pred[i])/len(truth[i])
        df[i]['F1'] = (2*df[i]['P']*df[i]['R'])/(df[i]['P']+df[i]['R'])
     
    return pd.DataFrame(df).T


def elm(truth, pred, aggfunc=np.mean):
    assert len(truth) == len(pred)  # same amount of pages
    truth, pred = make_index(truth), make_index(pred)
    df = {i: {'size': len(truth[i]), 'P': 0, 'R': 0, 'F1': 0} for i in truth}
    for i in truth:
        TP = len((truth[i] & pred[i])-{i})
        FP = len(pred[i]-truth[i])
        FN = len(truth[i]-pred[i])
        if pred[i] == {i}:
            df[i]['P'] = 1
        else:
            df[i]['P'] = TP/len(pred[i]-{i})
        if truth[i] == {i}:
            df[i]['R'] = 1
        else:
            df[i]['R'] = TP/len(truth[i]-{i})

        df[i]['F1'] = TP / (TP + .5 * (FP + FN)) if not pred[i] == {i} == truth[i] else 1
    return  pd.DataFrame.from_dict(df, orient='index')


def calculate_metrics_one_stream(gold_vec, prediction_vec):
    
    out = {}
    
    gold_vec = np.array(gold_vec)
    prediction_vec = np.array(prediction_vec)
    scores = {
             'Bcubed': bcubed(gold_vec, prediction_vec)['F1'].mean(),
             'ELM': elm(gold_vec, prediction_vec)['F1'].mean()}
    
    scores_precision = {
             'Bcubed': bcubed(gold_vec, prediction_vec)['P'].mean(),
             'ELM': elm(gold_vec, prediction_vec)['P'].mean()}

    scores_recall = {
             'Bcubed': bcubed(gold_vec, prediction_vec)['R'].mean(),
             'ELM': elm(gold_vec, prediction_vec)['R'].mean()}
        
    out['precision'] = scores_precision
    out['recall'] = scores_recall
    out['F1'] = scores
        
    return out


def calculate_scores_df(gold_standard_dict, prediction_dict):
    all_scores = defaultdict(dict)
    for key in gold_standard_dict.keys():
        metric_scores = calculate_metrics_one_stream(gold_standard_dict[key],
                                                     prediction_dict[key])
        for key_m in metric_scores.keys():
            all_scores[key_m][key] = metric_scores[key_m]  
    return {key: pd.DataFrame(val) for key, val in all_scores.items()}


def calculate_mean_scores(gold_standard_dict, prediction_dict,
                          show_confidence_bounds=True):
    scores_df = {key: val.T.mean().round(2) for key, val in
                 calculate_scores_df(gold_standard_dict,
                                     prediction_dict).items()}
    scores_combined = pd.DataFrame(scores_df)
    test_scores = scores_combined

    confidence = 0.95

    # total number of documents is the number of ones in the binary array
    n = sum([np.sum(item) for item in prediction_dict.values()])

    z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)
    ci_length = z_value * np.sqrt((test_scores * (1 - test_scores)) / n)

    ci_lower = (test_scores - ci_length).round(2)
    ci_upper = (test_scores + ci_length).round(2)

    precision_ci = ci_lower['precision'].astype(str) + '-' + ci_upper[
        'precision'].astype(str)
    recall_ci = ci_lower['recall'].astype(str) + '-' + ci_upper[
        'recall'].astype(str)
    f1_ci = ci_lower['F1'].astype(str) + '-' + ci_upper['F1'].astype(str)

    out = pd.DataFrame(scores_df)
    out = out.rename({0: 'value'}, axis=1)
    out['support'] = sum(
        [np.sum(item).astype(int) for item in gold_standard_dict.values()])
    if show_confidence_bounds:
        out['CI Precision'] = precision_ci
        out['CI Recall'] = recall_ci
        out['CI F1'] = f1_ci

    return out

    
def evaluation_report(gold_standard_json, prediction_json, round_num=2,
                     title="", show_confidence_bounds=True):
    # 1. print the mean scores
    display(Markdown("<b> Mean scores of the evaluation metrics for %s </b>" % title))
    display(calculate_mean_scores(gold_standard_json, prediction_json,
                                 show_confidence_bounds=show_confidence_bounds).round(round_num))
    
# TODO: See if I can add the standard deviation column as string to the mean column.
def create_score_dataframe(dataframe_bcubed, dataframe_elm):
    
    # First we do this for the BCUBED values
    bcubed_vals = dataframe_bcubed.describe().iloc[1:3, :].round(2)
    bcubed_vals.iloc[0, :] =  '$\mu=$' + bcubed_vals.iloc[0].astype(str) + ", $\sigma$=" + bcubed_vals.iloc[1, :].astype(str)
    bcubed_vals.drop(bcubed_vals.tail(1).index, inplace=True)
    bcubed_vals.rename(index={'mean': "BCUBED"}, inplace=True)
    
    # Next we do this for the ELM values
    elm_vals = dataframe_elm.describe().iloc[1:3, :].round(2)
    elm_vals.iloc[0, :] =  '$\mu=$' + elm_vals.iloc[0].astype(str) + ", $\sigma$=" + elm_vals.iloc[1, :].astype(str)
    elm_vals.drop(elm_vals.tail(1).index, inplace=True)
    elm_vals.rename(index={'mean': "ELM"}, inplace=True)
    
    display_df = pd.concat([bcubed_vals, elm_vals], axis=0)
    
    return display_df
    
    
# We create this function so that we can get the predictions of al lthe streams.
# Because we also want to be able to use this to make plots of the individual plots
# we also use the original stream names. We can then use Pandas groupign to obtain the predictions
# for the streams.
def get_bcubed_and_elm_scores(gold_standard_dict, prediction_dict):
    # Here we calculate the scores for BCUBED and ELM reporting precision, recall and f1
    bcubed_scores = []
    elm_scores = []

    # Save the scores of each element seperately
    for gold_clustering in gold_standard_dict.keys():
        bcubed_score = bcubed(gold_standard_dict[gold_clustering], prediction_dict[gold_clustering])
        elm_score = elm(gold_standard_dict[gold_clustering], prediction_dict[gold_clustering])
        
        # give names to the elements in the stream so that we can later use groupby to get the stream means
        # for the tables
        bcubed_score['name'] = gold_clustering
        elm_score['name'] = gold_clustering
        
        bcubed_scores.append(bcubed_score)
        elm_scores.append(elm_score)
        
    bcubed_df = pd.concat(bcubed_scores)
    elm_df = pd.concat(elm_scores)
    
    return bcubed_df, elm_df


# We can drop the size column, this doesn't make sense for aggregated values.
def get_mean_stream_predictions(score_dataframe):
    mean_scores = score_dataframe.groupby('name').mean()
    return mean_scores.drop('size', axis=1)

# TODO: some things will probably go wrong here if I get the axis information like I do know and still
# try to plot two distribution on the same plot, I will have to think of a way to fix this.
# TODO: make parametrs for the location of the mean and standard deviation.
def plot_kde_with_mean_and_std(dataseries, axis, legend_label: str, color: str, 
                              fontsize: int = 10, first=True, mean_label_location: float = 2.0,
                              std_label_location: float = 0.6):
    # We first create the basic kde plot, to which we can then add the information about mean and 
    # standard deviation
    basic_kdeplot = sns.kdeplot(data=dataseries, ax=axis, cut=0, clip=[0, 1], label=legend_label,
                             shade=False, color=color)
    
    # Get the information about the mean and standard deviation for the lines
    line_idx = 0 if first else 1
    xline = basic_kdeplot.lines[line_idx].get_xdata()
    yline = basic_kdeplot.lines[line_idx].get_ydata()
    data_mean = dataseries.mean()
    data_std = dataseries.std()

    height = np.interp(data_mean, xline, yline)
    # Plot the dotted mean line
    basic_kdeplot.vlines(data_mean, 0, height, color=color, ls=':')
    
    # Use the annotate function from matplotlib to create a double headed arrow showing the standard deviation
    axis.annotate("", (data_mean-data_std, std_label_location-0.2), (data_mean+data_std, std_label_location-0.2), arrowprops=dict(arrowstyle='<->'))

    # Create labels for the actual values of the mean and standard deviation of the KDE plot
    axis.text(max(0.1, data_mean-data_std), std_label_location, s="$\sigma=%.2f$" % data_std, fontsize=fontsize)
    axis.text(max(0.1, data_mean), mean_label_location, s="$\mu=%.2f$" % data_mean, fontsize=fontsize)

    # Showing standard deviation as a shaded plot a bit tricky we have to calculate which values fall between it and
    # only select those.
    plot_values = pd.Series(xline).between(data_mean-data_std, data_mean+data_std)
    # select only the rang of values that fall between 1 times the standard deviation on both sides
    # of the mean
    min_idx, max_idx = plot_values.idxmax(), plot_values[::-1].idxmax()
    basic_kdeplot.fill_between(xline[min_idx:max_idx], 0, yline[min_idx:max_idx], facecolor=color, alpha=0.2)
    # Return the plot for further customization options later
    return basic_kdeplot

# TODO: The main change I want to make to this function is that we now return the different axes instead of just plotting
# So that we can make adjustements for the final 3 by 3 plot.
def plot_p_r_f1_difference_kdes(dataframe_bcubed, dataframe_elm, mean_label_locations: List[float] = [2.5, 3.6],
                                std_label_locations: List[float] = [0.6, 1.0], list_of_axes: list=[], title=""):
    
    assert sorted(list(dataframe_bcubed.columns)) == sorted(list(dataframe_elm.columns))
    
    # TODO: return the plots so that I can later put them in the 3 by 3 plot
    if not len(list_of_axes):
        _, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    else:
        axes=list_of_axes
        
    for i, col_name in enumerate(dataframe_bcubed.columns):
        col_data_bcubed = dataframe_bcubed[col_name]
        col_data_elm = dataframe_elm[col_name]
        if col_data_elm.nunique() == 1:
            # Both will just indicate 1, so just plot a boxplot with both set at 1
            axes[i].bar(0.40, 1, 0.1, label='BCUBED', color='crimson', alpha=0.5)
            axes[i].bar(0.50, 1, 0.1, label="ELM", color='blue', alpha=0.5)
            axes[i].set_xlim(left=0, right=1)
            axes[i].set_xticklabels([])    
        else:
            plot_kde_with_mean_and_std(col_data_bcubed, axis=axes[i], legend_label="BCUBED", color="crimson", mean_label_location=mean_label_locations[0],
                                      std_label_location=std_label_locations[0])
            plot_kde_with_mean_and_std(col_data_elm, axis=axes[i], legend_label="ELM", color="blue", first=False, std_label_location=std_label_locations[1],
                                      mean_label_location=mean_label_locations[1])
    plt.suptitle(title)