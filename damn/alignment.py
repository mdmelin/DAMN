import numpy as np
from collections.abc import Iterable
from functools import partial
from tqdm import tqdm
from multiprocessing.pool import Pool,ThreadPool

def construct_timebins(pre_seconds, post_seconds, binwidth_s):
    pre_event_timebin_centers = -np.arange(0, pre_seconds, binwidth_s)[1:][::-1]
    post_event_timebin_centers = np.arange(0, post_seconds, binwidth_s) 
    timebin_centers = np.append(pre_event_timebin_centers, post_event_timebin_centers)
    timebin_edges = np.append(timebin_centers - binwidth_s/2, timebin_centers[-1] + binwidth_s/2)
    event_index = pre_event_timebin_centers.size # index of the alignment event in psth_matrix
    return timebin_centers, timebin_edges, event_index

def generate_master_alignment_bin_times(master_alignment_times, master_pre_s, master_post_s, binwidth_s):
    all_trial_bins = []
    for t in master_alignment_times:
        bins = construct_timebins(master_pre_s, master_post_s, binwidth_s)[0] + t
        all_trial_bins.append(bins)
    return np.concatenate(all_trial_bins)

def resample_to_timebins(master_times, sample_times, sample_values):
    # resample sample_values at sample_times to master_times using interpolation
    # sample_values is n_samples x n_features
    resampled_values = np.empty_like(master_times, shape=(master_times.size, sample_values.shape[1]))
    for i in range(sample_values.shape[1]):
        resampled_values[:,i] = np.interp(master_times, sample_times, sample_values[:,i], left=0, right=0)
    return resampled_values

'''
Many of these functions below are adopted from https://github.com/spkware/spks (Couto and Melin, 2025).
Please see this repo for the original implementations, and for many other related functions for working
with spiking data.
'''
def binary_spikes(spks,edges,kernel = None):
    ''' 
    """
    Create a vector of binary spikes, optionally convolved with a kernel.

    Parameters:
    -----------
    spks : list of array-like
        List of spike times for each unit.
    edges : array-like
        Time bin edges for discretizing spikes.
    kernel : array-like, optional
        Kernel to convolve with binary spike trains. If None, no convolution is performed.
        Note: When using a kernel, the function pads the spike trains to minimize edge effects.

    Returns:
    --------
    numpy.ndarray
        2D array of binary or convolved spike trains. Each row corresponds to a unit.

    Examples:
    ---------
    # Basic binary spikes
    binsize = 0.001
    edges = np.arange(0, 5, binsize)
    bspks = binary_spikes(spks, edges) / binsize
    ---------
    # Convolved with alpha function kernel
    binsize = 0.001
    t_decay = 0.025
    t_rise = 0.001
    decay = t_decay / binsize
    kern = alpha_function(int(decay * 15), t_rise=t_rise, t_decay=decay, srate=1./binsize)
    edges = np.arange(0, 5, binsize)
    bspks = binary_spikes(spks, edges, kernel=kern) / binsize
    ---------
    # Correct timing for plotting
    time = edges[:-1] + np.diff(edges[:2]) / 2

    Joao Couto - March 2016 , Modified by Max Melin
    '''
    
    if kernel is not None:
        n_pad = int(kernel.size / 2)
        bins = [np.histogram(sp,edges)[0] for sp in spks]
        bins = [np.convolve(np.pad(a,n_pad,'reflect'),kernel,'same')[n_pad:-n_pad] for a in bins] # padding deals with artifacts
    else:
        bins = [np.histogram(sp,edges)[0] for sp in spks]
    return np.vstack(bins)


def align_raster_to_event(event_times, spike_times, pre_seconds, post_seconds):
    """create aligned rasters relative to event_times

    Parameters
    ----------
    event_times : list or ndarray
        a list or numpy array of event times to be aligned to
    spike_times : list or ndarray
        a list spike times for one cluster
    pre_seconds : float, list
        grab _ seconds before event_times for alignment, by default 1
        can also be a list of different pre_seconds for each event
    post_seconds : float, list
        grab _ seconds after event_times for alignment, by default 2
        can also be a list of different pre_seconds for each event

    Returns
    -------
    list
        a list of aligned rasters for each event_times
    """    
    event_rasters = []
    pre_iterable = isinstance(pre_seconds, Iterable)
    post_iterable = isinstance(post_seconds, Iterable)
    for i, event_time in enumerate(event_times):
        relative_spiketimes = spike_times - event_time
        pre = pre_seconds[i] if pre_iterable else pre_seconds
        post = post_seconds[i] if post_iterable else post_seconds
        spks = relative_spiketimes[np.logical_and(relative_spiketimes <= post, relative_spiketimes >= -pre)]
        event_rasters.append(np.array(spks))
    return event_rasters

def compute_spike_count(event_times, spike_times, pre_seconds, post_seconds, binwidth_s, pad=0, kernel=None):
    #event_times = discard_nans(event_times) 
    rasters = align_raster_to_event(event_times, 
                                   spike_times,
                                   pre_seconds+pad,
                                   post_seconds+pad)

    _, timebin_edges, _ = construct_timebins(pre_seconds+pad, post_seconds+pad, binwidth_s) 

    psth_matrix = binary_spikes(rasters, timebin_edges, kernel=kernel) #/ binwidth_s # divide by binwidth to get a rate rather than count
    
    # recreate timebins without the pad
    timebin_centers, timebin_edges, event_index = construct_timebins(pre_seconds, post_seconds, binwidth_s)
    valid_inds = (timebin_centers > -pre_seconds) & (timebin_centers < post_seconds)
    timebin_centers = timebin_centers[valid_inds] # strip off pad from timebin_centers
    #if timebin_edges.size == psth_matrix.shape[1]: # handle case of odd bin count 
    #    psth_matrix = psth_matrix[:,:-1]
    return psth_matrix, timebin_centers, event_index