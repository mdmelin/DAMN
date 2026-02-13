import numpy as np

def bits_per_spike(y_true, y_pred):
    '''an implementation of bits per spike from
    Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. Chapter 10. “Evaluating Goodness-of-fit”. 2014.
    '''
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, None)

    # Model log-likelihood
    ll_model = np.sum(y_true * np.log(y_pred) - y_pred)

    # Null model (mean firing rate)
    mean_rate = np.mean(y_true)
    ll_null = np.sum(y_true * np.log(mean_rate + eps) - mean_rate)

    total_spikes = np.sum(y_true)

    return (ll_model - ll_null) / (np.log(2) * total_spikes)

def bits_per_spike_multi_target(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(bits_per_spike(y_true[:, i], y_pred[:, i]))
    return np.array(scores)