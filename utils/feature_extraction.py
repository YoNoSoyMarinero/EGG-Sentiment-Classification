import numpy as np
from scipy.signal import welch
from scipy.stats import entropy 

"""
The feature_extraction is used for extracting features from raw EEG signals.
The extracted features include spectral entropy, signal variance, dominant frequency, maximum value, and zero crossings.
"""

def featutre_extraction(channels ,labels, fs, t):
    """
    Feature extraction method

    Parameters:
    channels (np.array): Channels of interest to be processed
    labels (np.array): Labels of each sample of channels of interest
    fs (np.array): Sampling frequency of features
    t (np.array): Time vector of channels

    Returns:
    final_matrix (np.array): Feature matrix
    """


    time = 1
    step = int(fs*time)
    final_matrix = []
    current_vector = []
    for i in range(0, len(channels[0]), step):
        for ch in channels:
            freqs, psd = welch(ch[i:i+step], fs=fs, nperseg=fs*2)
            spectral_entropy = entropy(psd)
            signal_variance = np.var(ch[i:i+step])
            peak_frequency = freqs[np.argmax(psd)]
            max_value = np.max(ch[i:i+step])
            zero_crossings = np.sum(np.diff(np.sign(ch[i:i+step])) != 0)
            time_since_start = t[i]
            current_vector.append(spectral_entropy)
            current_vector.append(signal_variance)
            current_vector.append(peak_frequency)
            current_vector.append(max_value)
            current_vector.append(zero_crossings)
        current_vector.append(time_since_start)

        unique_elements, counts = np.unique(labels[i:i+step], return_counts=True)
        most_frequent_index = np.argmax(counts)
        most_frequent_element = unique_elements[most_frequent_index]
        current_vector.append(most_frequent_element)
        final_matrix.append(current_vector)
        current_vector = []

    return np.array(final_matrix)

def create_feature_names(channel_names):
    """
    Returns feature names of extracted features

    Paramters:
    channel_names (str): Channel nammes 
    """

    feature_names = ["spectral_entropy", "signal_varaince", "peak_frequency", "max_value", "zero_crossing"]
    names = []

    for cn in channel_names:
        for fn in feature_names:
            names.append(f"{cn}_{fn}")

    names.append('time_since_start')
    names.append('label')

    return names
