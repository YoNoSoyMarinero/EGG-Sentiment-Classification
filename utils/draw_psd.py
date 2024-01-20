from scipy.signal import welch
import matplotlib.pyplot as plt

"""
The `draw_psd` function is used to display the spectral densities of specific EEG rhythms for an individual channel.``
"""

def draw_psd(vector, fs):
    """
    Draws power spectral density graph

    Parameters:
    vector (np.array): Plotting channel
    fs (int): Sampling frequency
    """
    frequencies, psd = welch(vector, fs, nperseg=1024)
    plt.figure(figsize=(10, 6))
    plt.semilogy(frequencies, psd, label='EEG PSD')
    plt.axvspan(0.5, 4, color='red', alpha=0.2, label='Delta (0.5-4 Hz)')
    plt.axvspan(4, 8, color='orange', alpha=0.2, label='Theta (4-8 Hz)')
    plt.axvspan(8, 13, color='yellow', alpha=0.2, label='Alpha (8-13 Hz)')
    plt.axvspan(13, 30, color='green', alpha=0.2, label='Beta (13-30 Hz)')
    plt.axvspan(30, 40, color='blue', alpha=0.2, label='Gamma (30-40 Hz)')

    plt.title('Power Spectral Density (PSD) of EEG')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.legend()
    plt.grid(True)
    plt.show()
