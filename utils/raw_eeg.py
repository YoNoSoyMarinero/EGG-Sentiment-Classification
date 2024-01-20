import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import morlet, spectrogram, butter, filtfilt


class RawEEG:
    """
    Wraps raw eeg signals, allows ploting, channel selection by name and data filtering
    """

    def __init__(self, eeg_raw, channels_of_interest) -> None:
        """
        Loads and subsamples channels of interest

        Parameters:
        eeg_raw (mne.io.cnt.cnt.RawCNT): Raw loaded eeg signal
        channels_of_interest (list): List of channels that needs to be processed
        """
        self.matrix = eeg_raw.get_data()
        self.ch_names = eeg_raw.ch_names
        self.channels_of_interest = channels_of_interest
        self.sampling_frequency = 1000
        self.t = np.linspace(0, (self.matrix.shape[1]-1)/self.sampling_frequency, self.matrix.shape[1])
        self.raw_eeg_data = self.matrix[[index for index, value in enumerate(eeg_raw.ch_names) if value in self.channels_of_interest]]

    def plot_raw_eeg(self):
        """
        Plots all channels of intereset from RawEEG object
        """
        _, axs = plt.subplots(len(self.channels_of_interest), 1, figsize=(10, 6), sharex=True)
        for channel_idx in range(len(self.channels_of_interest)):
            axs[channel_idx].plot(self.t, self.matrix[channel_idx, :], label=f'Channel {self.channels_of_interest[channel_idx]}')
            axs[channel_idx].set_ylabel('Amplitude')
            axs[channel_idx].legend()
        plt.xlabel('Time (seconds)')
        plt.suptitle('EEG Data - Multiple Channels')
        plt.show()

    def plot_single_channel(self, channel_name, start_stamps = None, end_stamps = None, labels = None):
        """
        Plots single channel of choice from channels of interest:

        Paramters:
        channel_name (str): Channel name
        start_stamps (list): Start time stamps
        end_stamps (list): End time stamps
        labels (list): Class values between time stamps
        """
        channel_idx = self.channels_of_interest.index(channel_name)
        plt.figure(figsize=(10, 4))
        plt.plot(self.t, self.matrix[channel_idx, :], label=f'Channel {channel_name}')
        if start_stamps is not None:
            start_indices = np.where(np.isin(self.t, start_stamps))[0]
            plt.scatter(self.t[start_indices], np.zeros_like(self.t)[start_indices], color='red', label='Start stamps', zorder=2)
        if end_stamps is not None:
            end_indices = np.where(np.isin(self.t, end_stamps))[0]
            plt.scatter(self.t[end_indices], np.zeros_like(self.t)[end_indices], color='yellow', label='End stamps', zorder=2)
        if labels is not None:

            unique_classes = np.unique(labels)
            class_colors = {0: 'red', 1: 'cyan', 2: 'green', 3: 'orange', 4: 'purple',np.nan: 'white' }  

            for class_label in unique_classes:
                class_indices = np.where(labels == class_label)[0]
                
                if np.isnan(class_label):
                    color = class_colors[np.nan]
                else:
                    color = class_colors[class_label]
    
                plt.fill_between(self.t, y1=np.min(self.matrix[channel_idx, :]), y2=np.max(self.matrix[channel_idx, :]),
                                where=np.isin(np.arange(len(self.t)), class_indices),
                                color=color, alpha=0.4, label=str(class_label))
            
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title(f'EEG Data - Channel {channel_name}')
        plt.legend()
        plt.show()
    
    def get_single_channel(self, channel_name):
        """
        Getting 1-D values from single channel by name

        Paramaters:
        channel_name (str): Channel name
        """
        return self.matrix[self.ch_names.index(channel_name)]
    
    def filter_data(self):
        """
        Filters channels of interest
        """
        fcutlow = 8
        fcuthigh = 30
        nyq = 0.5 * self.sampling_frequency
        Wn = [fcutlow/nyq, fcuthigh/nyq]
        b, a = butter(2, Wn, btype='band')
        filtered_data = np.zeros_like(self.matrix)
        for channel_idx in range(self.matrix.shape[0]):
            filtered_data[channel_idx, :] = filtfilt(b, a, self.matrix[channel_idx, :])

        self.matrix = filtered_data