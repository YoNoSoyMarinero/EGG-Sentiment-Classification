import json
import numpy as np

"""
The `LabelsContainer` class is used for marking individual samples of EEG signals.
Alongside the SEED-V database, temporal instances in which specific video clips utilizing particular semantics begin are obtained.
Based on the vector length of the signal duration and the temporal features, a feature vector is generated at each temporal instance.

"""


class LabelsContainer:
    def __init__(self, session, time_stamps_path):
        """
        Loads time stamps and labels

        Parameters:
        session (int) : Number of session, can take only values 1, 2, 3
        time_stamps_path (str): Patho to time stamps json file
        """
        with open(time_stamps_path, "r") as file:
            data = json.load(file)
            self.session = data[str(session)]

    def create_labels_vector(self, t):
        """
        Creates label vector as same length as signal, creates label vector of object's session

        Parameters:
        t (np.array): Time vector of signal that needs to labeled

        Returns:
        labels (np.array): Labels (semantic classes) of each sample  
        """
        labels = np.full(t.shape, np.nan)
        for start, end, label in zip(self.session['start'], self.session['end'], self.session['labels']):
            mask = (t >= start) & (t <= end)
            labels[mask] = label

        return labels