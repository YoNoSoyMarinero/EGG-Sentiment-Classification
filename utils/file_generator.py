import os

def eeg_file_generator(folder_path):
    """
    Creates list of paths of RawEEG folder from EEGRaw folder

    Paramters:
    folder_path(str): RawEEG
    """

    paths = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            paths.append(file_path)

    return paths