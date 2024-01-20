# EEG Sentiment Classification

---

## Problem Definition

The goal of this project is to classify raw EEG signals based on the sentiment of a video the participant is watching. The classification involves five emotional classes: disgust, fear, sadness, neutral state, and happiness. The objective is to identify relevant features for training machine and deep learning models to achieve accurate sentiment classification.

---

## Dataset

For model training, the SEED-V database is utilized. The SEED-V database is a valuable resource for researchers in the field of emotional state analysis through electroencephalography (EEG). It encompasses five emotional classes: happiness, sadness, disgust, fear, and neutrality. The database consists of EEG signals recorded using a 62-channel NCI NeuroScan system at a sampling rate of 1 kHz. It includes data from 16 participants, each exposed to short clips in three sets of 15 clips, with each clip corresponding to one of the mentioned emotions. Access to the database requires a license, available at [SEED-V Database Access](https://bcmi.sjtu.edu.cn/home/seed/downloads.html#seed-v-access-anchor).

---

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository.
2. Obtain a license for the SEED-V database.
3. Download the necessary data from [SEED-V Database](https://bcmi.sjtu.edu.cn/home/seed/downloads.html#seed-v-access-anchor).
4. Install the required dependencies using `pip install -r requirements.txt`.

---

## Usage

In folder data create folder raw_eeg_cnt_files and there store all .cnt files downloaded from link above. Folder structure should look like this:

data/
├── dataset_1000/
│   └── ... (contents of dataset_1000)
├── dataset_2000/
│   └── ... (contents of dataset_2000)
├── eeg_features_1000/
│   └── ... (contents of eeg_features_1000)
├── eeg_features_2000/
│   └── ... (contents of eeg_features_2000)
├── raw_eeg_cnt_files/
│   └── ... (contents of raw_eeg_cnt_files)
├── spectrograms/
│   └── ... (contents of spectrograms)
└── session_time_stamps.json

---

## Acknowledgments

Giving credit to Shanghai Jiao Tong University for providing us a Dataset.

