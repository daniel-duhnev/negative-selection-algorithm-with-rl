# Study of the Performance of Negative Selection Algorithm (NSA) with Enhancement via Reinforcement Learning (RL)

This repository contains the implementation and experimental setup for studying how Reinforcement Learning can enhance the Negative Selection Algorithm's performance in anomaly detection tasks.

## Overview

This research explores the application of NSA and RL across three different domains:
- Unix system call sequences
- Twitter sentiment analysis
- AG News classification

## Repository Structure

```
├── data/
│   ├── syscalls/         
│   └── twitter-sentiment/ 
├── scripts/
│   ├── NSA Implementation/
│   │   ├── classifier_nsa_twitter.py
│   │   ├── classifier_nsa_unix.py
│   │   └── classifier_nsa_and_rl_agnews.py
│   ├── RL Enhancement/
│   │   ├── classifier_rl_twitter.py
│   │   ├── classifier_rl_unix.py
│   │   └── classifier_rl_unix_feature_plots.py
│   └── results/          
└── Final_Revised_Project_Report.pdf  
```

## Running Experiments

The python scripts can be executed using:
```bash
python <script-name>
```
Example:
```bash
python classifier_rl_unix.py
```

### Configuration
- Parameters can be adjusted within each script

## Datasets

1. **Unix System Calls**: Located in `data/syscalls/`
   - Pre-processed sequences for anomaly detection
   - Includes training and test sets with labels

2. **Twitter Sentiment**: Located in `data/twitter-sentiment/`
   - Sentiment analysis dataset
   - Pre-processed for binary classification

3. **AG News**: Accessed via Hugging Face
   - Automatically downloaded during script execution
   - Requires internet connection

## Example Output
Below is an example of a successful run using the Twitter dataset:
![Screenshot 2025-06-14 165056](https://github.com/user-attachments/assets/8a372bc0-6968-4a5d-8d48-b5f37e18e04d)
