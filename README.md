# DS 4002 Project 1

## Software and Platform
We ran the scripts using VSCode using Windows on an NVIDIA GPU. We also used the following Python packages that can be installed using pip:
- matplotlib
- numpy
- pandas
- pathlib
- seaborn
- sklearn
- torch
- tqdm
- transformers

## Documentation Map
DS4002PROJ
- `README.md`: This file
- `LICENSE`: Description of the license for this project
- 📁 Data Folder: contains data used throughout the pipeline
  - `METADATA.md`: Description of the data being used
  - `yelp_sample_250k.csv`: Sample of 250k reviews (for improved processing speeds)
  - `yelp_with_sentiment.csv`: Reviews with analyzed sentiment labels/strengths, output from sentiment script
- 📁 Output Folder
  - [insert description of plots here]
- 📁 Scripts Folder
  - `0_processing.py`: Initial processing script to generate exploratory plots
  - `1_sentiment.py`: Sentiment analysis script, taking in raw Yelp reviews and outputting the reviews with their associated sentiments
  - `2_classify.py`: Classification script, taking in reviews with associated sentiments and training/testing a decision tree based on that data

## How to Reproduce Results
[In this section, you should give explicit step-by-step instructions to reproduce the Results of your study. These instructions should be written in straightforward plain English, but they must be concise, but detailed and precise enough, to make it possible for an interested user to reproduce your results without much difficulty.]