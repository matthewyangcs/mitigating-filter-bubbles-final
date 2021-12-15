# mitigating-filter-bubbles-final

If you wish to run the data pre-processing, then you must download the 100k_a.csv file located from(https://cseweb.ucsd.edu//~jmcauley/datasets.html#twitch) and put it in twitch_preprocessing/raw_data/

Next, use `conda env create -f environment.yml` to create the environment.

Then, you can run all of our analysis and experiments via the notebooks.

twitch_preprocessing/ contains the following notebooks:
 1. Explores all of the data and performs experiments on optimizing community generation
 2. Generates the final communities used and additional pre-processing

recsys-filterbubbles/ contains:
 1. Preprocessing all of the data to the input format of the LSTM
 2. Trains a baseline model, explores and verifies the filter bubble effect
 3. Buckets all of the training and validation data into the categories mentioned in our paper (diverse, filter bubble, breaking bubble, etc.)
 4. Performs training data modification and retrains models, then evaluates on the final test set
 5. Performs TracIn experiments on the categories of items
 6. Performs TracIn experiments on self-influence
