This is the dataset containing tweets from parody and real accounts. If you use this data please cite the paper: A. Maronikolakis, D. Sanchez Villegas, D. Preoţiuc-Pietro and N. Aletras (2020). Analyzing Political Parody in Social Media. In ACL.

To comply with Twitter's policy (no sharing of more than 50k tweets at once), we have split the data into three parts (data_split1.csv, data_split2.csv, data_split3.csv). Here we only publish the first split, if you require the rest you can contact us. Each row represents one tweet, containing {person, account, tweet_id, tweet_pp, tweet_raw, label} data.

person: Name of politician (eg. Boris Johnson)
account: Twitter account name (eg. @BorisJohnson)
tweet_id: The id of the given tweet
tweet_pp: The tweet text, pre-processed
tweet_raw: The tweet text, in its original form
label: Whether the tweet is parody (0) or real (1)

Alongside the csv files, we publish json files containing tweets merged by person. Each row in the json files represents a given person, with all their real and parody tweets. The files are also split in three, to comply with Twitter policy.

All the tweet ids are provided in tweet_ids.txt, where each line contains one tweet_id and the corresponding label. These can be used to retrieve the original tweet, alongside any other information you may require. Note that some of the tweets may have been deleted or set to private.

The data for each experiment is provided in each one separate folder (gender/location/persons). In each folder, subfolders denote the different setups for each experiment. In these subfolders, the tweet ids and labels for the train/dev/test sets are provided. The original text/information can be extracted from the csv files, or from querying the Twitter API.
