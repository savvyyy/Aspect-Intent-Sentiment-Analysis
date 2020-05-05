import pandas as pd
import datetime, csv, os, json
import os.path
from os import path

from utils.functions import sentimentData, getAspectTwitter, getPolarity, getTwitterData

def absa(hashTagSubject):
    print('hashTagSubject', hashTagSubject)
    public_tweets_path = os.getcwd() + '/' + hashTagSubject + '.csv'
    if path.exists(public_tweets_path):
        print('hai')
        
        data = pd.read_csv(os.path.realpath(public_tweets_path))
        data["tweet"] = data["tweet"].astype(str)
        data["cleaned_tweet"] = data["cleaned_tweet"]
        data["Polarity"] = data["tweet_without_url"].apply(getPolarity)
        data['Sentiment'] = data.apply(sentimentData, axis=1)
        data["Aspects"] = data["cleaned_tweet"].apply(getAspectTwitter)
        

        aspect_list = data.to_dict(orient='records')
        return aspect_list

    else:
        print('nai hai')
        public_tweets_file = getTwitterData(hashTagSubject)
        public_tweets_path = public_tweets_file + '/' + hashTagSubject + '.csv'
        
        data = pd.read_csv(os.path.realpath(public_tweets_path))
        data["tweet"] = data["tweet"].astype(str)
        data["cleaned_tweet"] = data["cleaned_tweet"]
        data["Polarity"] = data["tweet_without_url"].apply(getPolarity)
        data['Sentiment'] = data.apply(sentimentData, axis=1)
        data["Aspects"] = data["tweet_without_url"].apply(getAspectTwitter)
        
        aspect_list = data.to_dict(orient='records')
        return aspect_list