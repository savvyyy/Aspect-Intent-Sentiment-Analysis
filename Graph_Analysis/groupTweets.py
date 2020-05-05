from datetime import datetime
import re
import os.path
from os import path
import pandas as pd
from utils.functions import getTwitterData, getPolarity

remove_ms = lambda x:re.sub("\+\d+\s","",x)
make_date = lambda x:datetime.strptime(remove_ms(x), "%a %b %d %H:%M:%S %Y")
cleaned_date_format = lambda x:"{:%Y-%m-%d %H:%M:%S:%f}".format(make_date(x))

def createGroup(hashTagSubject):
    print('hashTagSubject',hashTagSubject)
    public_tweets_path = os.getcwd() + '/' + hashTagSubject + '.csv'

    if path.exists(public_tweets_path):
        print('hai')
        public_tweets =  pd.read_csv(os.path.realpath(public_tweets_path))
        tweetText = []
        polarity = []
        for i in range(0, len(public_tweets)):
            tweetText.append({
                'tweet' : public_tweets['tweet'][i],
                'date' : cleaned_date_format(public_tweets['created_at'][i])
            })
        sortedTweets = sorted(tweetText,key=lambda x: x['date'], reverse=False)

        for tweet in sortedTweets:
            polarity.append({
                'date' : tweet['date'],
                'polarity' : getPolarity(tweet['tweet'])
            })
    
        return polarity

    else:
        print('nai hai')
        public_tweets_file = getTwitterData(hashTagSubject)
        public_tweets_path = public_tweets_file + '/' + hashTagSubject + '.csv'
        public_tweets =  pd.read_csv(os.path.realpath(public_tweets_path))

        tweetText = []
        polarity = []
        for i in range(0, len(public_tweets)):
            tweetText.append({
                'tweet' : public_tweets['tweet'][i],
                'date' : cleaned_date_format(public_tweets['created_at'][i])
            })
        sortedTweets = sorted(tweetText,key=lambda x: x['date'], reverse=False)

        for tweet in sortedTweets:
            polarity.append({
                'date' : tweet['date'],
                'polarity' : getPolarity(tweet['tweet'])
            })
    
        return polarity