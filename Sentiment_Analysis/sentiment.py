import csv
import datetime
import os.path
from os import path
import pandas as pd
from utils.functions import calculatePolarity, getTwitterData

# cwd = os.getcwd()
# print(cwd)


# def sentimentAnalysisAllTweets(hashTagSubject):
#     print('hashTagSubject',hashTagSubject)
#     data = []
#     sentimentTweetAll = []
#     polarityAll = []
#     tweetTextAll = []
#     userAll = []
#     createdAtAll = []
#     public_tweets_path = os.getcwd() + '/' + hashTagSubject + '.csv'
#     if path.exists(public_tweets_path):
#         print('hai')
#         public_tweets = pd.read_csv(os.path.realpath(public_tweets_path))
#         for orig_tweet in public_tweets['tweet']:
#             tweetTextAll.append(orig_tweet)
#         for orig_tweet in public_tweets['userName']:
#             userAll.append(orig_tweet)
#         for orig_tweet in public_tweets['created_at']:
#             createdAtAll.append(orig_tweet)
#         for tweet in public_tweets['tweet_without_url']:
#             polarity_of_tweet = calculatePolarity(tweet)
#             polarityAll.append(polarity_of_tweet)
#             if polarity_of_tweet < -0.5 and polarity_of_tweet > -1:
#                 sentimentTweetAll.append('Very Negative')
#             elif polarity_of_tweet < 0 and polarity_of_tweet > -0.5:
#                 sentimentTweetAll.append('Negative')
#             elif polarity_of_tweet == 0 or polarity_of_tweet == 0.0:
#                 sentimentTweetAll.append('Neutral')
#             elif polarity_of_tweet > 0 and polarity_of_tweet <= 0.5:
#                 sentimentTweetAll.append('Positive')
#             elif polarity_of_tweet > 0.5 and polarity_of_tweet <= 1:
#                 sentimentTweetAll.append('Very Positive')
#             else:
#                 raise ValueError('sentiment error')
#         for i in range(0, len(tweetTextAll)):
#             data.append({
#                 'tweet' : tweetTextAll[i],
#                 'polarity' : polarityAll[i],
#                 'sentiment' : sentimentTweetAll[i],
#                 'username' : userAll[i],
#                 'created_at' : createdAtAll[i]
#             })
#         return data
#     else:
#         print('nai hai')
#         public_tweets_file = getTwitterData(hashTagSubject)
#         public_tweets_path = public_tweets_file + '/' + hashTagSubject + '.csv'
#         public_tweets =  pd.read_csv(os.path.realpath(public_tweets_path))
#         for orig_tweet in public_tweets['tweet']:
#             tweetTextAll.append(orig_tweet)
#         for tweet in public_tweets['tweet_without_url']:
#             polarity_of_tweet = calculatePolarity(tweet)
#             polarityAll.append(polarity_of_tweet)
#             if polarity_of_tweet < -0.5 and polarity_of_tweet > -1:
#                 sentimentTweetAll.append('Very Negative')
#             elif polarity_of_tweet < 0 and polarity_of_tweet > -0.5:
#                 sentimentTweetAll.append('Negative')
#             elif polarity_of_tweet == 0 or polarity_of_tweet == 0.0:
#                 sentimentTweetAll.append('Neutral')
#             elif polarity_of_tweet > 0 and polarity_of_tweet <= 0.5:
#                 sentimentTweetAll.append('Positive')
#             elif polarity_of_tweet > 0.5 and polarity_of_tweet <= 1:
#                 sentimentTweetAll.append('Very Positive')
#             else:
#                 raise ValueError('sentiment error')
#         for i in range(0, len(tweetTextAll)):
#             data.append({
#                 'tweet' : tweetTextAll[i],
#                 'polarity' : polarityAll[i],
#                 'sentiment' : sentimentTweetAll[i],
#                 'username' : userAll[i],
#                 'created_at' : createdAtAll[i]
#             })
#         return data
        



def sentimentAnalysis(hashTagSubject, source):
    print('hashTagSubject',hashTagSubject)
    polarity= []
    totalCount = 0
    tweetText = []
    dataAll = []
    sentimentTweetAll = []
    polarityAll = []
    tweetTextAll = []
    userAll = []
    createdAtAll = []
    public_tweets_path = os.getcwd() + '/' + hashTagSubject + '.csv'
    if path.exists(public_tweets_path):
        print('hai')
        public_tweets =  pd.read_csv(os.path.realpath(public_tweets_path))
        for orig_tweet in public_tweets['tweet']:
            tweetTextAll.append(orig_tweet)
        for orig_tweet in public_tweets['userName']:
            userAll.append(orig_tweet)
        for orig_tweet in public_tweets['created_at']:
            createdAtAll.append(orig_tweet)
        for tweet in public_tweets['tweet_without_url']:
            polarity_of_tweet = calculatePolarity(tweet)
            polarityAll.append(polarity_of_tweet)
            if polarity_of_tweet < -0.5 and polarity_of_tweet > -1:
                sentimentTweetAll.append('Very Negative')
            elif polarity_of_tweet < 0 and polarity_of_tweet > -0.5:
                sentimentTweetAll.append('Negative')
            elif polarity_of_tweet == 0 or polarity_of_tweet == 0.0:
                sentimentTweetAll.append('Neutral')
            elif polarity_of_tweet > 0 and polarity_of_tweet <= 0.5:
                sentimentTweetAll.append('Positive')
            elif polarity_of_tweet > 0.5 and polarity_of_tweet <= 1:
                sentimentTweetAll.append('Very Positive')
            else:
                raise ValueError('sentiment error')
        for i in range(0, len(tweetTextAll)):
            dataAll.append({
                'tweet' : tweetTextAll[i],
                'polarity' : polarityAll[i],
                'sentiment' : sentimentTweetAll[i],
                'username' : userAll[i],
                'created_at' : createdAtAll[i]
            })
        for tweet in public_tweets['tweet_without_url']:
            totalCount = totalCount+1
            tweetText.append(tweet)
            data = calculatePolarity(tweet)
            polarity.append(data)

        if(totalCount > 0):
            Sum = sum(polarity)
            average = Sum/len(polarity)
            if(average > 0 and average < 0.5):
                # print("happy")
                return {
                    'average': average,
                    'sentiment': 'positive',
                    'text': dataAll,
                    'source' : source
                }
            elif(average > 0.5 and average <= 1):
                # print("very happy")
                return {
                    'average': average,
                    'sentiment': 'very positive',
                    'text': dataAll,
                    'source' : source
                }
            elif(average == 0 or average == 0.0):
                # print("Neutral")
                return {
                    'average': average,
                    'sentiment': 'Neutral',
                    'text': dataAll,
                    'source' : source
                }
            elif(average < 0 and average > -0.5):
                # print('Negative')
                return {
                    'average': average,
                    'sentiment': 'Negative',
                    'text': dataAll,
                    'source' : source
                }
            elif(average < -0.5 and average > -1):
                # print('Very Negative')
                return {
                    'average': average,
                    'sentiment': 'Very Negative',
                    'text': dataAll,
                    'source' : source
                }
        
        else:
            print("No Result Found")
            return {
                'result': 'No Result Found'
            }
    else:
        print('nai hai')
        public_tweets_file = getTwitterData(hashTagSubject)
        public_tweets_path = public_tweets_file + '/' + hashTagSubject + '.csv'
        public_tweets =  pd.read_csv(os.path.realpath(public_tweets_path))
        for orig_tweet in public_tweets['tweet']:
            tweetTextAll.append(orig_tweet)
        for orig_tweet in public_tweets['userName']:
            userAll.append(orig_tweet)
        for orig_tweet in public_tweets['created_at']:
            createdAtAll.append(orig_tweet)
        for tweet in public_tweets['tweet_without_url']:
            polarity_of_tweet = calculatePolarity(tweet)
            polarityAll.append(polarity_of_tweet)
            if polarity_of_tweet < -0.5 and polarity_of_tweet > -1:
                sentimentTweetAll.append('Very Negative')
            elif polarity_of_tweet < 0 and polarity_of_tweet > -0.5:
                sentimentTweetAll.append('Negative')
            elif polarity_of_tweet == 0 or polarity_of_tweet == 0.0:
                sentimentTweetAll.append('Neutral')
            elif polarity_of_tweet > 0 and polarity_of_tweet <= 0.5:
                sentimentTweetAll.append('Positive')
            elif polarity_of_tweet > 0.5 and polarity_of_tweet <= 1:
                sentimentTweetAll.append('Very Positive')
            else:
                raise ValueError('sentiment error')
        for i in range(0, len(tweetTextAll)):
            dataAll.append({
                'tweet' : tweetTextAll[i],
                'polarity' : polarityAll[i],
                'sentiment' : sentimentTweetAll[i],
                'username' : userAll[i],
                'created_at' : createdAtAll[i]
            })
        for tweet in public_tweets['tweet_without_url']:
            totalCount = totalCount+1
            tweetText.append(tweet)
            data = calculatePolarity(tweet)
            polarity.append(data)
    

        if(totalCount > 0):
            Sum = sum(polarity)
            average = Sum/len(polarity)
            if(average > 0 and average < 0.5):
                # print("happy")
                return {
                    'average': average,
                    'sentiment': 'positive',
                    'text': dataAll,
                    'source' : source
                }
            elif(average > 0.5 and average <= 1):
                # print("very happy")
                return {
                    'average': average,
                    'sentiment': 'very positive',
                    'text': dataAll,
                    'source' : source
                }
            elif(average == 0 or average == 0.0):
                # print("Neutral")
                return {
                    'average': average,
                    'sentiment': 'Neutral',
                    'text': dataAll,
                    'source' : source
                }
            elif(average < 0 and average > -0.5):
                # print('Negative')
                return {
                    'average': average,
                    'sentiment': 'Negative',
                    'text': dataAll,
                    'source' : source
                }
            elif(average < -0.5 and average > -1):
                # print('Very Negative')
                return {
                    'average': average,
                    'sentiment': 'Very Negative',
                    'text': dataAll,
                    'source' : source
                }
        
        else:
            print("No Result Found")
            return {
                'result': 'No Result Found'
            }