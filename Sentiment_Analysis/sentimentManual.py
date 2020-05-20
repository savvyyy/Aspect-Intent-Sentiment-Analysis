from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def sentimentData(text, source):
    polarity = calculatePolarity(text)
    data = []
    if polarity < -0.5 and polarity > -1:
        sentiment = 'Very Negative'
    elif polarity < 0 and polarity > -0.5:
        sentiment = 'Negative'
    elif polarity == 0 or polarity == 0.0:
        sentiment = 'Neutral'
    elif polarity > 0 and polarity <= 0.5:
        sentiment = 'Positive'
    elif polarity > 0.5 and polarity <= 1:
        sentiment = 'Very Positive'
    data.append({
        'polarity': polarity,
        'sentiment': sentiment
    })
    return {
        'text' : data,
        'source' : source
    }


def calculatePolarity(text):
    sent_analyser = SentimentIntensityAnalyzer()
    return sent_analyser.polarity_scores(text)['compound']