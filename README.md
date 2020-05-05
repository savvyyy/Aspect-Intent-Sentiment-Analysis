# Sentiment Analysis

Sentiment analysis refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information.



## Cloning

```bash
git clone https://github.com/savvyyy/Sentiment-Analysis.git
```



## Installation

```bash
1. sudo chmod +x ./setup.sh
2. Run ./setup.sh
```
## Usage
Change .env.example to .env
```
Run python main.py 
```
main.py will start pointing to 127.0.0.1:5000. Run this url in postman to see the result.

1. Endpoint '/getSentiment' is an Api that provides Sentiment Analysis.
2. Endpoint '/absa' is an Api that provides Abstract Based Sentiment Analysis.
3. Endpoint '/intent' is an Api that provides Intent Based Sentiment Analysis(need more work).
4. Endpoint '/graph' is an Api that provides Date wise tweets and their polarity to plot in a graph to study the variations in tweets.

## Description

1. ```Sentiment Analysis :-``` Sentiment analysis refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. It returns Polarity,relevance of subjectivity and Sentiment(Very Positive, Positive, Neutral, Negative, Very Negative ).

2. ```Aspect Sentiment :-``` Aspect-based sentiment analysis is a text analysis technique that breaks down text into aspects (attributes or components of a product or service), and then allocates each one a sentiment level (positive, negative or neutral).
Here’s a breakdown of what aspect-based sentiment analysis can extract :- a) Sentiments: positive or negative opinions about a particular aspect.
b) Aspects: the thing or topic that is being talked about.

3. ```Intent Analysis :-```Intent Analysis acknowledges the intentions from the text. It can be any intentions such as the intention to sell, or intention to complain or intention to purchase etc.

## UI
Refer to ```https://github.com/savvyyy/Sentiment-Analysis-UI.git``` 