from flask import Flask, jsonify, make_response
from flask_restful import Api, Resource, reqparse
from Sentiment_Analysis.sentiment import sentimentAnalysis
from Aspect_Twitter.absa import absa
from intent import intentPrediction
from Graph_Analysis.groupTweets import createGroup
from flask_cors import CORS
from Aspect_Analysis.aspect import getAspect, Model
from Sentiment_Analysis.sentimentManual import sentimentData, calculatePolarity

app = Flask(__name__)
CORS(app, support_credentials=True)
api = Api(app)

class SentimentAnalysisResult(Resource):
    def __init__(self):
        pass

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('text', type=str, required=True)
        parser.add_argument('source', type=str, required=True)
        args = parser.parse_args()

        if args.source == 'twitter':
            return sentimentAnalysis('#'+args.text, args.source), 200
        else:
            return sentimentData(args.text, args.source), 200

# class SentimentAnalysisAll(Resource):
#     def __init__(self):
#         pass

#     def get(self):
#         parser = reqparse.RequestParser()
#         parser.add_argument('hashtag', type=str, required=True)
#         parser.add_argument('source', type=str, required=True)
#         args = parser.parse_args()

#         if args.source == 'twitter':
#             return sentimentAnalysisAllTweets('#'+args.hashtag), 200
#         else:
#             return {'error'}

# class AspectSentimentAnalysis(Resource):
#     def __init__(self):
#         pass

#     def get(self):
#         parser = reqparse.RequestParser()
#         parser.add_argument('hashtag', type=str, required=True)
#         args = parser.parse_args()

#         return absa('#'+args.hashtag), 200

class AspectSentimentAnalysisDomainSpecific(Resource):
    def __init__(self):
        pass

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('source', type=str, required=True)
        parser.add_argument('text', type=str, required=True)
        args = parser.parse_args()

        if args.source == 'twitter':
            return absa('#'+args.text), 200
        else:
            data = getAspect(args.source, args.text)
            print('dataaaaaa', data)
            return data, 200

class IntentSentimentAnalysis(Resource):
    def __init__(self):
        pass

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('hashtag', type=str, required=True)
        args = parser.parse_args()

        return intentPrediction('#'+args.hashtag), 200


class PlotGraphApi(Resource):
    def __init__(self):
        pass

    def get(self):
        parser = reqparse.RequestParser()
        # parser.add_argument('source', type=str, required=True)
        parser.add_argument('hashtag', type=str, required=True)
        args = parser.parse_args()

        return createGroup('#'+args.hashtag), 200

        # if args.source == 'twitter':
        #     return createGroup('#'+args.hashtag), 200
        # else:
        #     return {'error'}

api.add_resource(SentimentAnalysisResult, '/getSentiment')
# api.add_resource(SentimentAnalysisAll, '/getSentimentAll')
# api.add_resource(AspectSentimentAnalysis, '/absaTwitter')
api.add_resource(AspectSentimentAnalysisDomainSpecific, '/aspect/domain')
api.add_resource(IntentSentimentAnalysis, '/intent')
api.add_resource(PlotGraphApi, '/graph')


if __name__ == "__main__":
    app.run(debug=True)