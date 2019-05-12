from flask_restful import Resource
from tools.predict import Lane
lane = None


def setup(api_key, **kwargs):
    global lane
    lane = Lane(api_key, **kwargs)


class Cnn(Resource):
    global lane

    def get(self, name_or_id):
        try:
            prediction = lane.predict(name_or_id)
        except Exception as e:
            print('get: ' + str(e))
            return {'message': f'An error occurred by predicting {name_or_id}'}, 500
        if prediction:
            return prediction, 200
        return {'message': f'Nothing to predict for {name_or_id} not found'}, 404
