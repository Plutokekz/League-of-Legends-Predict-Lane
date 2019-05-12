from flask import Flask
from flask_restful import Api
from resources.predict import Cnn, setup


app = Flask(__name__)
api = Api(app)
setup("Api-Key")


api.add_resource(Cnn, '/predict/<name_or_id>')


if __name__ == '__main__':
    app.run(port=5000)
