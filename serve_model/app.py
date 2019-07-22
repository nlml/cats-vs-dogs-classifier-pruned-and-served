from flask import Flask, url_for
from flask import request
from flask import json


app = Flask(__name__)


class Model():
    def __init__(self):
        self.model = 1

    def classify_json_data(self, data):
        return {'result': 'cat', 'confidence': 0.95}


model = Model()


@app.route('/')
def api_root():
    return 'Welcome'


@app.route('/classify', methods=['POST'])
def api_message():

    if request.headers['Content-Type'] == 'text/plain':
        return "Text Message: " + request.data

    elif request.headers['Content-Type'] == 'application/json':
        result = model.classify_json_data(request.json)
        return json.dumps(result)

    elif request.headers['Content-Type'] == 'application/octet-stream':
        with open('./binary', 'wb') as f:
            f.write(request.data)
        return "Binary message written!"

    else:
        return "415 Unsupported Media Type ;)"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
