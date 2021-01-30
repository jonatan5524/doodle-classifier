from flask import Flask
from flask import render_template
from flask_cors import CORS, cross_origin
from flask import request
from model import Model

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = Model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load')
@cross_origin()
def load():
    model.load()

    return "loaded"

@app.route('/capture', methods=['POST'])
@cross_origin()
def capture():
    data = request.stream.read()
    data = data.decode("utf-8").split(',')
    
    return model.predict(data)


if __name__ == "__main__":
    app.run()