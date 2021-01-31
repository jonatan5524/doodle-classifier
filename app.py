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
    """ Index route of the server

    Returns:
        Text: The index.html file
    """
    return render_template('index.html')

@app.route('/load')
@cross_origin()
def load():
    """ Loads the model from the last checkpoint

    Returns:
        Str: Loaded approval
    """
    model.load()

    return "loaded"

@app.route('/capture', methods=['POST'])
@cross_origin()
def capture():
    """ Predict the current drawing of the user

    Returns:
        Str: The model prediction
    """
    data = request.stream.read()
    data = data.decode("utf-8").split(',')
    
    return model.predict(data)


if __name__ == "__main__":
    app.run()