from flask import Flask,request,jsonify
from inference import get_model
from config import *
app = Flask(__name__)

model = get_model()


@app.route("/predict", methods=['POST'])
def predict():
    context = request.json["context"]
    attribute = request.json["attribute"]
    try:
        out = model.predict(context, attribute)
        return jsonify({"result": out})
    except Exception as e:
        return jsonify({"result": "Model Failed"})


if __name__ == "__main__":
    app.run('0.0.0.0', port=PORT_NUMBER)

