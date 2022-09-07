import useModel
from flask import Flask, jsonify, request

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    input = request.get_json(silent=True, force=True)
    output = useModel.predict_emo(input['emotions'])

    if input == None:
        return jsonify({'code': '400'})

    return jsonify({'code': '200', 'result': output})

if __name__ == "__main__":
    app.run()
