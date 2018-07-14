import json
from flask import Flask, request, jsonify
from pymagnitude import Magnitude
from auto import avg_len, distinct_1, distinct_2, greedy_match, extrema_score, average_embedding_score

class Word2Vec:
    def __init__(self, vectors):
        self.vectors = vectors
        self.layer1_size = self.vectors.dim
    
    def __getitem__(self, word):
        return self.vectors.query(word)
    
    def __contains__(self, word):
        return word in self.vectors
    
    def dim(self):
        return self.vectors.dim
 
app = Flask(__name__)
vectors = Magnitude('vectors.magnitude')
w2v = Word2Vec(vectors)

@app.route("/avg_len", methods=['GET', 'POST'])
def avg_len_route():
    model_responses = json.loads(request.get_json())['model_responses']
    value = avg_len(model_responses)
    return jsonify({'name': 'Average Length', 'value': value})

@app.route("/distinct_1", methods=['GET', 'POST'])
def distinct_1_route():
    model_responses = json.loads(request.get_json())['model_responses']
    value = distinct_1(model_responses)
    return jsonify({'name': 'Distinct 1', 'value': value})

@app.route("/distinct_2", methods=['GET', 'POST'])
def distinct_2_route():
    model_responses = json.loads(request.get_json())['model_responses']
    value = distinct_2(model_responses)
    return jsonify({'name': 'Distinct 2', 'value': value})

@app.route("/greedy_match", methods=['GET', 'POST'])
def greedy_match_route():
    model_responses = json.loads(request.get_json())['model_responses']
    baseline_responses = json.loads(request.get_json())['baseline_responses']
    value = greedy_match(model_responses, baseline_responses, w2v)
    return jsonify({'name': 'Greedy Match', 'value': value})

@app.route("/extrema_score", methods=['GET', 'POST'])
def extrema_score_route():
    model_responses = json.loads(request.get_json())['model_responses']
    baseline_responses = json.loads(request.get_json())['baseline_responses']
    value = extrema_score(model_responses, baseline_responses, w2v)
    return jsonify({'name': 'Extrema Score', 'value': value})

@app.route("/average_embedding_score", methods=['GET', 'POST'])
def average_embedding_score_route():
    model_responses = json.loads(request.get_json())['model_responses']
    baseline_responses = json.loads(request.get_json())['baseline_responses']
    value = average_embedding_score(model_responses, baseline_responses, w2v)
    return jsonify({'name': 'Average Embedding Score', 'value': value})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)