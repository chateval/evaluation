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

def get_automatic_evaluations():
    model_responses = json.loads(request.get_json())['model_responses']
    baseline_responses = json.loads(request.get_json())['baseline_responses']
    
    automatic_evaluations = dict()
    automatic_evaluations['avg_len'] = 1.*avg_len(model_responses)
    automatic_evaluations['distinct_1'] = 1.*distinct_1(model_responses)
    automatic_evaluations['distinct_2'] = 1.*distinct_2(model_responses)
    automatic_evaluations['greed_match'] = 1.*greedy_match(model_responses, baseline_responses, w2v)[0]
    automatic_evaluations['extrema_score'] = 1.*extrema_score(model_responses, baseline_responses, w2v)[0]
    automatic_evaluations['average_embedding_score'] = 1.*average_embedding_score(model_responses, baseline_responses, w2v)[0]

    return automatic_evaluations

@app.route("/auto", methods=['GET', 'POST'])
def auto():
    return json.dumps(get_automatic_evaluations())

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8001, debug=True)
