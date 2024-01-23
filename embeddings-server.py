from flask import Flask, jsonify, request
from flask_cors import CORS

import embed;

# create the api endpoint
app = Flask(__name__)
CORS(app)

@app.route('/get-embedding', methods=['POST', 'GET'])
def get_embedding():
    data = request.json
    text = data['text']
    
    embedding = embed.create_embeddings(text)
    return jsonify(embedding.tolist())

if __name__ == '__main__':
    app.run(debug=True)