from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = Flask(__name__)

class CVMatcher:
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)

    def segment_text(self, text, max_length=256):
        segments, current, count = [], [], 0
        for line in text.split('\n'):
            line = line.strip()
            if not line: continue
            current.append(line)
            count += len(line.split())
            if count >= max_length:
                segments.append(' '.join(current))
                current, count = [], 0
        if current:
            segments.append(' '.join(current))
        return segments

    def compute_similarity(self, cv_text, job_text):
        cv_chunks = self.segment_text(cv_text)
        job_chunks = self.segment_text(job_text)
        cv_emb = self.model.encode(cv_chunks, convert_to_tensor=True)
        job_emb = self.model.encode(job_chunks, convert_to_tensor=True)
        sim_matrix = util.cos_sim(cv_emb, job_emb).cpu().numpy()
        return float(np.max(sim_matrix)) * 100

    def rank_jobs(self, cv: str, jobs: dict):
        results = []
        for title, desc in jobs.items():
            score = self.compute_similarity(cv, desc)
            results.append((title, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

matcher = CVMatcher()

@app.route('/match', methods=['POST'])
def match():
    data = request.json
    cv_text = data.get('cv')
    jobs = data.get('jobs')
    
    if not cv_text or not jobs:
        return jsonify({"error": "Missing 'cv' or 'jobs' in request."}), 400

    ranked = matcher.rank_jobs(cv=cv_text, jobs=jobs)
    return jsonify(ranked)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

