import json
import time
import sys
from pathlib import Path

import numpy as np
# Ensure the repo root is importable when this file is run directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.rag_pipeline import RAGPipeline

class AnswerEvaluator:
    def __init__(self):
        self.rag = RAGPipeline()
        self.embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')

    def compute_similarity(self, reference_answer, generated_answer):
        ref_embedding = np.asarray(
            self.embedding_model.encode(reference_answer, normalize_embeddings=True)
        ).reshape(1, -1)
        gen_embedding = np.asarray(
            self.embedding_model.encode(generated_answer, normalize_embeddings=True)
        ).reshape(1, -1)
        similarity = cosine_similarity(ref_embedding, gen_embedding)[0][0]
        return float(similarity)
    
    def evaluate(self, dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        results = []
        total_similarity = 0
        total_generation_time = 0

        for idx, item in enumerate(dataset):
            question = item["question"]
            reference_answer = item["reference_answer"]
            print(f"Evaluating question {idx + 1}/{len(dataset)}")

            start = time.time()
            rag_result = self.rag.answer_query(question)
            generation_time = time.time() - start
            generated_answer = rag_result["answer"]

            similarity = self.compute_similarity(reference_answer, generated_answer)
            total_similarity += similarity

            results.append({
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "similarity": round(similarity, 4),
                "sources": rag_result["sources"],
                "generation_time": round(generation_time, 2)
            })
            total_generation_time += generation_time

        average_similarity = total_similarity / len(dataset) if dataset else 0
        average_generation_time = total_generation_time / len(dataset) if dataset else 0

        summary = {
            "average_similarity": round(average_similarity, 4),
            "average_generation_time": round(average_generation_time, 2),
            "total_questions": len(dataset),
            "results": results
        }

        return summary
    
if __name__ == "__main__":
    evaluator = AnswerEvaluator()
    report = evaluator.evaluate("evaluation/questions.json")
    Path("evaluation").mkdir(exist_ok=True)
    with open("evaluation/results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Evaluation completed. Results saved to evaluation/results.json")
    print(f"Average Similarity: {report['average_similarity']}")
    print(f"Average Generation Time: {report['average_generation_time']} seconds")