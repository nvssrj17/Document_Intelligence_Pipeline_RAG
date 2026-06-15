import ollama
from config import OLLAMA_MODEL

class LocalLLM:
    def __init__(self):
        self.model = OLLAMA_MODEL

    def generate(self, prompt):
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response ["message"]["content"]
    
        