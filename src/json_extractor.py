import json
from llm import LocalLLM

class JSONExtractor:
    def __init__(self):
        self.llm = LocalLLM()

    def extract(self, context, schema):
        prompt = f"Extract the following information from the context:\n\nContext: {context}\n\nSchema: {json.dumps(schema)}\n\nProvide the extracted information in JSON format."
        response = self.llm.generate(prompt)
        try:
            extracted_data = json.loads(response)
            return extracted_data
        except json.JSONDecodeError:
            raise ValueError("Failed to parse the response as JSON. Response was: " + response)