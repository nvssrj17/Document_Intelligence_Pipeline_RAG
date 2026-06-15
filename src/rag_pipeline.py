from retrieve import Retriever
from context_builder import ContextBuilder
from llm import LocalLLM

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.context_builder = ContextBuilder()
        self.llm = LocalLLM()

    def answer_query(self, query):
        retrieved_docs = self.retriever.retrieve(query)
        context, pages = self.context_builder.build_context(retrieved_docs)
        prompt = f"""
        You are answering questions about
        the Mahabharata.

        Use ONLY information present in
        the retrieved context.

        Answer in 2-4 concise sentences.

        If the answer is not found,
        say:
        'I could not find that information.'

        Context:
        {context}
        Question:
        {query}
        """
        answer = self.llm.generate(prompt)
        return {"answer": answer, "sources": pages}