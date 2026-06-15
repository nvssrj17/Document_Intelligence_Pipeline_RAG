from rag_pipeline import RAGPipeline

rag = RAGPipeline()
while True:
    query = input("Enter your question: ")
    if query.lower() == "exit":
        break
    result = rag.answer_query(query)
    print("\nAnswer:\n", result["answer"])
    print("\nSources:", result["sources"])
