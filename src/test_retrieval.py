from retrieve import Retriever

query = input("Ask Question: ")

results = Retriever().retrieve(query)
for idx, result in enumerate(results):
   
   doc = result[0]
   score = result[1]

   print("\n")
   print("=" * 60)

   print(f"Result {idx+1}")
   print(f"Page: {doc['page']}")
   print(f"Score: {score:.4f}")
   print(doc["text"][:500])