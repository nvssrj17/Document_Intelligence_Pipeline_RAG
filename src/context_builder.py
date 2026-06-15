class ContextBuilder:
    def build_context(self, retrieved_docs):
        context = []
        sources = []
        for item in retrieved_docs:
            doc = item[0] if isinstance(item, tuple) else item
            page = doc['page']
            context.append(f"[Page {page}]\n{doc['text']}")
            sources.append(page)
        return "\n\n".join(context), sorted(list(set(sources)))