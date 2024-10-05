# RAG

## Evolution
![RAG types][RAGEvolution]
1. Naive RAG
**Components**:
- Indexing: preprocessing and embedding in Vector DB.
- Retrieval: same process and return most similar documents.
- Generation: prompting all together
**Drawbacks**:
- Retrieval Challenges: The retrieval phase often struggles
with precision and recall, leading to the selection of misaligned
or irrelevant chunks, and the missing of crucial information.
- Generation Difficulties: hallucinations and mixing irrelevant information.

2. Advanced RAG
[Visual Article](https://pub.towardsai.net/advanced-rag-techniques-an-illustrated-overview-04d193d8fec6)
Pre-retrieval and post-retrieval strategies:
**pre-retrieval**:
- Enhancing data granularity, optimizing index structures, adding metadata,
alignment optimization, and mixed retrieval.
- Query rewritting or expansion.
**Post- retrieval**: Integrate correctly with the query:
- Rerank chunks and context compressing
- Re-ranking the retrieved information to relocate the most relevant content to the edges of the prompt is a key strategy: LlamaIndex or Langchain implement them 
3. Modular RAG:
- **New Modules**   :
Specified components: Ex: Several retrieval indexes and the system decides which one to use.
RAG-fusion: Combine the previous results, obtained in parallel, into one context with the distilled information of all sources.
- **New Patterns**:
Rewrite-Retrieve-Read (arXiv:2305.14283, 2023), Generate-Read ( arXiv:2209.10063, 2022, Paper Filippo), Recitation-Read: emphasizes retrieval from model weights (arXiv:2210.01296, 2022.)
sub-queries (smaller and more specific queries) and hypothetical document embeddings (LLM generated labels like possible queries)
 Demonstrate-Search-Predict, Retrieve-Read-Retrieve-Read flow of ITERRETGEN: boost modules with the output of another module. 
Flexible orchestation: FLARE (arXiv:2305.06983, 2023) and Self-Rag (self reflection Retrieve  arXiv:2310.11511, 2023)




[RAGEvolution]: DifferentRags.png