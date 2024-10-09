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
Flexible orchestation: FLARE (methods that actively decide when and what to retrieve across the course of the generation) (arXiv:2305.06983, 2023: https://arxiv.org/abs/2305.06983) and **Self-Rag** (Es necesario lo que he recuperado? Como podría combinarlo todo de manera que las ideas queden más claras?) (self reflection Retrieve  arXiv:2310.11511, 2023: https://arxiv.org/abs/2310.11511)

## Retrieval:
### Type o data to retrieve:
- Unstructured: Text
- Semi-structured: Table (SQL)
- Structured: Knowledge graph -> Harder but better understanding
- LLMs-Generated Content: GenRead -> Generate instead of retrieve

### Data Granularity:
- Small: Can lose context and semantic meaning.
- Big: Redundant or not important text that can distract the LLM

## Indexing:

1. Chunking strategy:
- Fixed size
- Small2Big: sentences (small) are used as the retrieval unit, and the preceding and following sentences are provided as (big) context to LLMs.
2. Metadata Attachments:
- artificially constructed: HyDE
3. Hierarchical Indexing: Higher to Smaller chunks with summaries at each node.
4. **Knowledge Graph index**



## Query Enhancing:

### Query Expansion
Query expansion involves broadening a single query into multiple queries to add context and ensure optimal relevance. This process includes several approaches that enhance the quality of the generated answers.

- **Multi-Query**: Using prompt engineering with LLMs, a query is expanded into several parallel queries. These are carefully designed, not random, to increase the likelihood of accurate responses.
  
- **Sub-Query**: This involves decomposing a complex question into simpler sub-questions. The least-to-most prompting method is used to address these sub-questions, allowing for a more complete and contextual answer.

- **Chain-of-Verification (CoVe)**: After expansion, the queries are validated by LLMs to reduce hallucinations and improve reliability, ensuring that the expanded queries provide more trustworthy results.

### Query Transformation::

- Query Rewrite: LLM rewrites -> recall enhancing
- HyDE: Create hypothetical chunks that would be recoverend
### Query Routing:
Metadata/Semantic Router/Filter

## Embedding:

1. Mix/hybrid Retrieval: 
Sparse retrieval:  initial search then Dense retrieval: Specifically, it also demonstrates that sparse retrieval models can enhance the zero-shot retrieval capability of dense retrieval models and assist dense retrievers in handling queries containing rare entities, thereby improving robustness

2. Fine-tuning Embedding Model:
Context significantly deviates from pre-training corpus: Technical words like legal practice
Promptagator: LLM as a few-shot query generator to create task-specific retrievers (the queries)
LLM-Embedder: LLM generates rewards.
RLHF: Human feedback. Not practical.


### Adapter:

UPRISE: Retrieve prompts from a pool (https://arxiv.org/abs/2303.08518).
Augmentation-Adapted Retriever/BGM: FineTune a small LM to adapt the query with the retrieve sources.



## Generation:



[RAGEvolution]: DifferentRags.png





# Consejos juristas:

1. Identificar Rama del derecho y filtrar fuentes.

2. Traducción a lenguaje jurista- Ex: Discapacitado -> Persona con discapacidad

3. Personas jurídicas (empresas)
Personas físicas (Persona en sí)
Entidades sin personalidad jurídica (Comunidad de bienes:  un grupo de amigos que compran conjuntamente una propiedad para alquilarla.)

Constitución (muy ambigua para resistir sin cambios)

Derecho internacional (Normativa Unión Europea) Pacta Sunt Servanda (Si firmas un un convenio internacional)
Tiene que cumplirlo pero quien lo regula.

**Está en vigor?**: 
- BOE: Disposición Derogada

**Ámbito de aplicación**: Al principio generalmente (un artículo puede concretar)

**Definiciones**: Hay definiciones de los términos más extaños