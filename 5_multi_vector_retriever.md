# Multi-Vector Retriever for Semi-Structured & Multi-Modal RAG - Technical Summary

## Core Problem Statement

Standard RAG works well for plain text but struggles with real-world documents that contain:

1. **Semi-structured data**: Tables mixed with text (e.g., financial reports, research papers)
2. **Multi-modal content**: Images, diagrams, charts alongside text
3. **Complex layouts**: Information distributed across different content types

Traditional chunking strategies often **split tables**, **ignore images**, or **lose structural context**, leading to poor retrieval and answer quality.

---

## The Multi-Vector Retriever Concept

### **Core Idea: Decoupling**

The multi-vector retriever decouples two things:

1. **What you retrieve with** (reference/summary) - optimized for semantic search
2. **What you pass to the LLM** (raw document) - complete information for answer synthesis

**Why this matters**: You can create search-optimized representations (summaries, descriptions) while preserving full context for the LLM.

### **Simple Example**

```
Document: [10-page verbose technical report]

Traditional RAG:
- Chunk the 10 pages into 50 chunks
- Embed all 50 chunks
- Retrieve top-5 chunks
- Problem: Context fragmented, may miss important info

Multi-Vector RAG:
- Create 1 concise summary of the report
- Embed the summary for retrieval
- Store reference to FULL 10-page report
- When summary matches query → return entire report to LLM
- Benefit: Easy retrieval + complete context
```

### **Architecture**

```
┌──────────────────────────────────────────────┐
│              Document Processing              │
├──────────────────────────────────────────────┤
│                                               │
│  PDF/Doc → Parse → Extract:                  │
│                    ├─ Text blocks             │
│                    ├─ Tables                  │
│                    └─ Images                  │
└──────────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────┐
│          Create Search References             │
├──────────────────────────────────────────────┤
│  Text    → Summary/Chunk                     │
│  Table   → Text summary of table             │
│  Image   → Text description (via MLLM)       │
└──────────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────┐
│            Dual Storage System                │
├──────────────────────────────────────────────┤
│  Vector Store:    Doc Store:                 │
│  - Summaries      - Raw text                 │
│  - Descriptions   - Original tables          │
│  - Embeddings     - Original images          │
└──────────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────┐
│              Retrieval + Synthesis            │
├──────────────────────────────────────────────┤
│  Query → Semantic Search (vectors)           │
│       → Retrieve IDs                         │
│       → Fetch RAW content from docstore      │
│       → Pass to LLM with full context        │
└──────────────────────────────────────────────┘
```

---

## Document Loading with Unstructured

**Tool**: [Unstructured.io](https://unstructured.io) - specialized library for document parsing

**Capabilities**:
- Extracts tables, images, text from PDFs, Word docs, PPTs, etc.
- Uses layout models (YOLOX) to detect bounding boxes
- Identifies document structure (titles, sections, captions)
- Handles multiple file formats

**Processing Pipeline for PDFs**:
1. Remove embedded image blocks
2. Detect tables using layout models
3. Identify sections via title detection
4. Aggregate text under each section
5. Chunk text based on configurable parameters
6. Return structured elements by type

**Output Example**:
```python
elements = [
    {'type': 'Title', 'text': 'Introduction'},
    {'type': 'NarrativeText', 'text': 'This paper discusses...'},
    {'type': 'Table', 'data': <table_html>},
    {'type': 'Image', 'data': <image_bytes>},
    {'type': 'NarrativeText', 'text': 'As shown in Figure 1...'}
]
```

---

## Semi-Structured RAG (Tables + Text)

### **The Problem**

Tables contain structured information that doesn't work well with:
- Standard text chunking (may split table across chunks)
- Direct embedding (structure gets lost)
- Semantic search (querying about "Q3 revenue" doesn't match table cells)

### **The Solution**

1. **Extract tables** using Unstructured
2. **Generate text summaries** of each table using LLM
3. **Embed summaries** in vector store
4. **Store raw tables** in docstore with references
5. **At retrieval**: If table summary matches query → return raw table to LLM

### **Example**

```python
# Original Table
| Quarter | Revenue | Growth |
|---------|---------|--------|
| Q1 2023 | $45M    | 12%    |
| Q2 2023 | $52M    | 15%    |
| Q3 2023 | $61M    | 17%    |

# Generated Summary (stored in vector DB)
"This table shows quarterly revenue for 2023, 
with Q1 at $45M (12% growth), Q2 at $52M (15% growth), 
and Q3 at $61M (17% growth), demonstrating consistent 
revenue acceleration throughout the year."

# User Query
"What was the revenue growth in Q3 2023?"

# Retrieval Flow
1. Query embedding matches table summary
2. Retrieve table ID from vector store
3. Fetch RAW table from docstore
4. Pass original table to LLM
5. LLM extracts: "17% growth, $61M revenue"
```

### **Why This Works**

- **Summaries**: Natural language, semantically rich, good for retrieval
- **Raw tables**: Structured, precise, complete information for LLM
- **Best of both worlds**: Easy to find + accurate to use

---

## Multi-Modal RAG (Text + Tables + Images)

### **Three Approaches**

#### **Option 1: Multimodal Embeddings**

**Technique**: Use models like CLIP that embed both images and text into the same vector space

**Flow**:
```
Text/Images → CLIP encoder → Unified embedding space
           ↓
    Retrieve via similarity search
           ↓
    Link to raw images in docstore
           ↓
    Pass raw images + text to multimodal LLM (GPT-4V)
```

**Pros**:
- Joint retrieval across modalities
- Preserves visual information
- No need to generate text from images

**Cons**:
- Requires multimodal embeddings (not all providers support)
- Need multimodal LLM for answer generation (expensive)
- CLIP quality varies for domain-specific images

---

#### **Option 2: Image-to-Text Summarization (No Image Retrieval)**

**Technique**: Convert images to text summaries, embed as text, exclude images from final synthesis

**Flow**:
```
Images → Multimodal LLM → Text summaries → Text embeddings
       ↓
    Retrieve text summaries
       ↓
    Pass only text + tables to TEXT-ONLY LLM
       (images excluded from synthesis)
```

**Use Case**: When you can't use multimodal LLMs for synthesis (cost, latency, privacy)

**Example**:
```python
# Image: Graph showing stock price trends
# Generated summary:
"Line chart depicting AAPL stock price from Jan-Dec 2023, 
showing steady growth from $130 to $190, with notable dip 
in May to $145 before recovering."

# User query: "How did Apple stock perform in 2023?"
# Retrieved: Text summary only
# LLM synthesis: Using text-only model (GPT-3.5, Claude Sonnet)
```

**Pros**:
- Can use cheaper text-only LLMs
- Easier to deploy (no multimodal infrastructure)
- Better privacy (no raw images sent to LLM)

**Cons**:
- Loses visual details
- Summary quality depends on image-to-text model
- Can't handle complex visual reasoning

---

#### **Option 3: Image Summarization + Image Retrieval** (Recommended)

**Technique**: Best of both worlds - text summaries for retrieval, raw images for synthesis

**Flow**:
```
Images → Multimodal LLM → Text summaries
       ↓
    Embed summaries in vector store
       ↓
    Store RAW images in docstore with references
       ↓
    Query matches summary → Retrieve image ID
       ↓
    Fetch raw image from docstore
       ↓
    Pass raw image to multimodal LLM for synthesis
```

**Why This Is Best**:
- Text summaries: Easy to search with standard embeddings
- Raw images: Full visual information preserved for accurate answers
- Flexible: Can use text embeddings (cheaper) but still leverage multimodal LLMs

**Example Workflow**:
```python
# 1. Image Processing
chart_image = load_image("sales_chart.png")
summary = gpt4v.summarize(chart_image)
# Summary: "Bar chart showing regional sales..."

# 2. Storage
vector_id = embed(summary)
docstore[vector_id] = chart_image  # Store raw image

# 3. Retrieval
query = "What were the regional sales differences?"
matching_id = vector_search(query)  # Matches summary
raw_image = docstore[matching_id]   # Get original

# 4. Synthesis
answer = gpt4v.generate(query, context=[raw_image, text_chunks])
```

**Pros**:
- High accuracy (full visual information)
- Good retrieval (text summaries work well with standard embeddings)
- Flexible LLM choice (can upgrade/downgrade as needed)

**Cons**:
- Requires multimodal LLM for both summarization and synthesis
- Higher cost than text-only approaches
- More complex pipeline

---

## Practical Implementation Details

### **Image Summarization with LLaVA**

The blog demonstrates using **LLaVA 7B** (open-source multimodal model):

**Hardware Requirements**:
- Mac M2 Max, 32GB RAM
- ~45 tokens/sec
- Can run locally for privacy

**Example Output**:
```python
# Input: Image of fried chicken arranged like a world map
# LLaVA Output:
"The image features a close-up of a tray filled with 
various pieces of fried chicken. The chicken pieces are 
arranged in a way that resembles a map of the world, 
with some pieces placed in the shape of continents and 
others as countries."
```

**Quality Observations**:
- Captures visual humor and creativity
- Reasonable accuracy for most images
- Good enough for retrieval purposes
- May miss fine details for technical diagrams

---

### **Fully Local/Private Pipeline**

For privacy-sensitive applications, entire pipeline can run locally:

```
Component Stack:
├─ Document Parsing: Unstructured (local)
├─ Image Summarization: LLaVA 7B (local via llama.cpp)
├─ Text Embeddings: GPT4All (local)
├─ Vector Store: Chroma (local)
├─ Multi-Vector Retriever: LangChain (local)
└─ Answer Generation: LLaMA2-13B (local via Ollama)
```

**Use Cases**:
- Healthcare (HIPAA compliance)
- Legal (confidentiality)
- Enterprise (proprietary data)
- Government (classified information)

**Trade-offs**:
- Lower accuracy than GPT-4V/Claude
- Higher latency (45 tokens/sec vs 100+)
- Requires powerful hardware (32GB+ RAM)
- But: Complete data privacy

---

## Comparison with Other RAG Techniques

| Technique | What It Optimizes | Multi-Vector Retriever Position |
|-----------|-------------------|--------------------------------|
| **Base RAG** | Simple chunking + embedding | Baseline |
| **Summary Embedding** | Uses summaries for retrieval | ✅ Uses this approach |
| **Windowing** | Retrieves chunks, returns expanded context | Similar concept, different implementation |
| **Metadata Filtering** | Pre-filters by structured metadata | Complementary technique |
| **Fine-tuned Embeddings** | Better semantic matching | Complementary technique |
| **2-stage RAG** | Keyword + semantic search | Orthogonal approach |

**Multi-Vector Retriever's Unique Value**:
- Handles heterogeneous data types (text, tables, images)
- Decouples retrieval optimization from synthesis optimization
- Flexible: Works with any content type that can be summarized

---

## Interview Notes / Specialist Notes

### **Core Concepts to Master**

1. **The Decoupling Principle**:
   - Retrieval representation ≠ Synthesis representation
   - Optimize each independently
   - Bridge via reference IDs

2. **Why Not Just Embed Everything?**:
   - Tables: Structure gets lost in embeddings
   - Images: Visual information can't be captured by text embeddings
   - Long documents: Chunking loses context
   - Multi-Vector: Preserves structure + context

3. **Storage Architecture**:
   ```
   Vector Store: Fast semantic search (summaries/descriptions)
      ↕ (linked by IDs)
   Doc Store: Full fidelity storage (raw content)
   ```

### **Implementation Considerations**

#### **1. When to Use Which Option?**

| Scenario | Recommended Option |
|----------|-------------------|
| Cost-sensitive, text-only LLM | Option 2 (no image retrieval) |
| Privacy-critical, local deployment | Option 3 with open models |
| Highest accuracy needed | Option 3 (summary + raw images) |
| Already using CLIP embeddings | Option 1 (multimodal embeddings) |

#### **2. Summary Quality is Critical**

```python
# Bad summary (too generic)
"This is a table showing financial data."

# Good summary (specific, searchable)
"Quarterly revenue table for 2023 showing growth from 
$45M in Q1 to $61M in Q3, with YoY growth rates of 
12%, 15%, and 17% respectively."
```

**Tips for Good Summaries**:
- Include key numbers/facts
- Mention trends and patterns
- Add context (time periods, comparisons)
- Use domain-specific terminology
- Keep length 2-5 sentences

#### **3. Chunking Strategy for Mixed Documents**

```python
# Document structure
sections = [
    {'type': 'text', 'content': '...', 'id': 'text_1'},
    {'type': 'table', 'content': <table>, 'id': 'table_1'},
    {'type': 'text', 'content': '...', 'id': 'text_2'},
    {'type': 'image', 'content': <img>, 'id': 'img_1'}
]

# Processing strategy
for section in sections:
    if section['type'] == 'text':
        # Normal chunking
        chunks = chunk_text(section['content'])
        for chunk in chunks:
            vector_store.add(embed(chunk), ref=chunk)
    
    elif section['type'] == 'table':
        # Summarize table
        summary = llm.summarize_table(section['content'])
        vector_store.add(embed(summary), ref=section['content'])
    
    elif section['type'] == 'image':
        # Describe image
        description = mllm.describe(section['content'])
        vector_store.add(embed(description), ref=section['content'])
```

#### **4. Retrieval Strategy**

**Challenge**: How many documents to retrieve from each type?

```python
# Strategy 1: Fixed allocation
results = {
    'text': retrieve(query, k=5),
    'tables': retrieve(query, k=2),
    'images': retrieve(query, k=3)
}

# Strategy 2: Dynamic allocation (better)
all_results = retrieve(query, k=10)  # Mixed types
# Rerank by relevance score
top_results = rerank(all_results)[:7]
```

**Best Practice**: Use hybrid approach with minimum guarantees per type

---

### **Common Interview Questions**

**Q: "Why not just use OCR to extract text from images and tables?"**

A: Several reasons:
1. **OCR loses structure**: Tables become unstructured text
2. **Visual information lost**: Charts, diagrams contain patterns OCR can't capture
3. **Quality issues**: OCR errors in handwriting, low-res images, complex layouts
4. **Context loss**: Spatial relationships and visual emphasis matter

Multi-vector approach preserves structure and visual information while enabling retrieval.

**Q: "How do you handle cases where summaries don't match user queries?"**

A: Several mitigation strategies:
1. **Multiple summaries per item**: Generate 2-3 diverse summaries emphasizing different aspects
2. **Hierarchical summaries**: Short (1 sentence) + medium (2-3 sentences) + long (5+ sentences)
3. **Keyword extraction**: Add explicit keywords to summaries
4. **Fallback to metadata**: Use traditional metadata filtering as backup
5. **Query expansion**: Generate multiple query variations (ties back to Query Transformations blog)

**Q: "What's the latency overhead of this approach?"**

A: Break down by component:
```
Traditional RAG: ~200ms
├─ Embedding: 50ms
├─ Vector search: 50ms
└─ LLM generation: 100ms

Multi-Vector RAG: ~700ms
├─ Embedding: 50ms
├─ Vector search: 50ms
├─ Docstore fetch: 100ms  ← Additional overhead
├─ Image loading: 200ms    ← For images
└─ LLM generation: 300ms   ← More context = longer generation
```

**Optimization strategies**:
- Cache frequently accessed raw documents
- Lazy load images (only when needed)
- Compress images before storage
- Use faster docstore (Redis vs disk)
- Parallel retrieval for multiple types

**Q: "How do you evaluate multi-modal RAG quality?"**

A: Multi-level evaluation:

1. **Component-level**:
   - Summary quality: Human eval of summaries vs original
   - Retrieval recall: Are relevant items found?
   - Synthesis quality: Does LLM use retrieved content correctly?

2. **End-to-end**:
   - Answer accuracy: Correct factual responses
   - Completeness: Uses all relevant modalities
   - Hallucination rate: Doesn't invent information

3. **Modality-specific**:
   - Table questions: Numerical accuracy
   - Image questions: Visual reasoning correctness
   - Cross-modal: "Explain the trend in Figure 3 and compare with Table 2"

**Q: "Can you combine this with other RAG techniques?"**

A: Absolutely! Multi-Vector Retriever is composable:

```python
# Combine with Query Transformation
transformed_query = rewrite(user_query)

# Combine with Metadata Filtering  
filtered_results = multi_vector_retrieve(
    transformed_query,
    metadata_filter={'date': '2023', 'dept': 'finance'}
)

# Combine with Reranking
reranked = cohere_rerank(filtered_results)

# Combine with Hypothetical Document Embeddings (HyDE)
hyde_doc = generate_hypothetical_answer(query)
hyde_results = multi_vector_retrieve(hyde_doc)
```

**Q: "What about video content?"**

A: Extension of the same principles:
1. **Frame extraction**: Sample keyframes from video
2. **Frame summarization**: Use multimodal LLM on keyframes
3. **Transcript extraction**: Speech-to-text for audio
4. **Temporal indexing**: Add timestamp metadata
5. **Multi-vector storage**: Summaries in vector store, video clips in docstore

Challenge: Much higher storage and processing requirements.

---

### **System Design Perspective**

**Production Architecture**:

```python
class MultiVectorRAG:
    def __init__(self):
        self.parser = Unstructured()
        self.image_summarizer = LLaVA()  # or GPT-4V
        self.table_summarizer = LLM()
        self.vector_store = ChromaDB()
        self.doc_store = MongoDB()  # For binary data
        self.embedder = OpenAIEmbeddings()
        self.llm = GPT4()
        
    def ingest_document(self, file_path):
        # 1. Parse
        elements = self.parser.partition(file_path)
        
        # 2. Process each element type
        for elem in elements:
            if elem.type == 'Table':
                summary = self.table_summarizer(elem.data)
                doc_id = self.doc_store.save(elem.data)
                vec_id = self.vector_store.add(
                    self.embedder.embed(summary),
                    metadata={'doc_id': doc_id, 'type': 'table'}
                )
            
            elif elem.type == 'Image':
                description = self.image_summarizer(elem.data)
                doc_id = self.doc_store.save(elem.data)
                vec_id = self.vector_store.add(
                    self.embedder.embed(description),
                    metadata={'doc_id': doc_id, 'type': 'image'}
                )
            
            elif elem.type == 'Text':
                chunks = chunk_text(elem.data)
                for chunk in chunks:
                    self.vector_store.add(
                        self.embedder.embed(chunk),
                        metadata={'content': chunk, 'type': 'text'}
                    )
    
    def query(self, question):
        # 1. Retrieve
        results = self.vector_store.similarity_search(
            self.embedder.embed(question),
            k=10
        )
        
        # 2. Fetch full content
        context = []
        for result in results:
            if result.metadata['type'] in ['table', 'image']:
                doc_id = result.metadata['doc_id']
                full_content = self.doc_store.fetch(doc_id)
                context.append(full_content)
            else:
                context.append(result.metadata['content'])
        
        # 3. Generate answer
        answer = self.llm.generate(
            question,
            context=context
        )
        
        return answer
```

---

### **Key Takeaways**

1. **Fundamental Principle**: Decouple what you search with from what you synthesize with

2. **When to Use**:
   - Documents contain tables, images, charts
   - Need to preserve structure and visual information
   - Long documents where chunking loses context
   - Privacy-sensitive scenarios (can run fully local)

3. **Three Multimodal Options**:
   - Option 1: Multimodal embeddings (CLIP)
   - Option 2: Image-to-text, text-only synthesis
   - Option 3: Image summaries for retrieval, raw images for synthesis (best)

4. **Critical Success Factors**:
   - High-quality summaries/descriptions
   - Efficient docstore for binary data
   - Proper ID linking between vector store and docstore
   - Good retrieval strategy across multiple content types

5. **Production Considerations**:
   - Latency overhead is real (~3-4x slower)
   - Storage requirements increase (raw + summaries)
   - More complex to debug and monitor
   - But: Significantly better accuracy for mixed-content documents

This approach represents a paradigm shift from "chunk everything" to "represent intelligently, retrieve precisely, synthesize completely."
