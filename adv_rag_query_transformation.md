# Query Transformations in RAG Systems - Technical Summary

## Core Problem Statement

Naive RAG systems face three fundamental limitations:

1. **Content Relevance**: Document chunks often contain irrelevant information that degrades retrieval quality
2. **Query Quality**: User questions are frequently poorly worded or structured for optimal retrieval
3. **Structural Mismatch**: Natural language queries need conversion to structured formats (SQL, metadata filters, etc.)

## What is Query Transformation?

Query transformation modifies the user's question **before** it reaches the embedding model, rather than passing it directly to the vectorstore. This isn't new in search engines (query expansion has existed for years), but LLMs now enable sophisticated, context-aware transformations.

**Standard RAG Flow:**
```
User Question → Embedding Model → Vector Search → Top-k Documents
```

**Query Transformation Flow:**
```
User Question → LLM Transformation → Embedding Model → Vector Search → Top-k Documents
```

---

## Key Techniques

### 1. **Rewrite-Retrieve-Read**

**Concept**: Use an LLM to rewrite the user's query into a more optimal form for retrieval.

**Why it works**: Original queries aren't always optimal for semantic search. A rewrite can clarify intent, add context, or restructure the question.

**Example:**
- Original: "What's that thing about climate?"
- Rewritten: "What are the primary causes and effects of climate change?"

**Prompt strategy**: Simple rewrite instruction that maintains intent while improving clarity.

**Implementation consideration**: Single-pass transformation; adds one LLM call to pipeline.

---

### 2. **Step-Back Prompting**

**Concept**: Generate a higher-level, more abstract "step back" question alongside the original question. Use both for retrieval.

**Why it works**: Abstract questions retrieve foundational knowledge, while specific questions get detailed information. Combined retrieval provides both context and specifics.

**Example:**
- Original: "Why does lithium-ion battery degrade after 500 cycles?"
- Step-back: "What are the principles of lithium-ion battery operation?"

Both questions retrieve documents, and combined results provide comprehensive grounding.

**Use cases**: 
- Complex technical questions requiring foundational understanding
- Questions with implicit prerequisites
- Multi-level reasoning tasks

**Implementation consideration**: Requires two separate retrievals that are then merged.

---

### 3. **Follow-Up Questions (Conversational Context)**

**Problem**: In multi-turn conversations, follow-up questions reference previous context.

**Three approaches:**

1. **Embed only follow-up**: Loses context
   - User: "What can I do in Italy?"
   - Follow-up: "What type of food is there?"
   - Problem: "there" has no reference

2. **Embed entire conversation**: May retrieve irrelevant results from unrelated prior turns
   - Wastes retrieval on off-topic history

3. **LLM query transformation** (recommended): Generate a standalone question from conversation history
   - Transforms: "What type of food is there?" → "What types of food are available in Italy?"

**Key engineering point**: Prompt design is critical. You need to balance:
- Preserving essential context
- Removing distracting information
- Maintaining query intent

**Production pattern**: This is standard in chat-based RAG applications (e.g., WebLangChain uses this approach).

---

### 4. **Multi-Query Retrieval**

**Concept**: Generate multiple related sub-queries from a single complex question, execute all retrievals in parallel, and combine results.

**Why it works**: Complex questions often require information from multiple angles or subtopics.

**Example:**
- Original: "Who won a championship more recently, the Red Sox or the Patriots?"
- Generated sub-queries:
  - "When did the Red Sox last win a championship?"
  - "When did the Patriots last win a championship?"

**Implementation**: 
- LLM generates 3-5 diverse queries
- Parallel retrieval for all queries
- Results combined (union or intersection depending on use case)

**Benefits**:
- Covers multiple aspects of complex questions
- Reduces risk of missing relevant information
- Handles ambiguous queries better

---

### 5. **RAG-Fusion**

**Concept**: Extension of Multi-Query Retrieval that uses **Reciprocal Rank Fusion (RRF)** to intelligently reorder retrieved documents.

**How it differs from Multi-Query**:
- Multi-Query: Retrieves from all queries and passes everything forward
- RAG-Fusion: Retrieves from all queries, then uses RRF to rerank based on which documents appear in multiple result sets

**Reciprocal Rank Fusion (RRF)**:
```python
# Simplified concept
for each document:
    score = sum(1 / (k + rank_in_query_i)) for all queries where doc appears
# Higher score = document is relevant to multiple sub-queries
```

**Why it works**: Documents appearing in multiple retrieval results are likely more relevant to the overall question.

**Example**:
If a document appears at rank 2 for query1 and rank 3 for query2, it gets higher combined score than documents only appearing in one query.

---

## Interview Notes / Specialist Notes

### **Conceptual Understanding**

1. **Why transform queries?**
   - User queries are natural language, often ambiguous or underspecified
   - Embedding models work better with clear, well-structured text
   - Single queries may not capture all information needs

2. **Trade-offs**:
   - **Latency**: Each transformation adds an LLM call (50-500ms typically)
   - **Cost**: Extra LLM calls increase operational costs
   - **Accuracy**: Can improve retrieval quality by 20-40% in practice
   - **Complexity**: More moving parts, harder to debug

3. **When to use which technique?**
   - **Rewrite**: Simple queries that are poorly worded
   - **Step-back**: Complex technical questions needing foundational context
   - **Follow-up transformation**: Always use in conversational RAG
   - **Multi-query**: Questions requiring multiple pieces of information
   - **RAG-Fusion**: When you need highest accuracy and can afford latency/cost

### **Implementation Details**

1. **Prompt Engineering is Critical**:
   - Each technique lives or dies by its prompt
   - Test prompts extensively with diverse queries
   - Use LangSmith or similar for prompt versioning

2. **Caching Strategies**:
   - Cache transformed queries for repeated questions
   - Consider semantic caching (similar questions share transformations)

3. **Evaluation Metrics**:
   - **Retrieval Recall**: Are relevant documents retrieved?
   - **Retrieval Precision**: Are irrelevant documents filtered out?
   - **End-to-end accuracy**: Does it improve final answer quality?
   - **Latency**: P95, P99 response times

4. **Production Considerations**:
   - Run transformations async where possible
   - Have fallback to direct retrieval if transformation fails
   - Monitor transformation quality (bad transformations = bad retrieval)

### **Common Interview Questions**

**Q: "Why not just use better embedding models instead of query transformation?"**

A: Better embeddings help, but they don't solve query quality issues. A poorly worded question is poorly worded regardless of embedding model. Transformations leverage LLM reasoning to improve the query itself. Also, embeddings can't handle multi-part questions or conversational context effectively.

**Q: "How do you handle the latency of extra LLM calls?"**

A: Several strategies:
- Use faster models for transformations (e.g., Claude Haiku instead of Opus)
- Parallelize multiple transformations (Multi-Query, RAG-Fusion)
- Cache transformed queries
- Stream responses to user while transformation happens
- Consider async processing for non-critical paths

**Q: "How do you evaluate if a query transformation is working?"**

A: Multi-level evaluation:
1. **Transformation quality**: Is the transformed query better than original? (human eval)
2. **Retrieval metrics**: Does it retrieve more relevant documents? (recall@k, MRR)
3. **End-to-end**: Does the final answer improve? (accuracy, relevance scores)
4. **A/B testing**: Compare with/without transformation in production

**Q: "What if the LLM generates a bad transformation?"**

A: Implement safety mechanisms:
- Confidence scoring on transformations
- Retrieve with both original and transformed queries
- Use multiple transformation strategies and vote
- Have human-in-the-loop for high-stakes applications
- Monitor and alert on transformation failures

### **System Design Perspective**

For a production RAG system with query transformation:

```python
# Conceptual architecture
def advanced_rag_pipeline(user_query, conversation_history=None):
    # 1. Choose transformation strategy based on query type
    strategy = select_strategy(user_query)
    
    # 2. Transform query (potentially multiple)
    transformed_queries = transform_query(
        user_query, 
        conversation_history, 
        strategy
    )
    
    # 3. Parallel retrieval
    all_docs = []
    for query in transformed_queries:
        docs = vectorstore.similarity_search(query)
        all_docs.extend(docs)
    
    # 4. Rerank if using RAG-Fusion
    if strategy == "rag_fusion":
        all_docs = reciprocal_rank_fusion(all_docs)
    else:
        all_docs = deduplicate(all_docs)
    
    # 5. Generate answer with context
    answer = llm.generate(user_query, context=all_docs)
    
    return answer
```

### **Key Takeaway**

Query transformation is a **pre-retrieval optimization technique** that uses LLMs to improve query quality before semantic search. It's most effective when:
- User queries are ambiguous or poorly structured
- Questions require multiple pieces of information
- Conversational context is important
- You can afford the latency/cost trade-off

The technique shifts from "retrieve better" to "ask better questions" - a fundamental mindset change in RAG system design.
