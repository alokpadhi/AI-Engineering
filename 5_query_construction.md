# Query Construction - Technical Summary

## Core Problem Statement

Most real-world data has structure - whether it's fully structured (SQL databases), semi-structured (documents with tables), or unstructured with metadata (vector databases with filters). Traditional RAG approaches that rely solely on semantic similarity don't take advantage of this structure.

**The Challenge**: Users express queries in natural language, but data systems require specific query languages (SQL, Cypher, metadata filters, etc.). How do we bridge this gap?

**Query Construction = Converting natural language into database-specific query syntax**

---

## The Data Type Spectrum

### **Three Data Types and Their Characteristics**

1. **Structured Data**
   - **Storage**: SQL databases, Graph databases
   - **Characteristics**: Predefined schemas, organized in tables/relations
   - **Query Language**: SQL (relational), Cypher (graph)
   - **Strength**: Precise queries, exact matches, complex joins

2. **Semi-Structured Data**
   - **Storage**: Hybrid databases (e.g., PostgreSQL + pgvector)
   - **Characteristics**: Mix of structured (tables, columns) and unstructured (text, embeddings)
   - **Query Language**: SQL + semantic operators
   - **Strength**: Combines exact filtering with semantic search

3. **Unstructured Data**
   - **Storage**: Vector databases
   - **Characteristics**: No predefined model, but often has structured metadata
   - **Query Language**: Metadata filters + semantic search
   - **Strength**: Semantic similarity search

```
┌────────────────────────────────────────────────┐
│           Data Type Spectrum                   │
├────────────────────────────────────────────────┤
│                                                │
│  Structured ←→ Semi-Structured ←→ Unstructured│
│     SQL          SQL+Vector        VectorDB    │
│    Graph         pgvector          +metadata   │
│                                                │
└────────────────────────────────────────────────┘
```

---

## Why Query Construction Matters

**Example Query**: "What are movies about aliens in the year 1980?"

**Decomposition**:
- Semantic component: "aliens" (semantic search)
- Exact component: "year == 1980" (metadata filter)

**Traditional RAG approach** (naive):
```python
# Just embed the entire query
embedding = embed("movies about aliens in the year 1980")
results = vector_search(embedding, k=10)
# Problem: May return movies about aliens from 2020
```

**Query Construction approach**:
```python
# Parse query into structured components
parsed = {
    'semantic_query': 'aliens',
    'filter': 'year == 1980'
}
results = vector_search(
    embed('aliens'),
    filter={'year': 1980},
    k=10
)
# Better: Returns only alien movies from 1980
```

---

## Four Query Construction Techniques

---

## 1. Text-to-Metadata-Filter (Self-Query Retriever)

### **Use Case**
Vector databases with metadata filtering (Chroma, Pinecone, Weaviate, etc.)

### **The Problem**
User queries often contain both semantic content and logical filtering conditions mixed together.

### **How It Works**

**Step 1: Define Data Source**
```python
# Define what metadata fields exist
metadata_field_info = [
    AttributeInfo(
        name="artist",
        description="The artist who performed the song",
        type="string"
    ),
    AttributeInfo(
        name="length",
        description="Length of song in seconds",
        type="integer"
    ),
    AttributeInfo(
        name="genre",
        description="Music genre",
        type="string"
    )
]
```

**Step 2: User Query Interpretation**
```python
user_query = "Songs by Taylor Swift or Katy Perry about teenage romance under 3 minutes long in the dance pop genre"

# LLM decomposes into:
structured_query = {
    "query": "teenager love",  # Semantic search term
    "filter": "and(or(eq('artist', 'Taylor Swift'), eq('artist', 'Katy Perry')), lt('length', 180), eq('genre', 'pop'))"
}
```

**Step 3: Logical Condition Extraction**

Available operators:
- `eq`: equals
- `ne`: not equals
- `lt`: less than
- `lte`: less than or equal
- `gt`: greater than
- `gte`: greater than or equal
- `and`: logical AND
- `or`: logical OR
- `not`: logical NOT

**Step 4: Execute Structured Query**
```python
# The vectorstore understands both semantic and filter
results = vectorstore.similarity_search(
    query="teenager love",
    filter=parsed_filter,
    k=10
)
```

### **Implementation**

```python
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

# Define metadata schema
metadata_field_info = [
    AttributeInfo(name="genre", description="The genre of the movie", type="string"),
    AttributeInfo(name="year", description="The year the movie was released", type="integer"),
    AttributeInfo(name="director", description="The director of the movie", type="string"),
    AttributeInfo(name="rating", description="Rating out of 10", type="float")
]

document_content_description = "Brief summary of a movie"

# Create self-query retriever
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True
)

# Use it
results = retriever.get_relevant_documents(
    "I want sci-fi movies from the 1980s rated above 8"
)
```

### **Why This Works**

Traditional approach:
```
"movies from 1980s rated above 8"
    ↓
Single embedding
    ↓
Semantic search (may return movies from 1970s, or lower rated films)
```

Self-query approach:
```
"movies from 1980s rated above 8"
    ↓
LLM decomposition
    ↓
semantic: "movies" + filter: "year >= 1980 AND year < 1990 AND rating > 8"
    ↓
Precise results
```

---

## 2. Text-to-SQL

### **Use Case**
Querying relational databases (PostgreSQL, MySQL, SQLite, etc.)

### **Key Challenges**

1. **Hallucination**: LLMs invent fake table/column names
2. **User Errors**: Misspellings, ambiguous references
3. **Schema Complexity**: Large databases with many tables

### **Solution Strategies**

#### **Strategy 1: Database Description (Grounding)**

Provide LLM with accurate schema information using CREATE TABLE format:

```sql
-- Prompt includes:
CREATE TABLE movies (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    year INTEGER,
    director TEXT,
    genre TEXT
);

-- Plus sample rows:
SELECT * FROM movies LIMIT 3;
/*
id | title           | year | director        | genre
1  | The Matrix      | 1999 | Wachowskis      | Sci-Fi
2  | Inception       | 2010 | Nolan           | Thriller
3  | Interstellar    | 2014 | Nolan           | Sci-Fi
*/
```

**Why this works**: LLM sees exact column names, types, and sample data

#### **Strategy 2: Few-Shot Examples**

Include question-query pairs in the prompt:

```python
examples = [
    {
        "question": "How many movies were released in 2010?",
        "query": "SELECT COUNT(*) FROM movies WHERE year = 2010;"
    },
    {
        "question": "List all sci-fi movies by Christopher Nolan",
        "query": "SELECT title FROM movies WHERE director = 'Nolan' AND genre = 'Sci-Fi';"
    }
]
```

**Research shows**: Few-shot examples improve accuracy by 20-30%

#### **Strategy 3: Error Handling with SQL Agents**

```python
# Basic text-to-SQL (no error handling)
query = llm.generate_sql(user_question)
result = db.execute(query)  # May fail

# SQL Agent (with error handling)
agent = SQLAgent(llm, db)
result = agent.run(user_question)

# Agent workflow:
# 1. Generate SQL
# 2. Execute query
# 3. If error → analyze error
# 4. Regenerate fixed SQL
# 5. Retry (up to N times)
```

**Error Recovery Example**:
```
User: "Show me sales for Q1"
Agent generates: SELECT * FROM quartely_sales WHERE quarter = 1
Error: Table 'quartely_sales' does not exist

Agent sees error → checks schema → finds 'quarterly_sales'
Agent regenerates: SELECT * FROM quarterly_sales WHERE quarter = 1
Success!
```

#### **Strategy 4: Proper Noun Correction**

**Problem**: User types "Franc Sinatra" instead of "Frank Sinatra"

**Solution**: Use vectorstore for fuzzy matching
```python
# Step 1: Store all artist names in vectorstore
artist_vectorstore.add_documents([
    "Frank Sinatra",
    "Elvis Presley",
    "The Beatles",
    ...
])

# Step 2: When user queries with misspelling
user_input = "Franc Sinatra"
corrected = artist_vectorstore.similarity_search(user_input, k=1)
# Returns: "Frank Sinatra"

# Step 3: Use corrected name in SQL
query = f"SELECT * FROM songs WHERE artist = '{corrected[0]}'"
```

### **Complete Text-to-SQL Pipeline**

```python
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase

# 1. Connect to database
db = SQLDatabase.from_uri("postgresql://localhost/movies")

# 2. Create agent with tools
agent = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,
    agent_type="openai-tools"
)

# 3. Query in natural language
response = agent.run(
    "What are the top 5 highest-rated sci-fi movies from the 1990s?"
)

# Behind the scenes:
# - Examines database schema
# - Generates SQL: SELECT title, rating FROM movies 
#                  WHERE genre='Sci-Fi' AND year BETWEEN 1990 AND 1999 
#                  ORDER BY rating DESC LIMIT 5
# - Executes query
# - Formats results in natural language
```

---

## 3. Text-to-SQL+Semantic (Hybrid Approach)

### **Use Case**
Databases that support both structured data AND vector embeddings (PostgreSQL + pgvector, Supabase)

### **What is pgvector?**

Open-source PostgreSQL extension that adds vector similarity search:

```sql
-- Create table with embedding column
CREATE TABLE tracks (
    id SERIAL PRIMARY KEY,
    name TEXT,
    artist TEXT,
    genre TEXT,
    name_embedding VECTOR(1536)  -- OpenAI embedding dimension
);

-- Similarity search using <-> operator
SELECT * FROM tracks 
ORDER BY name_embedding <-> '[0.1, 0.2, ...]'::VECTOR
LIMIT 10;
```

**Operators**:
- `<->`: L2 distance (Euclidean)
- `<#>`: Negative inner product
- `<=>`: Cosine distance

### **Why This Is Powerful**

**Pure vectorstore limitations**:
```python
# Can't do: "Find the 3 saddest songs AND the 90th percentile sad song"
# Vectorstores only support top-k
```

**Pure SQL limitations**:
```python
# Can't do: Semantic search for "sadness" in song titles
# SQL requires exact matches
```

**pgvector combines both**:
```sql
-- Get saddest song
SELECT name FROM tracks 
ORDER BY name_embedding <-> :sadness_embedding 
LIMIT 1;

-- Get 90th percentile sad song
SELECT name FROM tracks 
ORDER BY name_embedding <-> :sadness_embedding 
OFFSET (SELECT COUNT(*) * 0.9 FROM tracks)::INT 
LIMIT 1;

-- Filter by genre AND semantic similarity
SELECT name FROM tracks
WHERE genre = 'pop'
ORDER BY name_embedding <-> :love_embedding
LIMIT 5;
```

### **Real-World Example**

**Scenario**: Album-song database

**Complex Query**: "Find albums containing the most songs matching sadness, from albums with 'lovely' in the title"

**Breaking it down**:
1. Semantic search: "sadness" (on song embeddings)
2. Semantic search: "lovely" (on album title embeddings)
3. SQL aggregation: COUNT songs per album
4. SQL filter: Albums where title matches "lovely"

**SQL+Semantic Query**:
```sql
SELECT 
    albums.title,
    COUNT(*) as sad_song_count
FROM albums
JOIN songs ON albums.id = songs.album_id
WHERE 
    -- Semantic filter on album title
    albums.title_embedding <-> :lovely_embedding < 0.3
    AND
    -- Semantic filter on song content
    songs.lyrics_embedding <-> :sadness_embedding < 0.4
GROUP BY albums.title
ORDER BY sad_song_count DESC
LIMIT 10;
```

**This is IMPOSSIBLE with**:
- Pure vectorstores (can't do aggregations, complex joins)
- Pure SQL (no semantic understanding)

### **When to Use pgvector**

✅ Use when:
- Need semantic search + structured filtering/aggregation
- Have relational data with text columns to embed
- Want single database for everything (simpler architecture)
- Cost-sensitive (cheaper than separate vectorstore)

❌ Don't use when:
- Pure semantic search (dedicated vectorstore faster)
- No relational structure (vectorstore sufficient)
- Need specialized vector features (HNSW, product quantization)

---

## 4. Text-to-Cypher (Graph Databases)

### **Use Case**
Knowledge graphs (Neo4j, Neptune, etc.)

### **Why Knowledge Graphs?**

**SQL database limitations**:
- Many-to-many relationships are awkward
- Schema changes are costly
- Hierarchical data hard to query

**Vectorstore limitations**:
- Don't understand relationships between entities
- No graph traversal
- Limited to similarity

**Knowledge graphs excel at**:
- Modeling complex relationships
- Graph traversal (find paths, neighborhoods)
- Flexible schema evolution
- Multi-hop reasoning

### **What is Cypher?**

Visual query language for graph databases:

```cypher
// Node pattern
(:Person {name: "Alice"})

// Relationship pattern
(:Person)-[:KNOWS]->(:Person)

// Complete pattern
(:Person {name: "Alice"})-[:KNOWS]->(:Person {name: "Bob"})

// Multi-hop
(:Person {name: "Alice"})-[:KNOWS*2..4]->(:Person)
// Finds people 2-4 hops away from Alice
```

**Key Concepts**:
- **Nodes**: Entities (Person, Movie, Company)
- **Relationships**: Connections (KNOWS, ACTED_IN, WORKS_FOR)
- **Properties**: Attributes on nodes/relationships
- **Labels**: Node types

### **Text-to-Cypher Pipeline**

```python
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph

# 1. Connect to graph database
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password"
)

# 2. Refresh schema (important!)
graph.refresh_schema()

# 3. Create chain with dedicated models
cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(model="gpt-4", temperature=0),  # For generating Cypher
    qa_llm=ChatOpenAI(model="gpt-3.5-turbo"),  # For natural language answer
    graph=graph,
    verbose=True
)

# 4. Query in natural language
response = cypher_chain.run(
    "Who are Alice's friends who work at the same company as Bob?"
)
```

**Behind the scenes**:
```cypher
// Generated Cypher query
MATCH (alice:Person {name: "Alice"})-[:KNOWS]->(friend:Person)
MATCH (bob:Person {name: "Bob"})-[:WORKS_FOR]->(company:Company)
MATCH (friend)-[:WORKS_FOR]->(company)
RETURN friend.name
```

### **Real-World Example: DevOps RAG**

**Schema**:
```cypher
(Ticket)-[:ASSIGNED_TO]->(Person)
(Ticket)-[:RELATED_TO]->(Service)
(Service)-[:DEPENDS_ON]->(Service)
(Person)-[:WORKS_IN]->(Team)
```

**Natural Language Query**: "How many open tickets are there?"

**Generated Cypher**:
```cypher
MATCH (t:Ticket {status: "open"})
RETURN COUNT(t) as open_tickets
```

**Complex Query**: "Show me all tickets affecting the payment service, including dependent services"

**Generated Cypher**:
```cypher
MATCH (t:Ticket)-[:RELATED_TO]->(s:Service {name: "payment"})
OPTIONAL MATCH (s)<-[:DEPENDS_ON*]-(dependent:Service)<-[:RELATED_TO]-(dep_ticket:Ticket)
RETURN t.id, t.title, COLLECT(DISTINCT dependent.name) as affected_services
```

### **Why Use GPT-4 for Cypher Generation?**

Cypher is more complex than SQL:
- Pattern matching syntax is intricate
- Multiple ways to express same query
- Easy to generate invalid patterns
- Performance implications of bad queries

**Recommendation**: Use GPT-4 or Claude Opus for Cypher generation, cheaper model for final answer synthesis.

---

## Comparison Table

| Technique | Data Type | Query Language | Best For | Complexity |
|-----------|-----------|----------------|----------|------------|
| **Text-to-Metadata-Filter** | Unstructured + metadata | Metadata filters | VectorDB with structured metadata | Low |
| **Text-to-SQL** | Structured | SQL | Relational databases, exact queries | Medium |
| **Text-to-SQL+Semantic** | Semi-structured | SQL + vectors | Hybrid semantic + structured queries | High |
| **Text-to-Cypher** | Graph data | Cypher | Complex relationships, graph traversal | High |

---

## Interview Notes / Specialist Notes

### **Core Concepts**

1. **Query Construction vs Query Transformation**
   - **Transformation**: Rewrite query in same language (English → better English)
   - **Construction**: Convert to different language (English → SQL/Cypher/Filters)

2. **When to Use What?**
   ```
   Decision Tree:
   
   Is data purely unstructured?
   ├─ Yes → Multi-Vector Retriever (Blog 2) + Query Transformation (Blog 1)
   └─ No → Has structure, use Query Construction
       │
       ├─ Data in relational DB?
       │   ├─ Has embeddings? → Text-to-SQL+Semantic
       │   └─ No embeddings → Text-to-SQL
       │
       ├─ Data in graph DB? → Text-to-Cypher
       │
       └─ Data in vectorstore with metadata? → Text-to-Metadata-Filter
   ```

3. **The Fundamental Trade-off**
   - **Precision vs Recall**
   - Structured queries: High precision, may miss semantically similar results
   - Semantic search: High recall, may include irrelevant results
   - Hybrid (SQL+Semantic): Best of both worlds

### **Implementation Best Practices**

#### **1. Schema Management**

```python
# Bad: Hardcoded schema in prompt
prompt = "Database has tables: users, orders, products..."

# Good: Dynamic schema injection
def get_schema_description(db):
    tables = db.get_table_names()
    schema_desc = []
    for table in tables:
        columns = db.get_table_info(table)
        sample_rows = db.run(f"SELECT * FROM {table} LIMIT 3")
        schema_desc.append(f"Table: {table}\n{columns}\nSample:\n{sample_rows}")
    return "\n\n".join(schema_desc)
```

#### **2. Few-Shot Example Selection**

```python
# Static examples (simple but limited)
examples = [
    {"q": "How many users?", "sql": "SELECT COUNT(*) FROM users"},
    {"q": "Top 5 products", "sql": "SELECT * FROM products ORDER BY sales DESC LIMIT 5"}
]

# Dynamic example selection (better)
from langchain.prompts import SemanticSimilarityExampleSelector

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    vectorstore,
    k=3  # Select 3 most similar examples to current query
)

# Now each query gets the most relevant examples
```

#### **3. Query Validation**

```python
def validate_sql(query, db):
    # 1. Syntax check
    try:
        db.parse(query)
    except SyntaxError:
        return False, "Invalid SQL syntax"
    
    # 2. Table existence check
    tables_in_query = extract_tables(query)
    valid_tables = db.get_table_names()
    if not set(tables_in_query).issubset(valid_tables):
        return False, f"Unknown tables: {tables_in_query - valid_tables}"
    
    # 3. Dry run (if safe)
    try:
        db.run(query + " LIMIT 0")  # Check validity without returning data
    except Exception as e:
        return False, f"Query error: {e}"
    
    return True, "Valid"
```

#### **4. Caching Strategies**

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_query_construction(user_query, schema_hash):
    """Cache constructed queries"""
    return llm.construct_query(user_query)

# Usage
schema_hash = hashlib.md5(str(db.get_schema()).encode()).hexdigest()
constructed_query = cached_query_construction(user_query, schema_hash)
```

### **Common Interview Questions**

**Q: "How do you prevent SQL injection in text-to-SQL systems?"**

A: Multiple layers of defense:
1. **Parameterized queries**: Never concatenate user input into SQL
2. **Query validation**: Parse and validate before execution
3. **Allowlist operations**: Only permit SELECT, limit destructive operations
4. **Sandboxing**: Use read-only database connections
5. **Query review**: Log and monitor all generated queries

```python
# Bad (vulnerable)
query = f"SELECT * FROM users WHERE name = '{user_input}'"

# Good (parameterized)
query = "SELECT * FROM users WHERE name = ?"
db.execute(query, (user_input,))

# Best (validated + parameterized)
if validate_query(constructed_query):
    db.execute_safe(constructed_query, params)
```

**Q: "What if the LLM generates expensive queries that would timeout or crash the DB?"**

A: Implement query cost estimation and limits:

```python
def estimate_query_cost(query, db):
    # 1. Use EXPLAIN to get query plan
    plan = db.run(f"EXPLAIN {query}")
    
    # 2. Check for red flags
    red_flags = [
        "Sequential Scan on large_table",  # Full table scan
        "Nested Loop",  # Cartesian product
        "cost=" # Extract estimated cost
    ]
    
    # 3. Set limits
    MAX_COST = 10000
    if estimated_cost > MAX_COST:
        return False, "Query too expensive"
    
    return True, plan

# Alternative: Use query timeouts
db.execute(query, timeout=5)  # Kill query after 5 seconds
```

**Q: "How do you handle ambiguous queries in query construction?"**

A: Multi-step clarification:

```python
def handle_query(user_query, db):
    # 1. Parse query for ambiguities
    ambiguities = detect_ambiguities(user_query, db)
    
    if ambiguities:
        # 2. Ask for clarification
        clarifications = []
        if "multiple_tables" in ambiguities:
            clarifications.append(
                f"Did you mean {ambiguities['tables'][0]} or {ambiguities['tables'][1]}?"
            )
        
        return {"type": "clarification", "questions": clarifications}
    
    # 3. Proceed with query construction
    return construct_query(user_query, db)

# Example ambiguity
user_query = "Show me Johns"
# Ambiguous: johns table? Users named John? Products named John?
```

**Q: "Compare pgvector vs dedicated vectorstore (Pinecone, Weaviate) for production use"**

A: Trade-offs analysis:

| Aspect | pgvector | Dedicated Vectorstore |
|--------|----------|----------------------|
| **Speed** | Good for <1M vectors | Optimized for billions of vectors |
| **Features** | Basic ANN (HNSW, IVFFlat) | Advanced (product quantization, filtering) |
| **Cost** | Very cheap (shared infrastructure) | More expensive (dedicated service) |
| **Ops** | Single database (simpler) | Additional service (complex) |
| **SQL** | ✅ Full SQL capabilities | ❌ Limited/no SQL |
| **Scale** | Good to ~10M vectors | Scales to billions |
| **Use Case** | Hybrid semantic+structured queries | Pure vector search at scale |

**Recommendation**: 
- pgvector: Start here for most applications
- Dedicated vectorstore: When you need >10M vectors or advanced vector features

**Q: "What's the failure rate of query construction in production?"**

A: Typical production metrics:

```
Query Construction Success Rates (with proper setup):
├─ Text-to-Metadata-Filter: 85-95% (simplest)
├─ Text-to-SQL: 70-85% (depends on schema complexity)
├─ Text-to-SQL+Semantic: 65-80% (most complex)
└─ Text-to-Cypher: 60-75% (highest complexity)

Factors affecting success:
- Schema complexity (more tables = lower success)
- Query ambiguity
- LLM capability (GPT-4 > GPT-3.5)
- Few-shot examples quality
- Schema documentation quality
```

**Mitigation strategies**:
1. Fallback to semantic search if construction fails
2. Human-in-the-loop for failed queries
3. Continuous learning from corrections
4. A/B testing different construction strategies

### **Advanced Patterns**

#### **1. Cascading Query Construction**

Try multiple strategies in sequence:

```python
def cascading_retrieval(user_query, db, vectorstore):
    strategies = [
        ("structured", text_to_sql),
        ("hybrid", text_to_sql_semantic),
        ("semantic", semantic_search)
    ]
    
    for strategy_name, strategy_fn in strategies:
        try:
            results = strategy_fn(user_query, db, vectorstore)
            if results and len(results) > 0:
                return results, strategy_name
        except Exception:
            continue
    
    return [], "none"
```

#### **2. Query Decomposition for Complex Questions**

```python
# Complex query: "Compare revenue between Q1 and Q2 for products in the electronics category"

# Decompose into sub-queries:
sub_queries = [
    "SELECT SUM(revenue) FROM sales WHERE quarter = 1 AND category = 'electronics'",
    "SELECT SUM(revenue) FROM sales WHERE quarter = 2 AND category = 'electronics'"
]

# Execute sub-queries
results = [db.run(q) for q in sub_queries]

# Synthesize answer
answer = llm.synthesize(results, original_query)
```

#### **3. Confidence Scoring**

```python
def construct_with_confidence(user_query, db):
    # Generate multiple candidate queries
    candidates = [
        llm.generate_query(user_query, temperature=0.0),  # Deterministic
        llm.generate_query(user_query, temperature=0.3),  # Slight variation
        llm.generate_query(user_query, temperature=0.3)   # Another variation
    ]
    
    # Score by agreement
    unique_queries = set(candidates)
    if len(unique_queries) == 1:
        confidence = "high"
    elif len(unique_queries) == 2:
        confidence = "medium"
    else:
        confidence = "low"
    
    # If low confidence, ask for clarification
    if confidence == "low":
        return ask_clarification(user_query)
    
    # Use most common query
    from collections import Counter
    most_common = Counter(candidates).most_common(1)[0][0]
    return most_common, confidence
```

---

### **Key Takeaways**

1. **Query Construction is about translation**: Natural language → Database-specific query language

2. **Four main techniques** for different data types:
   - Metadata filters (vectorstore + metadata)
   - SQL (relational databases)
   - SQL+Semantic (hybrid databases)
   - Cypher (graph databases)

3. **Common challenges** across all techniques:
   - LLM hallucination (fake tables/fields)
   - Query validation
   - Error handling
   - Schema complexity

4. **Success factors**:
   - Good schema documentation
   - Quality few-shot examples
   - Robust error handling
   - Appropriate LLM choice (GPT-4 for complex queries)

5. **When to combine with other techniques**:
   - Use Query Transformation (Blog 1) before construction
   - Use Multi-Vector Retriever (Blog 2) for documents with structure
   - Cascade multiple strategies for robustness

6. **Production considerations**:
   - Cache constructed queries
   - Monitor success rates
   - Implement fallbacks
   - Use confidence scoring
   - Validate queries before execution

**The Big Picture**: Query Construction completes the RAG toolkit by enabling natural language interfaces to structured data, bridging the gap between how humans ask questions and how databases are queried.
