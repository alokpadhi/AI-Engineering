# AI Engineering System Design Master Guide

## 🎯 The Ultimate Reference for Production LLM Systems

This guide synthesizes AI Engineering best practices into a **systematic decision framework** for building production-ready LLM applications. Use this BEFORE starting any LLM project.

---

# Part 1: The 10-Step Production AI System Design Process

## Step 1: Problem Definition & Requirements Analysis

### 1.1 Core Task Definition
```
What problem are we solving?
- Customer-facing (chatbot, search, recommendations)
- Internal tooling (analysis, automation, code generation)
- Content generation (writing, summarization, translation)
- Data processing (extraction, classification, enrichment)
```

### 1.2 Success Metrics Definition
```
Business Metrics:
□ User engagement (retention, session length, DAU)
□ Task completion rate
□ Cost savings (vs human baseline)
□ Revenue impact

Quality Metrics:
□ Accuracy/F1 score (for classification)
□ ROUGE/BLEU (for generation)
□ User satisfaction (thumbs up/down rate)
□ Error rate (hallucinations, format failures)

Operational Metrics:
□ Latency (P50, P95, P99)
□ Cost per request
□ Uptime/availability
□ Mean Time to Detect/Repair (MTTD/MTTR)
```

### 1.3 Constraints & Requirements
```
Performance:
- Latency budget: <500ms? <2s? <10s? Async OK?
- Throughput: Requests/second expected?
- Availability: 99.9%? 99.99%? Can tolerate downtime?

Cost:
- Budget per request: <$0.01? <$0.10? <$1.00?
- Total monthly budget?
- Cost vs quality trade-off acceptable?

Data:
- Data sensitivity: Public? Internal? PII? HIPAA?
- Data volume: KB? MB? GB? TB?
- Data freshness: Real-time? Daily updates? Static?

Compliance:
- Regulatory requirements (GDPR, HIPAA, SOC2)
- Audit trail required?
- Explainability needed?
- Human oversight mandated?
```

### 1.4 Failure Mode Analysis
```
What happens if system fails?
- Critical (financial loss, safety risk) → Require HITL, extensive guardrails
- Important (user frustration) → Strong error handling, graceful degradation
- Minor (convenience feature) → Best-effort, fail open

Maximum acceptable error rate?
- 0.1% for high-stakes (medical, financial)
- 1% for moderate stakes (customer support)
- 5% for low stakes (recommendations)
```

---

## Step 2: Choose Your Foundation Model Strategy

### 2.1 Model Selection Matrix

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
| **Complex reasoning, coding** | GPT-4, Claude Opus 4, Gemini Ultra | Highest capability |
| **Balanced performance** | GPT-4o, Claude Sonnet 4.5 | Best quality/cost ratio |
| **High-volume, simple tasks** | GPT-4o-mini, Claude Haiku, Gemini Flash | Cost-effective |
| **Specialized domain** | Fine-tuned model | Domain expertise |
| **Low-latency, on-device** | Llama 3 8B, Phi-3 | Local inference |
| **Embeddings** | text-embedding-3-large, Cohere embed-v3 | Best retrieval |

### 2.2 Model Provider Decision
```
Use OpenAI if:
✓ Need GPT-4 reasoning
✓ Function calling critical
✓ Large ecosystem of tools

Use Anthropic if:
✓ Need long context (200K+)
✓ Safety critical (strong refusals)
✓ Best instruction following

Use Open Source if:
✓ Data privacy absolute requirement
✓ Need complete control
✓ Have infrastructure/expertise
✓ Cost is primary concern

Use Multiple Providers if:
✓ Different tasks need different models
✓ Want fallback/redundancy
✓ Cost optimization via routing
```

### 2.3 Fine-tuning Decision Tree
```
Should you fine-tune?

START → Can prompt engineering solve it?
        ├─ YES → Don't fine-tune (use prompting)
        └─ NO → Continue

        ↓
        Do you have 500+ high-quality examples?
        ├─ NO → Collect more data or use few-shot
        └─ YES → Continue

        ↓
        Is the task specialized/domain-specific?
        ├─ NO → Probably don't need fine-tuning
        └─ YES → Continue

        ↓
        Will you update frequently?
        ├─ YES → Fine-tuning may be impractical (use RAG)
        └─ NO → Continue

        ↓
        Cost/latency critical?
        ├─ YES → Fine-tune smaller model
        └─ NO → Use large model with prompting

Fine-tuning candidates:
- Consistent format/style (e.g., SQL generation)
- Domain-specific vocabulary (medical, legal)
- Behavior modification (tone, personality)
- Distillation (GPT-4 → smaller model)
```

---

## Step 3: Design Your AI Architecture (5-Step Evolution)

### The Five Core Layers (Apply Progressively)

```
Layer 5: Agent Patterns & Write Actions
         ↑
Layer 4: Caching (Latency/Cost Optimization)
         ↑
Layer 3: Router & Gateway (Model Management)
         ↑
Layer 2: Guardrails (Safety & Quality)
         ↑
Layer 1: Context Construction (RAG/Tools)
         ↑
Base: LLM API Call
```

---

### **LAYER 1: Context Construction**

#### Decision: Do you need RAG?

```
Use RAG when:
✓ Knowledge changes frequently (news, docs, databases)
✓ Information exceeds context window
✓ Need citations/sources
✓ Proprietary/private knowledge
✓ Factual accuracy critical

Don't use RAG when:
✗ LLM knowledge sufficient
✗ Static, general knowledge
✗ Real-time data available via API
✗ Context fits in prompt
```

#### RAG Design Decisions

**Chunking Strategy:**
```
Small chunks (256-512 tokens):
- Precise retrieval
- More granular citations
- Use for: Q&A, fact lookup

Medium chunks (512-1024 tokens):
- Balanced context
- Standard approach
- Use for: General purpose

Large chunks (1024-1500+ tokens):
- Preserve context
- Better for summarization
- Use for: Document understanding
```

**Retrieval Method:**
```
Embedding-based (semantic search):
- Default choice
- Best for: Conceptual similarity
- Tools: OpenAI embeddings, Pinecone, Chroma

Keyword-based (BM25):
- Exact term matching
- Best for: Technical terms, IDs, names
- Tools: Elasticsearch, Meilisearch

Hybrid (embedding + keyword):
- Combine both approaches
- Best for: Production systems
- Rerank results for best quality
```

**Advanced RAG Patterns:**
```
Query Rewriting:
User query → LLM generates better search query → Retrieve
Use when: User queries ambiguous/vague

Hypothetical Document Embeddings (HyDE):
User query → LLM generates hypothetical answer → Embed → Search
Use when: Query-document mismatch

Multi-query:
User query → Generate 3-5 variations → Retrieve all → Deduplicate
Use when: Want comprehensive coverage

Parent Document Retrieval:
Chunk for search, retrieve parent for context
Use when: Need surrounding context
```

#### Decision: Do you need Tool Calling?

```
Use Tools when:
✓ Need real-time data (weather, stock prices, search)
✓ Need to perform actions (send email, create ticket)
✓ Need specialized computation (calculator, code execution)
✓ Need to access APIs

Tool Design Principles:
1. Clear function signatures (name, description, parameters)
2. Idempotent when possible
3. Return structured data
4. Handle errors gracefully
5. Log all tool calls for debugging
```

**Tool Calling vs Pre-fetching:**
```
Tool Calling (ReAct):
- Dynamic: LLM decides when to call
- Flexible: Can call multiple times
- Use when: Unpredictable needs

Pre-fetching:
- Static: Always retrieve same data
- Predictable: Query patterns known
- Use when: Consistent information needs
```

---

### **LAYER 2: Guardrails (Safety & Quality)**

#### Input Guardrails

```
ALWAYS include:
1. Prompt injection detection
   - Reverse dictionary approach
   - Pattern matching for attacks
   - Commercial: Lakera, Rebuff

2. PII detection & masking
   - Detect: SSN, credit cards, emails
   - Mask before sending to LLM
   - Tools: Presidio, AWS Comprehend

3. Content moderation
   - Hate speech, violence, explicit content
   - Tools: OpenAI Moderation API, Perspective API

For high-stakes systems, add:
4. Input validation (format, length, allowed values)
5. Rate limiting (per user, per IP)
6. Authentication & authorization
```

#### Output Guardrails

```
ALWAYS include:
1. Format validation
   - JSON parsing for structured outputs
   - Regex for specific formats
   - Retry with different prompt if fails

2. Safety checks
   - Toxic content detection
   - PII leakage prevention
   - Harmful content blocking

For factual systems, add:
3. Hallucination detection
   - Check citations exist
   - Verify facts against knowledge base
   - Use LLM-as-judge to score accuracy

4. Quality scoring
   - Relevance to query
   - Completeness
   - Coherence

For write actions, add:
5. Confirmation before execution
   - Preview action to user
   - Require explicit approval
   - Undo capability
```

**Guardrail Response Strategies:**
```
If guardrail triggers:
1. Retry with rephrased prompt (3 attempts)
2. Parallel calls to different models
3. Escalate to human (customer support)
4. Fail gracefully with apology

Track false refusal rate:
- Legitimate requests blocked
- Adjust thresholds if >5%
```

---

### **LAYER 3: Router & Gateway**

#### Model Router

```
Use Router when:
✓ Different tasks need different models
✓ Cost optimization important (route simple → cheap model)
✓ Quality tiering (free users → fast model, paid → best model)
✓ Specialization (code → CodeLlama, analysis → GPT-4)

Router Architecture:
User query → Intent classifier → Route to appropriate model → Response

Intent Classifier:
- Small model (GPT-2, BERT, Llama 7B)
- Fast (<100ms)
- Cheap (<$0.0001 per call)
- High accuracy (>95%)

Classification categories:
- Simple Q&A → Haiku/GPT-4o-mini
- Complex reasoning → Opus/GPT-4
- Code generation → Specialized code model
- Creative writing → Claude Sonnet
```

#### Gateway Pattern

```
Use Gateway for:
✓ Multi-provider management (OpenAI + Anthropic + open source)
✓ Unified interface to different LLM APIs
✓ Centralized logging, monitoring, cost tracking
✓ Fallback policies (if OpenAI down, use Anthropic)

Gateway Components:
1. Request routing (load balancing, provider selection)
2. Response caching (avoid duplicate calls)
3. Access control (rate limits, quotas, API keys)
4. Monitoring (latency, errors, costs per provider)

Tools:
- LiteLLM (unified API)
- Portkey (gateway + observability)
- Kong/Nginx (API gateway)
```

---

### **LAYER 4: Caching Strategy**

#### Exact Match Caching

```
How it works:
Input hash → Lookup cache → If hit, return cached response

When to use:
✓ Repeated identical queries (FAQ, common questions)
✓ Expensive operations (long documents, complex reasoning)
✓ High traffic, low diversity

Implementation:
- In-memory (Redis): <10ms, hot data
- Database (PostgreSQL): <50ms, warm data
- Eviction: LRU (Least Recently Used)

CRITICAL: Never cache PII
- Can leak across users
- Detection + scoping + sanitization
```

#### Semantic Caching

```
How it works:
Embed query → Find similar embeddings (cosine >0.95) → Return cached response

When to use:
✓ Query variations ("What's weather?" vs "Tell me weather")
✓ Hit rate >30% expected
✓ Embedding overhead (10-50ms) acceptable

Similarity thresholds:
- 0.99+: Near-identical (very conservative)
- 0.95-0.98: General purpose (recommended)
- 0.90-0.94: Aggressive (higher hit rate, lower quality)

Cost-benefit:
Worth it if: (embedding cost + lookup) < (0.3 × LLM cost)
```

#### Multi-tier Caching Strategy

```
Tier 1 (Hot): In-memory (Redis)
- Most frequent queries
- TTL: 1 hour
- <10ms latency

Tier 2 (Warm): Database (PostgreSQL)
- Moderately frequent queries
- TTL: 24 hours
- <50ms latency

Tier 3 (Cold): Object storage (S3)
- Rare but cacheable queries
- TTL: 7 days
- <200ms latency

Promotion strategy:
- Access count >10 in hour → Promote to Tier 1
- Access count >3 in day → Promote to Tier 2
```

---

### **LAYER 5: Agent Patterns & Write Actions**

#### When to Use Agents

```
Use Agent Patterns when:
✓ Task requires multiple steps (research → analyze → write)
✓ Need iterative refinement (generate → critique → improve)
✓ Dynamic decision making (ReAct: decide next action based on observations)
✓ Tool use required (search, calculator, API calls)

Don't use Agents when:
✗ Single-step task (just use LLM call)
✗ Fixed workflow (use pipeline instead)
✗ Latency critical (<1s response needed)
```

#### Agent Reasoning Patterns (From Your Doc)

```
Chain-of-Thought (CoT):
- Step-by-step reasoning
- Use for: Math, logic, analysis
- Prompt: "Let's think step by step"

Tree-of-Thoughts (ToT):
- Explore multiple paths
- Use for: Creative tasks, strategic planning
- Pattern: Generate options → Evaluate → Select best

ReAct (Reasoning + Acting):
- Iterative: Thought → Action → Observation
- Use for: Tool use, web search, data retrieval
- Best for: Dynamic tasks

Reflection:
- Generate → Critique → Refine
- Use for: Code generation, content improvement
- Pattern: Self-correction loop
```

#### Write Actions: Critical Safety Requirements

```
Write actions = Irreversible changes (emails, orders, database updates)

ALWAYS include (non-negotiable):
1. Pre-execution validation
   - Verify parameters correct
   - Check authorization
   - Dry-run simulation

2. Human-in-the-loop (HITL)
   - Preview action to user
   - Require explicit confirmation
   - Timeout if no response

3. Action limits
   - Max $ amount per transaction
   - Rate limits (1 email/min)
   - Scope restrictions

4. Audit trail
   - Log all actions
   - Record approvals
   - Timestamp everything

5. Rollback mechanism
   - Undo capability
   - Compensation actions
   - Clear process

Example: Banking transfers
❌ Never allow: "Transfer $10,000 to account X" without approval
✓ Always require: Preview → User confirms → Execute → Confirm success
```

---

## Step 4: Design Prompt Engineering Strategy

### 4.1 Prompt Template Architecture

```
Anatomy of Production Prompt:

[System Message]
- Role definition ("You are an expert...")
- Behavioral guidelines
- Output format requirements
- Constraints & rules

[Context] (Optional)
- Retrieved documents (RAG)
- Conversation history
- User profile/preferences

[Few-shot Examples] (Optional)
- 2-5 high-quality examples
- Diverse coverage of edge cases

[User Query]
- Actual question/request

[Output Instructions]
- Format (JSON, markdown, etc.)
- Length requirements
- Citation requirements
```

### 4.2 Prompt Engineering Techniques

```
Basic Techniques:
1. Clear instructions ("Extract all dates in ISO format")
2. Provide examples (few-shot prompting)
3. Break complex tasks into steps
4. Specify output format explicitly

Advanced Techniques:
1. Chain-of-Thought: "Think step by step before answering"
2. Self-Consistency: Generate 3-5 answers, pick most common
3. Least-to-Most: Start simple, build complexity
4. Decomposition: Break into subtasks, solve separately

For better quality:
- Add "Explain your reasoning" → More thoughtful responses
- Use delimiters (""", ###) → Clearer structure
- Request confidence scores → Filter low-confidence outputs
- Ask for alternatives → Explore solution space
```

### 4.3 Prompt Management

```
DON'T: Hard-code prompts in application code
DO: Store prompts separately, version control

Structure:
prompts/
├── customer_support_v1.yaml
├── content_generation_v2.yaml
└── code_review_v3.yaml

Each prompt file:
version: v3
description: "Code review assistant"
system_message: |
  You are an expert code reviewer...
examples:
  - input: "Review this Python function..."
    output: "Issues found: ..."
parameters:
  temperature: 0.3
  max_tokens: 2000

Benefits:
✓ A/B testing prompts
✓ Rollback if new prompt worse
✓ Track performance by version
✓ Non-engineers can update
```

---

## Step 5: Implement Evaluation System

### 5.1 Evaluation Strategy

```
Three types of evaluation:

1. Unit Tests (Component Level)
   - Test individual functions
   - Mock LLM responses
   - Fast, deterministic
   - Run on every commit

2. Integration Tests (System Level)
   - Test full workflows
   - Use real LLM calls
   - Cover edge cases
   - Run daily/before deploy

3. User Evaluation (Production)
   - A/B testing
   - User feedback
   - Monitor metrics
   - Continuous
```

### 5.2 Evaluation Metrics by Task Type

**Classification Tasks:**
```
Metrics: Accuracy, Precision, Recall, F1
Ground truth: Labeled dataset (100-1000 examples)
Threshold: >90% for production

Evaluation:
- Confusion matrix analysis
- Per-class performance
- Error analysis (where it fails)
```

**Generation Tasks:**
```
Metrics:
- Automated: ROUGE, BLEU, BERTScore
- LLM-as-judge: GPT-4 rates quality 1-5
- Human evaluation: Expert ratings

Ground truth: Reference outputs or rubrics
Threshold: Depends on use case

Evaluation dimensions:
- Relevance (answers the question)
- Accuracy (factually correct)
- Completeness (covers all aspects)
- Coherence (logical flow)
- Style (appropriate tone)
```

**RAG Systems:**
```
Metrics:
- Retrieval: Precision@K, Recall@K, MRR
- Generation: Faithfulness, Answer relevance
- End-to-end: User satisfaction

Tools:
- RAGAS framework
- TruLens
- Custom eval harness

Key dimensions:
- Context relevance (retrieved docs useful?)
- Faithfulness (output grounded in context?)
- Answer relevance (addresses user question?)
```

### 5.3 LLM-as-Judge Pattern

```
Use GPT-4/Claude Opus as evaluator:

Evaluation prompt:
"""
Evaluate this response on a scale of 1-5:

User query: {query}
Response: {response}
Context: {context}

Rate these dimensions:
1. Relevance: Does it answer the question?
2. Accuracy: Is information correct?
3. Completeness: Covers all aspects?
4. Tone: Appropriate for context?

Provide:
- Overall score (1-5)
- Scores for each dimension
- Brief justification
"""

Best practices:
- Use chain-of-thought ("Explain your reasoning")
- Provide rubric/examples
- Run multiple judges, aggregate scores
- Validate against human labels periodically
```

### 5.4 Continuous Evaluation

```
Implement monitoring:

Real-time metrics:
- Latency (P50, P95, P99)
- Error rate (by error type)
- Cost per request
- Cache hit rate

Quality metrics (sampled):
- Run LLM-as-judge on 1% of traffic
- Track over time
- Alert if drops >10%

User feedback:
- Thumbs up/down rates
- Explicit feedback forms
- Implicit signals (edits, retries)

Dashboard:
- Daily quality score
- Error breakdown
- Cost trends
- User satisfaction
```

---

## Step 6: Design Monitoring & Observability

### 6.1 The Three Quality Metrics (Non-Negotiable)

```
1. MTTD (Mean Time To Detect)
   - Target: <5 minutes
   - How long until you know there's a problem?
   
2. MTTR (Mean Time To Repair)
   - Target: <30 minutes
   - How long to fix once detected?
   
3. CFR (Change Failure Rate)
   - Track: Failed deployments / Total deployments
   - Goal: <5%

Warning: "If you don't know your CFR, redesign platform for observability."
```

### 6.2 What to Monitor (Comprehensive)

```
Format Failures:
- JSON parsing errors
- Missing required fields
- Invalid data types
□ Track: Count, % of total, examples

Quality Metrics:
- LLM-as-judge scores
- User satisfaction ratings
- Task completion rate
□ Track: Distribution, trends, outliers

Safety Metrics:
- Toxicity rate
- PII leakage incidents
- Guardrail trigger rate
- False refusal rate
□ Track: Count, severity, resolution time

Performance Metrics:
- TTFT (Time To First Token)
- TPOT (Time Per Output Token)
- Total latency (P50, P95, P99)
- Cache hit rate
□ Track: By endpoint, model, time of day

Cost Metrics:
- Cost per request
- Cost by model
- Cache savings
- Monthly burn rate
□ Track: Trends, anomalies, budget vs actual

User Behavioral:
- Session length
- Retry rate
- Edit rate (user modifies output)
- Abandonment rate
□ Track: Cohorts, experiments, changes

Business Metrics:
- Task completion rate
- User retention (Day 1, Day 7, Day 30)
- Feature adoption
- Revenue impact
□ Track: Funnels, attribution, ROI
```

### 6.3 Logging & Tracing

```
What to log (ALWAYS):
{
  "request_id": "uuid-1234",
  "timestamp": "2025-01-06T10:30:00Z",
  "user_id": "user-5678",
  "model": "gpt-4",
  "prompt_tokens": 150,
  "completion_tokens": 200,
  "latency_ms": 2500,
  "cost_usd": 0.0185,
  "success": true,
  "error": null,
  "guardrails_triggered": [],
  "cache_hit": false
}

For debugging (sample 1-10%):
- Full prompt
- Full response
- Retrieved context (for RAG)
- Tool calls made
- Intermediate reasoning steps

Distributed tracing:
- Trace complete request flow
- Component-level latency
- Dependency failures
- Tools: OpenTelemetry, Datadog, Jaeger
```

### 6.4 Alerting Strategy

```
Critical alerts (page on-call):
- Error rate >5% for 5 minutes
- Latency P95 >2x baseline for 10 minutes
- Complete outage
- PII leakage detected
- Costs >2x expected

Warning alerts (email/Slack):
- Error rate >2%
- Quality score drops >10%
- Cache hit rate drops >20%
- Guardrail trigger rate >15%

Info alerts (dashboard):
- Daily summaries
- Weekly cost reports
- Monthly quality trends
```

---

## Step 7: Implement Drift Detection

### 7.1 Three Types of Drift

```
1. System Prompt Drift
   - Your prompts change over time
   - Detection: Hash comparison
   - Solution: Version control, change log

2. User Behavior Drift
   - Users adapt, change how they interact
   - Detection: Monitor query distribution, length, topics
   - Example: Users learn to game system
   - Solution: Retrain/adjust based on new patterns

3. Model Drift
   - Provider updates model (GPT-4 March → June)
   - Detection: Benchmark suite, regression tests
   - Example: Voiceflow saw 10% quality drop after update
   - Solution: Pin versions, maintain fallback, re-evaluate
```

### 7.2 Drift Detection System

```
Baseline per version:
- Run 100-500 eval examples
- Record performance metrics
- Store as "golden dataset"

Continuous monitoring:
- Run subset daily (20 examples)
- Run full suite weekly
- Compare to baseline

Statistical tests:
- T-test for mean differences
- Chi-square for distribution changes
- Alert if p-value <0.05

Regression suite:
- Critical test cases that must pass
- Run before every deployment
- Block if any failures

Canary deployments:
- Route 5% traffic to new version
- Monitor for 24-48 hours
- Rollback if quality drops >5%
```

### 7.3 Mitigation Strategies

```
For model drift:
1. Pin versions (e.g., gpt-4-0613 not gpt-4)
2. Test new versions before migration
3. Maintain fallback to stable version
4. Monitor after migration

For behavior drift:
1. Retrain classifiers quarterly
2. Update prompts based on patterns
3. Add new examples to few-shot
4. Adjust guardrails based on abuse patterns

For prompt drift:
1. Version all prompts
2. Require review for changes
3. A/B test changes
4. Easy rollback mechanism
```

---

## Step 8: Design User Feedback System

### 8.1 When to Collect Feedback

```
1. Calibration (Onboarding)
   - Required: Face ID, voice recognition
   - Optional: Skill level, preferences
   - Best practice: Allow skip, calibrate gradually

2. Error Recovery (Something Bad Happens)
   - Always enable: Downvote, regenerate, edit
   - Conversational: "You're wrong", "Too long"
   - Enable task completion: Transfer to human

3. Low Confidence (Model Uncertain)
   - Comparative evaluation (show 2-3 options)
   - Ask user to choose
   - Use as preference data

4. Success (Controversial - Use Sparingly)
   - Don't ask routinely (annoying)
   - Sample 1-5% of users
   - Only for novel/high-impact features
   - Implicit signals better (shares, bookmarks)
```

### 8.2 How to Collect Feedback

```
Design Principles:
✓ Seamlessly integrated into workflow
✓ Zero extra effort (implicit best)
✓ Easy to ignore
✓ User motivated (gets better results)

Excellent examples:

Midjourney:
- Generate 4 images
- Upscale = strong positive
- Variations = weak positive
- Regenerate = negative
→ Every action = feedback, no explicit rating

GitHub Copilot:
- Suggestion in gray
- Tab = accept (positive)
- Keep typing = reject (negative)
→ Natural workflow, complete signal

Bad examples:
- Ask after every query (annoying)
- Require written explanation (friction)
- Ask "Did you like this?" for factual questions
- Confusing UI (star placement wrong)
```

### 8.3 Feedback Types & Signals

```
Explicit feedback:
- Thumbs up/down (easy, sparse, biased)
- Star ratings (granular, effort required)
- Yes/no (simple, limited info)
- Written feedback (rich, very sparse)

Implicit feedback:
- Edits (strong signal: original losing, edit winning)
- Regenerations (ambiguous)
- Copy/paste (positive)
- Time spent (engagement)
- Conversation length (context-dependent)
- Task completion (ultimate signal)

Natural language signals:
- Early termination ("stop", exit) → Not going well
- Error correction ("No, I meant...") → Misunderstood
- Rephrasing attempts → Model confused
- Complaints ("This is wrong") → Specific issues
- Confirmation requests ("Are you sure?") → Lack trust
```

### 8.4 Privacy vs Context Trade-off

```
For shallow analytics (aggregates):
- No context needed
- Track: thumbs up %, error rate
- Privacy: No concerns

For deep improvements (root cause):
- Context required (last 10 turns, retrieval docs)
- Privacy: Major concerns

Solution: Tiered consent
Tier 1 (Default): Anonymous aggregates only
Tier 2 (Opt-out): Anonymized patterns (no PII)
Tier 3 (Opt-in): Full context with PII redacted
Tier 4 (Separate opt-in): Training data (fully anonymized)

Transparency:
"Your feedback helps improve Claude. You can choose to share:
□ Basic usage stats (always)
☑ Conversation patterns (recommended)
☐ Full conversation history for this feedback
☐ Allow use in model training"
```

### 8.5 Feedback Limitations & Biases

```
Leniency bias:
- Users rate higher than warranted
- Avoid conflict, be nice, skip follow-up
- Uber: Average 4.8/5, <4.6 = at risk
- Solution: Redesign scale (remove negative connotations)

Randomness:
- Users click without reading
- Lack motivation for thoughtful input
- Solution: Make feedback inevitable (action = signal)

Position bias:
- First option gets more clicks
- Solution: Randomize positions, model position effects

Preference bias:
- Favor longer responses (easier to notice than accuracy)
- Recency bias (last answer seen)
- Solution: Blind comparisons, ground truth validation

Degenerate feedback loops:
- Predictions → Feedback → Amplify biases
- Rich get richer (popular stays popular)
- Can change product identity
- Solution: Exploration, diversity constraints, ground truth anchoring

Sycophancy:
- Model learns to please, not inform accurately
- Training on feedback → Dishonesty rewarded
- Solution: Separate satisfaction vs accuracy metrics, reward honesty
```

---

## Step 9: Production Deployment Strategy

### 9.1 Deployment Checklist

```
Pre-deployment:
□ All tests passing (unit, integration, regression)
□ Eval metrics meet thresholds
□ Security review completed
□ Performance benchmarks validated
□ Cost estimates reviewed
□ Runbook prepared
□ Rollback plan documented

Deployment:
□ Canary deployment (5% traffic)
□ Monitor for 24-48 hours
□ Gradual rollout (5% → 25% → 50% → 100%)
□ Ready to rollback at any stage

Post-deployment:
□ Monitor all metrics hourly for 1 week
□ Daily review for 1 month
□ Document any issues/learnings
□ Update runbook based on experience
```

### 9.2 Error Handling & Resilience

```
Retry logic:
- Transient errors: Retry with exponential backoff
- Max 3 retries
- Add jitter to prevent thundering herd

Circuit breaker:
- After N consecutive failures, stop calling
- Wait cooldown period
- Try again (half-open state)
- Reset if successful

Graceful degradation:
- Primary model fails → Fallback model
- RAG fails → Answer without context (caveat)
- Tool fails → Continue without that tool
- Complete failure → Helpful error message

Timeout management:
- Set appropriate timeouts (not too short, not infinite)
- User-facing: 30s max
- Background: 5min max
- Async for long-running tasks
```

### 9.3 Rate Limiting & Quotas

```
Rate limiting:
- Per user: 100 requests/hour
- Per IP: 1000 requests/hour
- Per API key: Based on tier

Quota management:
- Free tier: 10 requests/day
- Pro tier: 1000 requests/day
- Enterprise: Custom

Response to limit exceeded:
- Return 429 status code
- Include Retry-After header
- Explain limits clearly
- Offer upgrade path
```

### 9.4 Scaling Strategy

```
Vertical scaling (per instance):
- Increase memory for caching
- More CPU for preprocessing
- GPU for local models

Horizontal scaling (more instances):
- Load balancer distributes requests
- Stateless design (share nothing)
- External cache (Redis) for shared state
- Database read replicas

Async processing:
- Long tasks → Queue (Celery, RabbitMQ)
- Return immediately with job ID
- User polls for results
- Webhook callback when done

Auto-scaling:
- Scale based on queue length
- Scale based on CPU/memory
- Scale based on latency
- Cooldown period (5 min)
```

---

## Step 10: Multi-Agent System Design (If Needed)

### 10.1 When to Use Multi-Agent

```
Use Single Agent when:
- Single clear task
- Linear workflow
- <5 steps
- No specialization needed

Use Multi-Agent when:
- Multiple distinct tasks
- Need specialization (coding + research + writing)
- Complex coordination required
- >5 steps with branching
```

### 10.2 Multi-Agent Patterns (From Your Doc)

```
Router Pattern (2-3 tasks, sequential):
User query → Classifier → Route to specialist → Response
Example: Customer support routing

Supervisor Pattern (3-7 tasks, coordination needed):
Manager agent → Coordinates workers → Aggregates results
Example: Research assistant with specialists

Hierarchical Pattern (>7 tasks):
Multi-level management structure
Example: Enterprise workflow automation

Subgraph Pattern (reusable components):
Modular workflows that can be composed
Example: Shared validation/processing steps
```

### 10.3 Communication Patterns

```
Direct Invocation:
- Agent A completes → Directly calls Agent B
- Use when: Simple, synchronous, same process

Shared State:
- All agents read/write common data structure
- Use when: Need common data access
- Warning: Requires coordination

Message Passing:
- Agents send messages via queue
- Use when: Distributed, async, decoupled
- Best for: Microservices architecture

Isolated State:
- Each agent has own workspace
- Use when: Privacy, testing, independence
```

### 10.4 Cooperation Patterns

```
Team Collaboration:
- Multiple agents work together
- Each contributes expertise
- Example: Code review team (security + performance + style)

Debate/Roleplay:
- Agents argue different perspectives
- Explore solution space
- Example: Pro/con analysis for decision

Consensus Building:
- Multiple agents vote on decision
- Majority or unanimous
- Example: Committee approval

Red Team / Blue Team:
- Adversarial testing
- One attacks, one defends
- Example: Security testing
```

---

# Part 2: Production AI System Architecture Template

## Complete System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                            │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    API GATEWAY / LOAD BALANCER                  │
│  • Authentication/Authorization                                  │
│  • Rate limiting                                                 │
│  • Request routing                                               │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT PROCESSING                              │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐ │
│  │ PII Detection│ Prompt       │ Content      │ Input        │ │
│  │ & Masking    │ Injection    │ Moderation   │ Validation   │ │
│  │              │ Detection    │              │              │ │
│  └──────────────┴──────────────┴──────────────┴──────────────┘ │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CONTEXT CONSTRUCTION                          │
│  ┌──────────────────────┬──────────────────────┐                │
│  │   RAG (if needed)    │  Tool Calls (if req) │                │
│  │  ┌────────────────┐  │  ┌────────────────┐  │                │
│  │  │ Query Rewrite  │  │  │ Function Calls │  │                │
│  │  │ Vector Search  │  │  │ API Integrations│  │                │
│  │  │ Rerank         │  │  │ Web Search     │  │                │
│  │  │ Retrieve Docs  │  │  │ Code Execution │  │                │
│  │  └────────────────┘  │  └────────────────┘  │                │
│  └──────────────────────┴──────────────────────┘                │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                        ROUTING (if multi-model)                  │
│  Intent Classification → Route to appropriate model              │
│  (Simple Q&A → Fast model, Complex → Best model)                │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      CACHE CHECK                                 │
│  ┌──────────────────────┬──────────────────────┐                │
│  │   Exact Match        │  Semantic Match      │                │
│  │   (Redis)            │  (Embedding search)  │                │
│  └──────────────────────┴──────────────────────┘                │
│  Hit → Return cached | Miss → Continue                           │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      LLM API CALL                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Model: GPT-4 / Claude / Gemini / Open Source              │ │
│  │  Prompt: System + Context + Examples + User Query          │ │
│  │  Parameters: Temperature, Max Tokens, etc.                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Agent Loop (if using agents):                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Thought → Action → Observation → [Repeat until done]      │ │
│  │  (ReAct, CoT, Reflection patterns)                         │ │
│  └────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT PROCESSING                             │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐ │
│  │ Format       │ Hallucination│ Safety       │ Quality      │ │
│  │ Validation   │ Detection    │ Check        │ Scoring      │ │
│  │ (JSON parse) │ (verify facts│ (toxicity,   │ (LLM judge)  │ │
│  │              │  & citations)│  PII leak)   │              │ │
│  └──────────────┴──────────────┴──────────────┴──────────────┘ │
│  If fail → Retry (max 3x) | If success → Continue               │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                 WRITE ACTIONS (if applicable)                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Pre-execution validation → Human approval (HITL) →        │ │
│  │  Execute action → Verify success → Audit log               │ │
│  └────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                     RESPONSE TO USER                             │
│  • Format for display                                            │
│  • Add citations if RAG                                          │
│  • Include feedback mechanisms (thumbs up/down)                  │
└───────────────────────────────┬─────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                 LOGGING & MONITORING                             │
│  ┌──────────────────────┬──────────────────────┐                │
│  │   Metrics Tracking   │   Observability      │                │
│  │  • Latency           │  • Distributed traces│                │
│  │  • Cost              │  • Error logs        │                │
│  │  • Quality scores    │  • Debug info        │                │
│  │  • User feedback     │  • Alert triggers    │                │
│  └──────────────────────┴──────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘

CONTINUOUS PROCESSES (Running in parallel):
┌─────────────────────────────────────────────────────────────────┐
│ • Drift Detection (monitor model/behavior changes)               │
│ • Evaluation Pipeline (run test suites)                          │
│ • Feedback Analysis (analyze user signals)                       │
│ • Cost Optimization (adjust routing, caching)                    │
│ • Security Monitoring (detect attacks, abuse)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

# Part 3: Decision Framework Cheatsheet

## Quick Reference: What Do I Need?

### ✅ ALWAYS Include (Non-Negotiable)

```
1. Input guardrails (PII detection, prompt injection protection)
2. Output validation (format, safety checks)
3. Error handling (retry logic, graceful degradation)
4. Logging (request ID, latency, cost, errors)
5. Monitoring (latency, error rate, cost)
6. Authentication & rate limiting
```

### 🤔 Probably Need (Most Production Systems)

```
1. RAG (if knowledge exceeds context or changes frequently)
2. Caching (if repeated queries expected, cost/latency matters)
3. Prompt versioning (for A/B testing, rollback)
4. Evaluation pipeline (for quality assurance)
5. User feedback collection (for improvement)
6. Drift detection (for model/behavior monitoring)
```

### 🎯 Maybe Need (Depends on Use Case)

```
1. Multi-agent system (if task complex, needs specialization)
2. Router (if different tasks need different models)
3. Gateway (if using multiple LLM providers)
4. HITL (if high-stakes decisions, compliance required)
5. Fine-tuning (if specialized domain, cost critical)
6. Semantic caching (if high traffic, query variations)
```

### ❌ Probably Don't Need (Start Simple)

```
1. Complex multi-agent orchestration (start with single agent)
2. Custom fine-tuned models (prompt engineering often sufficient)
3. Real-time streaming (unless UI requires it)
4. Advanced agent patterns (ToT, multi-turn debate) for simple tasks
5. Elaborate consensus mechanisms (overkill for most cases)
```

---

## Decision Matrix: My Use Case → What I Need

```
┌──────────────────────────────────────────────────────────────────┐
│ CHATBOT / CONVERSATIONAL AI                                      │
├──────────────────────────────────────────────────────────────────┤
│ ✅ MUST: Input/output guardrails, conversation memory           │
│ ✅ SHOULD: RAG (for knowledge), caching (common Qs)             │
│ 🤔 MAYBE: Router (intent-based), HITL (sensitive topics)        │
│ ❌ SKIP: Complex multi-agent, fine-tuning (unless specialized)  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ CODE GENERATION / COPILOT                                        │
├──────────────────────────────────────────────────────────────────┤
│ ✅ MUST: Output validation (syntax check), security scan        │
│ ✅ SHOULD: Caching (common patterns), RAG (codebase context)    │
│ 🤔 MAYBE: Reflection (iterative improvement), multi-agent       │
│ ❌ SKIP: Input guardrails less critical, HITL (slows down)      │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ CONTENT GENERATION (Marketing, Writing)                         │
├──────────────────────────────────────────────────────────────────┤
│ ✅ MUST: Output guardrails (brand safety), quality scoring      │
│ ✅ SHOULD: Reflection (improve quality), style guidelines       │
│ 🤔 MAYBE: Multi-agent (writer + editor), HITL (approval)        │
│ ❌ SKIP: RAG (unless need brand docs), complex agents           │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ DATA ANALYSIS / INSIGHTS                                         │
├──────────────────────────────────────────────────────────────────┤
│ ✅ MUST: Data validation, error handling (bad data)             │
│ ✅ SHOULD: RAG (connect to databases), tools (SQL, Python)      │
│ 🤔 MAYBE: Multi-agent (analyze + visualize), ReAct pattern      │
│ ❌ SKIP: Real-time (can be async), elaborate agents             │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ CUSTOMER SUPPORT                                                 │
├──────────────────────────────────────────────────────────────────┤
│ ✅ MUST: RAG (knowledge base), HITL (escalation to human)       │
│ ✅ SHOULD: Router (intent classification), caching (FAQs)       │
│ 🤔 MAYBE: Multi-agent (triage + specialist), sentiment analysis │
│ ❌ SKIP: Complex reasoning (keep simple), write actions         │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ DOCUMENT PROCESSING (Extraction, Summarization)                 │
├──────────────────────────────────────────────────────────────────┤
│ ✅ MUST: Format validation, error handling (bad inputs)         │
│ ✅ SHOULD: Structured output (JSON), chunking strategy          │
│ 🤔 MAYBE: Multi-agent (extract + validate), reflection          │
│ ❌ SKIP: RAG (doc itself is context), complex orchestration     │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ RESEARCH ASSISTANT                                               │
├──────────────────────────────────────────────────────────────────┤
│ ✅ MUST: RAG (documents), tools (web search), multi-step        │
│ ✅ SHOULD: Multi-agent (search + analyze + write), ReAct        │
│ 🤔 MAYBE: Reflection (improve quality), supervisor pattern      │
│ ❌ SKIP: Real-time (can be async), simple single-agent          │
└──────────────────────────────────────────────────────────────────┘
```

---

# Part 4: Implementation Roadmap

## Phase 1: MVP (Week 1-2)

```
Goal: Get something working end-to-end

✅ Must build:
1. Single LLM API call (OpenAI/Anthropic)
2. Basic prompt template
3. Input validation (basic)
4. Output format validation
5. Simple error handling
6. Basic logging (print statements OK)
7. Manual testing (10-20 test cases)

🚫 Don't build yet:
- Multi-agent systems
- RAG (use prompt if knowledge fits)
- Caching
- Complex guardrails
- Monitoring dashboards

Deliverable: Working prototype you can demo
```

## Phase 2: Quality & Safety (Week 3-4)

```
Goal: Make it safe and reliable

✅ Must build:
1. Input guardrails (PII, prompt injection)
2. Output guardrails (safety, quality)
3. Retry logic with exponential backoff
4. Structured logging (JSON)
5. Evaluation dataset (50-100 examples)
6. Basic metrics (accuracy, latency, cost)

🎯 Consider adding:
- RAG if knowledge doesn't fit in prompt
- Prompt versioning
- A/B testing framework

Deliverable: Safe for internal testing
```

## Phase 3: Production Ready (Week 5-8)

```
Goal: Scale and monitor

✅ Must build:
1. Monitoring dashboard (Grafana, Datadog)
2. Alerting (PagerDuty, Slack)
3. Caching (Redis)
4. Rate limiting
5. Authentication
6. Comprehensive testing (100+ cases)
7. Deployment pipeline (CI/CD)
8. Runbook & documentation

🎯 Consider adding:
- Model router (if using multiple models)
- Gateway (if multiple providers)
- Advanced caching (semantic)
- Drift detection

Deliverable: Production-grade system
```

## Phase 4: Optimization & Scaling (Week 9-12)

```
Goal: Improve quality and reduce cost

✅ Must do:
1. Continuous evaluation
2. User feedback collection
3. Cost optimization (routing, caching)
4. Performance tuning
5. A/B testing prompts

🎯 Consider adding:
- Multi-agent (if complexity warrants)
- Fine-tuning (if specialized need)
- Advanced agent patterns
- HITL workflows

Deliverable: Optimized, cost-effective system
```

---

# Part 5: Common Pitfalls & How to Avoid Them

## Top 10 Mistakes

### 1. **Starting Too Complex**
```
❌ Wrong: "Let's build a multi-agent system with 5 specialized agents"
✅ Right: "Let's get a single agent working first, then add complexity"

Principle: Start simple, add complexity only when needed
```

### 2. **Ignoring Evaluation**
```
❌ Wrong: "It looks good in manual testing, ship it!"
✅ Right: "Let's build eval dataset and track metrics over time"

Principle: Can't improve what you don't measure
```

### 3. **Not Monitoring Production**
```
❌ Wrong: "We deployed it, it's working fine... I think?"
✅ Right: "Dashboard shows P95 latency 1.2s, error rate 0.3%, all green"

Principle: Observability is not optional
```

### 4. **Trusting LLM Output Blindly**
```
❌ Wrong: Execute LLM-generated code/SQL/commands directly
✅ Right: Validate, preview, require human approval for write actions

Principle: Never trust, always verify
```

### 5. **Prompt Injection Vulnerability**
```
❌ Wrong: {user_input} directly in prompt without checks
✅ Right: Detect attacks, use delimiters, validate input first

Principle: Users will try to jailbreak your system
```

### 6. **Ignoring Cost**
```
❌ Wrong: "GPT-4 for everything, monitor cost later"
✅ Right: Route simple queries to cheaper models, cache aggressively

Principle: Cost can spiral out of control quickly
```

### 7. **No Graceful Degradation**
```
❌ Wrong: One LLM call fails → Entire system crashes
✅ Right: Fallback models, retry logic, helpful error messages

Principle: Systems will fail, plan for it
```

### 8. **Treating Feedback as Ground Truth**
```
❌ Wrong: "Users gave thumbs up, must be perfect!"
✅ Right: "Understand biases, validate against objective metrics"

Principle: Feedback is signal, not truth
```

### 9. **Not Planning for Drift**
```
❌ Wrong: "Model working great today, will work great forever"
✅ Right: "Monitor performance, regression tests, ready to adapt"

Principle: Everything drifts over time
```

### 10. **Over-Engineering**
```
❌ Wrong: "We might need X someday, let's build it now"
✅ Right: "YAGNI (You Aren't Gonna Need It) - build when needed"

Principle: Premature optimization is root of all evil
```

---

# Part 6: Tool & Framework Recommendations

## By Category

### **LLM Providers**
```
Production-ready:
- OpenAI (GPT-4, GPT-4o) → Best overall
- Anthropic (Claude) → Long context, safety
- Google (Gemini) → Multimodal, fast

Open source:
- Llama 3 → General purpose
- Mistral → European alternative
- CodeLlama → Code specialized
```

### **Orchestration & Agents**
```
Full-featured:
- LangChain → Most popular, lots of integrations
- LlamaIndex → RAG-focused
- Haystack → NLP pipelines

Lightweight:
- Instructor → Structured outputs
- Guidance → Constrained generation
- LMQL → Query language for LLMs
```

### **Vector Databases**
```
Managed:
- Pinecone → Easy, expensive
- Weaviate → Fast, open source option
- Qdrant → High performance

Self-hosted:
- Chroma → Simple, embedded
- Milvus → Scalable
- pgvector → PostgreSQL extension
```

### **Observability**
```
LLM-specific:
- Langfuse → Open source LLM tracing
- LangSmith → LangChain's platform
- Helicone → Proxy with monitoring

General:
- Datadog → Enterprise grade
- Grafana → Open source dashboards
- Sentry → Error tracking
```

### **Evaluation**
```
Frameworks:
- RAGAS → RAG evaluation
- TruLens → Comprehensive eval
- Giskard → Testing & validation

LLM-as-judge:
- OpenAI GPT-4 → Best quality
- Claude Opus → Alternative
- Open source → Cost-effective
```

### **Guardrails**
```
Commercial:
- Lakera → Prompt injection detection
- Rebuff → Security focused
- NeMo Guardrails → NVIDIA

Open source:
- Presidio → PII detection
- LLM Guard → Multiple checks
- Custom regex → Simple patterns
```

---

# Part 7: Final Checklist Before Launch

## Pre-Launch Checklist

```
FUNCTIONALITY:
□ Core functionality works for 95% of test cases
□ Edge cases handled gracefully
□ Error messages helpful and user-friendly
□ Latency acceptable (P95 < your SLA)

SAFETY:
□ Input guardrails prevent prompt injection
□ Output guardrails catch hallucinations, toxicity
□ PII detection & masking working
□ No unsafe actions possible without approval

QUALITY:
□ Evaluation dataset (100+ examples) established
□ Baseline metrics recorded
□ Quality meets minimum threshold
□ Human evaluation conducted (10+ reviewers)

MONITORING:
□ Logging captures all important events
□ Metrics dashboard set up
□ Alerts configured for critical issues
□ On-call rotation established

COST:
□ Cost per request estimated
□ Monthly budget established
□ Alerts for cost overruns
□ Optimization strategy in place

SECURITY:
□ Authentication implemented
□ Rate limiting configured
□ API keys secured (not in code)
□ Data encryption at rest and in transit

COMPLIANCE:
□ Privacy policy updated
□ Terms of service reviewed
□ GDPR/CCPA compliance verified (if applicable)
□ Data retention policy established

OPERATIONS:
□ Runbook documented
□ Rollback plan prepared
□ Incident response process defined
□ Team trained on monitoring & debugging

SCALABILITY:
□ Load testing completed
□ Auto-scaling configured
□ Database capacity planned
□ CDN/caching strategy implemented

USER EXPERIENCE:
□ Feedback mechanisms in place
□ Loading states handled
□ Streaming implemented (if applicable)
□ Mobile experience tested
```

---

# Part 8: Quick Start Templates

## Template 1: Simple Q&A Chatbot

```python
from openai import OpenAI
import json
import logging
from datetime import datetime

# Setup
client = OpenAI(api_key="your-api-key")
logging.basicConfig(level=logging.INFO)

class SimpleChatbot:
    def __init__(self):
        self.system_prompt = """You are a helpful assistant.
        Be concise, accurate, and friendly."""
        
    def validate_input(self, user_input: str) -> bool:
        """Basic input validation"""
        if len(user_input) > 2000:
            return False
        # Add more checks (PII, prompt injection, etc.)
        return True
    
    def generate_response(self, user_input: str) -> dict:
        """Generate response with monitoring"""
        request_id = f"req_{datetime.now().timestamp()}"
        
        # Validate
        if not self.validate_input(user_input):
            return {"error": "Invalid input", "request_id": request_id}
        
        # Log request
        logging.info(f"Request {request_id}: {user_input[:100]}")
        
        try:
            # Call LLM
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # Extract & validate output
            output = response.choices[0].message.content
            
            # Log response
            logging.info(f"Response {request_id}: Success")
            
            return {
                "response": output,
                "request_id": request_id,
                "model": "gpt-4o-mini",
                "tokens": response.usage.total_tokens
            }
            
        except Exception as e:
            logging.error(f"Error {request_id}: {str(e)}")
            return {
                "error": "Something went wrong",
                "request_id": request_id
            }

# Usage
chatbot = SimpleChatbot()
result = chatbot.generate_response("What is the capital of France?")
print(result)
```

## Template 2: RAG System

```python
from openai import OpenAI
import chromadb
from typing import List

class RAGSystem:
    def __init__(self):
        self.client = OpenAI()
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection("docs")
        
    def add_documents(self, documents: List[str]):
        """Add documents to vector store"""
        for i, doc in enumerate(documents):
            self.collection.add(
                documents=[doc],
                ids=[f"doc_{i}"]
            )
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return results['documents'][0]
    
    def generate(self, query: str) -> str:
        """RAG: Retrieve + Generate"""
        # Retrieve relevant docs
        docs = self.retrieve(query)
        context = "\n\n".join(docs)
        
        # Generate with context
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer based on context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"}
            ]
        )
        
        return response.choices[0].message.content

# Usage
rag = RAGSystem()
rag.add_documents([
    "Paris is the capital of France.",
    "London is the capital of the UK."
])
answer = rag.generate("What is the capital of France?")
print(answer)
```

## Template 3: Multi-Agent System

```python
from typing import List, Dict
from openai import OpenAI

class Agent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.client = OpenAI()
    
    def execute(self, task: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are {self.role}"},
                {"role": "user", "content": task}
            ]
        )
        return response.choices[0].message.content

class Supervisor:
    def __init__(self):
        self.client = OpenAI()
        self.agents: Dict[str, Agent] = {}
    
    def add_agent(self, agent: Agent):
        self.agents[agent.name] = agent
    
    def route_task(self, task: str) -> str:
        """Decide which agent should handle task"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Which agent should handle this?\nAgents: {list(self.agents.keys())}\nTask: {task}\nRespond with just the agent name."
            }]
        )
        return response.choices[0].message.content.strip()
    
    def execute(self, task: str) -> str:
        # Route to appropriate agent
        agent_name = self.route_task(task)
        agent = self.agents.get(agent_name)
        
        if agent:
            return agent.execute(task)
        return "No suitable agent found"

# Usage
supervisor = Supervisor()
supervisor.add_agent(Agent("researcher", "research expert"))
supervisor.add_agent(Agent("writer", "content writer"))

result = supervisor.execute("Research recent AI developments")
print(result)
```

---

# Summary: Your AI Engineering Workflow

```
1. DEFINE
   ↓
   - What problem?
   - What metrics?
   - What constraints?
   
2. DESIGN
   ↓
   - Choose model
   - Design architecture (5 layers)
   - Plan evaluation
   
3. BUILD (MVP)
   ↓
   - Single agent first
   - Basic guardrails
   - Manual testing
   
4. SECURE
   ↓
   - Input/output guardrails
   - Error handling
   - Safety checks
   
5. MONITOR
   ↓
   - Logging & metrics
   - Alerting
   - Dashboards
   
6. EVALUATE
   ↓
   - Test dataset
   - Continuous eval
   - User feedback
   
7. OPTIMIZE
   ↓
   - Cost (routing, caching)
   - Quality (prompts, RAG)
   - Latency (caching, streaming)
   
8. SCALE
   ↓
   - Load balancing
   - Auto-scaling
   - Multi-region (if needed)
   
REPEAT: Iterate based on data
```

---

**Key Principle**: Start simple, add complexity only when justified by data. Every component should earn its place in the architecture through measured improvement in metrics that matter.

**Remember**: The best AI system is the simplest one that meets your requirements. Complexity is a liability, not an asset.
