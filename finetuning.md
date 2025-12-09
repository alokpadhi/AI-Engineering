# Finetuning Overview  
*Balanced revision notes for interviews & practical reference*

---

## What Is Finetuning?

**Finetuning** starts with a **base (pre-trained) model** that already has general capabilities but isn’t yet optimal for your specific task. The goal is to **adapt and refine** this model so it performs well for a **target use case** (e.g., legal QA, text-to-SQL, summarization).

Finetuning is a form of **transfer learning**—reusing knowledge learned from one task to speed up and improve learning on a related task.

---

## Transfer Learning: Core Idea

**Transfer learning** (Bozinovski & Fulgosi, 1976) studies how knowledge from one task can be reused for another.

### Human analogy
- Knowing piano → easier to learn guitar

### Landmark example
- **Google Multilingual Translation (Johnson et al., 2016)**  
  A single model learned Portuguese–Spanish translation **without** direct training examples by transferring knowledge from:
  - Portuguese ↔ English
  - English ↔ Spanish

---

## Why Transfer Learning Matters for LLMs

LLMs are pre-trained on **massive, cheap, unlabeled text data** (text completion).  
This knowledge can then be transferred to **specialized, data-scarce tasks**, such as:
- Legal QA
- Medical summarization
- Text-to-SQL
- Code understanding

### Key benefit: **Sample efficiency**
- Training from scratch: **millions** of examples
- Finetuning a strong base model: **hundreds** of examples

> This is why foundation models are so valuable for real-world applications.

---

## How to Think About Finetuning Conceptually

According to **InstructGPT (OpenAI, 2022)**:
- Finetuning often *does not add entirely new capabilities*
- Instead, it **unlocks and surfaces capabilities** that the model already has but:
  - Are hard to access via prompting
  - Behave inconsistently or misalign with human expectations

---

## Finetuning vs Other Transfer Learning Approaches

### 1. Finetuning (Parameter Update)
- The model’s weights are updated
- Most common method for LLMs

### 2. Feature-Based Transfer
- Use the model as a **feature (embedding) extractor**
- Train a separate model (e.g., classifier head) on top

📌 Very common in **computer vision**
- ImageNet-pretrained CNNs used for detection/segmentation

---

## Where Finetuning Fits in the Training Pipeline

1. **Pre-training**
   - Self-supervised
   - Large-scale, unlabeled data
   - Learns general language/world knowledge

2. **Finetuning (Post-training)**
   - Task-specific adaptation
   - Can include multiple stages and objectives

> Any training done *after* pre-training is technically finetuning.

---

## Types of Finetuning (Quick Recap)

### 1. Self-Supervised Finetuning (Continued Pre-training)

- Also called **domain-adaptive pretraining**
- Uses **cheap, unlabeled but task-related data**

**Examples**
- Legal QA → finetune on raw legal documents
- Vietnamese summarization → finetune on Vietnamese text corpus

✅ Improves domain familiarity **before** expensive supervised data

---

### 2. Supervised Finetuning (SFT)

- Uses high-quality **(input, output)** pairs
- Inputs: instructions or prompts
- Outputs: expected responses

**Examples**
- Instruction → summary
- Question → answer
- Text → class label

✅ Aligns model behavior with:
- Human intent
- Formatting expectations
- Task-specific correctness

⚠️ Challenges:
- Data is **expensive**
- Needs:
  - Domain expertise
  - Factual accuracy
  - Safety & political correctness

---

### 3. Preference Finetuning (RLHF-style)

- Uses **comparative feedback**
- Data format:
```

(instruction, preferred response, rejected response)

```

- Trains the model to **prefer human-favored outputs**

✅ Improves:
- Helpfulness
- Safety
- Harmlessness
- User satisfaction

---

### 4. Infilling (Fill-in-the-Blank) Finetuning

- Instead of predicting next token, predict **missing spans**
- Especially useful for:
- Text editing
- Code debugging
- Refactoring

📌 Even autoregressive models can be finetuned this way.

---

### 5. Long-Context Finetuning

- Extends maximum context length
- Often requires **architectural changes**
- Positional embeddings
- Attention scaling

**Example**
- Code Llama:
- Context length increased from **4k → 16k tokens**

⚠️ Trade-offs:
- Harder to train
- May degrade performance on short inputs

---

## Who Does Finetuning?

### Model Developers
- Perform extensive post-training
- Release:
- Base models
- Instruction-tuned variants
- Safety-aligned variants
- Domain-specific variants

### Application Developers
- Usually finetune an **already post-trained model**
- The more refined the model:
- The less finetuning is needed
- The cheaper and safer adaptation becomes

---

## Key Interview Takeaways

- Finetuning is **transfer learning applied to foundation models**
- It dramatically improves **sample efficiency**
- Common sequence:
```

Pre-training → continued pre-training → supervised finetuning → preference finetuning

```
- Finetuning often *refines behavior*, not raw knowledge
- Choose finetuning type based on:
- Data availability
- Task complexity
- Alignment requirements
- Long-context finetuning is powerful but risky

---

### One-Line Interview Summary

> “Finetuning adapts a pre-trained foundation model to a specific task via transfer learning, improving sample efficiency and alignment by refining existing capabilities rather than learning from scratch.”

---
# When to Finetune  
*A decision framework for practitioners & interview-ready notes*

---

## Finetuning vs Prompting: First Principles

Before deciding **how** to finetune, you must decide **whether** finetuning is even the right approach.

### Key reality check
- **Finetuning is expensive**
  - More data
  - More compute
  - More ML expertise
- **Prompting is cheaper and faster**
  - Should always be tried first
  - Includes: prompt engineering, RAG, structured prompts, tools, agents

✅ **Industry rule of thumb**  
> Try prompt-based approaches exhaustively before finetuning.

⚠️ **Important**  
Finetuning and prompting are **not mutually exclusive**.  
Most real-world systems use **both**.

---

## Primary Reasons to Finetune

### 1. Improve Output Quality (General or Task-Specific)

Finetuning is done when the base model:
- Performs well on benchmarks
- ❌ Performs poorly on *your specific task*

**Common signals**
- The model:
  - Hallucinates on domain-specific queries
  - Fails edge cases consistently
  - Doesn't generalize to your data distribution

---

### 2. Enforce Strict Output Formats

Finetuning is especially strong at enforcing **structured outputs**:
- JSON
- YAML
- SQL
- Domain-specific schemas

✅ Much more reliable than prompting alone for:
- Production APIs
- Downstream machine parsing
- Deterministic workflows

---

### 3. Domain or Dialect Adaptation

Out-of-the-box models usually learn:
- Standard syntax
- Common patterns

They often fail on:
- Organization-specific logic
- Non-standard SQL dialects
- Enterprise-specific schemas

**Examples**
- Model works for ANSI SQL → fails on vendor-specific SQL
- Model works on common queries → fails on customer-specific logic

✅ Finetuning on **your exact data distribution** fixes this.

---

### 4. Bias Mitigation & Alignment

Finetuning can help **reduce biases inherited from pre-training**.

**Research-backed findings**
- Exposure to curated counterexamples can reduce:
  - Gender bias
  - Racial bias
  - Occupational stereotypes

**Examples**
- If a model associates CEOs with male names:
  - Finetune on datasets with female CEOs
- Finetuning on:
  - Text written by women → reduces gender bias
  - Text written by African authors → reduces racial bias

📌 Finetuning ≠ perfect fairness  
But it’s a **practical bias mitigation lever**

---

## Why Smaller Models Are Often Finetuned (Not Bigger Ones)

### Smaller Models Win in Practice

- Lower memory footprint
- Cheaper training
- Faster inference
- Easier experimentation

✅ **Most production finetuning happens on smaller models**

---

### Distillation: Big → Small

A common industry pattern:

1. Use a **large, powerful model**
2. Generate high-quality outputs
3. Finetune a **smaller model** to replicate that behavior

This is called **knowledge distillation**.

✅ Benefits
- Retains performance
- Strong reduction in cost & latency
- More controllable in production

---

## Real-World Evidence: Small > Large (Sometimes)

📌 **Grammarly case study**
- Finetuned **Flan-T5**
- Data used: **82,000 instruction–output pairs**
- Result:
  - Outperformed a GPT-3–based text editing model
  - Despite being **60× smaller**

✅ Key insight  
> A well-finetuned small model can beat a generic large model on a focused task.

---

## Ecosystem Shift: Why Finetuning Is More Viable Today

Early foundation model era:
- Best models were closed
- Limited or no finetuning access

Today:
- Rich open-source ecosystem
- High-quality models at multiple scales
- Domain-specific base models available

✅ Result  
Finetuning is now:
- More accessible
- More cost-effective
- More attractive for startups and enterprises

---

## Decision Checklist: Should You Finetune?

✅ Finetune if:
- Prompting + RAG isn’t good enough
- Your task is narrow and repetitive
- You need strict output formats
- You have domain-specific data
- Latency and cost matter at scale
- Bias mitigation is a requirement

❌ Avoid finetuning if:
- Prompting already works
- Task changes frequently
- You lack clean training data
- You can tolerate small errors
- Time-to-market is critical

---

## Interview-Ready One-Liners

- **Why finetune?**  
  > To adapt a general-purpose model to a specific domain, format, or data distribution when prompting alone is insufficient.

- **Big vs small models?**  
  > Smaller finetuned models often outperform larger generic models on narrow tasks.

- **Bias mitigation?**  
  > Finetuning on curated data can reduce biases inherited from pre-training.

- **Prompting vs finetuning?**  
  > Prompting is cheaper and faster; finetuning is justified when reliability, structure, or domain alignment is critical.

---
# Reasons *Not* to Finetune  
*Practical constraints, trade-offs, and interview-grade insights*

---

## Big Picture Takeaway

> **Finetuning is powerful, but often unnecessary and sometimes counterproductive.**  
Many gains attributed to finetuning can be achieved with **prompting, RAG, tools, and system design**—at much lower cost and risk.

In industry, **over-finetuning is a common mistake**, especially early in a project.

---

## 1. Prompting Can Replace Many Benefits of Finetuning

Finetuning can improve:
- Task performance
- Structured outputs
- Domain adaptation

✅ But so can:
- Well-designed prompts
- Few-shot examples
- Chain-of-thought
- RAG
- Tool use & agents
- Output validation layers

**Key insight**  
Before finetuning, ask:
> *Have we exhausted prompt design and evaluation systematically?*

In many cases, the answer is **no**.

---

## 2. Task-Specific Finetuning Can Hurt Other Capabilities

### The “Alignment Tax” Problem

Finetuning improves performance on **one task**, but can **degrade performance on others**.

**Example**
- Model handles:
  - Product recommendations ✅
  - General feedback ✅
  - Order changes ❌
- You finetune only on *order changes*
- Result:
  - Order changes ✅
  - Recommendations ❌
  - Feedback ❌

This is frustrating if:
- The application expects **diverse prompts**
- One model handles multiple tasks

### Mitigation strategies
- Finetune on **all relevant tasks** (more data, more complexity)
- Use **multiple specialized models**
- Merge models if needed (model merging / adapters)

✅ Interview insight  
> Finetuning optimizes *locally*, not *globally*.

---

## 3. Finetuning Has High Upfront & Ongoing Costs

### a) Data Cost
- High-quality annotated data is:
  - Slow to collect
  - Expensive
  - Requires domain expertise
- AI-generated and open-source data help, but:
  - Quality varies
  - Bias & noise risks remain

---

### b) ML Expertise Required
Finetuning isn’t “click-and-go”:
- You must understand:
  - Base model selection
  - Optimizers
  - Learning rates
  - Overfitting vs underfitting
  - Evaluation & debugging
- Tools simplify workflows but don’t remove **ML complexity**

✅ For teams without ML maturity, this is a major risk.

---

### c) Serving & Operations Complexity
After finetuning, you must:
- Decide how to host:
  - Self-host vs API
- Handle:
  - Inference optimization
  - Scaling
  - Latency
  - Reliability

If you already host models in-house → easier  
If not → **large operational leap**

---

### d) Long-Term Maintenance Burden

The ecosystem moves fast:
- New base models release frequently
- New models may:
  - Outperform your finetuned model
  - Be cheaper & faster
  - Require less maintenance

Key unresolved questions:
- When do you switch to a new base model?
- Is the improvement worth refinetuning?
- Do you revalidate old data?
- Do you keep multiple models alive?

✅ Many teams underestimate this hidden cost.

---

## 4. Finetuning Is Rarely the Right First Step

**Industry best practice**
1. Start with prompting
2. Add RAG
3. Add tools/agents
4. Improve evaluation & metrics
5. Only then consider finetuning

### Common anti-pattern
> “Prompting doesn’t work → we must finetune”

Reality (very common in practice):
- Prompts were:
  - Poorly written
  - Barely tested
  - Not representative of real data
- Metrics unclear or missing

After systematic prompt iteration → performance becomes sufficient.

✅ Interview-ready insight  
> Most “prompting doesn’t work” claims are actually *prompting wasn’t done seriously*.

---

## 5. Beware the “Domain-Specific Model” Fallacy

### Myth
> “General-purpose models don’t work for domain-specific tasks. We must finetune or train our own.”

### Reality
- General-purpose models improve rapidly
- They often outperform specialized models—even in niche domains

#### Case study: BloombergGPT vs GPT-4
- BloombergGPT:
  - 50B parameters
  - ~$1.3–2.6M compute cost (excluding data)
- GPT-4-0314:
  - Zero-shot
  - **Significantly outperformed BloombergGPT** on financial benchmarks

✅ Lesson  
> A strong general-purpose model can beat a costly domain-specific model.

That said:
- Bloomberg still gained **in-house control & expertise**
- Specialized models may still win in proprietary or internal contexts

---

## 6. Prompting & Finetuning Are Complementary, Not Competing

Prompt experimentation builds:
- Evaluation pipelines
- Annotation standards
- Metrics
- Experiment tracking

These are **prerequisites** for successful finetuning.

✅ Prompting failure without structure ≠ finetuning justification

---

## 7. Token Cost Is No Longer a Strong Finetuning Argument

**Old benefit**
- Finetuning reduced prompt size
- Fewer examples → fewer tokens → lower cost

**Today**
- Prompt caching mitigates this
- Reused prompt segments can be cached

Remaining limitation:
- Prompt examples still limited by context length
- Finetuning has *no such limit*

✅ Net takeaway  
Token savings alone is **not** a strong finetuning justification anymore.

---

## Decision Summary: When *Not* to Finetune

❌ Avoid finetuning when:
- You’re early in experimentation
- Prompting hasn’t been explored systematically
- Tasks are diverse
- Team lacks ML & MLOps maturity
- Maintenance budget is unclear
- General-purpose models already perform well
- ROI is marginal

✅ Prefer finetuning only when:
- Prompting + RAG clearly hit a ceiling
- Task is narrow, repetitive, high-volume
- Structured reliability is critical
- You can afford long-term maintenance

---

## Interview-Ready One-Liners

- **Why not finetune immediately?**  
  > Finetuning is expensive, brittle, and task-specific; prompting often delivers sufficient performance faster.

- **Main finetuning risk?**  
  > Improving one task while degrading others—especially in multi-task applications.

- **Domain-specific models vs GPT-4?**  
  > Strong general-purpose models frequently outperform costly domain-specific models.

- **Best practice?**  
  > Exhaust prompting and system design before committing to finetuning.

---
# Finetuning vs RAG  
**How to decide, when to use what, and how they work together (Interview-ready summary)**

---

## Core Decision Principle

> **Choose RAG or finetuning based on *why* the model fails — not on preference.**

Model failures fall into **two fundamental categories**:

| Failure Type | Root Cause | Best First Fix |
|-------------|-----------|---------------|
| **Information-based** | Model lacks or has outdated knowledge | **RAG** |
| **Behavior-based** | Model knows facts but behaves incorrectly | **Finetuning** |

This distinction is critical and frequently tested in interviews.

---

## 1. Information-Based Failures → Prefer RAG

### What are information-based failures?
Failures caused by **missing, private, or outdated knowledge**, for example:

- The model **doesn’t have the information**
  - Private company data
  - Internal policies
  - User-specific data
- The model has **stale knowledge**
  - Training cutoff before recent events
  - New product releases
  - Current statistics

📌 Example  
> *“How many studio albums has Taylor Swift released?”*  
Model answers **10 instead of 11** due to a training cutoff.

---

### Why RAG Works Better Here
- Injects **fresh and relevant knowledge at inference time**
- Avoids retraining the model
- Reduces hallucinations
- Scales better for frequently changing data

📚 **Key research finding**  
**Ovadia et al. (2024)** show that:

- **Base model + RAG** consistently outperforms:
  - Finetuned models
  - Finetuned + RAG (in many cases)

#### Evidence (Current Events QA)

| Model Setup | Performance |
|-----------|-------------|
| Base model | Low |
| **Base + RAG** ✅ | **Highest** |
| Finetuned only | Low–Moderate |
| Finetuned + RAG | Often worse than Base + RAG |

✅ Interpretation  
> Finetuning can *specialize* a model but also reduce its flexibility and general knowledge.

---

## 2. Behavior-Based Failures → Prefer Finetuning

### What are behavior-based failures?
Failures where the model:
- Produces **factually correct but unusable outputs**
- Fails to follow required **formats or schemas**
- Generates **irrelevant or underspecified responses**

📌 Examples
- Technical specs are correct but miss required detail
- HTML code doesn’t compile
- SQL / DSL / JSON is syntactically invalid
- Model ignores style, tone, or safety constraints

---

### Why Finetuning Helps
Finetuning:
- Aligns model **behavior**, not knowledge
- Improves:
  - Output format adherence
  - Task-specific relevance
  - Consistency and style
- Is especially effective for:
  - **Semantic parsing**
  - Domain-specific syntax
  - Structured outputs

🧠 Reminder  
Strong models already do well on:
- JSON
- YAML
- Regex

They struggle more with:
- Custom DSLs
- Rare programming languages
- Complex enterprise schemas

---

## 3. Key Mental Model (Very Interview-Friendly)

> 🧠 **“Finetuning is for *form*, RAG is for *facts*.”**

| Technique | Improves |
|---------|----------|
| **RAG** | Knowledge, accuracy, freshness |
| **Finetuning** | Format, behavior, style |

⚠️ Important nuance  
- Finetuning *can* reduce hallucinations with **very high-quality data**
- Poor-quality finetuning data can **increase hallucinations**

---

## 4. If Both Issues Exist → Start with RAG

When a model has **both**:
- Missing knowledge ❌
- Behavioral issues ❌

✅ **Start with RAG**, because:
- Easier to implement
- No training data required
- No hosting or inference changes
- Often larger performance gains

📌 Practical advice  
Begin with **simple retrieval**:
- BM25 / keyword search
- Document filtering

Don’t jump straight to:
- Vector DBs
- Hybrid retrieval
- Complex pipelines

---

## 5. RAG vs Finetuning: Engineering Trade-offs

| Aspect | RAG | Finetuning |
|-----|----|-----------|
| Engineering complexity | Higher inference complexity | Higher training complexity |
| Inference pipeline | Retriever + model | Model only |
| Maintenance | Update data | Retrain model |
| Cost sensitivity | Retrieval + tokens | Training + hosting |
| Adaptability | High | Lower |

---

## 6. RAG and Finetuning Can Be Combined (But Not Always Helpful)

- RAG + finetuning can help **~43% of the time**
- In **~57% of cases**, RAG alone performs better

✅ Insight  
> Finetuning does not universally improve RAG — it can even hurt it.

---

## 7. Recommended Workflow (Industry Best Practice)

⚠️ **Evaluation must exist at every step**

### Step-by-step Adaptation Path

1. **Prompting only**
   - Clear instructions
   - Good examples
   - Prompt versioning
2. **Few-shot prompting**
   - 1–50 examples depending on task
3. **Add RAG**
   - Start with term-based retrieval
4. **Based on failure mode**
   - Information gaps → advanced RAG
   - Behavioral issues → finetuning
5. **Optional: RAG + finetuning**
   - Validate with metrics, not intuition

🎯 Embedding-based retrieval  
- Adds *inference* complexity

🎯 Finetuning  
- Adds *model development* complexity

---

## 8. Interview-Ready One-Liners

- **When should you choose RAG over finetuning?**  
  > When failures are caused by missing or outdated information.

- **What does finetuning improve that RAG can’t?**  
  > Output behavior, structure, syntax, and style.

- **Which usually gives bigger gains first?**  
  > RAG, especially for factual or knowledge-intensive tasks.

- **Golden rule?**  
  > Diagnose failure modes before choosing adaptation techniques.

---

## Final Takeaway

✅ **RAG answers “What should the model know?”**  
✅ **Finetuning answers “How should the model behave?”**

A strong AI engineer chooses **based on failure diagnosis, not hype.**
# Memory Bottlenecks in Finetuning  
**Why finetuning is memory-intensive, what exactly consumes memory, and how engineers mitigate it**

---

## Why Memory Is a Core Bottleneck

> **Memory, not compute, is the primary bottleneck for finetuning large foundation models.**

This applies to:
- **Inference** ✅
- **Finetuning** ✅✅ (much more severe)

Understanding *why* this happens is essential for:
- Choosing the right finetuning technique (full FT vs PEFT)
- Estimating required GPU hardware
- Making trade-offs between cost, speed, and model quality

---

## Key Takeaways (High-Impact Summary)

1. **Finetuning requires significantly more memory than inference**
   - Inference runs only the **forward pass**
   - Training requires **both forward and backward passes**

2. **Primary contributors to memory usage during finetuning**
   - Number of model parameters
   - Number of **trainable** parameters
   - Numerical precision (FP32, FP16, INT8, etc.)

3. **Trainable parameters dominate memory usage**
   - More trainable parameters → more gradients + optimizer state
   - This is why **freezing parameters reduces memory**

4. **PEFT (Parameter-Efficient Finetuning) exists to reduce memory**
   - Train only a small subset of parameters
   - Keep most weights frozen

5. **Quantization directly reduces memory footprint**
   - FP32 (4 bytes) → FP16 (2 bytes) → INT8 (1 byte) → INT4 (0.5 byte)
   - Example:
     - 13B parameters × FP32 = **52 GB**
     - 13B parameters × FP16 = **26 GB**

6. **Inference can run at very low precision**
   - FP16, INT8, INT4 are commonly used
   - Accuracy trade-offs are usually acceptable

7. **Training is sensitive to numerical precision**
   - Pure low-bit training is unstable
   - Mixed precision is the norm (FP16 + FP32)

---

## Why Finetuning Needs More Memory Than Inference

### Inference
- Only **forward pass**
- No parameter updates
- No gradients
- No optimizer states

✅ Memory mostly used for:
- Model weights
- Activations (temporary)

---

### Finetuning (Training)
Requires **backpropagation**, which adds major memory overhead.

Each training step has **two phases**:

1. **Forward pass**
   - Compute model outputs
2. **Backward pass**
   - Compute errors
   - Compute gradients
   - Update weights

✅ Each *trainable* parameter requires storing:
- Weight value
- Gradient
- Optimizer states (often multiple)

---

## Trainable vs Frozen Parameters

### Definitions
- **Trainable parameters**
  - Updated during finetuning
  - Require gradients + optimizer states
- **Frozen parameters**
  - Not updated
  - No gradient computation
  - Much cheaper memory-wise

📌 **Critical Insight**
> Memory usage during finetuning scales with the number of *trainable* parameters, not total parameters.

---

## Backpropagation: Where Memory Explodes

During the **backward pass**, for each trainable parameter:

1. Loss is computed  
2. Gradient is calculated  
   - One gradient per trainable parameter
3. Optimizer updates parameters  
   - Uses additional internal state

### Example: Adam Optimizer
For each parameter, Adam typically stores:
- Current weight
- Gradient
- First moment estimate
- Second moment estimate

➡️ **~3–4× memory overhead per trainable parameter**

This is why:
- Full-model finetuning is often infeasible on limited GPUs
- PEFT methods are widely adopted

---

## Numerical Precision and Memory

### Why precision matters
Each parameter is stored as a numeric value with fixed bit width.

| Precision | Bytes per Weight |
|---------|-----------------|
| FP32 | 4 bytes |
| FP16 / BF16 | 2 bytes |
| INT8 | 1 byte |
| INT4 | 0.5 byte |

### Example (13B-parameter model)

| Precision | Memory for Weights |
|---------|-------------------|
| FP32 | ~52 GB |
| FP16 | ~26 GB |
| INT8 | ~13 GB |
| INT4 | ~6.5 GB |

📌 **Important**
- These numbers are *just for weights*
- Finetuning adds:
  - Gradients
  - Optimizer states
  - Activations

---

## Why Mixed Precision Is Standard for Training

- Low precision → faster + cheaper
- But too low → unstable gradients

✅ Common strategy: **Mixed precision training**
- Forward pass: FP16 / BF16
- Backward pass + optimizer: FP32 where needed

This balances:
- Stability
- Performance
- Memory usage

---

## Why PEFT Exists (Big Picture)

> **Reducing trainable parameters is the single most effective way to reduce finetuning memory.**

PEFT techniques:
- Freeze most of the model
- Train a small number of parameters
- Avoid storing gradients for the full network

This directly attacks the dominant memory contributors:
- Gradients
- Optimizer states

---

## Interview-Ready One-Liners

- **Why does finetuning need more memory than inference?**  
  > Because backpropagation requires storing gradients and optimizer states for each trainable parameter.

- **What mainly determines finetuning memory usage?**  
  > The number of trainable parameters and numerical precision.

- **Why is PEFT so effective?**  
  > It reduces the number of parameters that need gradients and optimizer states.

- **Why can inference use lower precision than training?**  
  > Training is numerically sensitive; inference is more tolerant of approximation.

---

## Final Mental Model

✅ **Inference**  
> Weights only → forward pass → low memory

✅ **Finetuning**  
> Weights + gradients + optimizer state → backward pass → high memory

Understanding this is foundational for:
- Choosing finetuning strategies
- Estimating GPU needs
- Explaining PEFT and quantization convincingly in interviews
  
# Memory Math for Large Models
**How to estimate inference and training memory requirements (back-of-the-napkin math)**

Knowing how much memory a model needs helps you:
- Choose the right GPU/accelerator
- Decide whether a model fits on existing hardware
- Understand *why* training and inference require very different resources

Because real systems apply many optimizations, the formulas below are **approximations**, but they’re extremely useful for planning.

---

## Why Inference and Training Have Different Memory Profiles

- **Inference**
  - Only the **forward pass**
  - Mainly stores model weights + temporary activations
- **Training**
  - Forward + **backward pass**
  - Stores weights, activations, gradients, and optimizer states

This difference is one reason why:
- Some chips are optimized for **inference**
- Others are optimized for **training** (covered later in system design discussions)

---

## Memory Needed for Inference

### 1. Model Weights

Let:
- **N** = number of model parameters
- **M** = memory per parameter (bytes)

Memory for weights:

```

Weights memory = N × M

```

Examples:
- FP32 → 4 bytes
- FP16 / BF16 → 2 bytes
- INT8 → 1 byte

---

### 2. Activations and KV Cache

During inference, memory is also needed for:
- Activation values
- Key–value (KV) vectors for attention

These grow with:
- Sequence length
- Batch size

👉 A common rule of thumb:
> **Activation + KV memory ≈ 20% of weights memory**

---

### 3. Total Inference Memory (Approximation)

```

Inference memory ≈ N × M × 1.2

```

---

### Example: 13B-Parameter Model (FP16)

- Parameters: 13B
- Bytes per parameter: 2 (FP16)

```

Weights = 13B × 2 bytes = 26 GB
Total inference memory ≈ 26 × 1.2 = 31.2 GB

```

✅ **Conclusion**  
A GPU with **24 GB VRAM is not sufficient** to run this model for inference.

---

### Why Model Size Quickly Becomes a Problem

- 70B parameters × 2 bytes = **140 GB just for weights**
- Add activations → even more

📌 This is why:
- Multi-GPU inference
- Tensor/model parallelism
- Quantization  
are necessary at scale

---

## Memory Needed for Training (Finetuning)

Training memory includes **everything from inference**, plus more.

### Total Training Memory

```

Training memory = model weights + activations + gradients + optimizer states

```

---

## Gradients and Optimizer States

During backpropagation, **each trainable parameter** requires extra storage.

### Per-parameter overhead (depends on optimizer):

| Optimizer | Extra Values Stored |
|---------|---------------------|
| SGD | 0 |
| Momentum SGD | 1 |
| Adam | 2 |

Additionally:
- Each trainable parameter needs **1 gradient value**

So with **Adam**:
```

Total extra values = 1 (gradient) + 2 (optimizer states) = 3

```

---

### Example: Full Finetuning, 13B Parameters, Adam, FP16

- Trainable params: 13B
- Extra values per param: 3
- Bytes per value: 2

```

Gradients + optimizer memory =
13B × 3 × 2 bytes = 78 GB

```

That is **in addition to**:
- Weights
- Activations

---

### Example: PEFT with 1B Trainable Parameters

```

1B × 3 × 2 bytes = 6 GB

```

✅ This single calculation explains why **PEFT massively reduces memory usage**.

---

## The Hidden Giant: Activation Memory

Earlier, we assumed:
> Activation memory < weights memory

⚠️ **This is often false during training.**

- Activations must be stored for gradient computation
- For large transformers, activations can **dominate memory**

Research (Korthikanti et al., 2022) shows:
- Activation memory can be **larger than model weights**
- Especially for large models and long sequences

---

## Gradient Checkpointing (Activation Recomputation)

### Idea
Instead of storing all activations:
- Store only a subset
- Recompute others during the backward pass

### Trade-off

| Benefit | Cost |
|------|------|
| Much lower memory usage | Slower training |
| Enables larger models | More compute |

This technique is known as:
- **Gradient checkpointing**
- **Activation recomputation**

It is widely used in large-scale training.

---

## Quick Comparison: Inference vs Training

| Aspect | Inference | Training |
|-----|---------|----------|
| Forward pass | ✅ | ✅ |
| Backward pass | ❌ | ✅ |
| Gradients | ❌ | ✅ |
| Optimizer states | ❌ | ✅ |
| Activation storage | Limited | Large |
| Memory usage | Lower | Much higher |

---

## Interview-Ready Mental Models

- **Inference memory ≈ weights + small activation overhead**
- **Training memory ≈ weights + activations + (gradients × optimizer states)**
- **Adam ≈ 3× memory per trainable parameter**
- **PEFT works by shrinking the “trainable parameters” term**
- **Gradient checkpointing trades compute for memory**

---

## Practical Rule of Thumb

> If a model barely fits in memory for inference,  
> **it almost certainly will not fit for full finetuning on the same hardware.**

That’s why:
- Full finetuning is rare for large models
- PEFT + mixed precision + checkpointing are the standard stack
``
# Numerical Representations & Their Impact on Model Memory
**(High-signal summary for revision & interviews)**

Numerical representation (precision format) determines **how many bytes each model value occupies**, which directly impacts:
- Model memory footprint
- Training stability
- Inference quality
- Hardware compatibility

Reducing precision is one of the **most powerful levers** for scaling large models.

---

## Why Numerical Representation Matters

Memory used by model weights is:

```

model parameters × bytes per value

```

So:
- Halving bytes per value ⇒ **halving model weight memory**
- This affects **weights, gradients, optimizer states, activations**

---

## Floating-Point Formats (IEEE 754 Family)

Neural networks traditionally use **floating-point numbers**.

| Format | Bits | Bytes | Common Usage |
|-----|----|-----|---------------|
| FP64 | 64 | 8 | Scientific computing (NumPy default) |
| FP32 | 32 | 4 | Traditional deep learning |
| FP16 | 16 | 2 | Modern training & inference |

FP64 is **rarely used** for neural networks due to high memory cost.

---

## Specialized AI-Oriented Float Formats

Modern hardware introduced formats optimized for ML workloads:

| Format | Bits | Designed By | Notes |
|-----|-----|------------|------|
| BF16 | 16 | Google | Larger range, less precision |
| TF32 | ~19 | NVIDIA | FP32-like behavior on GPUs |

- **BF16** → Prioritizes *range*
- **FP16** → Prioritizes *precision*
- **TF32** → Trades precision for training speed on NVIDIA GPUs

---

## Integer (Fixed-Point) Formats

Increasingly popular for inference and sometimes finetuning:

| Format | Bits | Use Case |
|-----|----|---------|
| INT8 | 8 | Inference (widely used) |
| INT4 | 4 | Extreme compression (emerging) |

➡️ Integer formats drastically reduce memory but increase numerical error risk.

---

## Range vs Precision (Core Concept)

Each floating-point number consists of:
- **1 sign bit**
- **Range bits (exponent)** → how large/small values can be
- **Precision bits (significand/mantissa)** → numerical accuracy

### Trade-off:

| More Bits For | Effect |
|-------------|-------|
| Range | Can represent very large values |
| Precision | More accurate small differences |

---

## FP16 vs BF16: Critical Difference

Although both use **16 bits**, they behave very differently.

| Property | FP16 | BF16 |
|-------|-----|------|
| Range | Smaller | Larger |
| Precision | Higher | Lower |
| Overflow risk | Higher | Lower |
| Popular for | Inference | Training |

### Example Insight:
- FP16 may overflow to **INF**
- BF16 can still represent large values (less precise but safer)

---

## Precision Loss in Practice (Key Insight)

Converting from FP32 → lower precision **changes values**.

Example:
- FP32: `1234.56789`
- FP16: `1235.0` (≈0.035% error)
- BF16: `1232.0` (≈0.208% error)

➡️ **BF16 sacrifices precision to preserve range**

---

## Real-World Pitfall (Interview-Grade Example)

⚠️ **Loading a model in the wrong numerical format can destroy quality**

- Llama 2 weights were released in **BF16**
- Many users loaded them in **FP16**
- Result: noticeably worse performance

✅ Lesson:
> Always load models in the **format they were trained for**

---

## Quantization = Precision Reduction

Reducing precision is called **quantization**:

| Conversion | Effect |
|--------|------|
| FP32 → FP16 | 2× memory reduction |
| FP16 → INT8 | 4× memory reduction |
| FP16 → INT4 | 8× memory reduction |

Quantization is:
- ✅ Extremely effective for inference
- ⚠️ Risky for training (numerical stability)

---

## Training vs Inference Precision

### Inference
- Can use **lower precision safely**
- INT8 / INT4 increasingly common

### Training
- More sensitive to numerical errors
- Uses **mixed precision**
  - Some ops in FP32
  - Others in FP16 / BF16

---

## Choosing the Right Format (Decision Factors)

Depends on:
1. **Value distribution**
   - Do values span a wide range?
2. **Sensitivity to numerical error**
   - Is small drift acceptable?
3. **Hardware**
   - GPUs favor TF32 / FP16
   - TPUs favor BF16

---

## Interview-Ready Takeaways

- **Precision ↔ memory is a direct trade-off**
- **FP16 ≠ BF16** (same bits, different behavior)
- **BF16 is safer for training**
- **INT8/INT4 are powerful for inference**
- **Wrong precision loading = silent model degradation**
- **Quantization lowers memory but raises numerical risk**

---

## One-Line Mental Model

> *Precision controls memory, stability, and performance — and the “best” format depends on whether you are training or inferring.*

This understanding is **essential** for:
- Finetuning strategies
- Inference optimization
- System design interviews

# Quantization (Precision Reduction)

Reducing the number of bits needed to represent a model’s values directly reduces its
memory footprint. For example:
- A **10B-parameter model in FP32 (32-bit)** requires **40 GB** for weights.
- The same model in **FP16 (16-bit)** requires **20 GB**.

This reduction in precision—commonly referred to as **quantization**—is one of the
cheapest and most effective ways to reduce memory usage. It is:
- Straightforward to apply
- Largely architecture-agnostic
- Widely supported by modern ML frameworks

In ML literature, *low precision* generally refers to any format smaller than FP32.

---

## Quantization vs. Reduced Precision

Strictly speaking:
- **Quantization** → converting floats to **integer formats** (e.g., INT8, INT4)
- **Reduced precision** → converting to **lower-bit formats**, float or integer

In practice, *quantization* is used loosely to mean **any precision reduction**.
This document follows that convention.

---

## What to Quantize

Ideally, you quantize what consumes most memory **without significantly degrading
quality**.

### Major memory contributors at inference time:
- **Model weights**
- **Activations**
- **KV cache** (for transformer attention, discussed later)

### Common practice:
- ✅ **Weight quantization** (most common, most stable)
- ⚠️ **Activation quantization** (harder, more sensitive to errors)

Weight quantization usually offers the best memory savings with the least accuracy loss.

---

## When to Quantize

Quantization can happen:
1. **After training** → *Post-Training Quantization (PTQ)*
2. **During training** → *Training-aware quantization*

For most application developers, **PTQ is the dominant and recommended approach**.

---

## Inference Quantization

Originally, models were trained and served using FP32.
Since the late 2010s, inference precision has steadily decreased.

### Common inference precisions:
- FP16 / BF16 (standard)
- INT8 (widely deployed)
- INT4 (increasingly popular)

### Notable work:
- **LLM.int8()** – Dettmers et al. (2022)
- **QLoRA (4-bit)** – Dettmers et al. (2023)

### Mixed-precision inference
Different parts of the model use different precisions:
- Apple (2024): mixture of **2-bit and 4-bit**, averaging **3.5 bits/weight**
- NVIDIA Blackwell GPUs: native **4-bit float inference**

---

## Float vs Integer Quantization (8-bit and Under)

Below 8 bits, representation choices become critical.

Two main approaches:
- **Minifloats** (FP8, FP4)
- **Integers** (INT8, INT4) — more common and efficient

> Note: The smallest fully IEEE-compliant floating-point format is **4-bit**.

---

## Limits of Quantization

- Minimum possible: **1 bit**
- Extreme approaches have been explored:
  - BinaryConnect
  - Xnor-Net
  - BitNet

### BitNet b1.58 (Microsoft, 2024)
- Uses **1.58 bits per parameter**
- Comparable performance to **16-bit Llama 2**
- Demonstrated feasibility up to **3.9B parameters**

This suggests a future where **ultra-low-bit LLMs** are practical.

---

## Performance Impact of Reduced Precision

Reduced precision offers **two main speed advantages**:
1. **Larger batch sizes** (less memory per model copy)
2. **Faster arithmetic** (fewer bits → faster ops)

Example:
- 32-bit addition → ~32t time
- 16-bit addition → ~16t time

⚠️ However:
- Precision conversion overhead can cancel latency gains
- Performance depends on hardware support

---

## Downsides of Quantization

- Small rounding errors accumulate
- Out-of-range values can become:
  - `INF`
  - Arbitrary values
- Many small numerical errors can cause **large quality drops**

Reducing precision safely is an active research area spanning:
- Model architecture
- Training procedures
- Hardware design

---

## Post-Training Quantization (PTQ)

**Standard industry practice**:
1. Train the model in higher precision
2. Quantize for inference

Supported by:
- PyTorch
- TensorFlow
- Hugging Face Transformers

PTQ is often a **few lines of code**.

### On-device inference
Some devices only support quantized inference:
- TensorFlow Lite
- PyTorch Mobile

These frameworks provide built-in PTQ support.

---

## Training Quantization

Training-time quantization is less common but increasingly important.

### Two goals:
1. Improve **low-precision inference quality**
2. Reduce **training cost and time**

---

### Quantization-Aware Training (QAT)

- Simulates low precision (e.g., INT8) during training
- Training still runs in high precision
- Helps model adapt to quantization noise

Pros:
- Better inference quality at low precision

Cons:
- No training speedup
- Can increase training time

---

### Low-Precision Training

Training directly in lower precision can:
- Reduce memory usage
- Speed up training
- Eliminate train–serve precision mismatch

Examples:
- Character.AI trained models fully in **INT8**

Challenges:
- Backpropagation is **numerically fragile**
- Small rounding errors can compound
- Loss computation is especially sensitive

---

### Mixed-Precision Training (Most Common)

Typical strategy:
- Keep **master weights** in higher precision
- Use lower precision for:
  - Gradients
  - Activations
  - Some weights

Examples:
- LLM-QAT: weights + activations in 4-bit, embeddings in 16-bit

Most frameworks provide **Automatic Mixed Precision (AMP)** to manage this safely.

---

## Precision Across Training Phases

A common workflow:
1. **Pretraining** → high precision (FP32 / BF16)
2. **Finetuning** → lower precision (FP16 / INT8)

This allows:
- Large organizations to train models reliably
- Smaller teams to finetune models affordably

---

## Key Takeaways (Interview-Ready)

- Quantization is the **most effective way** to reduce model memory
- PTQ is the **default choice** for application developers
- Weight quantization is more stable than activation quantization
- Below 8 bits, representation choice matters significantly
- Reduced precision improves both **memory and compute**
- Training in low precision is harder than inference
- Mixed precision is the practical sweet spot

> **Rule of thumb:**  
> *Train high, serve low — unless you really know what you’re doing.*
# Finetuning Techniques (Memory-Efficient Adaptation)

Finetuning large models is **memory-intensive** and expensive. As models scale,
traditional finetuning approaches become inaccessible to most practitioners.
This section focuses on **memory-efficient finetuning techniques**, especially
**Parameter-Efficient Finetuning (PEFT)**, which make adaptation practical on
commodity hardware.

It also introduces **model merging**, a complementary (and more experimental)
approach for combining capabilities of multiple models.

---

## Why Finetuning Is Expensive

Finetuning cost scales with:
- Number of **trainable parameters**
- **Precision** (FP32 vs FP16 vs lower)
- **Optimizer states**
- **Activation memory**

### Full Finetuning (Baseline)

**Full finetuning**:
- Updates *all* model parameters
- Same number of trainable parameters as total parameters

Example: **7B-parameter model (FP16 + Adam)**

| Component | Memory |
|---------|--------|
| Model weights | 14 GB |
| Gradients + optimizer states | 42 GB |
| **Total (excluding activations)** | **~56 GB** |

➡️ Exceeds most consumer GPUs (12–24 GB, even many 48 GB cards once activations are included).

---

## Strategies to Fit Finetuning on Limited Hardware

Two broad approaches:

### 1. Reduce the Memory Footprint
- **Quantization**
- **Parameter-Efficient Finetuning (PEFT)**

### 2. Use Hardware More Efficiently
- **CPU offloading**
- **ZeRO / DeepSpeed / FSDP**
  - Store gradients, optimizer states, or parameters on CPU instead of GPU

---

## Partial Finetuning (Early Attempt)

Instead of updating all parameters:
- Freeze most layers
- Finetune only a subset (e.g., last layer(s))

Example:
- 10-layer model → finetune last 1 layer → **10% trainable parameters**

### Problem
Partial finetuning is **parameter-inefficient**:
- Requires **many parameters** to match full finetuning
- Empirically wastes capacity

🔬 **Houlsby et al. (2019)**:
- BERT-large needed **~25% of parameters updated**
- to reach full-finetuning performance on GLUE

➡️ This defeats the purpose of reducing memory.

---

## Parameter-Efficient Finetuning (PEFT)

**Goal**:  
Achieve performance close to full finetuning using **orders of magnitude fewer
trainable parameters**.

> A method is considered PEFT if it:
> - Matches full finetuning performance
> - Uses **~1–5% trainable parameters** (or less)

---

## Adapters: The Birth of PEFT

Introduced by **Houlsby et al. (2019)**

### Core Idea
- **Insert small trainable modules (adapters)** into the pretrained model
- **Freeze original weights**
- Train **only adapters**

### Architecture
- Two adapter modules per transformer block
- Placed after key sublayers
- Very small compared to the full model

### Results (BERT, GLUE benchmark)
- ✅ Performance within **0.4%** of full finetuning
- ✅ Only **~3% trainable parameters**

---

## Trade-offs of Adapter-Based PEFT

### ✅ Pros
- Massive reduction in memory usage
- Enables finetuning on affordable hardware
- Often **sample-efficient**
  - Thousands of examples instead of millions

### ❌ Cons
- **Inference latency increases**
  - Extra layers added
  - More computation per forward pass

This latency cost motivated newer PEFT methods that avoid adding layers.

---

## Key Insights for Interviews / Practice

- **Full finetuning does not scale** for large models
- Partial finetuning is **parameter-inefficient**
- **PEFT is the practical standard** for adapting LLMs
- Adapters proved PEFT is viable but introduced inference overhead
- Modern PEFT techniques aim to:
  - Minimize trainable parameters
  - Avoid increasing inference cost
  - Maintain full-finetune quality

➡️ **Next logical step**:  
Understanding modern PEFT methods like **LoRA**, which dominate current practice.

---

## Quick Comparison

| Approach | Trainable Params | Memory | Inference Cost | Practical Today |
|--------|------------------|--------|----------------|-----------------|
| Full finetuning | 100% | ❌ Very high | ✅ Same | Rare |
| Partial finetuning | 10–25% | ⚠️ High | ✅ Same | Rare |
| Adapter PEFT | ~3% | ✅ Low | ❌ Higher | Sometimes |
| Modern PEFT (LoRA, etc.) | <1–2% | ✅✅ Very low | ✅ Same | ✅ Standard |

---

### Mental Model (Good for Interviews)

> **Full finetuning** retrains the brain.  
> **PEFT** adds tiny skill-specific “muscle memory” on top of a frozen brain.

# Parameter-Efficient Finetuning (PEFT): Techniques Overview

PEFT methods aim to adapt large foundation models **without updating all parameters**.
They dramatically reduce memory, compute, and data requirements while retaining
performance close to full finetuning.

At a high level, **PEFT techniques fall into two major categories**:

1. **Adapter-based (additive) methods**
2. **Soft prompt-based methods**

These categories reflect *how* the model is adapted:  
by **changing the architecture** vs **changing how inputs are processed**.

---

## 1. Adapter-Based Methods (Additive Methods)

### Core Idea
- **Add small, trainable modules** to a frozen pretrained model
- Train **only the added parameters**
- Original model weights remain unchanged

Because new parameters are *added*, these are also called **additive methods**.

---

### 🔹 LoRA (Low-Rank Adaptation) — *Dominant Standard*

- **Most popular PEFT method by far**
- Introduced by **Hu et al. (2021)**
- Replaces expensive weight updates with **low-rank matrix decompositions**
- Avoids adding latency-heavy layers (unlike early adapters)

📌 **Why LoRA dominates**
- Very small number of trainable parameters
- No inference-time latency increase
- Simple to integrate
- Works well across NLP, vision, and multimodal models

➡️ Because of its importance, **LoRA deserves (and gets) its own deep dive**.

---

### Other Adapter-Based Methods

| Method | Key Idea | Notable Strength |
|------|--------|----------------|
| **BitFit** (Zaken et al., 2021) | Train only bias terms | Extremely lightweight |
| **IA³** (Liu et al., 2022) | Scale attention & FFN activations | Excellent for multi-task tuning; can outperform LoRA & full finetuning |
| **LongLoRA** (Chen et al., 2023) | LoRA + attention modifications | Efficient long-context finetuning |

📌 **Important Interview Insight**  
> Adapter-based ≠ LoRA only. LoRA is popular, but **IA³ has shown superior results
in some multi-task settings**.

---

## 2. Soft Prompt-Based Methods

### Core Idea
- Keep model weights **fully frozen**
- Introduce **trainable vectors (tokens)** that guide model behavior
- These tokens are learned via backpropagation

These vectors are called **soft prompts**.

---

### Soft Prompts vs Hard Prompts

| Aspect | Hard Prompts | Soft Prompts |
|-----|-------------|-------------|
| Representation | Discrete text tokens | Continuous vectors |
| Human-readable | ✅ Yes | ❌ No |
| Trainable | ❌ No | ✅ Yes |
| How they guide model | Instructions in text | Learned embeddings |

➡️ Soft prompting sits **between prompt engineering and finetuning**.

---

### Popular Soft Prompt Variants (Often Confusing)

| Method | Where Soft Prompts Are Inserted |
|-----|-------------------------------|
| **Prompt Tuning** (Lester et al., 2021) | At input embeddings only |
| **Prefix Tuning** (Li & Liang, 2021) | At every transformer layer |
| **P-Tuning** (Liu et al., 2021) | Flexible insertion strategies |

📌 Even experts struggle to recall exact differences on the spot—**frameworks usually abstract this away**.

---

## Adoption Trends (Real-World Signal)

Analysis of **1,000+ GitHub issues** in `huggingface/peft` (Oct 2024):

- **LoRA overwhelmingly dominates usage**
- Soft prompting is:
  - Less common today
  - Seeing **growing interest** from teams wanting more flexibility than prompt engineering
  - Attractive when full finetuning is too expensive

📌 **Interpretation**
- LoRA = production workhorse  
- Soft prompts = niche but promising middle ground

---

## Choosing Between PEFT Methods

### Use Adapter-Based (LoRA / IA³) When:
- You need **strong task adaptation**
- Inference latency matters
- You want industry-proven robustness

### Use Soft Prompt-Based Methods When:
- You want to avoid modifying model internals
- You prefer fast iteration
- You need more control than prompts but less than finetuning

---

## Key Interview Takeaways (Must-Remember)

- PEFT methods fall into **adapter-based** and **soft prompt-based**
- **LoRA is the de facto industry standard**
- Adapter-based methods *add parameters*, soft prompts *add trainable tokens*
- Soft prompts are **not human-readable**
- IA³ can outperform LoRA in **multi-task scenarios**
- Most PEFT libraries support all these methods out of the box

---

## Mental Model (Quick Recall)

> **Adapter-based PEFT** changes *how the model computes*.  
> **Soft prompt PEFT** changes *what the model is conditioned on*.

---

✅ **Next logical step:**  
A **deep dive into LoRA** — how it works mathematically, why it avoids latency,
and when to prefer it over other PEFT techniques.
``
# LoRA (Low-Rank Adaptation)

LoRA (Low-Rank Adaptation), introduced by **Hu et al. (2021)**, is a **parameter-efficient
finetuning (PEFT)** technique that adapts large models **without increasing inference
latency**.

Unlike earlier adapter-based methods (e.g., Houlsby et al., 2019), LoRA **does not add
new layers** to the model. Instead, it modifies existing weight matrices in a way that
can be **merged back into the original model weights** after finetuning.

---

## Core Idea

LoRA applies **low-rank factorization** to selected weight matrices and **trains only
the low-rank components**, keeping the original weights frozen.

This dramatically reduces:
- Number of trainable parameters
- Training memory usage
- Training compute

While preserving:
- Inference speed
- Base model capacity

---

## How LoRA Works (Step by Step)

Consider a weight matrix:

- **W ∈ ℝⁿˣᵐ** (original pretrained weight matrix)

### 1. Low-Rank Decomposition

Choose a rank **r**, where `r ≪ min(n, m)`.

Create two trainable matrices:
- **A ∈ ℝⁿˣʳ**
- **B ∈ ℝʳˣᵐ**

Their product:
```

W_AB = A · B   ∈ ℝⁿˣᵐ

```

This product has the **same shape as W**, but far fewer parameters:
```

Params(W_AB) = r(n + m)  ≪  nm

```

---

### 2. Modify the Weight Matrix

Instead of replacing `W`, LoRA **adds a low-rank update**:

```

W′ = W + (α / r) · (A · B)

```

Where:
- **α (alpha)** = scaling factor
- **r** = LoRA rank

The `(α / r)` factor stabilizes training.

---

### 3. Finetuning Phase

- ✅ **Train A and B**
- ❌ **Freeze W**

Gradients are computed **only for A and B**, drastically reducing memory usage.

---

### 4. Inference Phase (Key Advantage)

After training:
- Merge `A · B` into `W`
- Discard A and B

✅ **No extra layers**  
✅ **No additional inference latency**  
✅ **Model behaves like a fully finetuned model**

---

## Why Low-Rank Factorization Works

LoRA relies on a well-studied concept: **low-rank approximation**.

### Example

A **9 × 9 matrix**:
- Full-rank parameters: `81`

Low-rank (`r = 1`) factorization:
- `9 × 1` + `1 × 9` = `18` parameters

➡️ **~78% reduction**, at the cost of approximation.

Higher rank → better approximation → more parameters.

---

## Why Is LoRA So Effective?

### The Intrinsic Dimension Hypothesis

Several studies (Li et al., 2018; Aghajanyan et al., 2020; Hu et al., 2021) show that:

- Large neural networks have **low intrinsic dimensionality**
- Most task-specific adaptations live in **a small subspace**
- Pretraining **compresses the model’s intrinsic dimension**

> **Surprisingly, larger pretrained models tend to have _lower_ intrinsic dimension
for downstream tasks.**

This explains:
- Why finetuning needs far fewer parameters than pretraining
- Why LoRA works so well with small `r`

---

## Performance Efficiency

From the LoRA paper (GPT-3 experiments):

- **~4.7 million trainable parameters**
- Only **0.0027%** of full finetuning parameters
- Achieves **comparable or better performance** than full finetuning on multiple tasks

This makes LoRA:
- ✅ Parameter-efficient
- ✅ Sample-efficient
- ✅ Compute-efficient

---

## Why Not Use LoRA for Pretraining?

If low-rank adaptation works so well, why not pretrain models in low rank?

### Attempts at Low-Rank Training

Historical and recent efforts:
- Sainath et al. (2013)
- Jaderberg et al. (2014)
- SqueezeNet (2016)
- ReLoRA (2023): up to 1.3B models
- GaLore (2024): promising at 1B–7B scale

### The Limitation

Low-rank training works well at smaller scales, but:

- Pretraining needs **full-rank capacity** to:
  - Explore the hypothesis space
  - Learn rich representations
  - Compress intrinsic dimensionality

> Pretraining appears to **require full-rank flexibility first**, after which
> low-rank methods become highly effective.

An open research question:
> *How much full-rank pretraining is required before switching to low-rank training?*

---

## Key Takeaways (Interview-Ready)

- LoRA adapts models via **low-rank updates to existing weights**
- Only matrices **A and B** are trained; base weights are frozen
- Updates can be **merged**, causing **zero inference latency**
- Works because pretrained models have **low intrinsic dimension**
- LoRA achieves near full-finetune performance with **orders of magnitude fewer parameters**
- Full-rank pretraining likely remains necessary before low-rank adaptation

---

## One-Line Mental Model

> **LoRA fine-tunes directions in weight space that matter, instead of re-learning the whole space.**

---
``
# LoRA Configurations

To apply **LoRA (Low-Rank Adaptation)** effectively, you must make two key decisions:

1. **Which weight matrices to apply LoRA to**
2. **What rank (`r`) to use for each factorization**

The quality, memory footprint, and efficiency of LoRA finetuning depend heavily on
these choices.

---

## Where to Apply LoRA

### Architecture Considerations

LoRA can be applied to individual weight matrices. Therefore, its efficiency depends
on:
- The **model architecture**
- The **types of weight matrices present**

Although LoRA has been explored in other architectures (e.g., CNNs), it is used
**primarily with transformer models**, where it is both simple and effective.

---

### Attention Matrices (Most Common)

In transformer-based models, LoRA is most commonly applied to the **attention
module**, specifically to these matrices:

- **Wq** – Query projection  
- **Wk** – Key projection  
- **Wv** – Value projection  
- **Wo** – Output projection  

Typically:
- LoRA is applied **uniformly** across all layers for the chosen matrix type  
- For example, applying LoRA to `Wq` means applying it to *every* query matrix in
  the model

---

### Memory-Constrained Decisions

In practice, hardware constraints often limit the total number of **trainable LoRA
parameters**. Given a fixed parameter budget, the question becomes:

> **Which matrices should receive LoRA to maximize performance?**

---

## Empirical Findings from the LoRA Paper (Hu et al., 2021)

When finetuning **GPT-3 175B**, the authors constrained LoRA to a budget of:

- **18 million trainable parameters**
- ≈ **0.01% of total model parameters**

GPT-3 175B details:
- **96 transformer layers**
- **Model dimension = 12,288**

---

### Parameter Budget Breakdown

Applying LoRA with rank `r = 2` to **all four attention matrices** yields:

```

Per layer:
(12,288 × 2 × 2) × 4 = 196,608 parameters

Whole model:
196,608 × 96 = 18,874,368 parameters

```

---

### Performance Comparison (18M Parameter Budget)

| LoRA Target | Rank (r) | WikiSQL | MultiNLI |
|------------|----------|---------|----------|
| Wq         | 8        | 70.4    | 91.0     |
| Wk         | 8        | 70.0    | 90.8     |
| Wv         | 8        | 73.0    | 91.0     |
| Wo         | 8        | 73.2    | 91.3     |
| Wq + Wk    | 4        | 71.4    | 91.3     |
| Wq + Wv    | 4        | 73.7    | 91.3     |
| Wq + Wk + Wv + Wo | 2 | **73.7** | **91.7** |

**Key takeaway**
- Applying LoRA to **all four attention matrices**, even with smaller rank, yielded
  the best overall results
- If limited to only **two matrices**, **Wq + Wv** is usually the best choice

---

## Beyond Attention: Feedforward Layers

Subsequent empirical results suggest expanding LoRA beyond attention matrices:

- **Databricks (2023)**  
  - Largest performance gains came from applying LoRA to **feedforward layers**
- **Fomenko et al. (2024)**  
  - Feedforward LoRA complements attention-based LoRA
  - Attention-based LoRA is usually more effective under strict memory limits

**Practical Insight**
> If memory allows, applying LoRA to *both attention and feedforward layers* often
yields the best performance.

---

## Choosing the LoRA Rank (`r`)

### Typical Range

Most applications work well with:

- `r ∈ [4, 64]`

Lower `r`:
- ✅ Fewer trainable parameters
- ✅ Lower memory usage

---

### Does Higher `r` Always Help?

Surprisingly, **no**.

Findings:
- Hu et al. (2021): Increasing `r` did **not** consistently improve performance
- Databricks (2023): Beyond a threshold, increasing `r` showed no quality gain
- Higher `r` can even cause **overfitting**

However:
- Some tasks benefit from higher rank  
- Raschka (2023) found **r = 256** optimal for certain workloads

✅ **Conclusion**: Start small and scale only if necessary.

---

## The Scaling Factor α (Alpha)

LoRA merges matrices using:

```

W′ = W + (α / r) · (A · B)

```

### Practical Guidelines

- The **ratio α : r** often works well between **1:8 and 8:1**
- Smaller `r` → larger `α`
- Larger `r` → smaller `α`

There is **no universal best value**—experimentation is required.

---

## Serving LoRA Adapters

LoRA significantly simplifies **model serving**, especially when handling multiple
specialized models.

---

### Option 1: Merge LoRA into Base Model (Offline)

- Merge `A · B` into `W` before deployment
- ✅ Zero inference latency overhead
- ❌ One merged model per task/customer

✅ Best for **single-LoRA deployment**

---

### Option 2: Keep Adapters Separate (Online Merge)

- Store:
  - Base matrix `W`
  - Multiple LoRA adapters `(A, B)`
- Merge happens during inference
- ✅ Massive storage savings
- ✅ Fast task switching
- ❌ Small inference latency overhead

✅ Best for **multi-LoRA serving**

---

### Storage Comparison Example

Given:
- `W`: 4096 × 4096 = **16.8M parameters**
- LoRA rank `r = 8`
- `A + B`: `4096 × 8 × 2` = **65,536 parameters**

For **100 LoRA adapters**:

**Option 1 (Merged models)**
```

16.8M × 100 = 1.68B parameters

```

**Option 2 (Shared base + adapters)**
```

16.8M + (65,536 × 100) = 23.3M parameters

```

✅ Over **70× reduction** in storage

---

### Operational Benefits of Multi-LoRA Serving

- Rapid switching between tasks/customers
- Only adapter weights need to be loaded
- Ideal for:
  - Per-customer models
  - Per-task specialization
  - On-device deployment

**Example**
- Apple (2024) used multiple LoRA adapters on a shared 3B base model
- Combined with quantization, they served multiple features **on-device**

---

## LoRA Ecosystem and Reusability

- LoRA adapters are modular and reusable
- Public LoRA adapters available on:
  - **Hugging Face**
  - **AdapterHub**
- Can be treated similarly to pretrained models

---

## Trade-offs and Limitations

### Pros
- Massive memory savings
- Fast finetuning
- Easy multi-model serving
- Strong performance with small parameter counts

### Cons
- Slightly weaker than full finetuning
- Requires understanding model internals
- Needs architectural hook points

✅ These issues are largely mitigated by PEFT frameworks:
- Hugging Face **PEFT**
- **Axolotl**
- **Unsloth**
- **LitGPT**

For popular base models, LoRA support is usually plug-and-play.

---

## Summary Cheat Sheet

- Default choice: **Apply LoRA to Wq + Wv**
- Best overall (if memory allows): **Attention + FFN**
- Typical rank: **4–64**
- Alpha tuning matters as much as rank
- Use **merged adapters** for single-task serving
- Use **separate adapters** for multi-task / multi-tenant systems

---
``
# Quantized LoRA (QLoRA and Variants)

## Motivation: Where Memory Really Goes

Although LoRA is **parameter-efficient**, its memory footprint is already *tiny*
compared to the base model’s weights. Reducing LoRA parameters further brings **very
little overall memory benefit**.

### Memory Breakdown (Illustrative)

| Model | Model Weights (16-bit) | LoRA Params (r=2, Q+K) | LoRA Memory |
|------|------------------------|------------------------|------------|
| Llama 2 (13B) | 26 GB | 3.28M | **6.55 MB** |
| GPT-3 (175B)  | 350 GB | 18.87M | **37.7 MB** |

✅ **Key insight**  
> LoRA adapters consume **<0.2%** of total memory.  
Optimizing LoRA size alone does not meaningfully reduce memory usage.

---

## The Real Lever: Quantization

Instead of shrinking LoRA further, **quantizing the base model** yields massive
memory gains.

This insight led to **Quantized LoRA** approaches, most notably **QLoRA**.

---

## QLoRA (Dettmers et al., 2023)

### Core Idea

QLoRA combines:
- **LoRA adapters** (trainable, low-rank)
- **4-bit quantized base model weights**

📌 During training:
- Model weights are stored in **4-bit**
- During forward/backward pass, weights are **temporarily dequantized to BF16**
- LoRA parameters remain trainable and high precision

---

### NF4: NormalFloat-4

QLoRA introduces a custom 4-bit format:

- **NF4 (NormalFloat-4)**
- Designed specifically for pretrained LLM weights
- Based on the observation that weights:
  - Follow a **normal distribution**
  - Have **median ≈ 0**

✅ NF4 preserves accuracy better than uniform INT4

---

### Paged Optimizers

QLoRA adds **paged optimizers** to handle limited GPU memory:

- Automatically offloads optimizer states between **CPU ↔ GPU**
- Especially useful for:
  - Long sequence lengths
  - Large models

✅ Enables finetuning **65B models on a single 48 GB GPU**

---

## Practical Impact

Using QLoRA, the authors finetuned:
- LLaMA 7B → 65B
- All in **4-bit**
- Resulting models: **Guanaco family**

---

## Performance: Guanaco vs Commercial Models (May 2023)

*(Evaluated using GPT-4 as judge)*

| Model | Size (GB) | Elo Score |
|-----|-----------|-----------|
| GPT-4 | – | **1348 ± 1** |
| **Guanaco 65B** | 41 | **1022 ± 1** |
| Guanaco 33B | 21 | 992 ± 1 |
| Vicuna 13B | 26 | 974 ± 1 |
| ChatGPT | – | 966 ± 1 |
| Guanaco 13B | 10 | 916 ± 1 |
| Bard | – | 902 ± 1 |
| Guanaco 7B | 6 | 879 ± 1 |

✅ **Key takeaway**
- Guanaco 65B **often outperformed ChatGPT**
- Achieved using:
  - 4-bit base model
  - LoRA adapters
  - Single GPU finetuning

---

## Trade-offs and Limitations

### Pros
✅ Drastically reduced memory footprint  
✅ Enables finetuning of very large models on limited hardware  
✅ Competitive performance with closed models  

### Cons
⚠️ **NF4 quantization is computationally expensive**  
⚠️ Training can be **slower** due to quantization/dequantization  
⚠️ More complex training pipeline  

> QLoRA trades **memory savings** for **increased training time**

---

## Beyond QLoRA: Emerging Quantized LoRA Variants

Quantized LoRA is an **active research area**. Notable alternatives include:

- **QA-LoRA** (Xu et al., 2023)
- **ModuLoRA** (Yin et al., 2023)
- **IR-QLoRA** (Qin et al., 2024)

Each explores different trade-offs between:
- Quantization granularity
- Training speed
- Stability
- Accuracy

---

## Interview-Ready Summary

**Why Quantized LoRA exists**
- LoRA is already small → shrinking it further doesn’t help much
- The real memory bottleneck is the **base model**

**What QLoRA does**
- Stores base model in **4-bit (NF4)**
- Trains LoRA adapters in higher precision
- Dequantizes weights only when needed

**Why it matters**
- Makes **single-GPU finetuning of 65B models possible**
- Enables open-source models to compete with ChatGPT-class systems

**When to use QLoRA**
- Limited GPU memory (≤48 GB)
- Want large-model performance
- Willing to trade training speed for feasibility

---

## Quick Reference

- LoRA memory ≪ model memory → optimize model weights
- QLoRA = LoRA + 4-bit base model
- NF4 optimized for pretrained weights
- Paged optimizers handle GPU memory limits
- Training slower, but **memory savings are massive**

---
# Model Merging and Multi-Task Finetuning

Finetuning customizes a **single model**; **model merging** creates a new model by
**combining multiple models**. Merging provides greater flexibility and can be used
with or without additional finetuning.

A merged model can:
- Perform **better** than its individual components
- Require **less memory** than serving multiple models
- Support **multi-task learning** without catastrophic forgetting
- Enable **on-device deployment** where memory and connectivity are limited

---

## Why Model Merging?

### 1. Performance Gains
Different models often excel at different parts of a task.

**Example**
- Model A answers 60% of questions (early subset)
- Model B answers a different 60% (later subset)
- A merged model might answer **80%+ overall**

---

### 2. Reduced Memory Footprint
Instead of deploying multiple task-specific models:
- Merge them into **one model**
- Save memory and inference cost

This is especially effective for **adapter-based finetuning (LoRA)**:
- If multiple LoRA adapters share the same base model,
- They can often be merged into a **single adapter**

---

### 3. Multi-Task Finetuning Without Forgetting

Without model merging, multi-task finetuning usually follows one of two approaches:

#### A. Simultaneous Finetuning
- Train on all tasks at once
- Requires **more data and training**
- Harder optimization problem

#### B. Sequential Finetuning
- Train on task A, then task B, etc.
- Suffers from **catastrophic forgetting**
- Model forgets earlier tasks (Kirkpatrick et al., 2016)

---

### ✅ Model Merging as an Alternative

1. Finetune separate models **in parallel**, each on a different task
2. Merge them afterward

Benefits:
- Each task is learned independently
- Reduced risk of catastrophic forgetting
- No need for massive multi-task datasets

---

## Model Merging for On-Device Deployment

On-device environments include:
- Phones
- Laptops
- Cars
- Smartwatches
- Warehouse robots

Constraints:
- Limited memory
- Limited or unreliable internet
- Privacy requirements

Instead of deploying **multiple models**:
- Merge them into **one multi-capability model**
- Reduce memory usage and inference cost
- Improve data privacy by keeping inference local

💡 Offloading inference to user devices also reduces cloud costs.

---

## Model Merging and Federated Learning

Model merging can act as a form of **federated learning** (McMahan et al., 2016):

1. Deploy a base model to many devices
2. Each device fine-tunes the model on local data
3. Periodically merge all resulting models into a new global model

Each merged model captures:
- Knowledge from multiple data sources
- Without centrally collecting raw data

---

## Model Merging vs Ensembling

Model merging evolved from **ensemble methods**, but the two differ fundamentally.

### Ensembling
- Keeps all models separate
- Combines outputs (e.g., voting, averaging)
- Multiple inference calls per request
- Higher inference cost

### Model Merging
- Combines **parameters**
- Produces a **single model**
- One inference call
- Lower deployment and inference cost

📌 Many top models on the Hugging Face Open LLM Leaderboard are **merged models**.

---

## High-Level Model Merging Approaches

Model merging methods differ in how parameters are combined. At a high level, there
are **three main approaches**:

### 1. Summing
- Combine parameters by weighted averaging or summation
- Common for models with matching architectures
- Simple and computationally cheap

---

### 2. Layer Stacking
- Take layers from different models and stack them
- Increases depth
- Can preserve specialized behaviors per layer block

---

### 3. Concatenation
- Combine representations or parameters side-by-side
- Often followed by projection layers
- Flexible but may increase model size

---

### Hybrid Approaches
You can combine methods:
- Sum some layers
- Stack others
- Concatenate task-specific components

---

## Key Takeaways

- **Model merging ≠ finetuning**, but they complement each other
- Enables **multi-task learning without forgetting**
- Reduces memory → ideal for **on-device deployment**
- Avoids costly multi-model inference (vs ensembling)
- PEFT + merging = powerful, low-cost customization strategy

---
# Summing

**Summing** is a model-merging approach that combines the *parameters* of multiple
models by adding them together. This section covers two common summing methods:

- **Linear combination**
- **Spherical Linear Interpolation (SLERP)**

Before summing, it’s often useful to **rescale models** so their parameter magnitudes
are comparable. If one model’s weights are much larger than another’s, naive summing
can bias the merged model toward the larger-scale one.

---

## 1. Linear Combination

Linear combination includes both **simple averaging** and **weighted averaging**.

Given two models **A** and **B**, their weighted merge is:

\[
\text{Merge}(A, B) =
\frac{w_A A + w_B B}{w_A + w_B}
\]

When \( w_A = w_B = 1 \), this reduces to a simple average.

> **Intuition**: Each model contributes proportionally to its weight.

Linear combination works surprisingly well despite its simplicity. The idea that
multiple models can be combined linearly dates back to at least the early 1990s
(Perrone, 1993). Today, it is widely used in:
- **Federated learning** (Wang et al., 2020)
- **Model soups** (Wortsman et al., 2022)

### Model Soups
Model soups demonstrate that *averaging the full weights of multiple finetuned
models* (sharing the same base model) can improve accuracy **without increasing
inference cost**.

In practice, however, developers more often merge **specific components** rather than
entire models:
- LoRA adapters
- Task-specific layers
- Submodules

---

## Task Vectors and Task Arithmetic

Linear combination works best when models are finetuned from the **same base
model**. In this case, merging can be understood through **task vectors**.

- **Task vector (delta parameters)**  
  \[
  \text{Task Vector} = \text{Finetuned Model} - \text{Base Model}
  \]

If finetuning is done with LoRA, the task vector can be constructed directly from the
LoRA weights.

Task vectors enable **task arithmetic** (Ilharco et al., 2022):
- **Task addition** → combine capabilities
- **Task subtraction** → remove capabilities or behaviors

**Example use cases**
- Remove undesirable behaviors (e.g., facial recognition)
- Reduce specific biases acquired during pre-training

---

## Merging Models with Different Architectures

Linear combination is simplest when:
- Models have the **same architecture**
- Models have the **same parameter shapes**

However, it *can still work* when architectures differ:
- Project layers to a shared dimension
- Resize or align embeddings
- Merge only compatible submodules

### Parameter Alignment
Some research proposes aligning parameters before summing so that semantically
corresponding weights are merged together:
- *Model Fusion via Optimal Transport* (Singh & Jaggi, 2020)
- *Git Re-Basin* (Ainsworth et al., 2022)
- *Merging by Matching Models in Task Parameter Subspaces* (Tam et al., 2023)

While theoretically appealing, alignment is complex and expensive, so **naive linear
combination** remains far more common in practice.

---

## 2. Spherical Linear Interpolation (SLERP)

Another popular summing method is **SLERP** (Spherical Linear Interpolation).

> **Interpolation** means estimating an unknown value using known values.
> In model merging:
> - Known values → constituent models
> - Unknown value → merged model

Linear averaging is one interpolation method; **SLERP is another**.

### Intuition Behind SLERP

- Treat each model (or task vector) as a **point on a sphere**
- Find the **shortest path** between the two points along the sphere’s surface
- Pick a point along that path based on an **interpolation factor** \( t \in [0, 1] \)

Interpretation of the interpolation factor:
- \( t < 0.5 \): closer to the first model
- \( t = 0.5 \): exact midpoint
- \( t > 0.5 \): closer to the second model

Unlike linear averaging, SLERP preserves **vector magnitude and angular structure**,
which can lead to smoother transitions between model behaviors.

### Practical Notes

- SLERP is defined for **two vectors at a time**
- To merge more than two models:
  - Apply SLERP sequentially (e.g., merge A+B, then merge the result with C)
- Most model-merging tools implement SLERP internally, so you rarely need to deal
  with the math directly

---

## Summary

**Summing-based model merging**:
- Is simple and computationally cheap
- Works especially well for models sharing the same base
- Enables task arithmetic (addition and subtraction)
- Includes:
  - **Linear combination** → simple, flexible, widely used
  - **SLERP** → geometry-aware interpolation, smoother behavior blending

In practice, linear combination is the default starting point, with SLERP used when
developers want more controlled interpolation between model behaviors.
## Pruning Redundant Task-Specific Parameters

During finetuning, many model parameters are adjusted. However, **most of these
adjustments are minor** and contribute little to task performance. Parameters whose
changes don’t meaningfully affect performance are considered **redundant**.¹

In *“TIES-Merging: Resolving Interference When Merging Models”* (Yadav et al.,
2023), the authors showed that a **large fraction of task-vector parameters can be
reset** with minimal performance degradation.

**Resetting** a parameter means reverting the finetuned parameter to its original
value in the base model, effectively setting the corresponding task-vector value to
zero (recall that a task vector is obtained by subtracting the base model from the
finetuned model).

> **Key result:** keeping only the **top 20% of task-vector parameters** often gives
> comparable performance to keeping 100%.

*(See Figure 7-17 in the original text.)*

### Why Pruning Helps Model Merging

While redundant parameters may not hurt an individual finetuned model, they can
**interfere when multiple models are merged**.

Modern merging techniques such as:
- **TIES** (Yadav et al., 2023)
- **DARE** (Yu et al., 2023)

first **prune redundant task-vector parameters** and then merge the remaining sparse
vectors. Both papers show that pruning significantly improves merged-model quality.

The **more models you merge**, the more important pruning becomes, because:
- Each task introduces redundant parameters
- Redundant parameters increase cross-task interference

> ⚠️ Note:  
> Task-vector pruning improves *performance*, not memory or latency.  
> The finetuned model itself is not pruned—only the task vector used for merging.

---

## Layer Stacking

**Layer stacking** merges models by taking **entire layers from different models and
stacking them together**. This approach is also known as:
- *Passthrough merging*
- *Frankenmerging*

### Characteristics
- Produces models with **new architectures**
- Parameter count increases or changes irregularly
- Typically **requires further finetuning** to perform well
- Very different from parameter summing

### Early Success: Goliath-120B

One early example is **Goliath-120B** (alpindale, 2023):
- Built from two finetuned Llama 2-70B models (Xwin and Euryale)
- Took **72 out of 80 layers** from each model
- Stacked them to create a much larger model

---

## Layer Stacking for Mixture-of-Experts (MoE)

Layer stacking can also be used to construct **Mixture-of-Experts (MoE)** models.

In *“Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints”*
(Komatsuzaki et al., 2022), the authors:
1. Took a pretrained dense model
2. Made **multiple copies of certain layers**
3. Added a **router** to select which copy (expert) to use per input
4. Trained the router and experts further

This approach can outperform MoE models trained from scratch.

### Real-World Example

Using this idea, **Together AI** merged six weaker open-source models into
**Mixture-of-Agents**, achieving performance comparable to GPT-4o on certain
benchmarks (Wang et al., 2024).

*(See Figure 7-18 in the original text.)*

---

## Model Upscaling via Layer Stacking

Layer stacking can also be used for **model upscaling**—creating a larger model from
an existing one without training from scratch.

### Depthwise Scaling: SOLAR-10.7B

Kim et al. (2023) proposed **depthwise scaling**, demonstrated in **SOLAR-10.7B**:

- Base model: **7B parameters**, 32 layers
- Target model: **10.7B parameters**, 48 layers

**Procedure:**
1. Duplicate the pretrained model
2. Merge the two copies:
   - **Sum some layers**
   - **Stack the rest**
3. Finetune the resulting larger model

For SOLAR-10.7B:
- 16 layers were summed
- Final depth = \(32 × 2 − 16 = 48\) layers

*(See Figure 7-19 in the original text.)*

---

## Concatenation

Instead of summing parameters, you can **concatenate them**.

### How It Works
- Parameter counts **add up**
- No compression effect

**Example (LoRA):**
- Two LoRA adapters with ranks `r₁` and `r₂`
- Concatenated adapter has rank `r₁ + r₂`

*(Illustrated in Figure 7-20.)*

### Why Concatenation Is Rarely Recommended

- ❌ No memory savings
- ❌ No inference efficiency gains
- ✅ Possible performance improvements, but often marginal

In most cases, the **performance gains do not justify the additional parameters**,
making concatenation less attractive than summing or pruning-based merging.

---

### Summary

| Technique | Key Idea | Pros | Cons |
|---------|--------|------|------|
| Task-vector pruning | Remove small delta parameters | Reduces interference | No memory savings |
| Layer stacking | Combine full layers | Enables scaling & MoE | Needs finetuning |
| Concatenation | Combine parameters directly | Simple | High memory cost |

---

¹ Assumption: parameters with the **largest changes during finetuning** are the most
important for the task.
# Finetuning Tactics (Practical Guide)

This section focuses on **practical, decision-oriented finetuning tactics**—what to
finetune, how to finetune, and with what tools—optimized for **real-world
engineering constraints** and **interview-level understanding**.

---

## 1. Key Decisions in Finetuning

To finetune a model, you must choose **three things**:

1. **Base model**
2. **Finetuning method**
3. **Finetuning framework**

While data collection, evaluation, and maintenance are often the hardest parts,
**the actual finetuning process is relatively straightforward** once these choices
are made.

---

## 2. Choosing a Base Model

Base model selection principles are the same as prompt-based systems:
- Model size
- License constraints
- Benchmark performance
- Cost and latency constraints

### Best Practice: Start Strong, Then Scale Down
At the start of a project:
- Use the **strongest model you can afford**
- If it fails, weaker models will almost certainly fail
- If it succeeds, use it as a **benchmark** to test cheaper models

---

## 3. Finetuning Development Paths

### A. Progression Path (Engineering-First Approach)

Recommended when:
- You want stable, predictable experimentation
- You’re validating both **code** and **data quality**

**Steps:**
1. **Start with the cheapest and fastest model**
   - Purpose: validate training pipeline correctness
2. **Finetune a mid-tier model**
   - If training loss does not decrease → data or setup issue
3. **Experiment with the best available model**
   - Identify upper-bound performance
4. **Train multiple models**
   - Map the **price–performance frontier**
   - Choose best cost-effective solution

✅ This reduces wasted GPU time and debugging risk.

---

### B. Distillation Path (Model Compression Strategy)

Recommended when:
- You need a **small, cheap model**
- You can afford a strong model temporarily

**Steps:**
1. Finetune the **strongest model** on a small dataset
2. Use it to **generate additional high-quality training data**
3. Finetune a **smaller / cheaper model** on the generated data

✅ This is **knowledge distillation**: transferring capabilities from large to small
models.

---

## 4. Choosing a Finetuning Method

### Full Finetuning vs PEFT (e.g., LoRA)

| Aspect | Full Finetuning | PEFT (LoRA, Adapters) |
|-----|---------------|----------------|
| Cost | Very high | Low |
| Memory | Very high | Low |
| Data required | Thousands to millions | Hundreds to thousands |
| Performance ceiling | Highest | Slightly lower |
| Multi-model serving | Expensive | Very efficient |

### Practical Guidelines

- **Start with LoRA / PEFT**
- Try full finetuning **only if needed**
- If you have **few hundred examples**, full finetuning often underperforms LoRA
- PEFT is usually **sample-efficient and cost-effective**

### Serving Consideration (Often Asked in Interviews)

- **Full finetuning** → one full model per task or customer
- **LoRA** → one base model + many adapters  
  ✅ Easier multi-tenant and multi-task deployment

---

## 5. Choosing a Finetuning Framework

### Option 1: Finetuning APIs
Examples:
- OpenAI
- Cloud providers
- Third-party platforms

**Pros**
- Quick to start
- Minimal infrastructure management

**Cons**
- Limited base models
- Limited hyperparameter control
- Black-box behavior

✅ Best for quick prototypes or small teams  
❌ Frustrating for advanced optimization

---

### Option 2: Open-Source Finetuning Frameworks

Popular frameworks:
- **LLaMA-Factory**
- **unsloth**
- **Hugging Face PEFT**
- **Axolotl**
- **LitGPT**

**Advantages**
- Full control over training
- Supports PEFT, LoRA, QLoRA, etc.
- Works with many open models

✅ Best choice for serious ML engineers

> If you need **full finetuning**, many models publish official training scripts
> (e.g., Meta Llama repos).

---

## 6. Scaling Finetuning (Distributed Training)

When a single GPU isn’t enough:

### Frameworks for Distributed Training
- **DeepSpeed**
- **PyTorch Distributed**
- **ColossalAI**

These handle:
- Data parallelism
- Model parallelism
- Memory offloading
- Multi-node training

✅ Essential knowledge for **senior ML / AI engineer roles**

---

## 7. Interview-Ready Takeaways

### High-Level Strategy
- Try **prompting → RAG → PEFT → full finetuning**
- Always define evaluation **before** finetuning
- Prefer **LoRA unless proven insufficient**

### Common Interview Questions You Can Answer with This Section
- *“How do you choose a base model for finetuning?”*
- *“When would you use full finetuning over LoRA?”*
- *“How do you finetune models with limited data?”*
- *“How do you serve multiple finetuned models efficiently?”*
- *“How would you scale finetuning across GPUs?”*

---

## 8. Quick Reference Cheat Sheet

**If…**
- You’re early-stage → start with strongest model
- You have limited data → use PEFT
- You need many task-specific models → LoRA + shared base
- You want cheaper inference → distill into smaller models
- You hit memory limits → use LoRA / QLoRA / DeepSpeed

---

✅ **Mental Model to Remember:**  
> *Prompting explores behavior, RAG adds knowledge, finetuning changes behavior,
> PEFT makes finetuning affordable.*

---
# Finetuning Hyperparameters

Depending on the **base model** and the **finetuning method**, there are many
hyperparameters you can tune to improve finetuning efficiency and stability.  
For task-specific values, always consult the documentation of the base model
or the finetuning framework you are using.

Below are **core hyperparameters** that appear most frequently in practice and interviews.

---

## 1. Learning Rate

The **learning rate** controls how fast the model’s parameters are updated during
training.

Think of learning as moving toward a goal:
- **Too small** → training is slow and may stall
- **Too large** → training becomes unstable and may never converge

### Practical Guidelines
- Typical range: **`1e-7` to `1e-3`**
- Common heuristic:
  - Take the learning rate used at the **end of pretraining**
  - Multiply it by a constant in **[0.1, 1]**

### Diagnosing with the Loss Curve
- **Loss fluctuates wildly** → learning rate is likely too high
- **Loss decreases very slowly** → learning rate is likely too low
- Increase learning rate until loss becomes unstable, then back off slightly

### Learning Rate Schedules
Instead of a fixed learning rate:
- Use **larger learning rates early**
- Use **smaller learning rates near convergence**

Learning rate schedules automatically control this behavior (e.g., cosine decay,
linear warmup + decay).

---

## 2. Batch Size

The **batch size** controls how many examples are processed before updating model
weights.

### Effects of Batch Size
- **Too small (e.g., < 8)** → unstable gradient updates
- **Larger batch size** → smoother, more reliable parameter updates
- **Larger batch size = more memory usage**

This creates a **cost vs efficiency trade-off**:
- More memory → larger batches → faster and more stable training

---

### Gradient Accumulation

When hardware constraints force small batch sizes:

Instead of updating weights after every batch:
1. Accumulate gradients over multiple batches
2. Update weights only after enough gradients are collected

This technique is called **gradient accumulation**.

✅ Allows stable training with limited GPU memory  
✅ Common in LLM finetuning

---

## 3. Number of Epochs

An **epoch** is one full pass over the training dataset.

### General Rules of Thumb
- **Large datasets (millions of examples)**  
  → 1–2 epochs may be sufficient
- **Small datasets (thousands of examples)**  
  → 4–10 epochs may still improve performance

### Diagnosing Epoch Count
Monitor **training loss vs validation loss**:
- Both decreasing → training can benefit from more epochs
- Training loss ↓ but validation loss ↑ → overfitting  
  → reduce epochs, add regularization, or add data

---

## 4. Prompt Loss Weight (Instruction Finetuning)

In instruction finetuning, each example contains:
- **Prompt** (input)
- **Response** (output)

During inference:
- Prompts come from users
- The model only generates **responses**

Therefore, training should emphasize **response tokens**.

### Prompt Loss Weight
This hyperparameter controls how much prompt tokens contribute to total loss.

| Prompt Loss Weight | Behavior |
|---|---|
| 100% | Prompt and response contribute equally |
| 0% | Model learns only from responses |
| ~10% (typical default) | Model learns mostly from responses, slightly from prompts |

✅ Default (~10%) works well for most instruction-tuned models

---

## 5. Summary Cheat Sheet

| Hyperparameter | What It Controls | Common Practice |
|---|---|---|
| Learning rate | Update step size | `1e-7` – `1e-3`, tune via loss |
| LR schedule | LR over time | Warmup + decay |
| Batch size | Stability vs memory | As large as memory allows |
| Gradient accumulation | Effective batch size | Used when memory-limited |
| Epochs | Data reuse | 1–2 (large data), 4–10 (small data) |
| Prompt loss weight | Prompt vs response learning | ~10% |

---

### Interview-Ready Insight

> **Most finetuning failures are caused not by architecture, but by poorly chosen
> learning rates, batch sizes, or epoch counts.**


