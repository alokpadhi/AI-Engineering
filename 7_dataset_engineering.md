# Dataset Engineering

The quality of a model depends on the **quality of its training data**. Even the best
ML team in the world with infinite compute cannot finetune a good model without
good data.

The goal of **dataset engineering** is to create a dataset that enables you to train
the best possible model—**ideally within your budget constraints**.

As fewer companies can afford to develop foundation models from scratch, more
organizations are turning to **data** as the primary lever for differentiating AI
performance. As models demand more data, data handling becomes increasingly
challenging, requiring greater investments in **talent, tooling, and infrastructure**.¹

As a result, data operations have evolved:
- From side tasks handled opportunistically
- To **dedicated roles and teams**

Many AI companies now employ:
- Data labelers
- Dataset creators
- Data quality engineers  

These roles often operate alongside or are embedded within core ML engineering
teams.

---

## The Growing Complexity of the Data Landscape

If the model landscape is confusing—crowded with architectures, checkpoints, and
APIs—the **data landscape is even more complex**.

There is an ever-growing array of:
- Public datasets
- Synthetic data techniques
- Data processing pipelines
- Evaluation strategies

This chapter provides:
- An overview of the modern data landscape
- Key considerations when building your own datasets

---

## What This Chapter Covers

The discussion begins with **data curation**, including questions such as:
- What data do you need?
- How much data do you need?
- What qualifies as *high-quality* data?

It then moves on to:
- Data synthesis
- Data processing and refinement

These steps are **not linear**. In practice, you will move back and forth between
them repeatedly.

Different training phases require **different dataset attributes**:
- **Pretraining** data quantity is often measured in *tokens*
- **Supervised finetuning** data quantity is often measured in *examples*

Despite these differences, the underlying curation principles remain largely the
same.

This chapter focuses primarily on **post-training data**, as it is most relevant for
application developers, while selectively borrowing insights from pretraining when
they are useful.

---

## Hard Truth About Data Work

While there are:
- Best practices to follow
- Tools to automate parts of the pipeline

Data work will mostly involve:
> **Toil, tears, and sweat**

---

## A Data-Centric View of AI

The increasing emphasis on data has led to a shift from **model-centric AI** to
**data-centric AI**.

### Model-Centric AI
Focuses on improving performance by:
- Designing better architectures
- Increasing model sizes
- Developing new training algorithms

### Data-Centric AI
Focuses on improving performance by:
- Improving data quality
- Creating better datasets
- Developing better data processing techniques

The goal is to enable **better models with fewer resources**.

---

## Evolution of Benchmarks

### Earlier Benchmarks (Model-Centric)
- Fixed dataset (e.g., ImageNet)
- Compete by training better models on the same data

### Recent Benchmarks (Data-Centric)
- Fixed model
- Compete by building better datasets

---

## Notable Data-Centric Initiatives

### Andrew Ng’s Data-Centric AI Competition (2021)
Participants improved performance using the same model by:
- Fixing incorrect labels
- Adding edge cases
- Augmenting data
- Improving consistency

### DataComp (2023)
- Goal: Create the best dataset for training a **CLIP** model
- Evaluation:
  - A standardized script trains CLIP on each dataset
  - Performance measured across **38 downstream tasks**

### DataComp for Language Models (2024)
- Evaluated datasets for LMs ranging from **412M to 7B parameters**

### Other Data-Centric Benchmarks
- **DataPerf** (MLCommons, 2023)
- **dcbench** (Eyuboglu & Karlaš, 2022)

---

## Key Takeaway

> As models become increasingly commoditized, **data quality and curation** are
> turning into the strongest competitive advantages in AI systems.

---

¹ **Data investment example: GPT-3 vs GPT-4**
- GPT-3 (2020): 2 people credited for data collection and filtering
- GPT-4 (2023): 80 people credited across data pipelines
- Even simple formats like *ChatML* involved 11 contributors, many senior researchers
``
# Data Curation — Interview & Revision Notes

Data curation is one of the **highest-leverage activities** in modern AI development.
While not all model issues can be solved with data alone, **most production-quality
improvements depend critically on the right data**.

Good data can make models:
- More capable
- Safer and more aligned
- Better at long-context reasoning

Poor data can:
- Increase hallucinations
- Amplify biases
- Waste expensive compute
- Actively harm model behavior

---

## What Is Data Curation?

**Data curation** is the disciplined process of:
- Deciding **what data to collect**
- Deciding **what data to remove**
- Deciding **how data should be structured**
- Ensuring data supports the *exact behaviors* you want the model to learn

It is both:
- A **technical science** (understanding how models learn)
- A **product discipline** (understanding application goals)

Because of this, dataset builders should work **closely with application and model developers**.

In small teams:
- One person often owns both modeling and data

In large organizations:
- Dedicated roles exist:
  - Data labelers
  - Dataset creators
  - Data quality engineers
  - Data compliance specialists²

---

## Data Formats Depend on Training Objective

Different finetuning techniques require **different data structures**:

| Training Type | Required Data Format |
|---------------|---------------------|
| Self-supervised finetuning | Raw sequences |
| Instruction finetuning | `(instruction, response)` |
| Preference finetuning | `(instruction, winning_response, losing_response)` |
| Reward model training | `((instruction, response), score)` |

👉 **Interview note**: Mismatch between data format and training objective is a common root
cause of failed finetuning.

---

## Training Data Must Exhibit Desired Behaviors

A core principle:
> **Models learn behaviors they observe in data — not behaviors you wish they had.**

This becomes especially challenging for **complex behaviors**, such as:

---

## Case Study 1: Chain-of-Thought (CoT) Reasoning

**Chain-of-thought (CoT)** teaches models to reason step-by-step before answering.

### Key Findings
- Including CoT in finetuning data **dramatically improves reasoning accuracy**
- Chun et al. (2024) showed **accuracy nearly doubled** for certain tasks when CoT data was used

### Why CoT Data Is Hard
- Generating step-by-step explanations is:
  - Time-consuming
  - Cognitively demanding
- Humans find it easier to give *answers* than *reasoning traces*

### Example

**Without CoT**
```text
Q: What is the boiling point of nitrogen?
A: -320.4°F
````

**With CoT**

```text
The cafeteria had 23 apples.
They used 20 → 23 - 20 = 3.
They bought 6 more → 3 + 6 = 9.
```

📌 **Result**: CoT datasets are rarer, more expensive, but often higher leverage.

---

## Case Study 2: Tool Use

Many models develop **implicit tool knowledge** during pretraining, but **demonstrated tool use**
dramatically improves reliability.

### Challenges in Tool Use Data

* Requires **expert annotations**
* Requires **procedural accuracy**
* Human experts often:

  * Skip steps unconsciously
  * Explain workflows imperfectly

Often, **observation > explanation**.

### Human vs AI Efficiency Mismatch

What works for humans may be inefficient for AI:

* Humans click GUIs
* Models prefer APIs
* Humans search sequentially
* Models can query and parse results in parallel

👉 For this reason, **simulations and synthetic data** are often better than human annotations
for tool use.

---

## Data Format Complexity for Tool Use

Tool use often requires **non-traditional conversation formats**:

* Multiple messages per turn
* Messages routed to:

  * Tools (e.g., code interpreter)
  * User (explanations, progress)

Example:

* Llama 3 introduced **multi-message chat formats**
* Includes:

  * Headers specifying message source/destination
  * Special tokens to delimit human and AI turns

📌 **Interview note**: Tool-use finetuning is as much a *format problem* as a data problem.

---

## Single-Turn vs Multi-Turn Data

| Type        | What It Teaches                | Difficulty |
| ----------- | ------------------------------ | ---------- |
| Single-turn | Instruction following          | Easier     |
| Multi-turn  | Task completion, collaboration | Harder     |

**Multi-turn data is critical for real-world tasks**:

* Clarifying intent
* Handling corrections
* Iterative refinement

However:

* Harder to collect
* Usually requires scenario-based generation

---

## Data Removal Is Also Data Curation

Data curation is not just **adding good data**, but also **removing bad data**.

### Example: Unwanted Model Behavior

Complaint:

> “The chatbot adds unsolicited advice and rewrites text unnecessarily.”

Root cause:

* Training data contains many examples of unsolicited suggestions

Fix:

1. Remove problematic examples
2. Add counterexamples that demonstrate correct behavior (e.g., fact-checking only)

📌 **Key insight**: Models do not self-correct; they reflect statistical patterns in data.

---

## The Three Pillars of Data Curation

All data curation decisions can be framed around **three criteria**:

### 1. Data Quality

* Correctness
* Consistency
* Signal-to-noise ratio

> Bad data poisons learning regardless of scale.

### 2. Data Coverage

* Diversity of scenarios
* Edge cases
* Correct distribution of behaviors

> Too much of one behavior can be as harmful as too little.

### 3. Data Quantity

* Number of tokens or examples
* Balanced against cost and diminishing returns

---

## Intuition: The Cooking Analogy

Think of model training like cooking:

| Concept       | Cooking Analogy        |
| ------------- | ---------------------- |
| Data quality  | Ingredient freshness   |
| Data coverage | Correct ingredient mix |
| Data quantity | Portion size           |

No amount of cooking skill can fix:

* Rotten ingredients
* Missing key ingredients
* Completely unbalanced recipes

---

## Interview-Ready Takeaways

* **Data is the primary differentiator** in post-foundation-model AI systems
* Complex behaviors (CoT, tool use) require **explicit demonstrations**
* Multi-turn and tool-use data introduce **format complexity**, not just labeling cost
* Removing bad data is often as impactful as adding new data
* Effective data curation optimizes **quality, coverage, and quantity — in that order**

---

² **Compliance Note**
At scale, data compliance (privacy, licensing, consent, jurisdiction) can become a
**full-time responsibility**, especially in regulated industries.

# Data Quality — Practical & Interview Notes

A **small amount of high-quality data can vastly outperform a large amount of noisy data**.

Empirical evidence consistently supports this:

- **Yi model family**:  
  10K carefully crafted instructions outperformed *hundreds of thousands* of noisy ones  
  (Young et al., 2024).

- **LIMA: Less Is More for Alignment**:  
  A 65B LLaMA model finetuned on **just 1,000 high-quality prompts** produced answers that
  were *equivalent or preferred to GPT-4 in 43% of cases*, as judged by humans  
  (Zhou et al., 2023).

  ⚠️ Trade-off: LIMA was **less robust** than product-grade models due to limited coverage.

- **Llama 3 findings**:  
  Human-generated data is **error-prone and inconsistent**, especially for *nuanced safety rules*.  
  This motivated the use of **AI-assisted annotation tools** to improve data quality.

---

## What Does “High-Quality Data” Mean?

Short answer:
> **Data is high-quality if it helps you train a model efficiently and reliably for your task.**

Long answer:
- “Quality” is **task-dependent**
- Different users, tasks, and domains require different notions of quality³

For **finetuning**, high-quality data typically has **six key characteristics**:

---

## The Six Characteristics of High-Quality Data

### 1. Relevant

Training examples must be **relevant to the task**.

- ❌ 19th-century legal cases for *modern legal Q&A*
- ✅ 19th-century legal cases for *historical legal analysis*

📌 **Relevance is contextual, not absolute.**

---

### 2. Aligned with Task Requirements

Annotations must reflect **what you actually want the model to do**.

Examples:

- If the task requires:
  - **Factual correctness** → annotations must be factually correct
  - **Creativity** → annotations should demonstrate creativity
  - **Scores + justifications** → both must be provided
  - **Conciseness** → answers should be short, even if more detail is “correct”

> The term **“aligned”** is preferred over “accurate” or “correct” because  
> *what is accurate is not always what the user wants*.

---

### 3. Consistent

Annotations should be consistent:
- Across **examples**
- Across **annotators**

Examples of inconsistency:
- Two annotators assigning very different scores to the same essay
- Two examples with the same score but clearly different quality

Why it matters:
- Inconsistency introduces **label noise**
- Noisy supervision confuses the model and slows learning

✅ **Clear annotation guidelines** are essential for alignment *and* consistency.

---

### 4. Correctly Formatted

All data must strictly follow the **expected input format**.

Common formatting issues to fix:
- HTML tags in scraped text
- Redundant system tokens
- Trailing spaces or extra newlines
- Inconsistent casing
- Inconsistent numeric formats (e.g., “1,000” vs “1000”)

> Even small formatting noise can interfere with learning.

---

### 5. Sufficiently Unique

Training data should avoid excessive duplication.

Why duplicates are harmful:
- Introduce bias
- Skew token distributions
- Cause data contamination
- Reduce effective dataset diversity

The phrase **“sufficiently unique”** is intentional:
- Some use cases tolerate light duplication
- Others (e.g., evaluation sets) require near-perfect deduplication

---

### 6. Compliant

Data must comply with:
- Internal policies
- External regulations
- Legal constraints

Examples:
- No PII if policy prohibits it
- Proper licensing for scraped data
- Jurisdiction-specific data handling requirements

📌 At scale, **data compliance can become a full-time responsibility**.

---

## Key Planning Step (Often Missed)

Before collecting or generating data, you should **explicitly define what each quality dimension means for your use case**:

- What counts as “relevant”?
- How strict should consistency be?
- What duplication rate is acceptable?
- What formatting rules are mandatory?
- What compliance constraints apply?

The **data-processing techniques** you choose later only make sense once these definitions are clear.

---

## Interview-Ready Takeaways

- **Data quality beats data scale** for finetuning
- A few thousand high-quality examples can rival or beat massive noisy datasets
- “Quality” is *task-aligned*, not universally defined
- The six pillars:
  1. Relevant
  2. Aligned with task requirements
  3. Consistent
  4. Correctly formatted
  5. Sufficiently unique
  6. Compliant
- Many finetuning failures are **data problems disguised as model problems**

---

³ *Note on Definitions*  
Different organizations define data quality differently:
- IBM: completeness, uniqueness, validity, timeliness, accuracy, consistency, fitness for purpose
- Wikipedia adds: accessibility, credibility, comparability, plausibility

This section focuses specifically on **data quality for finetuning large language models**.
# Data Coverage (a.k.a. Data Diversity)

**Data coverage** means your training data should span the *full range of problems*
your users will ask your model to solve — and the *different ways* they’ll ask them.

High-quality data alone is not enough.  
If your data doesn’t represent real-world usage patterns, the model will fail in
predictable (and frustrating) ways.

---

## What Is Data Coverage?

Data coverage answers the question:

> **Does my dataset represent how users actually use my application?**

In practice, this means capturing **variation along multiple dimensions**, such as:

- Query length (short vs long)
- Writing style (formal vs casual)
- Error patterns (typos, grammar mistakes)
- Domains (math, code, general knowledge)
- Languages and cultures
- Output formats and response lengths
- Number of turns (single-turn vs multi-turn)

Because coverage is about *diversity*, the terms **data coverage** and **data diversity**
are often used interchangeably.

---

## Why Coverage Matters

Real users are diverse:
- Some write long, detailed prompts with references
- Others write one-line instructions
- Some make spelling mistakes
- Some mix languages or use informal slang

If your finetuning data fails to include these patterns, the model will behave well in
tests—but poorly in production.

---

## Examples of Coverage by Application Type

### Example 1: Instruction-Following App

If users:
- Write both detailed and terse instructions → include both
- Often include typos → include noisy text
- Use multiple programming languages → include only those your users care about

---

### Example 2: French → English Translation

- Language diversity: ❌ not needed
- Topic diversity: ✅ important
- Sentence length diversity: ✅ important
- Speaking style diversity: ✅ important

---

### Example 3: Global Product Recommendation Chatbot

- Domain diversity: ❌ not critical
- Linguistic diversity: ✅ critical
- Cultural diversity: ✅ critical

---

## Data Coverage in General-Purpose Chatbots

For chatbots, **broad diversity is essential**:

- Topics
- Instruction styles
- Output formats
- Reasoning depth
- Response lengths
- Open-ended vs yes/no questions

> Ding et al. (2023):  
> *The most straightforward way to further improve chat language models is to increase
> the quality and diversity of training data.*

NVIDIA’s **Nemotron** dataset explicitly targeted:
- Task diversity  
- Topic diversity  
- Instruction diversity  
  - Multiple output formats
  - Variable output lengths
  - Open-ended and binary questions

⚠️ However, diversity can hurt if done carelessly:
- **“The Data Addition Dilemma”** (Shen et al., 2024) shows that *adding overly
  heterogeneous data can reduce performance* in some cases.

---

## Case Study: Llama 3 — Coverage Across Training Phases

Meta emphasized that **Llama 3’s performance gains came primarily from better data
quality and diversity**, not major architectural changes.

Llama 3 optimized **data coverage differently at each training stage**:

### Table: Domain Mix Across Training Phases

| Domain                    | Pre-training | Supervised FT | Preference FT |
|---------------------------|--------------|---------------|---------------|
| General knowledge (English) | 50%          | 52.66%        | 81.99%        |
| Math & reasoning          | 25%          | 21.19%        | 5.89%         |
| Coding                    | 17%          | 14.89%        | 6.93%         |
| Multilingual              | 8%           | 3.01%         | 5.19%         |
| Exam-like                 | ❌           | 8.14%         | ❌            |
| Long context              | ❌           | 0.11%         | ❌            |

---

### Key Observations

#### 1. Heavy Math & Code Emphasis Early
- ~50% of **pre-training + supervised finetuning** tokens are math or code
- This is far higher than their proportion on the internet
- Reason: **math and code are disproportionately effective at improving reasoning**

Meta confirmed:
> Annealing on small amounts of high-quality code and math data significantly boosts
> benchmark performance.

#### 2. Preference Finetuning Reflects Real Users
- Math + code drops to **~13%**
- Goal shifts from *capability building* → *preference alignment*
- Training distribution matches real user preferences

---

## How Do You Choose the Right Data Mix?

There is no universal formula. Common strategies include:

### 1. Usage-Matched Mix
- Reflect real-world user traffic distribution
- Best for preference finetuning and production stability

### 2. Experiment-Driven Mix
- Train **small models** on candidate data mixes
- Use scaling-law extrapolation to predict large-model performance
- Meta used this approach for Llama 3

---

## Quality vs Coverage: You Need Both

Zhou et al. (2023) ran a controlled experiment:

- Same model (7B)
- Same dataset size (2,000 examples)
- Different dataset characteristics:

| Dataset Type                    | Performance |
|--------------------------------|-------------|
| High-quality, low diversity    | ❌ weaker    |
| High diversity, low quality    | ❌ weaker    |
| High quality + high diversity  | ✅ best      |

**Conclusion**:
> Data quality and data coverage are *multiplicative*, not additive.

You need **both**.

---

## Practical Takeaways

- Data coverage = diversity across *how users ask* and *what they ask*
- Coverage requirements depend strongly on the application
- Over-diversification can hurt — coverage must be *controlled*
- Different training phases need different data mixes
- High-quality **math + code data punches above its weight**
- Best results come from datasets that are:
  - **High-quality**
  - **Well-covered**
  - **Intentionally mixed**

---

## Interview-Ready One-Liner

> *“Quality determines how well the model learns; coverage determines what it can handle
> in the real world. You need both to ship reliable systems.”*
# Data Quantity in Finetuning

**Data quantity** answers the question: *How many training examples do we actually need?*  
Like budget planning, the answer is highly context-dependent and shaped by multiple
interacting factors.

---

## There Is No Universal Number

Data needs range widely:

- **Extremely small**:  
  Experiments by Jeremy Howard & Jonathan Whitaker show that LLMs can sometimes
  learn from a *single example*.
- **Very large**:  
  Some finetuning runs use **millions of examples**.

However, even “millions” is small compared to **foundation model training**:
- Llama 2: **2 trillion tokens**
- Llama 3: **16 trillion tokens**

If one example ≈ 2,000 tokens:
- Llama 2 ≈ **1B examples**
- Llama 3 ≈ **15B examples**

---

## Should You Train from Scratch Instead?

If you already have *millions of high-quality examples*, it’s worth evaluating whether
**training from scratch** might perform better than finetuning.

### Ossification Problem

Finetuning can underperform when data is very large due to **ossification**
(Hernandez et al., 2021):

- Pre-training can *freeze* representational capacity
- Model struggles to fully adapt to new data
- **Smaller models are more susceptible** than larger ones

✅ Finetuning is *usually* more efficient  
⚠️ But for very large datasets, it may not always be optimal

---

## What Determines How Much Data You Need?

Beyond **data quality** and **data coverage**, three major factors matter:

---

### 1. Finetuning Technique

| Technique | Data Requirement |
|---------|------------------|
| **Full finetuning** | Tens of thousands → millions |
| **PEFT (LoRA, adapters)** | Hundreds → few thousands |

**Rule of thumb**:
- Large dataset → try **full finetuning**
- Small dataset → **PEFT** usually works better

---

### 2. Task Complexity

- **Simple tasks** (e.g., sentiment classification) → less data
- **Complex tasks** (e.g., QA over financial filings) → much more data

---

### 3. Base Model Strength

- Stronger base model → fewer examples needed
- Bigger models benefit more from small datasets
- This is the **opposite of pre-training**, where bigger models need more data

**OpenAI finetuning experiments**:
- With **100 examples**: stronger models perform far better
- With **550,000 examples**: all models converge to similar performance

✅ Strong base model matters most when data is scarce  
✅ Data volume matters most when it’s large

---

## Practical Strategy by Data Regime

### Small Data (≤ 1,000 examples)
- Use **advanced base model**
- Prefer **PEFT methods**
- Expect meaningful gains

### Large Data (≥ tens of thousands)
- Consider **full finetuning**
- Smaller base models may be cost-efficient
- Risk of ossification should be evaluated

---

## Always Start Small

Before investing heavily in dataset creation:

1. Start with **50–100 carefully crafted examples**
2. Finetune and evaluate improvement
3. Interpret results carefully:

- ✅ Clear improvement → more data likely helps
- ❌ No improvement → larger dataset rarely fixes the issue

⚠️ But don’t jump to conclusions:
- Learning rate issues
- Bad prompts
- Poor data quality
- Wrong loss weighting

In practice:
> **Most finetuning runs show improvements with 50–100 good examples**

---

## Reducing Data Requirements via Staged Finetuning

You can reduce the need for *high-quality labeled data* by first using *cheaper or weaker data*.

### 1. Self-Supervised → Supervised
- Pre-adapt using raw domain documents
- Then finetune on small labeled dataset

**Example**:  
Legal documents → legal Q&A

---

### 2. Less-Relevant → Task-Relevant
- Start with adjacent domains
- Then move to target task

**Example**:  
Tweet sentiment → product reviews

---

### 3. Synthetic → Real Data
- Generate synthetic data at scale
- Finetune first, then refine with real data

⚠️ High risk:
- Requires careful transitions
- Easy to waste compute and harm quality

---

## Estimating Returns from More Data

### Data Scaling Curve

Procedure:
1. Train on 25%, 50%, 100% of your dataset
2. Plot performance vs data size

Interpretation:
- **Steep slope** → more data helps a lot
- **Plateau** → diminishing returns

📉 Most real-world curves show *diminishing returns*

Example intuition:
- First 1,000 examples: +10% accuracy
- Next 1,000 examples: +5%
- Later chunks: even less

---

## Quantity vs Diversity

It’s not just *how much* data — but *how diverse* it is.

### Instruction Diversity Matters

“Scaling Instruction-Finetuned Language Models” (Chung et al., 2022):

- Performance jumps sharply from:
  - **9 tasks → 282 tasks**
- Still improves (but slower) up to:
  - **1,836 tasks**

✅ Task diversity improves generalization  
✅ Gains eventually plateau

---

## Budget Constraints Are Real

Data quantity is bounded by cost:

Example:
- Budget: $10,000
- Cost per annotation: $2
- Max examples: **5,000**

You must balance:
- **Data budget**
- **Compute budget**

Spending more on one usually means less for the other.

---

## Key Takeaways (Quick Interview Notes)

- There is **no fixed amount of data** needed for finetuning
- More data ≠ always better (ossification exists)
- Small data → strong base model + PEFT
- Large data → full finetuning may win
- Always run **small pilot finetunes**
- Data diversity often matters more than raw volume
- Performance gains usually show **diminishing returns**
- Budget constraints shape practical decisions

---

## Interview-Ready One-Liner

> *“When data is scarce, model choice dominates. When data is abundant, quantity and
diversity dominate. The hardest part is knowing which regime you’re in.”*
# Data Acquisition and Annotation

This section focuses on **how to acquire and annotate data** that is large enough,
high-quality, diverse, privacy-respecting, and regulation-compliant—while staying
within budget. For interviews and real systems, this is one of the *most critical and
underestimated* parts of AI development.

---

## Goal of Data Acquisition

The objective is to build a dataset that is:

- **Sufficiently large**
- **High quality**
- **Diverse**
- **Legally compliant** (privacy, licenses, regulations)
- **Cost-effective**

Data acquisition is not just an operational task—there is **active research** on
*data acquisition strategy*: how to optimally acquire data under constraints.

---

## The Most Valuable Data Source: Your Own Application

> **User-generated application data is usually the best data you can ever get.**

Why?
- Perfectly **task-aligned**
- Matches **real user distribution**
- Extremely hard to replicate with external datasets

### Types of Application Data
- User inputs (queries, prompts, uploads)
- System-generated logs (usage patterns, failures)
- User feedback (ratings, corrections, preferences)

If you can build a **data flywheel**—where user interactions continuously improve the
model—you gain a **massive long-term competitive advantage**.

> In practice, almost every successful AI product relies on a data flywheel.

---

## Before Creating Data: Use Existing Datasets

Always check **public or proprietary datasets** before building from scratch.

However:
- Never assume datasets are clean or usable as-is
- Thorough validation is mandatory
- Licenses must be carefully reviewed

⚠️ Even datasets labeled as “commercially usable” may contain sub-sources that are
not.

---

## Typical Dataset Creation Is Iterative (Not Linear)

A realistic dataset-building workflow looks like this:

1. Find an existing dataset with promising characteristics (e.g., 10k examples)
2. Remove low-quality instructions → 9k
3. Remove low-quality responses → 6k
4. Manually write responses for good instructions → back to 9k
5. Identify coverage gaps (e.g., missing topic X)
6. Create instruction templates for missing topics
7. Use AI to synthesize new instructions
8. Manually annotate synthetic data
9. Re-evaluate quality and diversity
10. Update guidelines → re-annotate data
11. Fact-check and correct errors
12. Repeat…

> This loop is where most time, money, and frustration goes.

---

## Common Real-World Pitfalls

- Annotation guidelines turn out to be unclear
- Annotators disagree → inconsistency
- Factually incorrect but fluent responses
- Overuse of synthetic data reduces diversity
- Bias or hallucinations are amplified instead of reduced

**Dataset engineering is feedback-heavy and messy by nature.**

---

## Public Dataset Resources (Must-Know for Interviews)

Always mention *both sources and caution*.

### General Dataset Repositories
1. **Hugging Face Datasets**
2. **Kaggle**
3. **Google Dataset Search**

### Government & Institutional Data
4. **Data.gov (US)** and **data.gov.in (India)**
5. **ICPSR (University of Michigan)** – social science data
6. **Stanford Large Network Dataset Collection** – graph data

### ML-Specific Repositories
7. **UCI ML Repository**
8. **OpenML**
9. **AWS Open Data**
10. **TensorFlow Datasets**
11. **EleutherAI lm-evaluation-harness**  
   - 400+ benchmark datasets  
   - Often large enough for PEFT finetuning

---

## Annotation Is the Hard Part (Not Labeling)

Annotation difficulty comes from **guideline design**, not just execution.

Key questions guidelines must answer:
- What makes a response *good*?
- Can a response be correct but unhelpful?
- What separates a score of 3 vs 4?
- When should the model refuse instead of answer?

Even large companies (e.g., LinkedIn) report:
> Annotation guidelines are among the hardest parts of the pipeline.

---

## Why Annotation Guidelines Fail in Practice

- They are vague
- They assume annotators “just know”
- They are not iteratively refined
- Teams abandon them halfway due to time pressure

Relying on the model to “figure it out” later is **high risk**, especially for:
- Safety
- Correctness
- Legal or medical domains

---

## Shared Foundation: Training Data & Evaluation Data

✅ **Annotation guidelines for training and evaluation should be aligned**

Benefits:
- Evaluation data can seed training data
- Evaluation examples can be augmented
- Reduces duplicated effort
- Improves consistency across the lifecycle

> Investing in evaluation data early pays off twice.

---

## Key Takeaways (Quick Interview Notes)

- Application data is the most valuable data source
- Data acquisition is iterative, not linear
- Public datasets save time but require deep validation
- Annotation guidelines are harder than annotation itself
- Poor guidelines → noisy data → poor models
- Training and evaluation data should share guidelines
- Data flywheels create durable AI advantages

---

## Interview-Ready One-Liner

> *“Models scale with compute, but products scale with data—and the hardest part
isn’t labeling data, it’s defining what ‘good’ actually means.”*
# Data Augmentation and Data Synthesis

Data—along with compute and talent—is one of the **hardest constraints in AI**. The
ability to generate training data programmatically has long been a goal of the
industry. Two key approaches enable this: **data augmentation** and **data synthesis**.

This section explains **what they are, why they matter, when to use them, and their
trade-offs**, with clear interview-ready framing.

---

## Definitions

### Data Augmentation
- Creates new data by **transforming existing real data**
- Example:
  - Flip, rotate, or crop an image
  - Paraphrase a sentence
- The underlying signal comes from **real data**

### Data Synthesis
- Generates **entirely new data** to mimic properties of real data
- Example:
  - Simulating web-bot movements
  - Generating synthetic medical records
- Not derived from a real instance

> In practice, the two terms are often used interchangeably because both aim to
automate dataset creation. In this chapter, **“data synthesis” is used as an umbrella
term**.

---

## Historical Context

Synthetic data predates modern AI:
- Traditionally used for **testing software**
- Tools like `Faker` and `Chance` generate:
  - Names
  - Addresses
  - Emails
  - Phone numbers

With generative AI, synthesis has expanded dramatically to:
- Legal contracts
- Medical notes
- Financial statements
- Product descriptions
- Images, videos, ads

This is a step-change in **data scalability**.

---

## Synthetic Data ≠ Replacement for Human Data

While synthetic data can greatly reduce reliance on human-generated data:
- It **does not fully replace human data**
- Best results often come from **mixing synthetic and human data**
- Limitations (distribution shift, feedback loops) still apply

This trade-off is critical in interviews.

---

## Why Use Data Synthesis?

Synthetic data is attractive because it can enhance the **Golden Data Trio**:
- **Quantity**
- **Coverage**
- **Quality**

It also addresses privacy and enables model distillation.

---

## 1. Increasing Data Quantity

**Most common motivation**

- Enables data at **scale**
- Useful when real data is:
  - Rare
  - Expensive
  - Dangerous
  - Non-existent

### Examples
- Rare weather events
- Deep-sea exploration
- Self-driving car accidents

> More data improves generalization *in principle*, especially when real data is
scarce.

---

## 2. Increasing Data Coverage

Synthetic data can **target gaps** in existing datasets.

### Typical Uses
- Generate:
  - Very short or very long inputs
  - Edge cases
  - Rare classes (class imbalance)
- Create **adversarial examples**
- Control specific behaviors:
  - Toxic language
  - Unsafe instructions
  - Factual inconsistencies

### Research Examples
- **TrueTeacher (Gekhman et al., 2022)**  
  Generated factually incorrect summaries to train fact-checking models.
- **Anthropic (Perez et al., 2022)**  
  Used LLMs to generate datasets probing 154 behaviors:
  - Ethics
  - Political views
  - Personality traits
  - Biases  

**Key finding:**  
> LM-generated datasets can approach—or even exceed—human-generated datasets
in quality for certain evaluation tasks.

---

## 3. Increasing Data Quality

Contrary to intuition, **synthetic data can sometimes be higher quality than human data**.

### Why?
Humans have inherent limitations:
- Memory gaps
- Inconsistencies
- Biases
- Fatigue

### Examples
- **Tool-use data**
  - Humans operate inefficiently from an AI perspective
  - AI-generated tool traces can be more precise and consistent
- **Complex math problems**
  - AI can generate problems beyond typical human creativity
- **Preference data**
  - Humans vary due to mood and motivation
  - AI-generated preferences are:
    - More consistent
    - More reproducible

---

## 4. Mitigating Privacy Concerns

Synthetic data is often the **only viable option** in regulated domains.

### Common Domains
- Healthcare
  - Synthetic patient records instead of real PHI
- Insurance
  - Synthetic claims instead of sensitive financial data

Benefits:
- Avoids PII exposure
- Easier regulatory compliance
- Enables broader experimentation

---

## 5. Model Distillation

Synthetic data enables **teacher–student training**.

### Workflow
1. Use a large, expensive model (teacher)
2. Generate outputs on many inputs
3. Train a smaller, cheaper model (student) to imitate behavior

Use cases:
- Reduce inference cost
- Improve latency
- Deploy on edge devices

Distillation via synthetic data is **standard practice** for production systems.

---

## Key Risks and Caveats (Interview-Important)

- Synthetic data can:
  - Reinforce model biases
  - Narrow distribution diversity
  - Cause model collapse if overused
- Human + synthetic mixtures often outperform either alone
- Validation against *real data* is still required

---

## Key Takeaways (Quick Revision)

- Data synthesis and augmentation automate data creation
- Synthetic data improves:
  - Quantity
  - Coverage
  - Sometimes even quality
- Critical for:
  - Privacy-sensitive domains
  - Rare-event modeling
  - Model distillation
- Best practice: **mix human and synthetic data**
- Synthetic data is a powerful tool, not a silver bullet

---

## Interview-Ready One-Liner

> *“Synthetic data helps us scale quantity, target coverage gaps, and sometimes even
outperform humans in consistency—but the best systems still rely on carefully mixing
synthetic and real data.”*
``
# Traditional Data Synthesis Techniques

Before modern generative AI, **data synthesis** was already widely used in software
testing, gaming, robotics, and simulations. These approaches rely on **procedural
generation**—algorithmic creation of data—rather than manual annotation.

This section covers the **two classical data synthesis approaches**:
1. **Rule-based synthesis**
2. **Simulation-based synthesis**

These techniques remain important today and often complement AI-powered synthesis.

---

## Procedural Data Generation (Background)

- **Procedural generation** = algorithm-driven data creation
- Widely used in:
  - Gaming (maps, levels, characters)
  - Robotics
  - Software testing
- Goal: create **large-scale, diverse, controllable data** efficiently

> Many modern AI data-generation techniques build directly on principles developed
in these older domains.

---

## 1. Rule-Based Data Synthesis

### What It Is
- Data is generated using **predefined rules, templates, and transformations**
- Often combined with random generators (e.g., Faker)

### Example: Transaction Data Template
Templates define structure; generators fill fields:

```text
Transaction ID: [Unique Identifier]
Date: [MM/DD/YYYY]
Time: [HH:MM:SS]
Amount: [Transaction Amount]
Merchant Name: [Merchant/Store Name]
Category: [Category Code]
Location: [City, State, Country]
Payment Method: [Credit/Debit/Cash/Online]
Status: [Completed/Pending/Failed]
````

### Common Use Cases

* Fraud detection (before accessing real financial data)
* Documents with fixed structure:

  * Invoices
  * Resumes
  * Contracts
  * Tax forms
  * Config files
* Grammar- or syntax-driven data:

  * Regex patterns
  * Math equations

**Real example:**
DeepMind trained **AlphaGeometry** using **100M synthetic geometry problems**.

---

## Rule-Based Data Augmentation (Transformations)

### Image Augmentation

* Rotate, crop, flip, scale, erase
* Preserves label semantics
* Popularized by **AlexNet (2012)** for ImageNet

### Text Augmentation

* Replace words with synonyms:

  * “fantastic” → “great”
* Use:

  * Dictionaries
  * Word embedding similarity

### Bias Mitigation via Augmentation

Rule-based swaps can reduce dataset bias:

| Original                | Augmented              |
| ----------------------- | ---------------------- |
| She’s a fantastic nurse | He’s a fantastic nurse |
| Mr. Alex Wang           | Ms. Alexa Wang         |
| My mom cooked dinner    | My dad cooked dinner   |

> Useful for **fairness**, but must be applied carefully to avoid artificial distortions.

---

## Perturbation-Based Augmentation

### What Is Perturbation?

* Add **small noise** to inputs without changing ground-truth semantics

### Key Findings

* Tiny perturbations can break models:

  * **One-pixel attacks** fooled:

    * ~68% of CIFAR-10 images
    * ~16% of ImageNet images
* Security and safety implications:

  * Face recognition spoofing
  * Autonomous driving failures

### Defensive Use

* Train models on perturbed data to improve robustness
* Examples:

  * Adversarial training (Goodfellow et al., 2013)
  * **ImageNet-C / ImageNet-P** (corruptions like snow, blur, noise)
  * BERT randomly replaced **1.5% of tokens**, improving performance

---

## Sophisticated Rule-Based Augmentation

Some domains use **advanced procedural pipelines**:

* Snap Inc. (2022):

  * Generated characters with:

    * Different skin tones
    * Body types
    * Clothing
    * Facial expressions
* Goal:

  * Reduce representation bias
  * Improve generalization to edge cases

---

## 2. Simulation-Based Data Synthesis

### What It Is

* Data generated via **virtual environments**
* Replaces costly, dangerous, or rare real-world experiments

---

## Simulation in Autonomous Driving

### Motivation

* Real-world testing is:

  * Expensive
  * Dangerous
  * Hard to scale

### Examples

* CARLA (open source)
* Waymo SimulationCity
* Tesla’s simulated San Francisco

> Example: testing how a self-driving car reacts to a horse on a highway—unsafe in
> reality, trivial in simulation.

---

## Simulation in Robotics

### Typical Workflow

1. Simulate multiple action trajectories
2. Execute virtually
3. Retain only successful outcomes
4. Train robot on filtered data

Use case:

* Learning manipulation tasks (e.g., pouring coffee)

**Key Insight:**

* If a system fails in simulation, it almost always fails in reality

---

## Sim2Real Gap

* Simulations are **approximations**, not reality
* **Sim2Real** research focuses on transferring:

  * Policies
  * Models
  * Control strategies
    from simulation to the real world

Failures are common due to:

* Physics mismatch
* Sensor noise differences
* Unrealistic environments

---

## Simulation for Tool-Use Data

* Humans are not always optimal planners
* Simulations can:

  * Enumerate action sequences
  * Execute and validate them
  * Select most efficient trajectory

Used to train:

* AI agents
* Tool-using LLMs
* Autonomous planners

---

## Simulation for Rare Events

Simulation excels where real data is scarce:

### Examples

* Finance:

  * IPO success
  * Bankruptcy cascades
* Manufacturing:

  * Defects
  * Equipment failure
* Climate science:

  * Extreme weather
  * Long-term climate scenarios

Synthetic simulation data enables models to learn from **potential futures**.

---

## Strengths & Limitations (Interview-Critical)

### Strengths

* Cheap at scale
* Safe
* Highly controllable
* Excellent for rare or dangerous events

### Limitations

* Simplified reality
* Distribution mismatch
* Performance depends on simulation fidelity

---

## Key Takeaways (Quick Revision)

* Traditional data synthesis predates generative AI
* Two main approaches:

  1. **Rule-based** (templates, transformations, perturbations)
  2. **Simulation-based** (virtual environments)
* Still widely used and highly effective
* Foundation for modern AI-powered data synthesis

---

## Interview One-Liner

> *“Before generative models, data synthesis relied on rule-based generation and
> simulations—still crucial today for safety, rare events, robustness, and bias control.”*
# AI-Powered Data Synthesis (Interview-Ready Summary Notes)

AI-powered data synthesis leverages **large, capable models** to generate, transform,
validate, and judge data at scale. Compared to traditional rule-based or simulation-based
methods, AI synthesis is **far more flexible, expressive, and scalable**, making it a
cornerstone of modern post-training pipelines for LLMs.

---

## Why AI-Powered Data Synthesis Matters

AI models can:
- Simulate **programs, tools, APIs, and humans**
- Generate **natural language, code, preferences, and reasoning traces**
- Validate and filter their **own outputs**
- Dramatically reduce **human annotation cost**

As a result, AI-powered synthesis is now **central to supervised finetuning (SFT),
preference tuning, and distillation**.

---

## AI as a Simulator

### 1. Simulating APIs and Programs
- AI can predict the **outputs of APIs without calling them**
- Example: **StableToolBench (Guo et al., 2024)**
  - Used AI to simulate API behavior
  - Avoided latency, cost, and rate limits
- Use case:
  - Training tool-using agents safely and cheaply

---

### 2. Simulating Humans (Self-Play)

**Self-play**: models interact with copies of themselves to generate data.

#### Canonical Examples
- **AlphaGo (DeepMind)** – millions of Go games via self-play
- **OpenAI Five (Dota 2)** – ~180 years of gameplay per day

#### Beyond Games
- Negotiation agents
- Buyer–seller simulations
- Customer support conversations
- Multi-agent planning

**Key Insight**  
Self-play enables exploration of strategies that never appear in human-generated data.

---

## AI for Data Augmentation

### 1. Paraphrasing
AI can rewrite queries to expand coverage:
- “How to reset my password?”
  - “I forgot my password.”
  - “Steps to reset passwords.”
  - “How can I change my password?”

Used to:
- Improve linguistic diversity
- Reduce overfitting to specific phrasings

---

### 2. Translation (Human & Code)

#### Natural Language Translation
- Translate from high-resource → low-resource languages
- Supports languages like Quechua, Lao, etc.

**Back-Translation for Quality Control**
1. Translate X → Y
2. Translate Y → X′
3. If X′ ≠ X → discard Y

---

#### Code Translation
- Translate code between programming languages
- Widely used in **Llama 3** training

---

### 3. Dataset Expansion via Rewriting
- **MetaMath (Yu et al., 2023)**
  - Rewrote 15K math problems into ~400K variants
  - Smaller models trained on MetaMath outperformed larger baselines

---

## Synthetic Data in Pre-Training vs Post-Training

### Pre-Training
- Goal: acquire *new knowledge*
- Synthetic data less effective
- Still used sparingly (e.g., **Cosmopedia – 25B synthetic tokens**)

### Post-Training (Much More Common)
- Instruction tuning
- Preference tuning
- Reward modeling
- Easier to validate correctness
- Higher ROI

---

## AI-Generated Preference Data

Used for:
- RLHF-style pipelines
- Preference finetuning datasets

### Bias Mitigation in AI Judges
- AI judges suffer from:
  - Position bias
  - Order bias

**Mitigation Strategy (NVIDIA, 2024)**
- Ask AI judge twice with swapped outputs
- Keep example only if:
  - Winner is consistent both times

---

## Instruction Data Synthesis (Core Use Case)

Instruction finetuning data = **(instruction, response)** pairs  
AI can generate:
- Instructions
- Responses
- Or both

---

### Forward Instruction Generation

#### Workflow
1. Define:
   - Topics
   - Keywords
   - Task types
2. Generate:
   - Instructions per topic/template
3. Generate:
   - One or more responses per instruction

#### Examples
- **UltraChat (Ding et al., 2023)**
  - Generate topics → subtopics → instructions → responses
- **Alpaca (Taori et al., 2023)**
  - Seeded with 175 examples
  - GPT-3 generated 52K instruction-response pairs

---

### Reverse Instruction Generation (High-Quality Trick)

**Key Insight**
- Long responses are harder to hallucinate than long instructions.
- Humans are better at writing high-quality long content.

#### Reverse Instruction Workflow
1. Start with high-quality long-form content (books, Wiki, stories)
2. Use AI to generate prompts that would elicit that content
3. Pair generated prompts with original human-written responses

#### Benefits
- Higher response quality
- Fewer hallucinations
- Less human annotation

**Used in:**
- Köksal et al. (2023)
- Li et al. (2023)
- Chen et al. (2023)

---

### Recursive Self-Improvement (Bootstrapping)

Li et al. (2023) showed:
1. Train weak model on small dataset
2. Use it to generate instructions for high-quality text
3. Retrain model on synthesized data
4. Repeat until convergence

> In theory, enables **continual self-improvement without new human labels**
> (practically challenging and unstable).

---

## Long-Context Instruction Synthesis

Goal: extend context length (e.g., 8K → 128K)

### Strategy
1. Split long documents into ≤8K chunks
2. Generate QA pairs per chunk
3. Reattach full long document as context during finetuning

This teaches models how to:
- Retrieve correct info from very long contexts

---

## Llama 3: A Gold-Standard Case Study

Llama 3 heavily relied on **synthetic post-training data**.

### Coding Data Synthesis Pipeline

#### Methods Combined
- Code generation
- Code translation
- Code back-translation
- AI-generated unit tests
- AI-driven error correction

---

### Step-by-Step Workflow

1. Generate programming problem descriptions
2. Generate solutions (multiple languages)
3. Generate unit tests (using AI)
4. Run:
   - Parsers
   - Linters
   - Tests
5. If failed:
   - Re-prompt AI with error feedback
6. Translate code into other languages
   - Keep only test-passing versions
7. Generate:
   - Code explanations
   - Documentation
8. Back-translate explanations → code
   - Keep only faithful explanations

✅ Only **fully verified examples** are included.

---

### Outcome
- **2.7M+ high-quality synthetic coding examples**
- Major contributor to Llama 3.1 performance

---

## Key Takeaways (Quick Revision)

- AI-powered synthesis is **dominant in post-training**
- Core capabilities:
  - Simulation
  - Self-play
  - Paraphrasing
  - Translation
  - Instruction + preference generation
- Quality control is critical:
  - Back-translation
  - Consistency checks
  - Unit tests
- Llama 3 demonstrates:
  - End-to-end AI-driven data pipelines
  - Synthetic data at massive scale

---

## Interview Sound Bites

- **Why synthetic post-training data?**
  > “It’s cheaper to generate, easier to validate, and directly aligned with downstream tasks.”

- **Biggest risk of AI-generated data?**
  > “Bias amplification and silent hallucinations—mitigated through verification loops.”

- **Why reverse instruction works?**
  > “Humans write better long content; AI writes better prompts.”

---
# Data Verification & Limits of AI-Generated Data (Interview Notes)

---

## 1. Data Verification

Goal: **Filter synthetic (and real) data** so only useful, reliable examples go into training.

### 1.1 Two Main Verification Strategies

1. **Functional correctness (programmatic checks)**
   - Works when outputs are *objectively checkable*:
     - Code: compile, run, unit tests
     - Math: recompute, check numeric equality
     - Formal formats: JSON schema, SQL execution, etc.
   - Reason coding data is so popular:
     - Easy to auto-verify → cheap, high-quality synthetic datasets
     - Llama 3: most synthetic data is code; all three synthesis methods give functionally verifiable outputs (execution, back-translation, tests).

2. **AI verifiers (AI judges / scorers)**
   - When functional checks aren’t possible.
   - Approaches:
     - **Scoring**: rate each example 1–5 or similar.
     - **Classification**: label as *good / bad / discard / keep*.
     - **Spec checking**: describe quality requirements in the prompt and ask:
       > “Does this example satisfy all of these requirements? Yes/No. Explain.”
   - Specialized verifiers:
     - **Factual consistency checkers** (catch hallucinations).
     - **Topic classifiers** (filter irrelevant topics).
     - **Style/quality classifiers** (e.g. “NeurIPS-worthy or not?”).

---

### 1.2 “Is Synthetic ≈ Real?” – Distribution Matching

- If you want synthetic data to “look like” real data:
  - Train a **real vs synthetic classifier**.
  - If classifier easily separates them → synthetic data is low fidelity.
- For domain-specific quality:
  - Example: academic writing
    - Train a classifier: *“Would this be accepted to NeurIPS?”*
    - Discard samples predicted to be clear rejects.

---

### 1.3 Heuristics & Anomaly Detection

**Heuristic filters** (cheap + effective baseline):

Typical filters:
- Empty or extremely short/long samples.
- Repetitive text (copy-pasta, obvious model drift).
- Same instruction with conflicting responses.
- Output that just repeats the input.
- Keyword / metadata based filters:
  - by topic, source, domain, author, date, etc.

Example from **Self-Instruct (Wang et al., 2022)**:
- Remove:
  - Repetitive instructions
  - Too short/too long instructions
  - Multiple different responses for same instruction
  - Output identical to input

**Anomaly detection**
- Fit a model over “normal” examples (embedding-based or statistical).
- Flag outliers as potentially low-quality.

---

### 1.4 Ultimate Test: “Does It Help the Model?”

- Regardless of metrics, the **true evaluation**:
  - Train with/without the synthetic data slice.
  - Check **downstream performance**.
- If it **doesn’t improve or hurts** performance:
  - Data is low quality or misaligned, even if local checks pass.

---

## 2. Limitations of AI-Generated Data

Even with good verification, synthetic data **cannot fully replace** human data (at least with current techniques). Four major limitations:

---

### 2.1 Quality Control & “Garbage In, Garbage Out”

- Synthetic data can be:
  - Incorrect, shallow, repetitive, biased.
- If we **can’t reliably measure/verify** its quality, it’s risky to use.
- Strong verification pipelines (execution tests, AI judges, heuristics) are **mandatory**, not optional.

---

### 2.2 Superficial Imitation (Style vs Substance)

Paper: **“The False Promise of Imitating Proprietary LLMs” (Gudibande et al., 2023)**

Key points:
- Distilled / imitation models often:
  - Copy **style** (tone, phrasing, format).
  - Struggle with **facts, reasoning, generalization** beyond training distribution.
- Example failure mode:
  - Teacher can solve hard math problems.
  - Student is trained on teacher’s final solutions only.
  - Student **learns to output “solution-shaped” answers** without actually doing the reasoning → hallucinated math.

Same can happen with human labels:
- If annotator uses knowledge the model doesn’t have, they’re effectively teaching the model to **confidently guess**.

**Takeaway**  
Imitation alone ≠ real capability improvement, especially for reasoning.
We still need **better base models + good reasoning data**, not just stylistic mimicry.

---

### 2.3 Model Collapse (Training on AI-Generated Data Repeatedly)

Paper: **“The Curse of Recursion: Training on Generated Data Makes Models Forget” (Shumailov et al., 2023)**

- If models are repeatedly trained on their own outputs:
  - They may **forget rare / tail events**.
  - Distribution collapses toward high-probability, generic patterns.
- Effects:
  - Irreversible degradation in:
    - Calibration
    - Coverage of rare cases
    - Overall diversity

Hypothesis:
- Models over-generate **common outcomes** (e.g., “no cancer”) and under-represent rare ones (“has cancer”).
- Over generations:
  - Common becomes **oversampled**, rare becomes **almost absent** → collapse.

Follow-up work:
- **Gerstgrasser et al. (2024)** and others:
  - Collapse is inevitable if training data is *purely synthetic*.
  - Mixing **real + synthetic** can avoid collapse.
  - But no clear formula yet for “safe percentage”.

Empirical counterpoints:
- Some works show **large synthetic fractions can still help**:
  - Llama 2-7B math finetuning with up to ~1M synthetic samples shows no saturation.
  - Nemotron-4 340B-Instruct used ~**98% synthetic data** for instruction + preference finetuning (but only one generation, not repeated cycles).

**Takeaway**  
Synthetic data is powerful, but **unbounded recursive reuse** is dangerous.  
We need:
- Real data mixed in
- Careful monitoring across generations

---

### 2.4 Bias Amplification in Feedback Loops

Paper: **“Data Feedback Loops: Model-driven Amplification of Dataset Biases” (Taori & Hashimoto, 2023)**

- Training repeatedly on model outputs can:
  - **Amplify existing biases** (political, demographic, stylistic, etc.).
- Interestingly:
  - The more *faithfully* synthetic data matches original distribution, the more stable the loop.
  - So “too much clever editing” can worsen things; careful mimicry can stabilize.

---

### 2.5 Obscured Data Lineage (Provenance Risk)

AI generation **hides where data really came from**:

Risks:
1. **Copyright / license contamination**
   - Teacher model trained on copyrighted text.
   - Its outputs are used as training data for your model.
   - You might inadvertently violate IP, even if your dataset appears “clean”.

2. **Benchmark contamination**
   - Teacher was trained on benchmark B.
   - Synthetic data leaks B’s content (memorized).
   - You train your student model on this data.
   - Student then “aces” B → but it’s just memorization via teacher.
   - Evaluation is no longer trustworthy.

Without transparent lineage:
- Hard to assess:
  - Legal safety
  - Generalization
  - True performance on “unseen” tasks

---

## 3. Practical Rules of Thumb

You can turn these into quick design heuristics:

1. **Always verify synthetic data**
   - Prefer **functional checks** where possible.
   - Use AI judges + heuristics elsewhere.

2. **Mix real + synthetic**
   - Don’t rely on **only** synthetic, especially across multiple generations of training.

3. **Don’t confuse style with capability**
   - Distillation = good for **interface & style**.
   - For reasoning, you still need:
     - Strong base models
     - High-quality reasoning data (CoT, proofs, unit tests, etc.).

4. **Track provenance**
   - Keep metadata on:
     - Source model
     - Source corpus
     - Generation date
     - Generation prompts
   - Helps with legal, safety, and evaluation integrity.

---
# Model Distillation (Knowledge Distillation) — Interview-Ready Notes

---

## 1. What Is Model Distillation?

**Model distillation (knowledge distillation)** is a technique where a **smaller model (student)** is trained to **mimic the behavior of a larger, more capable model (teacher)**.  
The goal is to *transfer knowledge* from the teacher to the student, usually via **synthetic data generated by the teacher**.

> Coined by Hinton et al. (2015): knowledge from a large model is “distilled” into a smaller one.

---

## 2. Why Distillation Is Used

### Primary motivations
- **Deployment efficiency**
  - Large models are expensive in:
    - Inference cost
    - Latency
    - Memory
    - Energy
- Distilled models:
  - Are **smaller**
  - Run **faster**
  - Are **cheaper to serve**
  - Often retain most of the teacher’s performance

### Classic example
- **DistilBERT (Sanh et al., 2019)**
  - ~40% smaller than BERT
  - Retains **~97%** of language understanding
  - **~60% faster** inference

---

## 3. How Distillation Is Done

### Two common student initialization strategies

1. **Train student from scratch**
   - Example: DistilBERT
   - Student architecture is smaller, trained directly on teacher-generated signals.

2. **Finetune a pre-trained student**
   - Example: Alpaca
   - Start with a pre-trained model (e.g., Llama-7B)
   - Finetune on teacher-generated instruction data.

---

## 4. Distillation via Synthetic Instruction Data

### Alpaca case study (Taori et al., 2023)
- **Teacher:** text-davinci-003 (~175B)
- **Student:** Llama-7B
- **Method:** Instruction finetuning on teacher-generated examples
- **Result:**
  - Student behaves similarly to teacher
  - Student is only **~4% the size** of the teacher

This pattern is now very common:
> **Large proprietary model → generate data → finetune smaller open model**

---

## 5. Distillation + PEFT (LoRA)

- Distillation is frequently combined with **adapter-based techniques** like **LoRA**
- Key benefit:
  - Further reduces training cost and memory usage
- Example:
  - **BuzzFeed (2023)**:
    - Used OpenAI’s text-davinci-003 to generate data
    - Finetuned Flan-T5 with **LoRA**
    - Achieved **~80% inference cost reduction**
    - (Exact performance trade-offs were not publicly detailed)

**Practical takeaway:**  
> Synthetic data + PEFT = *cheap, fast specialization*

---

## 6. Important Legal & License Constraints ⚠️

> **Not all models can legally be distilled.**

- Many proprietary model licenses:
  - **Explicitly forbid** using their outputs to train other models
  - Especially to train *competing* models
- This is **critical in production and enterprise settings**

✅ Always check:
- Model terms of service
- “No competitive training” clauses
- Output usage restrictions

---

## 7. Distillation ≠ All Synthetic Training

### Key distinction
- **Model distillation**:
  - Teacher’s behavior = **gold standard**
  - Student explicitly attempts to *match teacher*
- **General synthetic-data training**:
  - Teacher-generated data is just one ingredient
  - Student may surpass teacher

---

## 8. When Student Can Beat the Teacher

### Example 1: Reverse Instruction Bootstrapping
- Student generates better prompts for high-quality human content
- Iterative improvement without new human labels

### Example 2: Nemotron-4 (NVIDIA, 2024)
- **Base model:** 340B pretrained from scratch
- **Teacher for finetuning:** Mixtral-8x7B-Instruct (~56B MoE)
- **Student > Teacher**
  - Nemotron-4-340B-Instruct outperformed the teacher on many tasks

**Key insight:**  
> Teacher does **not** have to be bigger — just *better aligned* or *better at certain behaviors*.

---

## 9. Risks of Naive Distillation

From Llama 3 findings and prior research:

### Self-generated data pitfall
- Training **indiscriminately** on a model’s own outputs:
  - Does **not** reliably improve performance
  - Can **degrade** the model over time

### Why this happens
- Error amplification
- Bias reinforcement
- Hallucination feedback loops
- Loss of diversity (related to model collapse)

---

## 10. Safe Distillation: What Works

According to Llama 3 authors:

✅ Distillation **can** lead to continual improvement **if**:
- Synthetic data is **verified**
- Low-quality generations are filtered out
- Only **high-confidence, validated samples** are used

This reinforces a broader theme:
> **Distillation without verification is dangerous.  
> Verified synthetic data is powerful.**

---

## 11. Summary Table (Quick Revision)

| Aspect | Key Point |
|------|---------|
| Core idea | Small model learns from large model |
| Main goal | Reduce cost, latency, deployment size |
| Typical data | Teacher-generated synthetic instructions |
| PEFT synergy | Often combined with LoRA |
| Legal risk | Licenses may forbid competitive training |
| Not all synthetic data | ≠ distillation |
| Student > teacher | Possible (Nemotron-4) |
| Major risk | Naive self-training degrades models |
| Best practice | Verify synthetic data rigorously |

---

## 12. Interview Soundbites ✅

- “Distillation is about **behavior transfer**, not just size reduction.”
- “Synthetic data is powerful, but **verification is the difference between progress and model collapse**.”
- “Distillation works best when combined with **PEFT and strong data filtering**.”
- “Teacher superiority is *behavioral*, not strictly about parameter count.”
- “Licensing constraints are often the **real bottleneck**, not the technology.”

---
# Data Processing for Finetuning & Training — Interview-Ready Notes

This section focuses on **practical data processing steps** required to convert raw data into **model-ready datasets**, with emphasis on **efficiency, correctness, and inspection**. Data processing quality directly impacts model performance, training cost, and debugging complexity.

---

## 1. Why Data Processing Matters

- Raw data is **never ready** for training as-is.
- Poor processing can:
  - Introduce bias
  - Waste compute
  - Corrupt datasets permanently
  - Hide systemic annotation issues
- **Reading model papers** that disclose dataset processing steps is one of the best ways to learn real-world best practices.

---

## 2. Efficiency Principles for Large-Scale Data Processing

When working with large datasets, each processing step can take **hours or days**.

### Recommended best practices:

#### ✅ Order operations by cost
- Reorder processing steps to **minimize expensive operations**:
  - If cleaning is expensive → deduplicate first
  - If deduplication is expensive → filter low-quality data first
- There is **no fixed order** — choose what saves time and compute.

#### ✅ Always do trial runs
- Run pipelines on a **small subset first**
- Catch:
  - Logic errors
  - Schema mismatches
  - Unexpected data corruption

#### ✅ Never modify data in place
Keep raw data immutable:
- Enables reprocessing for future tasks
- Protects against irreversible bugs
- Supports reproducibility and auditing

> **Rule of thumb:** Raw data is sacred.

---

## 3. Step 1 — Inspect the Raw Data (Critical Step)

Before cleaning or transforming data, **understand what you have**.

### Key questions to answer:
- Where does the data come from?
- How was it gathered and processed?
- Has it been used elsewhere (risk of contamination)?
- What formats, tokens, or conventions does it use?

---

## 4. Statistical Exploration & Distribution Analysis

### Core distributions to inspect:
- Token frequency
- Input length
- Response length
- Special tokens usage
- Language distribution
- Topic distribution

These help answer:
- Is the data aligned with your task?
- Is there length or topic skew?
- Are there unexpected artifacts?

---

## 5. Advanced Exploratory Analysis (Used by Top Teams)

### Example: GPT-3 vs GPT-4 analysis (Microsoft, 2023)
Researchers compared:
- Distribution of **(verb, direct object, noun)** pairs
- Response length distributions

**Key observations:**
- GPT-4:
  - Uses broader and more diverse verb–noun combinations
  - Generates longer responses
- These statistics can be used to:
  - Compare datasets
  - Compare model outputs
  - Detect regressions or shifts

---

## 6. Slice the Data Along Multiple Axes

Always analyze data **by subgroups**, such as:
- Data source
- Time
- Annotator
- Language
- Topic
- Task type

Look for:
- Outliers
- Systematic biases
- Annotators who consistently:
  - Give higher or lower scores
  - Write shorter or longer responses

If scores are expected to be normally distributed:
- Check if **each annotator’s distribution** looks reasonable
- Annotator bias is common and dangerous if ignored

---

## 7. Handling Annotations

### Multiple annotations per example
- Compute **inter-annotator disagreement**
- Investigate examples with high disagreement:
  - Is the task ambiguous?
  - Are annotation guidelines unclear?
  - Is one annotator unreliable?

### Resolution strategies:
- Consensus labeling
- Tie-breaker annotators
- Adjust guidelines and re-annotate

---

## 8. Manual Inspection Is Non-Negotiable

> Greg Brockman (OpenAI):  
> **“Manual inspection of data has probably the highest value-to-prestige ratio in ML.”**

Why manual inspection matters:
- Tools show patterns, not meaning
- 10–15 minutes of reading examples often:
  - Reveals systemic issues
  - Exposes annotation flaws
  - Saves days of debugging

### What to manually check:
- Do examples make sense semantically?
- Are responses factual and relevant?
- Are instructions clear and realistic?
- Try annotating examples yourself:
  - Do your labels match existing ones?

---

## 9. Consistency & Uniqueness Checks

Look for:
- Same query → different responses
- Same response → different queries
- Duplicated or near-duplicated examples

These issues can:
- Confuse the model
- Introduce training instability
- Cause memorization or hallucinations

---

## 10. Key Takeaways (Quick Revision)

### ✅ Core principles
- Inspect before cleaning
- Optimize processing order
- Never mutate raw data
- Combine statistics + manual review

### ✅ What top teams do
- Analyze distributions deeply
- Slice data along multiple dimensions
- Track annotator behavior
- Treat data inspection as first-class work

### ✅ Interview-ready insight
> “Most model failures I’ve seen were **data failures**, not model failures.”

---

## 11. Interview Soundbites

- “Data processing order should be determined by **compute cost**, not convention.”
- “If you don’t understand your dataset’s distributions, you don’t understand your model.”
- “Manual data inspection saves more time than any automated tool.”
- “Annotation bias is a hidden failure mode in finetuning pipelines.”

---
# Data Processing (Continued): Deduplication, Cleaning, and Filtering

This section covers **why and how to deduplicate training data**, followed by **data cleaning and filtering practices** that directly impact model quality, safety, and training efficiency.

---

## 1. Deduplicate Data

### Why Deduplication Is Critical

Duplicated data can:
- **Skew data distributions**
- **Introduce spurious correlations**
- **Bias model behavior**
- **Cause train–test contamination**
- **Waste compute and annotation budget**

### Toy Example: Duplication Bias

| ID | Input (Product Description)            | Output (Price) |
|----|----------------------------------------|----------------|
| 1  | {item: pencil, color: red}             | $20            |
| 2  | {item: compass, color: green}          | $2             |
| 3  | {item: pencil, color: red}             | $20            |
| 4  | {item: pencil, color: red}             | $20            |
| 5  | {item: pencil, color: green}           | $1             |

Duplications (rows 1, 3, 4) may cause the model to **incorrectly associate red items with high prices**.

### Empirical Evidence

- Multiple studies show **performance degradation** due to duplication  
  (Lee et al., 2021; Tirumala et al., 2023)
- Anthropic (Hernandez et al., 2022):
  - Repeating **0.1% of data 100×** degraded an **800M model** to **400M-level** performance
- Even when performance doesn’t drop, **training cost increases unnecessarily**

---

## 2. Types of Data Duplication

Duplications can occur at multiple granularities:

- **Whole-document duplication**
  - Same document appears multiple times
- **Intra-document duplication**
  - Same paragraph repeated within a document
- **Cross-document duplication**
  - Boilerplate text, quotes, or templates reused across documents

### Definitional Questions You Must Decide

- What level?
  - Document / paragraph / sentence / token
- What similarity threshold?
  - Exact match?
  - ≥80% overlap?
- Ordering sensitivity?
  - Are lists duplicates if items are reordered?

There is no universal definition — it depends on **task and risk tolerance**.

---

## 3. Deduplication Techniques

Deduplication typically relies on **similarity measurement** (see Chapter 3).

### 3.1 Pairwise Comparison
- Compare every example with every other example
- Similarity options:
  - Exact match
  - n-gram overlap
  - Fuzzy matching
  - Semantic embeddings
- ❌ Expensive for large datasets (O(N²))

### 3.2 Hashing-Based Methods
- Hash examples into buckets
- Compare only within the same bucket
- Common techniques:
  - **MinHash**
  - **Bloom filters**
- ✅ Efficient and scalable

### 3.3 Dimensionality Reduction
- Embed data into lower-dimensional space
- Run similarity search or clustering
- Reuses vector search infrastructure (Chapter 6)

---

## 4. Useful Deduplication Libraries

Popular open-source tools include:
- `dupeGuru`
- `Dedupe`
- `datasketch`
- `TextDistance`
- `TheFuzz`
- `deduplicate-text-datasets`
- `lazyNLP` (supports Bloom-filter-based overlap estimation)

---

## 5. Clean and Filter Data

Deduplication alone is not sufficient. Data must also be **clean, compliant, and high-quality**.

---

## 6. Remove Extraneous Formatting

Many public datasets are scraped and contain:
- HTML tags
- Markdown artifacts
- Boilerplate formatting

Unless you want the model to **learn formatting**, remove them.

### Real-World Impact
- Databricks reported:
  - **+20% accuracy improvement**
  - **-60% input token length**
- Just from removing unnecessary Markdown and HTML tokens

---

## 7. Policy & Safety Cleaning

Remove or redact:
- Personally Identifiable Information (PII)
- Sensitive attributes (e.g., name, gender, ZIP code)
- Copyrighted content
- Toxic or unsafe text

Use detection and filtering techniques discussed in **Chapter 4**.

---

## 8. Remove Low-Quality Data

Low-quality data hurts models **more than having less data**.

### Detection Techniques
- AI-based verifiers
- Heuristics from data verification (see p. 391)
- Manual review

### Importance of Manual Inspection
- Visual inspection often reveals:
  - Repetitive patterns
  - Template abuse
  - Annotator shortcuts
- These insights can be converted into filtering heuristics

### Non-Obvious Quality Issues
- **Annotator fatigue**
  - Kern et al. (2024) found annotations created later in sessions are lower quality
- Time-based filtering or quality weighting can help

---

## 9. Filtering When Data Is Too Large

If you have **more data than compute budget**, further pruning is necessary.

### Techniques

#### 9.1 Active Learning
- Select examples that:
  - Maximize uncertainty
  - Improve decision boundaries
- Requires a reasonable evaluation signal

#### 9.2 Importance Sampling
- Weight or select examples most useful for training
- Depends on defining “importance” reliably

### Research Insight
- Meta (Sorscher et al., 2022):
  - High-quality data-pruning metrics can **dramatically reduce training costs**
  - Data efficiency is as important as model efficiency

---

## 10. Key Takeaways (Interview-Ready)

- Deduplication is **both a quality and efficiency requirement**
- Even tiny duplication rates can **severely degrade model performance**
- Cleaning formatting tokens can drastically improve accuracy and reduce cost
- Manual data inspection is essential for discovering hidden failure modes
- Smart filtering can save **millions in compute** without hurting performance

---

## 11. Strong Interview Soundbites

- “0.1% duplicated data repeated enough times can halve effective model capacity.”
- “Deduplication is a prerequisite for trustworthy evaluation.”
- “Formatting tokens are silent compute killers.”
- “Bad data costs more than no data.”

---
``
## Format Data

Once you’ve **deduplicated, cleaned, and filtered** your data, the next step is to
**format it exactly as expected by the model you’re finetuning**.

This step is deceptively important—**formatting mismatches are a common cause of
silent model failures**.

---

## 1. Match the Model’s Expected Template

Each model:
- Uses a **specific tokenizer**
- Expects a **specific chat or prompt template**

These templates (covered in Chapter 5) define how:
- System messages
- User messages
- Assistant responses  
are serialized into tokens.

❗ **Using the wrong template won’t usually crash training**—instead, it leads to:
- Poor convergence
- Unexpected behaviors
- Models ignoring parts of the input

---

## 2. Typical Supervised Finetuning Format

For supervised finetuning (SFT), data is usually stored as:

```

(instruction, response)

````

Where `instruction` can be decomposed into:
- **System prompt** (optional)
- **User prompt**

### Key Difference from Prompt Engineering

If you are moving from **prompt engineering → finetuning**:

- ✅ Prompt engineering often uses:
  - Long task descriptions
  - Few-shot examples
- ✅ Finetuning typically **does not require these**
  - The model learns behavior directly from many examples

---

## 3. Example: Converting a Few-Shot Prompt into Training Data

### Prompt Used with Base Model (3-shot)

```text
Label the following item as either edible or inedible.
Item: burger
Label: edible
Item: car
Label: inedible
Item: mushroom
Label: edible
Item: {INPUT}
Label:
````

### Converted Finetuning Dataset

| Example ID | Input    | Output   |
| ---------- | -------- | -------- |
| 1          | burger   | edible   |
| 2          | car      | inedible |
| 3          | mushroom | edible   |
| …          | …        | …        |

This converts **prompt examples into explicit supervision**.

---

## 4. Simpler Prompts After Finetuning

Once the model is finetuned, inference prompts can be much shorter:

```text
{INPUT} -->
```

✅ Benefits:

* Fewer input tokens
* Lower inference cost
* More predictable outputs

This is one reason **finetuning reduces serving cost**, especially for high-volume
applications.

---

## 5. Prompt–Training Format Alignment Is Mandatory

The **format used during inference must exactly match the training format**.

If the model was trained with:

```text
burger -->
```

Then **all of the following may cause issues**:

* ❌ `burger`
  (missing arrow)
* ❌ `Item: burger -->`
  (extra prefix)
* ❌ `burger --> `
  (extra trailing space)

These differences seem trivial to humans but **are different token sequences to the
model**.

---

## 6. Why Small Format Deviations Matter

* Tokenizers are sensitive to:

  * Whitespace
  * Prefixes
  * Separators
* The model learns **token-level patterns**, not conceptual intent
* Inference-time mismatches cause:

  * Reduced accuracy
  * Regression to base-model behavior
  * Hallucinations or refusals

---

## 7. Practical Best Practices

* ✅ Store a **canonical prompt template** alongside the model
* ✅ Validate inference prompts against training examples
* ✅ Include format validation in CI tests
* ✅ Run ablations to test format variants when performance is poor

---

## 8. Interview-Ready Takeaways

* “Formatting errors don’t raise exceptions—they degrade learning.”
* “Finetuning replaces long prompts with shorter, learned behaviors.”
* “Token-level mismatch is a common cause of silent performance regression.”
* “Inference prompts must be isomorphic to training prompts.”

---
