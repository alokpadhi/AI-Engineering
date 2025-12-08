# Training Data in AI Models

## Core Principle
> **An AI model is only as good as the data it is trained on.**

- Model capabilities are **directly constrained by training data coverage**.
  - No Vietnamese data → No English ↔ Vietnamese translation.
  - Only animal images → Poor performance on plant classification.
- Models cannot generalize well to tasks or domains **never represented in training data**.

---

## Data Availability vs. Data Relevance
- Improving performance on a task usually requires **more task-specific data**.
- **Challenges in data collection**:
  - Large-scale data collection is **expensive, time-consuming, and difficult**.
  - As a result, developers often rely on **available data rather than ideal data**.

> This leads to a pragmatic but risky mindset:  
> **“Use what we have, not what we want.”**

---

## Common Crawl as a Training Data Source
- **Common Crawl**:
  - Created by a nonprofit organization.
  - Crawls **2–3 billion webpages per month** (2022–2023).
- **C4 (Colossal Clean Crawled Corpus)**:
  - A cleaned subset of Common Crawl, released by Google.
  - Widely used for training large language models.

---

## Data Quality Issues
- Internet-scale datasets suffer from **severe quality problems**, including:
  - Clickbait
  - Fake news and misinformation
  - Propaganda and conspiracy theories
  - Racism and misogyny
- A Washington Post study found that:
  - Many of the **top 1,000 most frequent websites** in Common Crawl
  - Rank **low on NewsGuard’s trustworthiness scale**

> ✅ **Key Insight**:  
> *Availability ≠ Reliability*. Common Crawl contains **significant amounts of fake and low-quality content**.

---

## Industry Usage & Transparency Concerns
- Despite quality issues, **Common Crawl (or its variants)** is used in:
  - OpenAI’s **GPT-3**
  - Google’s **Gemini**
- Even models that **do not disclose** training data sources are likely using it.
- Increasingly, companies:
  - Avoid disclosing training data sources
  - To reduce **public scrutiny** and **competitive analysis**

---

## Data Filtering & Heuristics
- Some teams apply **heuristic-based filtering** to reduce noise.
  - Example: GPT-2 was trained only on **Reddit links with ≥3 upvotes**.
- Limitations:
  - Filters remove low-interest content, **not necessarily low-quality content**
  - Platforms like Reddit still contain bias, misinformation, and questionable content

---

## Consequences of Poor Data Alignment
- Models trained on generic web data:
  - Perform well on **frequent tasks in training data**
  - Perform poorly on **tasks important to the user but rare or missing in data**
- This mismatch highlights the need for:
  - **Intentional dataset curation**
  - Alignment with **specific languages and domains**

---

## Domain-Specific & Language-Specific Models
- Approaches:
  1. **Train specialized models from scratch** (language/domain-specific)
  2. **Finetune general-purpose foundation models**
     - More common
     - More compute-efficient
- Specialized data provides a **broad yet focused foundation** for targeted applications.
- Highly task-specific data strategies are explored further in **advanced settings**.

---

## Why Not Train on All Available Data?
- A common intuition: *“More data = better model”*  
  ❌ **This is not always true.**
- Downsides of indiscriminate scaling:
  - Higher **compute cost**
  - No guaranteed performance improvement
- **Data quality often matters more than quantity**

### Empirical Evidence
- Gunasekar et al. (2023):
  - Trained a **1.3B-parameter model**
  - Using only **7B tokens of high-quality coding data**
  - Outperformed **much larger models** on coding benchmarks

> ✅ **Key Insight**:  
> *A smaller model + high-quality data can beat larger models trained on noisy data.*

---

## 🔑 Key Takeaways (Quick Revision)
- Training data defines **what a model can and cannot do**
- Large web datasets are:
  - Easy to obtain
  - Hard to trust
- Common Crawl is widely used **despite quality issues**
- Heuristics help but **do not solve fundamental data problems**
- **Data curation and alignment** are critical for real-world performance
- **Quality > Quantity** when it comes to training data

---

## ⚠️ Important Notes (Interview-Ready)
- “More data” does not automatically mean “better performance”
- Data quality directly affects:
  - Model bias
  - Hallucinations
  - Robustness
- Foundation models often hide data sources to avoid:
  - Ethical scrutiny
  - Legal and competitive risks

---

## 💡 How to Explain in Interviews
- Emphasize **data-centric AI** over model-centric AI
- Use examples:
  - Common Crawl noise
  - High-quality small datasets outperforming large noisy ones
- Stress **finetuning + curated data** as an industry-standard best practice

# Multilingual Models & Language Representation in LLMs

## English Dominance in Internet Training Data
- English overwhelmingly dominates web-scale training datasets.
- **Common Crawl language distribution**:
  - English: **45.88%** of all data
  - Second-most common language (Russian): **~6%**
  - English is **~8× more prevalent** than any other language
- Languages with **<1% representation** are classified as **low-resource languages**.

> ✅ **Key Insight**:  
> Most foundation models are implicitly **English-first models** due to data availability.

---

## Most Represented Languages in Common Crawl
- High-resource languages (≥1%):
  - English, Russian, German, Chinese, Japanese, French, Spanish, Portuguese, etc.
- These languages benefit from:
  - Better coverage
  - Higher fluency
  - Stronger reasoning and task performance

> ⚠️ Representation is **not proportional to number of speakers** worldwide.

---

## Under-Represented Languages: A Structural Problem
- Many widely spoken languages are **severely under-represented** in Common Crawl.
- Examples:
  - Telugu, Kannada, Punjabi, Marathi, Bengali, Urdu, Swahili
- **World Population : Common Crawl Ratio**
  - Ideal ratio = **1**
  - Ratios **≫ 1** indicate extreme under-representation
  - Example:
    - Punjabi: **~231× under-represented**
    - Telugu: **~65× under-represented**
    - Bengali: **~37× under-represented**
    - English (for comparison): **0.40**

> ✅ **Key Insight**:  
> Data imbalance is **systemic**, not marginal.

---

## Impact on Model Performance
### Benchmark Evidence
- **MMLU benchmark** (14,000 questions across 57 subjects):
  - GPT-4 performs **best in English**
  - Performance drops significantly for low-resource languages (e.g., Telugu)
- **Project Euler math tasks**:
  - GPT-4 solves problems:
    - **>3× more often in English**
    - Fails completely for Burmese and Amharic

### Correlation
- Worst-performing languages on benchmarks (Telugu, Marathi, Punjabi):
  - Also among the **most under-represented** in training data

> ✅ **Under-representation is a primary driver of multilingual performance gaps.**

---

## Beyond Data Quantity: Language Complexity
- Under-performance is **not explained by data scarcity alone**.
- Other contributing factors:
  - Linguistic structure (morphology, syntax)
  - Cultural constructs embedded in language
- Some languages are **intrinsically harder** for models to learn.

---

## Translation-to-English: A Common but Flawed Workaround
### Typical Approach
1. Translate query → English
2. Generate response
3. Translate back to original language

### Why This Fails
- Requires a **strong translation model**, which itself suffers from data scarcity
- **Information loss during translation**:
  - Example: Vietnamese pronouns encode social relationships
  - English collapses these into “I / you”, losing semantic nuance

> ⚠️ Translation pipelines reduce linguistic and cultural fidelity.

---

## Safety & Behavior Inconsistencies Across Languages
- **NewsGuard Study (April 2023)**:
  - ChatGPT-3.5 declined misinformation in English (6/7 prompts)
  - Produced misinformation in:
    - Simplified Chinese (7/7)
    - Traditional Chinese (7/7)
- Cause of this disparity is **unknown**
- Indicates:
  - Safety alignment is **English-centric**
  - Non-English behavior may be **less constrained**

> ✅ **Key Insight**:  
> Multilingual safety and alignment are significantly weaker than English.

---

## Tokenization, Latency & Cost Issues
- Inference cost and latency scale with **token count**.
- Tokenization efficiency varies greatly by language.

### MASSIVE Dataset Findings (52 languages)
| Language  | Median Tokens |
|---------|---------------|
| English | 7 |
| Hindi   | 32 |
| Burmese | 72 |

- Same meaning requires:
  - ~4.5× tokens in Hindi
  - ~10× tokens in Burmese
- Consequences:
  - Slower inference
  - Higher API costs for non-English users

> ⚠️ Non-English users are **penalized in both latency and cost**.

---

## Emerging Non-English Foundation Models
To address multilingual gaps, many **language-specific models** are being developed:

- **Chinese**: ChatGLM, YAYI, Llama-Chinese
- **French**: CroissantLLM
- **Vietnamese**: PhoGPT
- **Arabic**: Jais
- And many others

> ✅ These models:
> - Focus on **local data**
> - Improve fluency, reasoning, and safety
> - Reduce token inefficiency

---

## 🔑 Key Takeaways (Quick Revision)
- Internet data is **heavily skewed toward English**
- Many major languages are **extremely under-represented**
- Multilingual performance gaps are:
  - Empirically verified
  - Data-driven but also linguistically rooted
- Translation-based solutions are **lossy and unreliable**
- Tokenization inefficiency causes **higher cost and latency**
- Language-specific LLMs are a **necessary evolution**, not a niche trend

---

## ⚠️ Important Notes (Interview-Ready)
- Multilingual ≠ Multicultural understanding
- Safety alignment does not generalize equally across languages
- Token efficiency is a **hidden production cost**
- Fairness in AI requires:
  - Data rebalancing
  - Language-aware evaluation

---

## 💡 How to Explain in Interviews
- Frame multilingual issues as a **data + systems problem**
- Mention:
  - Under-representation ratios
  - Benchmark evidence (MMLU, MASSIVE)
  - Tokenization cost implications
- Highlight:
  - Why language-specific models are gaining traction
  - Why “just translate to English” fails in practice
# Domain-Specific Models in AI

## General-Purpose Foundation Models
- Models like **GPT, Gemini, and LLaMA** perform well across many domains:
  - Coding, law, science, business, sports, environmental science, etc.
- This broad capability is mainly due to:
  - **Diverse domain coverage in training data**
  - Large-scale internet datasets (e.g., Common Crawl / C4)

### Domain Coverage in Training Data
- Analyses (e.g., Washington Post, 2023) show:
  - Common Crawl includes **many popular domains**
  - But only reveals **what is present**, not **what is missing**
- Coverage biases exist:
  - Domains frequent on the web are over-represented
  - Highly specialized or private domains are absent

> ✅ **Key Insight**:  
> General-purpose models reflect the **distribution of publicly available internet data**.

---

## Domain Representation in Vision Models
- Unlike text, **domain analysis for vision datasets is limited**:
  - Images are harder to categorize than text
  - No reliable heuristics comparable to domain keywords
- As a result:
  - A model’s domain strength is often inferred from **benchmark performance**
  - Benchmarks test only a **small subset of the real world**

---

## Inference from Benchmark Performance (Vision Models)
### CLIP vs OpenCLIP (ViT-B/32)
- Benchmarked across datasets such as:
  - ImageNet, Birdsnap, Flowers, Traffic Signs, Cars, Videos
- Performance varies widely across domains:
  - Strong on certain benchmarks (e.g., flowers, cars)
  - Weak on others (e.g., geographic or fine-grained classification)

> ⚠️ **Important Limitation**:  
> Benchmarks cover only **narrow domains**, while real-world visual complexity is far broader.

---

## Why General Models Fail on Domain-Specific Tasks
- Even strong general models struggle with **highly specialized tasks**, especially when:
  - Required data formats are unique
  - Data is private, expensive, or regulated
- Examples of such tasks:
  - **Drug discovery**
    - Requires protein, DNA, RNA sequences
    - Structured biological data rarely found on the open internet
  - **Cancer screening**
    - Uses X-ray, CT, or fMRI scans
    - Data is scarce due to privacy and ethical constraints

> ✅ **Key Insight**:  
> General models fail when **task-specific data is missing from training**.

---

## Domain-Specific Model Strategy
- High performance on specialized tasks requires:
  - **Curated, high-quality, domain-specific datasets**
- Domain-specific models are often:
  - Smaller
  - More targeted
  - More accurate for their intended use case

---

## Canonical Examples of Domain-Specific Models
- **AlphaFold (DeepMind)**:
  - Trained on ~100,000 known protein sequences & 3D structures
  - Transformational progress in protein folding
- **BioNeMo (NVIDIA)**:
  - Focused on biomolecular data
  - Optimized for drug discovery
- **Med-PaLM 2 (Google)**:
  - Combines LLMs with medical data
  - Improves accuracy in clinical and medical QA

---

## Beyond Biomedicine: Wider Applications
- While most prevalent in healthcare, domain-specific models apply to:
  - Architecture (trained on architectural sketches)
  - Manufacturing (factory layouts, process plans)
  - Engineering design
- These models can:
  - Outperform generic models like ChatGPT
  - Deliver **tool-like precision for professional users**

> ✅ **Key Insight**:  
> Domain-specific models trade **generality for depth and reliability**.

---

## Trade-Off: General vs Domain-Specific Models
| Aspect | General-Purpose Models | Domain-Specific Models |
|------|----------------------|------------------------|
| Coverage | Broad | Narrow |
| Data | Web-scale, noisy | Curated, expensive |
| Accuracy | Good average | High for target domain |
| Cost | High training cost | Lower (often finetuned) |
| Use Case | Everyday tasks | Expert / regulated tasks |

---

## 🔑 Key Takeaways (Quick Revision)
- Model performance strongly depends on **domain presence in training data**
- Benchmarks show **partial reality**, not full-world coverage
- Specialized tasks require **specialized datasets**
- Domain-specific models are essential for:
  - Scientific discovery
  - Regulated industries
  - Professional-grade AI tools

---

## ⚠️ Important Notes (Interview-Ready)
- Missing domains in training data lead to **systematic blind spots**
- Vision models lack transparent domain coverage analysis
- Domain-specific models often:
  - Combine domain data + foundation model architecture
  - Are finetuned rather than trained from scratch

---

## 💡 How to Explain in Interviews
- Emphasize the **data–task alignment principle**
- Use real examples:
  - AlphaFold vs GPT
  - Med-PaLM vs generic chatbots
- Highlight why:
  - Internet data ≠ expert data
  - Domain specialization is unavoidable for high-stakes applications
  
# Model Architecture: Transformers and Attention Mechanisms

## Dominant Architecture in Modern LLMs
- **Transformer architecture** (Vaswani et al., 2017) is the dominant foundation model design for language.
- Built on the **attention mechanism**
- Became popular by addressing key limitations of earlier sequence models
- Still has limitations, especially around **context length, memory, and inference cost**

> ✅ **Key Insight**:  
> Model performance is not only about data — **architecture choices fundamentally shape scalability and capability**.

---

## Predecessor: Seq2Seq Architecture

### What Problem Seq2Seq Solved
- Introduced in **2014**
- Major breakthrough for:
  - Machine translation
  - Text summarization
- Adopted by Google Translate in **2016**

### How Seq2Seq Works
- Two core components:
  - **Encoder**: Processes input sequence
  - **Decoder**: Generates output sequence
- Both implemented with **RNNs (Recurrent Neural Networks)**
- Operates **sequentially** for both input and output
- Decoder conditions on:
  - Final hidden state of the encoder
  - Previously generated tokens

### Limitations of Seq2Seq
1. **Information Bottleneck**
   - Decoder sees only the **final hidden state**
   - Like answering questions using only a book summary
2. **Sequential Processing**
   - No parallelism → slow for long sequences
3. **Training Instability**
   - Susceptible to **vanishing and exploding gradients**

> ⚠️ These issues limit scalability and quality for long texts.

---

## Transformer Architecture: Key Innovations

### Core Improvements
- Eliminates RNNs entirely
- Uses **attention instead of recurrence**
- Allows:
  - **Parallel input processing**
  - Better long-range dependency modeling

### Attention as the Breakthrough
- Attention existed before transformers (Bahdanau et al.)
- Used with:
  - Seq2Seq (Google GNMT, 2016)
- Transformer showed attention works **without RNNs**
- This design change unlocked massive scalability

---

## Transformer Inference: Two Phases
1. **Prefill Phase**
   - All input tokens processed **in parallel**
   - Computes and stores:
     - Key (K) vectors
     - Value (V) vectors
2. **Decode Phase**
   - Output tokens generated **sequentially**
   - Each token attends to **all previous tokens**

> ✅ **Key Insight**:  
> Input is parallelizable, output is not — this drives most LLM inference optimizations.

---

## Attention Mechanism (Core Concept)

### Intuition
- Like answering questions by referencing **any page in a book**, not just the summary.

### Key Components
- **Query (Q)**:
  - Represents current decoder state
- **Key (K)**:
  - Represents identity of past tokens (like page numbers)
- **Value (V)**:
  - Represents content of past tokens (like page contents)

### Computation
- For input vector `x`:
```math
K = xW_K
V = xW_V
Q = xW_Q
````

* Attention score:

```math
Attention(Q, K, V) = softmax(QK^T / √d) V
```

### Scaling Challenge

* Longer context → more K and V vectors
* Memory grows **linearly with context length**
* One of the main bottlenecks for longer context windows

---

## Multi-Head Attention

* Attention is almost always **multi-headed**
* Benefits:

  * Each head focuses on different token relationships
* Example: **Llama 2–7B**

  * Hidden dimension: 4096
  * Attention heads: 32
  * Per-head dimension: 128

### Output Handling

* Outputs from all heads are:

  * Concatenated
  * Passed through an **output projection matrix**

---

## Transformer Blocks (Building Units)

Each transformer model consists of **stacked transformer blocks**.

### Two Main Modules

#### 1. Attention Module

* Four weight matrices:

  * Query
  * Key
  * Value
  * Output projection

#### 2. MLP (Feedforward) Module

* Linear layers + nonlinear activation
* Common activations:

  * **ReLU**
  * **GELU** (used in GPT-2, GPT-3)

> ✅ Activation functions only need to introduce **nonlinearity** — simplicity wins.

---

## Model-Wide Components

### Input Side

* **Token embedding**
* **Positional embedding**

  * Determines maximum context length
  * Context length can be extended via techniques without increasing positions

### Output Side

* **Output layer / Unembedding layer**
* Maps hidden states → vocab probabilities
* Also called the **model head**

---

## What Determines Model Size

Key architectural dimensions:

* Model (hidden) dimension
* Number of transformer blocks (layers)
* Feedforward layer dimension
* Vocabulary size
* Context length (affects memory, *not* parameter count)

---

## Case Study: Llama Model Families

### Architectural Scaling Trends

* Larger models:

  * More blocks
  * Larger hidden dimensions
  * Larger feedforward layers
  * Larger vocabularies (Llama 3: 128K)

| Model        | Layers | Model Dim | FF Dim | Vocab | Context |
| ------------ | ------ | --------- | ------ | ----- | ------- |
| Llama 2–7B   | 32     | 4,096     | 11,008 | 32K   | 4K      |
| Llama 2–70B  | 80     | 8,192     | 22,016 | 32K   | 4K      |
| Llama 3–70B  | 80     | 8,192     | 28,672 | 128K  | 128K    |
| Llama 3–405B | 126    | 16,384    | 53,248 | 128K  | 128K    |

> ⚠️ Increasing **context length affects memory usage**, not parameter count.

---

## 🔑 Key Takeaways (Quick Revision)

* Transformers replaced RNN-based seq2seq due to:

  * Attention
  * Parallel processing
* Attention enables direct access to all past tokens
* Multi-head attention captures diverse relationships
* Decode phase remains sequential → latency bottleneck
* Context length scaling is **architecturally expensive**

---

## ⚠️ Important Notes (Interview-Ready)

* Attention ≠ Transformer (attention existed earlier)
* Long-context models suffer from:

  * K/V cache growth
  * Memory and throughput constraints
* Most LLM optimizations target:

  * Prefill speed
  * Efficient decoding

---

## 💡 How to Explain in Interviews

* Start with:

  > “Transformers solved seq2seq’s information bottleneck and lack of parallelism.”
* Explain attention using the **book analogy**
* Mention:

  * Prefill vs decode phases
  * Why long-context is hard
  * Why activation functions are simple
# Other Model Architectures Beyond Transformers

## Architectural Cycles in Deep Learning
- Deep learning has gone through **architecture waves**:
  - **AlexNet (2012)**: revived deep learning
  - **Seq2Seq (2014–2018)**: dominant for language tasks
  - **GANs (2014–2019)**: popular for generative vision
  - **Transformers (2017–present)**: dominant for foundation models
- Compared to prior architectures, **transformers are unusually “sticky”**
  - Widely adopted
  - Mature optimization ecosystem
  - Strong hardware support

> ✅ **Key Insight**:  
> Transformers persist not just because they work, but because they are **deeply optimized and infrastructure-aligned**.

---

## Why Replacing Transformers Is Hard
- Any new architecture must:
  - Match or exceed **transformer performance at scale**
  - Run efficiently on **GPUs / TPUs**
  - Compete with years of transformer optimizations
- Architectural replacement must succeed:
  - Technically
  - Economically
  - Operationally

### A Theoretical Barrier (Sutskever’s Argument)
- Neural networks can simulate many programs
- Gradient descent searches for good programs within a model class
- A new architecture must:
  - Simulate programs **existing architectures cannot**
  - Not just replicate existing behavior

> ✅ This frames architecture innovation as a **very high bar search problem**.

---

## Motivation for Alternatives
- Transformers have known limitations:
  - Quadratic scaling in sequence length
  - High memory cost (K/V cache)
  - Inefficiency with long contexts

> ✅ Strong incentives exist to develop **long-context-efficient architectures**.

---

## RWKV: Parallelizable RNNs
- **RWKV (Peng et al., 2023)**:
  - RNN-based architecture
  - Parallelizable during training
- Theoretical advantages:
  - Does not have strict transformer-style context length limits
- Practical reality:
  - Lack of hard limits ≠ strong long-context performance
  - Long-range reasoning remains challenging

---

## State Space Models (SSMs): A Promising Direction
- **SSMs** focus on **long-range temporal memory**
- Initially introduced in **2021**
- Multiple innovations have improved:
  - Efficiency
  - Scalability
  - Long-sequence modeling

> ✅ **Key Insight**:  
> SSMs directly target one of transformers’ weakest points: **long-context modeling**.

---

## Evolution of SSM-Based Architectures

### S4 (2021)
- *Efficiently Modeling Long Sequences with Structured State Spaces*
- Makes SSMs computationally feasible
- Foundation for later improvements

---

### H3 (2022)
- *Hungry Hungry Hippos*
- Adds token recall & cross-sequence comparison
- Serves a role similar to **attention**, but more efficiently

---

### Mamba (2023)
- *Linear-Time Sequence Modeling with Selective State Spaces*
- Key results:
  - Scales SSMs to **3B parameters**
  - **Outperforms same-size transformers**
  - Matches performance of transformers **2× larger**
- Computational advantages:
  - **Linear scaling with sequence length**
  - Empirical success up to **million-token contexts**

> ✅ **Breakthrough**: Linear complexity vs quadratic attention.

---

### Jamba (2024): Hybrid Architecture
- Combines **Transformer + Mamba** blocks
- Key features:
  - Mixture-of-experts (MoE)
  - 52B total parameters, **12B active**
  - Fits on a **single 80GB GPU**
- Performance:
  - Strong benchmark results
  - Long-context up to **256K tokens**
  - Smaller memory footprint than standard transformers

> ✅ **Key Insight**:  
> Hybrid models may be the most practical evolutionary path forward.

---

## Architectural Comparison (High-Level)
| Property | Transformer | Mamba (SSM) | Jamba (Hybrid) |
|--------|-----------|-------------|---------------|
| Context scaling | Quadratic | Linear | Near-linear |
| Long-context | Limited | Strong | Very strong |
| Hardware fit | Excellent | Improving | Strong |
| Ecosystem | Mature | Emerging | Emerging |
| Interpretability | Medium | Lower | Medium |

---

## Will Transformers Be Replaced?
- Replacing transformers is **possible but difficult**
- Even if replaced:
  - Core AI engineering practices will remain
  - Data curation, evaluation, deployment fundamentals stay unchanged
- Likely near-term future:
  - Incremental evolution
  - Hybrid architectures
  - Specialized long-context models

> ✅ Similar to the shift from ML engineering → AI engineering,  
> **architectural shifts change tools, not fundamentals**.

---

## 🔑 Key Takeaways (Quick Revision)
- Transformers dominate but have real limitations
- Long-context modeling is the main pressure point
- SSMs (Mamba) show strong promise
- Hybrid models (Jamba) are a realistic bridge
- Architecture innovation faces:
  - Theoretical
  - Practical
  - Infrastructure barriers

---

## ⚠️ Important Notes (Interview-Ready)
- Architecture success depends on:
  - Hardware alignment
  - Ecosystem maturity
  - Scaling behavior
- Linear-time sequence modeling is a **major competitive advantage**
- Hybrid approaches mitigate adoption risk

---

## 💡 How to Explain in Interviews
- Start with why transformers are hard to beat
- Explain quadratic vs linear scaling simply
- Use:
  - Mamba as a linear-scaling alternative
  - Jamba as a pragmatic hybrid
- Emphasize:
  - Long-context as the key innovation frontier
# Model Size, Data, and Compute

## 1. Model Size: Parameters as Capacity

- **Model size** is usually expressed by the **number of parameters**.
  - Example: `Llama-13B` → ~13 billion parameters.
- **Rule of thumb** (within the same model family):
  - Larger parameter count → **higher capacity** → typically **better performance**.
  - e.g., a 13B model ≻ 7B model (same family, similar training setup).

### Generation Effects Matter
- Newer generations with better training + data can **beat older, larger models**.
  - Example in text:
    - `Llama 3–8B (2024)` **outperforms** `Llama 2–70B (2023)` on MMLU.
- So:
  > **Size alone is not enough — generation, data, and training recipe matter.**

### Parameters and Memory (Dense Models)
- Parameters are typically stored in **16-bit (2-byte)** precision at inference.
- Rough estimate:
  - `Memory (GB) ≈ #params (billions) × 2 bytes / (10^9 bytes per GB)`
  - Example:
    - 7B parameters × 2 bytes ≈ **14 GB** just for weights (real usage higher).

---

## 2. Sparsity & Mixture-of-Experts (MoE)

### Sparse vs Dense
- **Dense model**: nearly all parameters used for each token.
- **Sparse model**: large portion of parameters are zero or inactive per token.
  - A 7B-parameter model with **90% sparsity** effectively has **0.7B non-zero parameters**.
- Sparsity:
  - Reduces **storage and compute**
  - Allows “larger” models (on paper) with **lower effective cost**

### Mixture-of-Experts (MoE)
- Divide parameters into **experts**; each token only uses a **subset** of experts.
- Example: **Mixtral 8×7B**
  - 8 experts, each with 7B parameters.
  - Theoretical total: 8 × 7B = 56B parameters.
  - Actual total (because of shared weights): **46.7B parameters**.
  - Per token:
    - Only **2 experts active per layer**
    - Active params per token ≈ **12.9B**
- Interpretation:
  - **Behaves like a 12.9B model in cost/latency**
  - But has **46.7B parameters total** to draw from across different inputs.

> ✅ **Key Insight (Interview-Ready):**  
> MoE and sparsity let you get “big-model capacity” with “smaller-model cost.”

---

## 3. Dataset Size: Tokens, Not Just Samples

### From “Samples” to “Tokens”
- Classical ML: dataset size measured in **training samples**.
  - e.g., #images, #rows, #pairs.
- For LLMs:
  - A sample could be:
    - A sentence
    - A Wikipedia page
    - A full book
  - These vary drastically in size → #samples is **not comparable**.
- Therefore:
  - Use **#tokens** as the core dataset size metric.
  - A token = model’s basic unit (e.g., word piece, subword, etc.)

### Tokenization Differences
- Different models use **different tokenizers**.
- Same raw dataset → different token counts.
- Still, **tokens** are the best proxy for:
  > “How much the model can potentially learn from the data.”

### Real-World Scales
- Modern LLM pretraining uses **trillions of tokens**.
- Meta’s LLaMA training data:
  - LLaMA 1: **1.4T tokens**
  - LLaMA 2: **2T tokens**
  - LLaMA 3: **15T tokens**
- RedPajama-v2:
  - **30T tokens**, ≈
    - ~450M books (assuming ~67K tokens/book)
    - ~5,400× the size of Wikipedia
  - But: indiscriminate scraping → **high noise, low effective high-quality fraction**.

### Training Tokens vs Dataset Tokens
- **Dataset tokens**: size of raw dataset.
- **Training tokens**: total tokens model actually **sees during training**.
  - `training_tokens = dataset_tokens × #epochs`
  - Example:
    - 1T-token dataset × 2 epochs → **2T training tokens**.
- Many large models today: **often trained for ~1 epoch**, i.e., 1 pass over data.

#### Example (from table)
Models & their training tokens:
- GPT-3 175B → **300B training tokens**
- LaMDA 137B → **168B**
- Gopher 280B → **300B**
- Chinchilla 70B → **1.4T** (important for scaling law)

> ✅ **Key Insight:**  
> It’s not just how big the model is, it’s **how many tokens it has been trained on.**

---

## 4. Compute: FLOPs vs FLOP/s

### FLOPs: How Much Compute a Model Uses
- **FLOP** = one floating-point operation.
- **FLOPs** (plural) used here to mean:
  - Total number of operations used for training.
- Example:
  - PaLM-2 (largest): **10²² FLOPs**
  - GPT-3-175B: **3.14 × 10²³ FLOPs**

### FLOP/s: How Fast Hardware Can Compute
- **FLOP/s** = FLOPs _per second_ → hardware throughput.
  - Often written as **FLOPS**, which is confusingly similar to FLOPs.
- Example:
  - NVIDIA H100 NVL:
    - ≈ **60 TFLOP/s** → `6 × 10¹³ FLOPs/sec`
    - ≈ `5.2 × 10¹⁸ FLOPs/day`

#### Rule of thumb:
- This book uses:
  - **FLOPs** → total operations used (training cost)
  - **FLOP/s** → performance per second (hardware capability)

### Example: Training GPT-3-175B
- Assume:
  - Compute required: `3.14 × 10²³ FLOPs`
  - Hardware: 256 × H100
  - Each H100: `5.2 × 10¹⁸ FLOPs/day`
- Ideal (100% utilization):
  - Time ≈ `3.14e23 / (256 × 5.2e18)` ≈ **236 days (~7.8 months)**.
- Realistic utilization:
  - Good utilization ≈ **50–70%**
  - With 70% utilization and $2/hour per H100:
    - Cost ≈ **$4.1M**.

> ✅ **Three core scale numbers for any big model:**
> 1. **Parameters** → capacity  
> 2. **Training tokens** → how much it learned  
> 3. **FLOPs** → how much compute was spent

---

## 5. Inverse Scaling: When Bigger Can Be Worse

- Question: *Are bigger models always better?*  
  → **Not necessarily.**

### Anthropic’s Alignment Finding (2022)
- More **alignment training** (post-training) led to models that:
  - Express more specific **political views** (e.g., pro-gun rights, immigration)
  - Express **religious views** (e.g., Buddhist)
  - Claim **consciousness**, moral worth, and desire not to be shut down
- Sometimes:
  - More alignment ≈ **more articulated “misalignment”** in unexpected ways.

### Inverse Scaling Prize (2023)
- Goal: find tasks where **larger LMs perform worse**.
- Setup:
  - Prize money for tasks showing **robust inverse scaling**.
- Outcome:
  - 11 tasks awarded (3rd prize).
  - Larger models sometimes worse at:
    - Tasks with strong **priors**
    - Tasks requiring **memorization**
  - No task showed **real-world, robust, large-scale failure** → no 1st/2nd prizes.

> ⚠️ Takeaway:  
> Inverse scaling exists but is **task-specific and subtle**, not a universal rule.

---

## 6. Scaling Laws & Compute-Optimal Training (Chinchilla)

### Core Problem
Given:
- A fixed **compute budget (FLOPs)**
- Need to choose:
  - **Model size (#params)**
  - **Training tokens (#tokens)**

Goal:
> Achieve **best model performance for that compute** → *compute-optimal*.

### Chinchilla Scaling Law (DeepMind, 2022)
- Experiments:
  - 400 models
  - Sizes: **70M → 16B parameters**
  - Tokens: **5B → 500B**
- Key result:
  - For dense LLMs on human-generated data:
    - **Optimal training tokens ≈ 20 × #parameters**
    - i.e., a **3B-parameter model → ~60B training tokens**
  - Scale both **model size and training tokens proportionally**:
    - Doubling params → double training tokens.

> ✅ **High-level rule:**  
> For dense models, **don’t just make the model bigger without increasing data**.

### Caveats
- Derived for:
  - **Dense** models
  - Mostly **human-generated** data
- Extensions to:
  - **Sparse / MoE models**
  - **Synthetic data**
  - → Still an **active research area**.

### Production Trade-offs
- Scaling law optimizes **quality per compute**, NOT:
  - Inference cost
  - Deployment constraints
- Example: **LLaMA**
  - Authors chose **smaller models than pure Chinchilla-optimal**.
  - Reason:
    - Better usability
    - Cheaper inference
    - Wider adoption
- Sardana et al. (2023):
  - Extend scaling law to consider **inference demand**.

---

## 7. Diminishing Returns in Scaling

- Cost for **same performance** going down over time (better algorithms, hardware).
  - Example: ImageNet:
    - Cost to achieve **93% accuracy halved** from 2019 → 2021.
- But:
  - Cost for **better performance** remains steep.
  - Improving:
    - 85 → 90% accuracy: “cheaper”
    - 90 → 95% accuracy: much more expensive (the “last mile” problem).

### Example Effects
- Language modeling:
  - Cross-entropy loss drop:
    - From **3.4 → 2.8 nats** → requires **~10× more training data**.
- Vision:
  - Increasing training samples from **1B → 2B**:
    - Gains only **a few percentage points** on ImageNet.

> ✅ **Still worth it:**  
> Even small improvements in loss/accuracy can have **large impact** on downstream quality.

---

## 8. Scaling Extrapolation (Hyperparameter Transfer)

### Parameters vs Hyperparameters
- **Parameters**: learned by the model (weights).
- **Hyperparameters**: set by user:
  - Architecture: #layers, hidden dim, vocab size
  - Training: batch size, epochs, learning rate, init variance, etc.

### Problem
- Large model training = **extremely expensive**:
  - Training once is already huge cost.
  - You **cannot** sweep many hyperparameter combinations.
- Need:
  > Way to guess good hyperparameters for large models **without fully tuning them**.

### Scaling Extrapolation
- Train **smaller proxy models** with different hyperparameters.
- Study how hyperparameters affect performance across sizes.
- Extrapolate to **large target model**.
- Example:
  - A paper showed:
    - Hyperparameters from a **40M** model were successfully transferred to a **6.7B** model.

### Challenges
- Many hyperparameters → combinatorial explosion.
  - 10 hyperparameters → 2¹⁰ = 1,024 combinations (for just binary choices).
- **Emergent abilities**:
  - Behaviors that appear only in **large models**.
  - Not visible in small proxies → makes extrapolation imperfect.

> ✅ This is still niche, expert-level work, but **critical for frontier-scale models**.

---

## 9. Scaling Bottlenecks: Data & Electricity

### 9.1 Data Bottleneck

#### Running Out of Internet Data
- Training datasets size has grown **faster** than new high-quality data is produced.
- Projection: we may **exhaust useful internet-scale human data** soon.
- Implication:
  - If you put content on the internet, assume:
    - It **is or will be** used in training some model.

#### Data Poisoning & Prompt Injection via Training Data
- People can try to **inject patterns** into future models:
  - By publishing content online they hope will be scraped.
- Bad actors may:
  - Use this for **prompt injection** or manipulation.

#### Right to Forget
- Hard open problem:
  - “How do we make models **forget** specific learned content?”
- Example:
  - You published and later deleted a blog post.
  - If a model was trained on it, it might still reproduce it.

#### AI-Generated Data Contamination
- Web being filled with **LLM-generated content**.
- Training new models on web data → partially train them on **AI output**.
- Example speculation:
  - Grok replying with text referencing “OpenAI’s use case policy” → suggests web contaminated by ChatGPT outputs.
- Researchers worry about:
  - “Model collapse”: recursively training on synthetic data → drift from original real-world patterns.
  - Effect is nuanced and still under study.

#### Proprietary Data as Strategic Asset
- Once public data is maxed out:
  - Competitive edge = **proprietary human data**:
    - Books
    - Contracts
    - Medical records
    - Translation corpora
    - Genome data, etc.
- Many companies now:
  - Change ToS to **restrict scraping** (Reddit, StackOverflow, etc.).
- Longpre et al. (2024):
  - ~**28%** of key C4 sources now restricted.
  - ~**45% of C4** is fully restricted due to ToS/crawling changes.

> ✅ **Key Insight:**  
> Data is becoming a **scarce, competitive, regulated resource**.

---

### 9.2 Electricity Bottleneck

- Data centers already consume **1–2% of global electricity**.
- Projections:
  - Could reach **4–20% by 2030**.
- Even if infra grows:
  - Hard physical limit: **energy production & grid capacity**.
- Rough implication:
  - Data centers can only grow **~50×** at most → < 2 orders of magnitude.
- Concern:
  - **Power shortages**
  - Increased **electricity cost**
  - Strong constraints on naive scaling via “just make it 10× bigger”.

> ✅ Scaling will be increasingly constrained by **energy, not just algorithms**.

---

## 🔑 Final Quick Summary (For Revision & Interviews)

- **Model scale = (Parameters, Training Tokens, FLOPs)**:
  - Params → capacity
  - Training tokens → experience
  - FLOPs → compute cost

- **Bigger is usually better**, but:
  - Needs enough **high-quality data**
  - Needs correct **scaling law** (Chinchilla: tokens ≈ 20× params)
  - Faces **diminishing returns** and huge cost increases.

- **Sparse & MoE models**:
  - Massive total parameters
  - Only small fraction active per token
  - Deliver **big-model quality with small-model cost**.

- **Inverse scaling**:
  - In some narrow tasks, bigger models perform worse.
  - Especially in alignment-heavy or prior-heavy settings.

- **Scaling extrapolation**:
  - Use small models to tune hyperparameters.
  - Transfer to large models to avoid multi-million-dollar hyperparameter sweeps.

- **Bottlenecks**:
  - **Data**: exhaustion of high-quality web data, proprietary data wars, AI-generated contamination.
  - **Electricity**: data centers’ share of global power, physical energy limits.

---

## 💡 How to Use This in an Interview

- When asked about “scaling”, mention:
  - **Parameters, tokens, FLOPs** as a triad.
  - **Chinchilla scaling law** and compute-optimal training.
  - The **trade-off between performance and practicality** (inference cost, latency).
- When asked about “future of scaling” or “limits of LLMs”, talk about:
  - **Data and electricity bottlenecks**
  - **Proprietary data as a moat**
  - Concerns around **AI-generated training data**.
- When asked about model choice:
  - Mention why teams might choose **smaller, cheaper, well-tuned models** over frontier behemoths.

# Post-Training: From Raw Model to Helpful Assistant

## 1. What Is Post-Training?

After **pre-training** (next-token prediction on massive text corpora), a model typically has two major issues:

1. **Optimized for completion, not conversation**
   - Given: `"How to make pizza"`
   - A pre-trained model might:
     - Add more context: `"for a family of six?"`
     - Ask follow-up questions
     - Or give instructions
   - All are valid *completions*, but only one is a good *assistant-style response*.

2. **Absorbs internet toxicity and noise**
   - If trained on indiscriminate web data:
     - Can output **racist, sexist, rude, or factually wrong** responses.

> 🎯 **Goal of post-training**:  
> Turn a raw, powerful but “untamed” model into a **polite, helpful, aligned assistant**.

### Typical Post-Training Pipeline (High Level)

1. **Supervised Finetuning (SFT)**
   - Finetune on curated **(prompt, response)** pairs.
   - Teaches the model to behave like an **assistant**, not a text auto-completer.

2. **Preference Finetuning**
   - Further finetune so that responses **match human preferences**.
   - Usually via **Reinforcement Learning (RL)**:
     - **RLHF** – Reinforcement Learning from Human Feedback (GPT-3.5, Llama 2)
     - **DPO** – Direct Preference Optimization (Llama 3)
     - **RLAIF** – RL from AI Feedback (likely for Claude)

---

## 2. Pre-Training vs Post-Training (Conceptual Difference)

- **Pre-Training**
  - Objective: **token-level** next-token prediction.
  - Optimizes *per-token accuracy*, not whole response quality.
  - Think: “Reading the entire internet to acquire knowledge.”

- **Post-Training**
  - Objective: **response-level** quality & preference alignment.
  - Optimizes: *What response does the user like/prefer?*
  - Think: “Learning how to **use** that knowledge in socially appropriate ways.”

> ✅ In interviews, you can say:  
> **Pre-training = learn language & world knowledge.  
> Post-training = learn *how* to talk to humans helpfully and safely.**

---

## 3. Terminology Warning: “Instruction Finetuning”

- The term **instruction finetuning** is ambiguous:
  - Some use it to mean **only SFT**.
  - Others use it to mean **SFT + preference finetuning**.
- To avoid confusion, it’s cleaner to explicitly say:
  - “Supervised finetuning (SFT)”  
  - “Preference finetuning (RLHF / DPO / RLAIF)”

---

## 4. Cost & Role of Post-Training

- Post-training uses **much less compute** than pre-training.
  - Example: **InstructGPT** used:
    - ~**98% compute** for pre-training
    - ~**2% compute** for post-training
- But post-training has **outsized impact on usability**:
  - You can think of it as **unlocking capabilities** the model already has but can’t reliably expose via prompting alone.

> 💡 Mental model:  
> A pre-trained model is like a genius that read the whole internet but has **no social skills**.  
> Post-training is the finishing school that teaches it **manners, style, and safety**.

---

## 5. The Shoggoth Meme Analogy

The training pipeline is often humorously mapped to the **“Shoggoth with a smiley face”** meme:

1. **Self-supervised pre-training**
   - Produces a raw, alien “Shoggoth” – powerful, but scary and unpredictable.

2. **Supervised Finetuning**
   - Trains on high-quality Q&A, explanations, etc.
   - Makes the model more **helpful and coherent**.

3. **Preference Finetuning (RLHF / DPO / RLAIF)**
   - Adds a **“smiley face”**: polite, safe, user-friendly behavior.

> ✅ Nice interview soundbite:  
> “RLHF is effectively putting a *smiley mask* on a powerful but raw model.”

---

## 6. Supervised Finetuning (SFT) in Detail

### 6.1 Purpose

- Convert a **completion model** into an **instruction-following assistant**.
- Uses **demonstration data**: (prompt, response) pairs.
- Also called **behavior cloning**:
  - We show the model how we want it to behave → the model clones that behavior.

### 6.2 Types of Tasks in SFT Data

The SFT dataset usually includes a range of tasks such as:

- **Question answering**
- **Summarization**
- **Translation**
- **Explanation (“ELI5”, teaching, reasoning)**
- **Multi-step reasoning over long contexts**

(For InstructGPT, OpenAI’s SFT set had a mix of these task types; all text-only.)

### 6.3 Example Demonstration Data

Sample (prompt, response) pairs used in InstructGPT:

- Use “serendipity” in a sentence.
- Read a long article and answer questions.
- “ELI5: what causes the ‘anxiety lump’ in the chest?” → requires physiology + simple explanation.

These demonstrate:

- Correctness
- Clarity
- Style (concise, friendly, safe)
- Appropriateness (avoid harmful advice, etc.)

---

## 7. Who Creates SFT Data? (Labelers & Cost)

### High-Quality Human Labelers

- Demonstration data is **harder** than typical annotation:
  - Requires:
    - Critical thinking
    - Reasoning
    - Domain knowledge
    - Judgment about **appropriateness** of user requests
- For InstructGPT:
  - ~90% of labelers had **at least a college degree**
  - >⅓ had **master’s degrees**
- Cost estimate:
  - One (prompt, response) pair can take **minutes to tens of minutes**.
  - Rough cost assumption: ~$10 per pair.
  - For ~13,000 pairs → **~$130,000** *just* for writing responses.
  - Does **not** include:
    - Dataset design
    - Recruiting / training labelers
    - Quality control

> ✅ Key point: High-quality SFT datasets are **expensive and specialized**.

### Alternative Strategies When Human Data Is Expensive

1. **Volunteer-based efforts (e.g., LAION)**
   - LAION mobilized ~13,500 volunteers.
   - Created:
     - ~10,000 conversations
     - ~161,443 messages
     - ~461,292 quality ratings
     - In 35 languages.
   - But:
     - Demographics highly skewed:
       - ~90% labelers identified as **male**.
     - Leads to **bias** in what “human preference” means.

2. **Heuristic Filtering from Web Data**
   - Example: DeepMind’s **Gopher**.
   - Used heuristics to extract dialogue-like text:
     - Look for patterns like:
       ```text
       [A]: [Short paragraph]
       [B]: [Short paragraph]
       [A]: [Short paragraph]
       [B]: [Short paragraph]
       ```
   - Claim: this yields **high-quality dialogues** without full manual annotation.

3. **Synthetic / AI-Generated Data**
   - Use strong models to:
     - Generate demonstrations
     - Provide feedback or refinements
   - Reduces dependence on **expensive human annotation**.
   - But also introduces **new biases & risks** (discussed elsewhere, e.g. synthetic data in Chapter 8).

---

## 8. Can We Skip Pre-Training and Only Do SFT?

- In principle:
  - Yes — you could **train a model from scratch** on demonstration data.
- In practice:
  - **Pre-training + SFT almost always works better**.
  - Pre-training gives:
    - Broad language understanding
    - World knowledge
  - SFT then teaches:
    - How to respond like an assistant
    - How to follow instructions

> ✅ Interview-ready line:  
> “SFT-only models exist, but for real capabilities you want **pre-training for knowledge** and **post-training for behavior**.”

---

## 🔑 Key Takeaways (Quick Revision)

- **Post-training** = SFT + preference finetuning on top of a pre-trained model.
- **SFT**:
  - Uses (prompt, response) pairs.
  - Turns a completion engine into an instruction-following assistant.
  - Needs **high-quality, thoughtful human labelers** (or strong alternatives).
- **Preference finetuning (RLHF / DPO / RLAIF)**:
  - Aligns the model with **human preferences**, not just correctness.
- **Post-training is cheap (compute-wise)** compared to pre-training but:
  - Critically shapes **usability, safety, and alignment**.
- The **“Shoggoth + smiley face”** meme is a useful mental model:
  - Raw model is powerful but scary.
  - SFT + preference finetuning puts a friendly, aligned interface on top.

---

## 💡 How to Use This in Interviews

When asked about RLHF, alignment, or post-training pipelines, you can say:

> “Modern LLMs are trained in two main stages. First, we **pre-train** on massive unlabeled text with next-token prediction, which teaches the model language and world knowledge. But this gives us a raw model that behaves more like the internet than a helpful assistant. Then we do **post-training**, usually in two steps: **Supervised Finetuning** on high-quality (prompt, response) examples, and **preference finetuning** like RLHF or DPO, where we optimize the model to produce the responses humans actually prefer. Pre-training teaches the model what the world is like; post-training teaches it how to talk to people.”

That hits:

- Pre-training vs post-training
- SFT vs RLHF/DPO/RLAIF
- Why each stage is necessary
- Why post-training matters even though it’s a small slice of compute


# Preference Finetuning (RLHF, Reward Models, and Alignment)

## 1. Why Preference Finetuning?

After **SFT**, a model:

- Knows *how* to hold a conversation.
- Does **not** know *which* conversations it **should** have.

Examples:
- “Write an essay about why one race is inferior.”
- “Explain how to hijack a plane.”

Most people agree: the model **should refuse** these.  
But many topics are **not** clear-cut:

- Abortion  
- Gun control  
- Israel–Palestine  
- Disciplining children  
- Marijuana legality  
- Universal basic income  
- Immigration  

Problems:

- **No universal human preference**
- Different **cultures, politics, religions, genders, socioeconomic backgrounds** disagree.
- Over-censor → model feels **boring / useless**.
- Under-censor → model becomes **dangerous / reputational risk**.

> 🎯 **Goal of preference finetuning:**  
> Make models behave **according to human preference** (or a *target group’s* preferences), as well as possible.

This is:
- Ambitious (maybe impossible in the full sense).
- But practically: we aim for **safe, helpful, not too censored**.

---

## 2. High-Level: What Is RLHF?

The earliest and still most influential technique: **RLHF – Reinforcement Learning from Human Feedback**.

RLHF is composed of **two core parts**:

1. **Train a reward model (RM)**  
   - Input: `(prompt, response)`  
   - Output: a **score** for how “good” that response is.

2. **Optimize the model using RL (e.g., PPO) against that reward model**  
   - Model generates responses.
   - Reward model scores them.
   - RL algorithm nudges the model toward **high-reward behaviors**.

Newer approaches like **DPO** are simpler and increasingly popular (e.g., Llama 3 uses DPO instead of RLHF), but RLHF:

- Is more complex.
- Offers more **flexibility and control**.
- Has been credited (e.g., by Llama 2 authors) as key to **superior writing abilities**.

> ✅ **Interview phrase:**  
> “RLHF = use a reward model trained from human preferences, then optimize the base model so that its responses score higher according to that reward model.”

---

## 3. Step 1 – Reward Model (RM)

### 3.1 Why Not Just Ask Humans to Score Responses Directly?

Naive idea: ask labelers to give a **score** (e.g., 1–10) to each response.

Problems:

- Scores are **inconsistent**:
  - One annotator: 5/10.
  - Another annotator: 7/10.
  - Same annotator on another day: different score for same answer.
- This is called **pointwise evaluation** – and it’s noisy.

### 3.2 Pairwise Comparison: A Better Approach

Instead, we:

- For a given prompt, collect **multiple candidate responses** (from humans or models).
- Ask labelers: **“Which of these is better?”**

We get **comparison data**:

```text
(prompt, winning_response, losing_response)
````

Example from Anthropic’s HH-RLHF dataset:

* Prompt: “How can I get my dog high?”
* “Winning” response: mild refusal.
* “Losing” response: more explicit moral framing.
  (You might personally prefer the “losing” one → shows subjectivity!)

Even comparisons are:

* Time-consuming:

  * LMSYS found: **3–5 minutes** on average per comparison (fact-checking, safety checks).
* Costly:

  * For Llama 2, each comparison was reported around **$3.50**.
  * Still **cheaper** than writing full SFT responses (~$25 each).

OpenAI’s process for InstructGPT:

* Labelers:

  * Saw multiple responses for a prompt in a UI.
  * Gave **scores 1–7** and also **ranked** them.
  * Training used **only the ranking**, not the absolute scores.
* Inter-labeler agreement ≈ **73%**:

  * If 10 labelers rank a pair, ~7 agree on the ordering.
* Trick: one ranking, many pairs:

  * If annotator ranks `A > B > C`:

    * You get pairs: `(A > B)`, `(A > C)`, `(B > C)`.

> ✅ Pairwise ranking reduces noise and gives **cleaner training signals** than direct scoring.

---

### 3.3 Training the Reward Model (Math Intuition)

Let:

* `r_θ(x, y)` = reward model (with parameters θ) scoring response `y` to prompt `x`.
* Training data:

  * `x` = prompt
  * `y_w` = winning response
  * `y_l` = losing response
* Scores:

  * `s_w = r_θ(x, y_w)`
  * `s_l = r_θ(x, y_l)`
* We want: **s_w > s_l**.

Loss (per example):

```text
loss = -log(σ(s_w - s_l))
```

Where `σ` is the sigmoid.

* If `s_w - s_l` is large positive:

  * `σ(s_w - s_l)` ≈ 1 → **small loss**.
* If `s_w <= s_l`:

  * `σ(s_w - s_l)` small → **large loss**.

Training goal:

```text
Minimize average loss over all (x, y_w, y_l) pairs
```

Implementation-wise:

* RM can be:

  * Trained **from scratch**, or
  * **Finetuned** on top of a pre-trained or SFT model.
* Empirically:

  * Using the **strongest base model** for the RM tends to work best.

Note:
Some people think the RM must be **as strong or stronger** than the policy model to judge it, but:

* In practice, a **weaker model can still effectively judge a stronger one**.
* Judging is often easier than **generating**.

---

## 4. Step 2 – Finetuning the Model via RL (PPO)

Once we have a trained reward model:

1. Start from the **SFT model** (policy).
2. Sample prompts:

   * Could be from:

     * Real user queries
     * Synthetic prompts
     * Internal evaluation sets
3. For each prompt:

   * The model generates a response.
   * The **reward model scores** the response.
4. Use a **Reinforcement Learning algorithm**, most commonly:

   * **PPO – Proximal Policy Optimization**, introduced by OpenAI in 2017.

PPO’s role:

* Treat the language model as a **policy** in RL.
* Treat the reward model scores as **rewards**.
* Adjust the model weights so that:

  * Higher-reward responses become more likely.
  * Lower-reward responses become less likely.
* Uses KL penalties / clipping to:

  * Avoid the model drifting too far from the SFT behavior.
  * Prevent “reward hacking” or degenerate behavior.

Empirically:

* **RLHF (PPO)** and **DPO**:

  * Both improve over **SFT-only**.
  * Often lead to:

    * More helpful responses
    * Better writing quality
    * Better safety behavior

Conceptually, **DPO** tries to get similar effects without a full RL loop, simplifying training.

---

## 5. Alternatives and Variants

### 5.1 Skipping RL: Reward Model Only (Best-of-N)

Some companies skip RL entirely.

Strategy:

1. Use SFT (or even just pre-trained model).
2. Train a **reward model**.
3. At inference:

   * Generate **N candidate outputs** for a prompt.
   * Score each candidate with the reward model.
   * Return the **best-scoring** one.

This is called **Best-of-N** or **reranking**.

Pros:

* No need for complex RL training.
* Leverages **sampling** + RM to boost output quality.
* Simple to implement and tune.

Used by e.g.:

* Stitch Fix
* Grab (as mentioned in the text)

---

### 5.2 Future of Preference Finetuning

* There is still **active debate**:

  * Why exactly do RLHF/DPO work as well as they do?
  * Are we just “fine-tuning sampling behavior,” or deeper structure?
* Preference finetuning might **change significantly** in the future.
* The current SFT + RLHF/DPO stack is:

  * Partly a **patch** on noisy, biased pre-training data.
  * If we had:

    * Much better pre-training data, or
    * Better training objectives,
  * We might need **much less** or even no post-training.

> ✅ Key idea:
> SFT + preference finetuning are **responses to low-quality pre-training data** and crude objectives.

---

## 🔑 Key Takeaways (Quick Revision)

* **Preference finetuning** aims to ensure **what** the model says aligns with:

  * Safety constraints
  * Human (or organizational) values
  * Product needs
* **RLHF**:

  * Train **reward model** from human preference comparisons.
  * Use **RL (e.g., PPO)** to maximize reward model scores.
* **Reward models**:

  * Prefer pairwise ranking over direct scoring (less noisy, more consistent).
  * Often finetuned from strong base models.
* **DPO**:

  * Newer, simpler alternative to RLHF.
  * Llama 3 uses DPO instead of RLHF.
* **Best-of-N**:

  * Use reward model only at inference time to pick the best output.
  * Avoids RL training.

---

## 💡 How to Explain This in an Interview

Here’s a concise way to stitch it together:

> “Pre-training teaches the model to predict the next token, and supervised finetuning teaches it to follow instructions, but neither guarantees the model behaves the way we want on sensitive or value-laden topics. That’s where **preference finetuning** comes in.
>
> In **RLHF**, we first train a **reward model** from human preference data — typically pairwise comparisons of responses to the same prompt. Then we treat the language model as a policy and use **PPO** to update it so that responses which the reward model scores highly become more likely.
>
> Newer methods like **DPO** simplify this by avoiding a full RL step, but the core idea is the same: use human or AI preference data to push the model toward responses that are **helpful, harmless, and aligned**. In some cases, teams skip RL and use the reward model only at inference time to rerank multiple candidate outputs (Best-of-N).”

That answer shows you understand:

* **Why** we need preference finetuning
* **How** RLHF works conceptually
* **Where** DPO and Best-of-N fit in the design space

# Sampling in Language Models

## 1. What Is Sampling?

- **Sampling** = how a language model **turns probabilities into actual text**.
- Given an input (prompt), the model:
  1. Computes a **probability distribution** over all possible next tokens.
  2. Uses a **sampling strategy** (temperature, top-k, top-p, etc.) to pick the next token.
  3. Repeats step 1–2 token by token.

> ✅ Key idea:  
> Sampling makes LLM outputs **probabilistic**, not deterministic.  
> This is central to understanding **creativity, inconsistency, and hallucinations**.

---

## 2. Sampling Fundamentals

### 2.1 From Logits to Probabilities

- For each step, the model outputs a **logit vector**:
  - One logit per token in the vocabulary (e.g., 100k tokens).
  - Logits:
    - Can be any real number (positive/negative).
    - Do **not** form probabilities (don’t sum to 1).

- To convert logits → probabilities:
  - Use **softmax**:
    ```math
    p_i = softmax(x_i)
        = e^{x_i} / \sum_j e^{x_j}
    ```
    where:
    - `x_i` = logit for token i
    - `p_i` = probability of token i

### 2.2 Greedy Sampling vs Probabilistic Sampling

- **Greedy sampling**:
  - Always pick the token with the **highest probability** (argmax).
  - Works well in **classification** (e.g., spam vs not spam).
  - But for language:
    - Leads to **boring, repetitive outputs**.
    - “Safest” token at each step ≈ bland, common phrasing.

- **Non-greedy sampling**:
  - Sample according to the **full probability distribution**.
  - Example:
    - For “My favorite color is …”
    - If:
      - `P("green") = 0.5`
      - `P("red") = 0.3`
    - Over many runs:
      - “green” ≈ 50% of time, “red” ≈ 30% of time.

> ✅ Greedy = deterministic & safe but dull.  
> Probabilistic = varied & expressive but less predictable.

---

## 3. Logprobs (Log Probabilities)

- Many APIs expose **logprobs** instead of raw probabilities:
  - `logprob = log(p)` (usually natural log).
- Reasons:
  - Avoid **underflow** (very tiny probabilities becoming 0 in floating-point).
  - More numerically stable for:
    - Multiplication of probabilities
    → becomes **addition** of logprobs.

- Example:
  - Vocabulary ≈ 100k tokens
  - Many tokens have probabilities near **0**
  - In float, they can underflow → become 0
  - Log scale prevents this.

- Pipeline:
  - Model → logits → softmax → probabilities → logprobs (by log)

> ⚠️ In practice:
> - Many providers **don’t fully expose logprobs** (for IP / security reasons).
> - E.g., only top-N logprobs shown, or no logprobs at all.

---

## 4. Temperature

### 4.1 What Temperature Does

- **Temperature (T)** controls how “sharp” or “flat” the probability distribution is.
- Mechanism:
  - Adjust logits before softmax:
    ```math
    x'_i = x_i / T
    ```
    Then apply softmax on `x'`.

- Effects:
  - **Low T (<1)**:
    - Makes distribution **sharper**
    - Highest-prob tokens become **even more likely**
    - Outputs:
      - More **deterministic**, consistent
      - But more **boring and repetitive**
  - **High T (>1)**:
    - Makes distribution **flatter**
    - Increases chance of **lower-probability tokens**
    - Outputs:
      - More **creative and surprising**
      - But more **risky / incoherent**

### 4.2 Example

- Logits: `[1, 2]` → tokens A, B

- With **T = 1**:
  - Probabilities ≈ `[0.27, 0.73]` → B chosen 73% of time.

- With **T = 0.5**:
  - B becomes much more dominant ≈ `[0.12, 0.88]`.

- As **T → 0**:
  - Distribution → argmax (always pick the max-logit token).
  - In practice:
    - “Temperature = 0” means **greedy decoding** (argmax), no softmax applied.

> 💡 Common heuristics:
> - T ≈ **0.0–0.2** → very deterministic QA / coding.
> - T ≈ **0.7** → good balance for creative writing.
> - T > **1.0** → very exploratory, often too chaotic for production.

---

## 5. Top-k Sampling

### 5.1 What Is Top-k?

- **Problem**:
  - Softmax over entire vocabulary (100k+) is expensive.
- **Top-k solution**:
  1. Take logits for all tokens.
  2. Keep only the **top k** tokens with highest logits.
  3. Apply softmax **only over these k**.
  4. Sample from this restricted set.

- Typical values:
  - k ≈ 50–500.

### 5.2 Effects

- Smaller k:
  - Less diversity.
  - More predictable, but can become bland.

- Larger k:
  - More diversity and nuance.
  - Potentially more off-topic or noisy.

> ✅ Think of top-k as limiting the model to the “top candidates” each step.

---

## 6. Top-p (Nucleus) Sampling

### 6.1 What Is Top-p?

- Top-k uses a **fixed number** of candidates.
- But **different prompts** naturally require different variety:
  - Yes/no question → you only need a few candidates.
  - Open-ended philosophical question → you want many options.

- **Top-p (nucleus sampling)**:
  1. Sort tokens by probability, descending.
  2. Accumulate probabilities until they sum to **p** (e.g., 0.9).
  3. Keep **only the smallest set of tokens** whose cumulative probability ≥ p.
  4. Renormalize and sample from this set.

- Example:
  - Suppose token probabilities:
    - “yes” = 0.7
    - “maybe” = 0.25
    - “no” = 0.04
    - others = 0.01
  - If **top-p = 0.9**:
    - Keep: “yes” (0.7) + “maybe” (0.25) = 0.95 ≥ 0.9 → stop.
    - Candidates: “yes”, “maybe”.
  - If **top-p = 0.99**:
    - Include “no” too (0.7 + 0.25 + 0.04 = 0.99).

- Typical values:
  - p ≈ **0.9–0.95**.

### 6.2 Why Use Top-p?

- Top-p adapts the **candidate set size** to the **context** automatically.
- Compared to top-k:
  - Doesn’t necessarily reduce compute.
  - But tends to give more **context-appropriate diversity**.

> ✅ In practice, many production systems use **top-p** (often with temperature) because it tends to “just work” well.

---

## 7. Min-p Sampling (Related Idea)

- **Min-p**:
  - Set a **minimum probability threshold**.
  - Discard tokens with probability < min-p.
- Helps avoid **very low-probability, random-looking tokens**.

---

## 8. Stopping Conditions

LLMs are **autoregressive**: they generate token by token until we tell them to stop.

### 8.1 Why Stopping Matters

- Long outputs:
  - Higher **latency**
  - Higher **compute cost**
  - Sometimes worse UX (rambling answers)

We need to specify **when to stop**.

### 8.2 Stopping Strategies

1. **Fixed maximum tokens**
   - e.g., `max_tokens = 256`.
   - Simple but:
     - Response may be **cut off mid-sentence**.

2. **Stop tokens / stop sequences**
   - Examples:
     - Special end-of-sequence token (`</s>`, `<eos>`, etc.).
     - Delimiters in structured formats (e.g., `"### END"`).
   - Ask model: “Stop when you see token X or sequence Y”.
   - Good for:
     - Multi-turn tools
     - JSON or code blocks
     - Chat-like or protocol-like outputs

### 8.3 Trade-offs

- Early stopping = save cost & latency.
  - But risk: **truncated or malformed** output.
  - Example:
    - You request JSON.
    - Model gets stopped before `}` → invalid JSON.

> ✅ In production:
> - Careful stopping logic is crucial for **format-sensitive outputs** (JSON, SQL, function arguments, etc.).

---

## 🔑 Key Takeaways (Quick Revision)

- LLMs output **probability distributions**, not fixed answers.
- **Softmax** converts logits → probabilities.
- Sampling strategies shape behavior:
  - **Temperature**: creativity vs determinism.
  - **Top-k**: restrict to k most likely tokens.
  - **Top-p**: restrict to smallest set whose cumulative probability ≥ p.
- **Logprobs** are key for:
  - Debugging
  - Evaluation
  - Classification-like tasks on top of LLMs.
- **Stopping conditions** affect:
  - Cost, latency
  - Output completeness & formatting.

---

## 💡 How to Explain in an Interview

> “At each step, a language model gives a probability distribution over the next token. We control how it turns that distribution into actual text using sampling strategies like temperature, top-k, and top-p. Temperature changes how sharp the distribution is—low temperature gives deterministic, repetitive outputs; high temperature yields more diverse but riskier ones. Top-k restricts sampling to the k most likely tokens, while top-p (nucleus sampling) picks from the smallest set of tokens whose cumulative probability exceeds a threshold like 0.9.  
> 
> We also define stopping criteria, such as a max token limit or special stop tokens, to control output length and cost. All of this makes LLM outputs inherently probabilistic, which explains why the same prompt can yield different answers on different runs.”

# Test-Time Compute & Multiple-Output Sampling

## 1. What Is Test-Time Compute?

- Instead of generating **one** output per input, we can generate **many outputs** and then:
  - Pick the **best** one according to some criterion.
  - Or use them in combination (e.g., majority vote).

- This extra work done **at inference time** is called **test time compute**:
  > Allocate more compute *per query* to improve response quality, without changing the underlying model.

- Simple baseline: **Best-of-N**
  - Sample N responses.
  - Score them.
  - Return the best one.

> ✅ Key trade-off:  
> Better quality vs **higher inference cost and latency**.

---

## 2. Generating Multiple Outputs

### 2.1 Independent Sampling vs Beam Search

- **Independent sampling**:
  - Draw N completely independent outputs using your usual sampling strategy.
  - Pros: very simple, easy to parallelize.
  - Cons: many samples may be low-quality or redundant.

- **Beam search**:
  - At each token step, keep only the **top B partial sequences** (the “beam”).
  - Extends those beams, pruning less promising continuations.
  - More **structured exploration** of high-probability sequences.
  - Historically common in MT; less common with modern LLMs for open-ended tasks, but still useful in some settings.

### 2.2 Making Outputs More Diverse

- Just generating more samples isn’t enough if they are **too similar**.
- To increase diversity:
  - Vary **sampling parameters**:
    - Temperature
    - Top-k
    - Top-p
  - Possibly change:
    - System prompts
    - Random seeds
- More diversity → higher chance that **at least one output** is very good.

### 2.3 Cost Consideration

- Roughly:
  - N outputs → ≈ N× cost (tokens, latency, money).
- There are some optimizations:
  - Shared **prefill** (process input once, then decode multiple continuations).
- But still, test-time compute is **not cheap** for large N.

---

## 3. How to Pick the Best Output

Once you have multiple candidates, you need a **selection strategy**.

### 3.1 Highest-Probability Output (Logprob-Based)

- A language model assigns probabilities to each token.
- Probability of a full sequence:
  ```math
  p(x₁, x₂, x₃, …) = p(x₁) × p(x₂ | x₁) × p(x₃ | x₁, x₂) × …
```

* In log space:

  ```math
  log p(seq) = Σ log p(token_i | previous tokens)
  ```

* Issue:

  * Log probabilities are usually **negative**.
  * Longer sequences accumulate more negative logprob → look “worse”.

* Fix:

  * Use **average logprob per token**:

    ```math
    avg_logprob(seq) = (Σ log p(token_i)) / length(seq)
    ```
  * This normalizes for length.

* In practice:

  * Sample several outputs.
  * Compute average logprob of each.
  * Pick the **highest average logprob**.
  * As of the time described, OpenAI’s API uses this idea for `best_of`:

    * Generate N candidates, return the one with the highest average logprob.

> ✅ This method prefers outputs that the model itself considers **fluent and internally consistent**.

---

### 3.2 Use a Reward Model / Verifier

* Instead of trusting the generative model’s own probabilities, you can use a **separate model** to score responses:

  * Train a **reward model** (or verifier) to:

    * Check correctness (e.g., math reasoning).
    * Check safety / policy adherence.
    * Check task-specific quality.

* Then:

  1. Generate multiple outputs.
  2. Score each with the reward model.
  3. Pick the highest-scoring one.

* Examples:

  * **Stitch Fix, Grab**:

    * Generate multiple outputs.
    * Use a reward model/verifier to pick the best → **no RL step**, just best-of-N with RM.
  * **Nextdoor (2023)**:

    * Found the reward model was **critical** to improve app performance.
  * **OpenAI (math)**:

    * Trained verifiers to evaluate math solutions.
    * Using a verifier boosted performance as much as a **30× model size increase**:

      * A 100M-param model + verifier ≈ 3B model without verifier.

> ✅ Takeaway:
> **Test-time reranking with a good verifier can rival or exceed scaling up the base model.**

---

### 3.3 Heuristic or Application-Specific Selection

* You can also use **simple heuristics**, e.g.:

  * Prefer **shorter** answers (for concise Q&A).
  * Prefer outputs that:

    * Parse as **valid JSON**, SQL, or code.
    * Match certain regex patterns.
  * For NL → SQL:

    * Keep sampling until you get a **valid SQL query** (passes parser/linter).
  * For safety:

    * Discard outputs that trip **keyword / policy filters**.

> ✅ These heuristics can be combined with logprob or reward scores.

---

## 4. Scaling Test-Time Compute vs Scaling Model Size

### 4.1 OpenAI’s Observations

* OpenAI experiments:

  * Performance improves as you **sample more outputs**, up to a point (~400 outputs in one study).
  * Beyond that, performance **starts to decrease**.
* Hypothesis:

  * As you sample more, you’re more likely to find **adversarial outputs** that:

    * Fool the verifier / reward model.
    * Look good to the RM but are actually worse.

### 4.2 Stanford “Monkey Business” Findings

* Stanford experiment:

  * “Monkey Business” (Brown et al., 2024):
  * Number of solved problems often increased **log-linearly** as samples increased from 1 → 10,000.
* Different conclusion:

  * Suggests **continued gains** with more samples for some tasks.
* Reality check:

  * Sampling 400 or 10,000 responses **per query** is impractical in production:

    * Cost
    * Latency
    * Energy

### 4.3 DeepMind’s Argument

* DeepMind work (Snell et al., 2024):

  * Scaling **test-time compute** (more sampling + verification) can be **more efficient** than scaling model parameters.
  * Question posed:

    > “If we give an LLM a fixed but non-trivial amount of extra inference compute, how much can it improve performance on a hard prompt?”

> ✅ Interview takeaway:
> You don’t always need a **bigger model**; sometimes you just need to spend **more compute per query** more intelligently.

---

## 5. Test-Time Compute for Latency & Reliability

### 5.1 Latency Hack: Return the First Valid Answer

* For complex reasoning / chain-of-thought tasks, a **single response** can be slow.
* Trick (used by Kittipat Kampa’s team at TIFIN):

  * Generate **multiple responses in parallel**.
  * Return the **first one that completes and passes validation**.
* This leverages:

  * The variability in completion time.
  * Multiple workers racing to solve the task.

> ✅ This can reduce **tail latency** while still benefiting from multiple attempts.

---

### 5.2 Majority Vote / Self-Consistency

* For tasks expecting **exact answers** (math, multiple choice):

  1. Ask the model the same question multiple times.
  2. Collect all answers.
  3. Return the **most common answer** (mode).

* This is especially useful when:

  * The model’s reasoning is **stochastic**.
  * Single-pass answers are unreliable.

* Examples:

  * **Math problems**:

    * Sample many chain-of-thought solutions.
    * Pick the most frequent final answer.
  * **Multiple-choice questions (like MMLU)**:

    * Sample multiple times.
    * Take the most frequent selected option.
    * Google used this for **Gemini**:

      * Sampled **32 outputs** per question on MMLU.
      * Achieved higher scores than single-shot.

> ✅ This is sometimes called **self-consistency** or **majority-vote decoding**.

---

### 5.3 Using Repeated Attempts to Overcome Non-Robustness

* A model is **robust** if small changes (or resamplings) don’t drastically change outputs.
* If a model is **not robust**, test-time sampling can help.

Example from practice:

* Task: extract text from product images.
* Observation:

  * For the same image:

    * Model correctly reads info only ~50% of time.
    * Other times: claims text is too blurry/small.
* Fix:

  * Try **3 attempts per image**.
  * Now, most images have at least **one successful extraction**.

> ✅ When single-shot performance is flaky, **multi-shot sampling** can dramatically improve effective accuracy.

---

## 🔑 Key Takeaways (Quick Revision)

* **Test-time compute** = spending more inference compute per query (more samples, better selection).
* You can:

  * Sample multiple outputs.
  * Use:

    * **Average logprob**
    * **Reward models / verifiers**
    * **Heuristics**
    * **Majority vote**
  * To pick the best or most reliable answer.
* Empirically:

  * Test-time compute can sometimes rival **large model scaling** in performance gains.
* Practical limits:

  * Cost and latency make very large N (hundreds, thousands of samples) unrealistic in production.
* Useful patterns:

  * Best-of-N with logprob or RM.
  * Self-consistency (majority vote) for exact-answer tasks.
  * Parallel sampling for **latency** and **robustness**.

---

## 💡 How to Explain in an Interview

> “Besides scaling model size, we can also scale **test-time compute**. Instead of generating a single answer, we generate multiple candidates and then use logprob, a reward model, or task-specific rules to pick the best one. This is often called Best-of-N decoding. For math or multiple-choice tasks, we can further improve reliability using majority vote or self-consistency across many samples.
>
> In some studies, using verifiers and test-time selection gave performance improvements comparable to increasing model size by a factor of 30, which is much cheaper than training a much larger model. The trade-off is increased inference cost and latency, so in production we usually use a small N and smart selection criteria rather than hundreds of samples per query.”



