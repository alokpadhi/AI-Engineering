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
````md
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

