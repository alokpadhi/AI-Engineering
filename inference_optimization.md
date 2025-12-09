# Inference Optimization

New models come and go, but one thing will always remain relevant: **making models
better, cheaper, and faster**.

Up until now, the discussion has focused on techniques for **improving model
capabilities**. This chapter focuses instead on **speed and cost**, which are just as
critical in real-world systems.

No matter how good a model is:
- If it’s **too slow**, users lose patience—or worse, predictions become useless  
  (e.g., *a next-day stock predictor that takes two days to run*).
- If it’s **too expensive**, the return on investment (ROI) may not justify deployment.

---

## Levels of Inference Optimization

Inference optimization can happen at **three levels**:

### 1. Model Level
Techniques that modify the model itself, such as:
- Reducing model size
- Lowering numerical precision
- Using more efficient architectures
- Removing computation bottlenecks (e.g., attention inefficiencies in transformers)

### 2. Hardware Level
Optimizations enabled by the underlying compute:
- Specialized AI accelerators
- More powerful or memory-efficient GPUs/TPUs
- Hardware-aware execution strategies

### 3. Service Level
How the model is **served in production**:
- Request batching
- Traffic-aware resource allocation
- Model-to-hardware matching
- Load balancing and autoscaling

The inference service must efficiently run models on hardware **under real traffic
conditions**, minimizing both **latency and cost**.

---

## A Highly Interdisciplinary Problem

Inference optimization often requires collaboration across disciplines:
- Model researchers
- Application developers
- Systems engineers
- Compiler designers
- Hardware architects
- Data center operators

This complexity is why inference optimization remains an active and evolving field.

---

## What This Chapter Covers

This chapter explores:
- Common **inference bottlenecks**
- Techniques to overcome them
- Performance metrics and trade-offs
- Optimization at the **model and service levels**
- A high-level overview of **AI accelerators**

---

## Speed vs Cost Trade-offs

Some optimizations improve **both speed and cost**:
- Example: **Lowering precision** → smaller models → faster inference → cheaper serving

Others involve trade-offs:
- Faster hardware may significantly increase cost
- Aggressive batching may lower cost but increase tail latency

Understanding these trade-offs is key to making informed engineering decisions.

---

## Why This Matters Even If You Use Managed Services

With the growing availability of open-source models, more teams are building their own
inference services.  

However, even if you rely on managed inference APIs:
- Understanding optimization techniques helps you **evaluate vendors**
- You can better **diagnose latency or cost issues**
- You’ll make more informed architectural decisions

If your application’s **latency or inference costs are limiting growth**, this chapter is
especially relevant.

---

## Understanding Inference Optimization

### Training vs Inference

There are two distinct phases in an AI model’s lifecycle:

- **Training**: Building (or finetuning) the model
- **Inference**: Using the trained model to produce outputs for inputs

Unless you actively train or finetune models, **inference is the phase you’ll spend most
of your time and money on**.

---
``
# Inference Overview

In production, the component that runs model inference is called an **inference
server**. It hosts the available models and has access to the necessary hardware.
Based on requests from applications (for example, user prompts), it allocates
resources to execute the appropriate models and returns the responses to users.

An inference server is part of a broader **inference service**, which is also
responsible for:
- Receiving requests
- Routing requests
- Possibly preprocessing requests before they reach the inference server

A simple inference service is illustrated in *Figure 9-1*.

> **Note**  
> As discussed in Chapter 7, inference involves only the **forward pass**, whereas
> training involves both the forward and backward passes.

### Training Cost vs Inference Cost

An interesting relationship exists between training cost and inference cost:

- Let **T** be the total training cost
- Let **p** be the price charged per inference
- Let **N** be the number of inference calls sold

Training a model only makes economic sense if:

```

T ≤ p × N

```

The more a model is used in production, the more model providers can reduce
inference cost. This logic generally does **not** apply to third-party API providers
who sell inference calls on top of open-source models.

---

## Model APIs and Inference Services

Model APIs provided by companies such as OpenAI and Google are full-fledged
**inference services**. If you use these services, you won’t be implementing most of
the techniques discussed in this chapter.

However, if you **host a model yourself**, you are responsible for:
- Building
- Optimizing
- Maintaining
the inference service.

---

## Computational Bottlenecks

Optimization is about identifying bottlenecks and addressing them.

Just as city planners alleviate traffic congestion by fixing bottlenecks, an inference
server should be designed around the **computational bottlenecks** of its workloads.

There are two main types:

### 1. Compute-Bound

A workload is **compute-bound** if its runtime is dominated by computation.

**Example**:
- Password decryption, which requires intensive mathematical operations

Performance here is limited by:
- FLOPs (floating-point operations per second)
- Parallel compute resources

---

### 2. Memory Bandwidth–Bound

A workload is **memory bandwidth–bound** if it is limited by data movement speed,
not computation.

**Example**:
- Moving data from CPU memory to GPU memory during model execution

In literature, this is often referred to as **memory-bound**, though terminology can
be confusing.

---

## Terminology Ambiguity: Memory-Bound vs Bandwidth-Bound

Some engineers use **memory-bound** to mean *bandwidth-limited*.
Others use it to mean *capacity-limited*.

### Memory capacity–bound
Occurs when hardware cannot hold all required data in memory, commonly resulting
in:

```

OOM (Out of Memory)

```

This can often be mitigated by:
- Splitting the workload into smaller chunks
- Offloading parts of the model to CPU memory

However, this introduces extra data transfer between CPU and GPU, which again
becomes a **bandwidth issue**. In practice, most memory capacity issues manifest as
bandwidth constraints.

> **Anecdote**  
> - Systems engineers typically use *memory-bound* to mean bandwidth-bound  
> - ML/AI engineers often use *memory-bound* to mean memory-capacity-bound

---

## Roofline Model

The concepts of compute-bound and memory-bandwidth-bound were formalized in the
**Roofline model** (Williams et al., 2009).

An operation is classified based on its **arithmetic intensity**:

```

Arithmetic intensity = number of arithmetic operations per byte of memory access

```

Profiling tools such as **NVIDIA Nsight** produce **roofline charts** that visualize
whether a workload is compute-bound or memory-bandwidth-bound (see *Figure 9-2*).

---

## Optimization Strategies by Bottleneck

- **Compute-bound workloads**
  - Scale across more chips
  - Use hardware with higher FLOP/s

- **Memory bandwidth–bound workloads**
  - Use hardware with higher memory bandwidth
  - Reduce memory access where possible

---

## Bottlenecks in Different Model Types

Different architectures produce different bottlenecks:

- **Image generation models** (e.g., Stable Diffusion)  
  → Typically **compute-bound**

- **Autoregressive language models**  
  → Typically **memory bandwidth–bound**

---

## Language Model Inference: Prefill vs Decode

Inference for transformer-based LLMs consists of two phases:

### 1. Prefill

- Processes input tokens **in parallel**
- Initializes the KV cache
- Limited by compute capacity

✅ **Compute-bound**

### 2. Decode

- Generates **one output token at a time**
- Requires repeated loading of large weight matrices
- Limited by memory transfer speed

✅ **Memory bandwidth–bound**

This process is visualized in *Figure 9-3*, where `<eos>` denotes the end-of-sequence
token.

---

## Production Implications

Because prefilling and decoding have **different computational profiles**:
- They are often **decoupled** in production
- Different machines or services may handle each phase

This technique is discussed further in **Inference Service Optimization**.

Key factors influencing bottlenecks in LLM inference:
- Context length
- Output length
- Request batching strategies

Long context lengths often cause memory bandwidth bottlenecks, though advanced
optimization techniques can mitigate this.

---

## Current and Future Trends

Due to:
- Transformer architectures
- Limitations of current accelerators

Many AI workloads today are **memory bandwidth–bound**.

However, future advances in:
- Hardware
- Compilers
- Model architecture

may shift AI inference back toward **compute-bound** regimes.
```
# Online vs Batch Inference APIs

Many model providers expose **two types of inference APIs** depending on application
requirements:

- **Online inference APIs** → optimize for **low latency**
- **Batch inference APIs** → optimize for **low cost and high throughput**

Understanding this distinction is critical for **system design interviews**, **cost
optimization**, and **production AI architecture**.

---

## 1. Online Inference APIs

### Definition
Online APIs process requests **as soon as they arrive**, prioritizing fast response
times.

### Characteristics
- ✅ Low latency (seconds or less)
- ✅ Suitable for interactive user experiences
- ⚠️ Higher cost per request
- ⚠️ Limited optimization opportunities due to latency constraints

### Typical Use Cases
- Chatbots
- Code generation
- AI copilots
- Search and question answering
- Any **customer-facing** application

### Internal Optimizations
Even online APIs may apply **micro-batching**, as long as:
- It does not noticeably increase latency
- User experience remains responsive

> The key goal is **latency first**, not maximum throughput.

---

## 2. Batch Inference APIs

### Definition
Batch APIs process requests **with relaxed latency constraints** to achieve **lower
cost per inference**.

### Characteristics
- ✅ High throughput
- ✅ Significantly cheaper
- ❌ Much higher latency (minutes to hours)
- ✅ Enables aggressive optimizations:
  - Large batching
  - Cheaper hardware
  - Scheduling during off-peak hours

### Cost Advantage
As of this writing:
- **Google Gemini** and **OpenAI batch APIs** offer  
  **~50% cost reduction**
- Turnaround time is typically **hours**, not seconds

---

## 3. When to Use Batch APIs

Batch inference is ideal when:
- Results are **not needed immediately**
- Workloads are large and repetitive

### Common Batch-Friendly Use Cases
- Synthetic data generation
- Periodic reporting
  - Summarizing Slack messages
  - Brand sentiment analysis on social media
  - Customer support ticket analysis
- Onboarding new customers
  - Processing all uploaded documents
- Migrating to a new model
  - Reprocessing historical data
- Large-scale personalization
  - Newsletters
  - Recommendations
- Knowledge base updates
  - Reindexing organizational documents

---

## 4. Streaming Responses (Online APIs)

### Problem
Autoregressive decoding can take a long time to finish, and users are impatient.

### Solution: Streaming Mode
- Tokens are returned **as soon as they are generated**
- Improves **time-to-first-token (TTFT)** dramatically

### Trade-offs
✅ Better perceived latency  
❌ You **cannot fully score or verify** the response before users see it

### Mitigation
- Retroactive moderation
- Post-generation filtering
- Immediate correction or removal if risk is detected

---

## 5. Why Providers Separate Online and Batch APIs

If an inference service can safely process **X requests/sec**:
- Total incoming traffic = **Y requests/sec**
- If **Y > X**, latency degrades

### Optimal Strategy
- Route **urgent, latency-sensitive** traffic → Online API
- Route **non-urgent** traffic → Batch API

This allows:
- SLA guarantees for interactive users
- Higher hardware utilization
- Lower overall operational cost

---

## 6. Batch APIs vs Traditional ML Batch Inference (Important Distinction)

### Traditional ML
- **Online inference:** compute predictions *after* request arrives
- **Batch inference:** precompute predictions *before* requests arrive

✅ Works well when:
- Inputs are finite
- Inputs are predictable  
  (e.g., recommendation systems)

### Foundation Models (LLMs)
- Inputs are **open-ended**
- User prompts are **unpredictable**
- Precomputation is generally **not feasible**

> Therefore, **batch APIs for LLMs ≠ traditional ML batch inference**

They are simply **delayed, cost-optimized execution**, not precomputed results.

---

## Interview & Production Takeaways (Quick Notes)

- ✅ Use **online APIs** for latency-critical user interactions
- ✅ Use **batch APIs** for large-scale offline or asynchronous workloads
- ✅ Streaming improves UX but increases safety complexity
- ✅ Batch APIs unlock:
  - 50%+ cost savings
  - Higher throughput
- ✅ Foundation-model batch inference ≠ traditional ML batch inference

**Interview tip:**  
If asked how to reduce inference cost at scale — *batch APIs should be one of the
first answers you mention*.
```
# Inference Performance Metrics (LLMs & Foundation Models)

Before applying inference optimizations, it’s critical to understand **what metrics
matter**. From a **user’s perspective**, latency dominates. From a **developer and
business perspective**, **throughput** and **utilization** determine scalability and
cost.

---

## 1. Core Inference Metrics

### 1.1 Latency (End-to-End)

**Latency** measures the total time from:
> 📩 *User sends a request* → 📤 *User receives the complete response*

For autoregressive models (LLMs), latency is not a single atomic value—it is composed
of multiple phases.

---

## 2. Autoregressive-Specific Latency Metrics

### 2.1 Time To First Token (TTFT)

**Definition**  
TTFT is the time between when a request is sent and when the **first visible token**
is generated.

**What it reflects**
- Corresponds to the **prefill phase**
- Strongly dependent on **input prompt length**
- Affected by:
  - Context size
  - Prompt caching
  - Prefill batching strategy
  - Compute allocation

**User expectations**
- ✅ Conversational chatbots → *Near-instant TTFT*
- ✅ Long-document tasks → Users tolerate higher TTFT

> ⚠️ TTFT is often the *most psychologically important* metric for perceived speed.

---

### 2.2 Time Per Output Token (TPOT)

**Definition**  
TPOT measures how long it takes to generate **each output token after the first
token**.

**Why it matters**
- Determines how long the full response takes
- Dominant factor for **long outputs**

**Example**
- TPOT = 100 ms
- Output = 1,000 tokens  
➡️ Total decode time = **100 seconds**

**Human reading constraint**
- Fast readers ≈ **120 ms/token**
- Practical target: **6–8 tokens/sec**

> Generating tokens faster than users can read rarely improves UX.

---

### 2.3 Time Between Tokens (TBT) / Inter-Token Latency (ITL)

- **TBT (LinkedIn)** / **ITL (NVIDIA)** measure spacing between output tokens
- Useful for diagnosing:
  - Jitter
  - Scheduling issues
  - Decode-phase bottlenecks

---

## 3. Latency Decomposition Formula

For standard autoregressive inference:

```

Total Latency = TTFT + TPOT × (# of output tokens)

```

---

## 4. Trade-Off: TTFT vs TPOT

Two applications with the **same total latency** can feel very different to users:

| Scenario | Experience |
|--------|-------------|
| Low TTFT, High TPOT | Fast start, slow streaming |
| Higher TTFT, Low TPOT | Slow start, fast generation |

### Engineering Trade-Off
- Shifting **compute resources** between:
  - Prefill (TTFT)
  - Decode (TPOT)
- Example insight:
  - **100 input tokens ≈ 1 output token** in latency impact
  - (Observed empirically by Anyscale)

> ✅ Optimal balance depends on **user studies**, not theory alone.

---

## 5. Model-Time vs User-Time (Hidden Token Generation)

TTFT observed by **users** may differ from TTFT measured **internally**.

### Example: Agentic / CoT Workflows
1. Model generates a **hidden plan**
2. Executes actions or tool calls (hidden)
3. Generates final visible answer

📌 From the **model’s view**  
- First token is generated early (Step 1)

📌 From the **user’s view**  
- First visible token appears much later (Step 3)

### Result
- **User-perceived TTFT ≫ Model-measured TTFT**

🔑 Some teams use:
> **Time To Publish** = time to first *user-visible* token

---

## 6. Latency Is a Distribution (Not a Single Number)

### Why averages are misleading

Example TTFT values (ms):
```

[100, 102, 100, 100, 99, 104, 110, 90, 3000, 95]

```

- **Average** ≈ 390 ms → misleadingly high
- One outlier skews perception

✅ Outliers can be caused by:
- Network issues
- Long prompts
- Resource starvation
- Cold starts

---

## 7. Percentile-Based Latency Metrics (Best Practice)

Instead of averages, **use percentiles**:

- **p50 (Median)** → Typical user experience
- **p90 / p95** → Tail latency (most users)
- **p99** → Worst-case experience (SLA-critical)

📌 Example:
- p50 TTFT = 100 ms → Half of users see faster responses
- p99 TTFT = 2,000 ms → Severe tail problems to investigate

### Recommended diagnostics
- Plot **TTFT vs input length**
- Track percentile trends over time
- Alert on p95/p99 regressions, not averages

---

## 8. Interview-Ready Key Takeaways

✅ Latency is user-facing; throughput & utilization drive cost  
✅ TTFT dominates perceived responsiveness  
✅ TPOT dominates long-output latency  
✅ TTFT + TPOT trade-off is a **resource allocation problem**  
✅ User-visible latency ≠ model-internal latency  
✅ Always use **percentiles**, not averages  

**Strong interview answer signal:**  
> “We optimize p95 TTFT and TPOT separately and monitor user-visible TTFT for
agentic flows.”

---

``
# Throughput and Goodput

## Throughput

**Throughput** measures how much work an inference service can complete over time.
For foundation model inference, throughput is most commonly expressed as:

- **Tokens per second (TPS)**

### Input vs Output Throughput

Some teams count both **input tokens (prefill)** and **output tokens (decode)** in
throughput calculations. However, because:

- Prefilling and decoding have **different computational bottlenecks**
- They are often **decoupled in modern inference servers**

👉 **Input and output throughput should be measured separately**.

When throughput is mentioned without modifiers, it usually refers to **output token
throughput**.

---

### User-Scaled Throughput

In multi-user systems, additional metrics are useful:

- **Tokens/s/user** – evaluates how well the system scales with more users
- **Requests per second (RPS)** – number of completed requests per second
- **Requests per minute (RPM)** – often preferred for LLMs since requests can take
  several seconds

Tracking RPS/RPM is important for understanding system behavior under concurrency
and avoiding provider throttling due to too many simultaneous requests.

---

## Throughput and Cost

Throughput is directly tied to **compute cost**.

### Example: Decode Cost

- Compute cost: **$2/hour**
- Output throughput: **100 tokens/sec**

Cost per 1M output tokens:

```

($2 / hour) ÷ (3600 sec/hour) ÷ 100 TPS × 1,000,000
≈ $5.56 per 1M tokens

```

If:
- Average output = **200 tokens/request**

Then:
- 1,000 requests = **200,000 tokens**
- Decode cost ≈ **$1.11**

---

### Example: Prefill Cost

- Compute cost: **$2/hour**
- Prefill capacity: **100 requests/min**

Cost for 1,000 prefills:

```

$0.33

```

---

### Total Cost per 1,000 Requests

```

Total = Prefill cost + Decode cost
= $0.33 + $1.11
= $1.44

```

---

## What Affects Throughput?

Throughput depends on:
- Model size
- Hardware (GPU/accelerator type)
- Workload characteristics
- Input/output length variance

### Notes on Comparisons

- Token counts vary across **tokenizers**
- Different models define tokens differently

👉 **Cost per request** is often a more reliable comparison metric than raw TPS.

---

## Latency–Throughput Trade-Off

As with most systems, inference services face a fundamental trade-off:

- ✅ Higher throughput → lower cost
- ❌ Higher throughput → worse TTFT and TPOT

Techniques such as batching can **double or triple throughput** if you’re willing to
sacrifice latency.

According to LinkedIn’s AI team (2024), such trade-offs are common in production
systems—but optimizing for throughput alone can hurt user experience.

---

## Goodput: Throughput That Actually Matters

To address this, many teams focus on **goodput**, a concept borrowed from
networking.

### Definition

**Goodput** measures the number of requests per second that **meet the Service Level
Objectives (SLOs)**.

### Example

Assume your application has:
- **TTFT ≤ 200 ms**
- **TPOT ≤ 100 ms**

If:
- The inference service completes **100 requests/min**
- Only **30 requests** meet the SLO

Then:

```

Goodput = 30 requests/min

```

Even though raw throughput is high, effective user-serving capacity is much lower.

---

## Key Takeaways (Interview-Ready)

- ✅ Throughput measures volume; **goodput measures usefulness**
- ✅ Input and output throughput should be tracked separately
- ✅ Higher throughput lowers cost but can degrade UX
- ✅ Always optimize for **SLO-compliant throughput**, not raw TPS
- ✅ Goodput is the right metric for production LLM systems

**Strong system-design framing:**

> “We optimize for goodput under p95 TTFT and TPOT constraints, not raw throughput.”

---
``
# Utilization, MFU, and MBU

Utilization metrics measure how efficiently hardware resources are used. They help
answer the question:

> *Out of the resources I’m paying for, how much useful work am I actually getting?*

However, not all “utilization” metrics are equally meaningful.

---

## GPU Utilization (Why `nvidia-smi` Is Misleading)

The most commonly cited metric is **GPU utilization**, reported by `nvidia-smi`
(SMI = System Management Interface).

**Definition (NVIDIA):**  
The percentage of time the GPU is actively processing *any* task.

### Why this is misleading

Consider a GPU that can perform **100 operations/second**:

- If it performs **1 operation/second**
- But does so continuously

👉 `nvidia-smi` reports **100% GPU utilization**

You are paying for 100 ops/sec and using **1% of the capacity**.

**Conclusion:**  
> GPU utilization tells you *whether* the GPU is busy, not *how efficiently* it is used.

---

## MFU — Model FLOP/s Utilization

**MFU (Model FLOP/s Utilization)** measures how effectively you use the GPU’s
*compute capacity*.

### Definition

```

MFU = (Observed throughput) / (Theoretical peak throughput)

```

### Example

- Hardware peak capability: **100 tokens/sec**
- Your inference service achieves: **20 tokens/sec**

```

MFU = 20 / 100 = 20%

```

MFU tells you how close you are to peak compute performance.

> MFU was popularized in the **PaLM paper (2022)**

---

## MBU — Model Bandwidth Utilization

Many LLM inference workloads are **memory bandwidth-bound**, not compute-bound.

**MBU (Model Bandwidth Utilization)** measures how efficiently you use memory
bandwidth.

### Memory bandwidth used by LLM inference

```

Bandwidth used = parameter_count × bytes_per_param × tokens/sec

```

### Example: 7B model, FP16

- Parameters: **7B**
- Precision: **FP16 = 2 bytes/param**
- Throughput: **100 tokens/sec**

```

Bandwidth used = 7B × 2 × 100 = 700 GB/s

```

If running on an **A100-80GB** GPU:

- Theoretical bandwidth: **2 TB/s**

```

MBU = 700 GB/s / 2,000 GB/s = 35%

```

> Fewer bytes per parameter → higher MBU  
> This is why **quantization** is so powerful.

---

## Relationship Between Throughput, MFU, and MBU

The relationships are approximately **linear**:

- Higher tokens/sec → higher MFU (compute-heavy workloads)
- Higher tokens/sec → higher MBU (bandwidth-heavy workloads)

Because of this, some teams loosely use *throughput* as a proxy for MFU/MBU—but
they are **not the same metric**.

---

## Compute-Bound vs Bandwidth-Bound Workloads

| Workload type | MFU | MBU |
|--------------|-----|-----|
| Compute-bound | High | Low |
| Bandwidth-bound | Low | High |

### In LLM inference

- **Prefill** → compute-bound → higher MFU
- **Decode** → memory-bandwidth-bound → higher MBU

Training typically has **higher MFU** than inference because:
- Workloads are more predictable
- Batching is more effective

---

## What Is “Good” MFU?

As of today:

- **Training MFU**
  - > **50%** is considered *good*
- **Inference MFU**
  - Typically much lower, especially during decoding

Achieving high MFU is harder than advertised because:

- Peak FLOP/s numbers are often measured under ideal or exotic conditions
- Real workloads include memory stalls, kernel launch overhead, and data movement

---

## Empirical MFU Examples (PaLM Paper)

| Model | Params | Hardware | MFU |
|------|--------|----------|-----|
| GPT-3 | 175B | V100 | 21.3% |
| Gopher | 280B | TPU v3 | 32.5% |
| MT-NLG | 530B | A100 | 30.2% |
| PaLM | 540B | TPU v4 | 46.2% |

---

## MBU in Practice

For **Llama-2-70B (FP16)** inference:

- MBU **decreases** as concurrency increases
- Reason:
  - More users → more compute per second
  - System shifts from bandwidth-bound → compute-bound

This behavior is expected and healthy.

---

## Key Takeaways (Production & Interview Ready)

- ❌ GPU utilization from `nvidia-smi` is not meaningful for efficiency
- ✅ MFU measures **compute efficiency**
- ✅ MBU measures **memory bandwidth efficiency**
- ✅ Inference alternates between compute-bound (prefill) and bandwidth-bound (decode)
- ✅ Quantization directly improves MBU
- ✅ High utilization ≠ good system
- ✅ **Cost and latency matter more than utilization**

**Strong framing:**

> “We track MFU and MBU to understand efficiency, but we optimize for goodput per
> dollar under latency constraints—not utilization.”

---
Below is a **clean, structured, interview-ready explanation** of the **AI Accelerators** section, rewritten for clarity while preserving the technical depth. I’ve also added practical insights that help connect hardware choices to inference optimization decisions.

---

# AI Accelerators (Inference-Oriented View)

Software performance is tightly coupled with the hardware it runs on. While some
optimization techniques are hardware-agnostic, **deep inference optimization
requires understanding accelerator architectures**.

Although this section focuses on inference, nearly all concepts apply to training as
well.

---

## Why Hardware Matters in AI

The evolution of AI has always been constrained—or unlocked—by compute:

* **1970s AI winter**: Limited compute made even simple neural networks impractical.
* **2012 deep learning revival**: AlexNet’s success was closely tied to using **GPUs**
  instead of thousands of CPUs.
* GPUs made large-scale neural networks accessible to researchers, triggering the
  modern deep learning era.

> Algorithmic breakthroughs often wait for hardware to catch up.

---

## What Is an Accelerator?

An **accelerator** is a chip designed to speed up a particular class of computations.

An **AI accelerator** is optimized specifically for machine learning workloads.

### GPUs vs CPUs

| CPUs                                     | GPUs                                   |
| ---------------------------------------- | -------------------------------------- |
| Few powerful cores (≤ ~64)               | Thousands of lightweight cores         |
| Excel at sequential, control-heavy tasks | Excel at massively parallel workloads  |
| OS, I/O, orchestration                   | Matrix multiplication, neural networks |

Neural networks rely heavily on **matrix multiplication**, which is highly parallelizable — making GPUs far more efficient than CPUs for ML workloads.

---

## GPUs as the Dominant AI Accelerator

GPUs remain the most widely used AI accelerators, with NVIDIA as the dominant vendor
during the 2020s AI boom.

However, GPU success has inspired many **alternative AI accelerators**, including:

* **AMD GPUs**
* **Google TPU (Tensor Processing Unit)**
* **Intel Habana Gaudi**
* **Graphcore IPU**
* **Groq LPU**
* **Cerebras Wafer-Scale Engine**
* **Custom cloud chips and startups**

Each makes different trade-offs around:

* Precision
* Memory bandwidth
* Latency
* Energy efficiency
* Scalability

---

## Training vs Inference: Why Specialized Chips Exist

Inference has different requirements than training:

### Training

* Backpropagation → high memory usage
* High numerical precision
* Throughput-oriented
* Large batch sizes

### Inference

* No gradients → much less memory
* Can use **lower precision** (INT8, FP8, INT4)
* Latency-sensitive
* Memory bandwidth is often the bottleneck

As a result:

* **Inference chips** emphasize:

  * Fast memory access
  * Low precision arithmetic
  * Low power consumption
* **Training chips** emphasize:

  * Large memory
  * High FLOP/s

> In many production systems, **inference cost exceeds training cost**, sometimes
> accounting for **up to 90% of total ML spend**.

---

## Inference-Specialized Accelerators

Examples include:

* **Apple Neural Engine** (on-device inference)
* **AWS Inferentia**
* **Meta MTIA**
* **Edge TPUs**
* **NVIDIA Jetson** (edge inference)

These chips are often:

* Cheaper per inference
* Lower power
* Optimized for INT8 / INT4
* Designed for production deployment, not research

---

## Compute Primitives: Scalars, Vectors, Tensors

Different accelerators are optimized around different **compute primitives**:

| Primitive | Optimized For     |
| --------- | ----------------- |
| Scalar    | CPUs              |
| Vector    | Traditional GPUs  |
| Tensor    | Modern GPUs, TPUs |

### Modern GPUs

* Started with vector units
* Now include **tensor cores** explicitly optimized for:

  * Matrix multiply–accumulate (MMA)
  * Low-precision arithmetic
  * Deep learning workloads

### TPUs

* Designed from the ground up for **tensor operations**
* Very high efficiency for large matrix multiplications

To run models efficiently, **software must match the hardware’s preferred compute primitive and memory layout**.

---

## Key Hardware Characteristics That Matter

Across inference and training, three hardware properties dominate decision-making:

### 1. Computational capability

* FLOP/s at different precisions (FP32, FP16, FP8, INT8, INT4)

### 2. Memory

* **Capacity** (Can the model fit?)
* **Bandwidth** (How fast can weights be streamed?)

> LLM inference is often **memory bandwidth-bound**, not compute-bound.

### 3. Power efficiency

* Cost per inference
* Thermal constraints
* Crucial for edge and mobile devices

---

## Interview & System-Design Takeaways

You can summarize this section concisely as:

> “AI accelerators optimize different points in the compute–memory–power trade-off.
> Training prioritizes compute throughput and memory capacity, while inference
> prioritizes memory bandwidth, latency, and low-precision arithmetic.”

Strong follow-up points:

* Why decoding is bandwidth-bound
* Why quantization improves inference speed
* Why CPUs are poor for LLM inference
* Why inference-specific chips exist

---
Here’s a **clear, structured, inference-oriented explanation** of the **Computational Capabilities, Memory Size, and Bandwidth** section—rewritten for understanding, interviews, and system design discussions.

---

# Computational Capabilities and Memory in AI Accelerators

Inference performance is determined not just by *how fast a chip can compute*, but also by *how fast it can move data*. This section explains **why FLOP/s alone is misleading** and why **memory bandwidth dominates LLM inference**.

---

## 1. Computational Capabilities (FLOP/s)

### What is FLOP/s?

* **FLOP/s (Floating Point Operations per Second)** measures how many numerical operations a chip can theoretically perform per second.
* Vendors usually advertise **peak FLOP/s**, but real workloads almost never reach this peak.
* The gap between peak and actual performance is captured by utilization metrics (e.g., MFU).

> Peak FLOP/s answers *“How fast could this chip be?”*
> Real FLOP/s answers *“How fast can my workload actually run?”*

---

### Precision Matters

The number of operations a chip can perform depends heavily on **numerical precision**:

* Lower precision → more operations per second
* Higher precision → fewer operations per second

Intuition:

> Adding two 32-bit numbers requires more work than two 16-bit numbers.

However, performance doesn’t scale linearly because:

* Modern chips have **specialized hardware units** (e.g., tensor cores)
* Some precisions are much more optimized than others

---

### H100 Example: FLOP/s by Precision

| Precision (Tensor Core) | TFLOP/s (with sparsity) |
| ----------------------- | ----------------------- |
| TF32 (≈19-bit)          | 989                     |
| BF16                    | 1,979                   |
| FP16                    | 1,979                   |
| FP8                     | 3,958                   |

**Key takeaways:**

* FP8 delivers **4× more compute** than TF32
* Lower precision is critical for inference throughput
* This is why **quantization** is so powerful for inference

> Increasing FLOP/s without increasing memory bandwidth often doesn’t improve LLM decoding speed.

---

## 2. Why Memory Matters More Than Compute

GPUs have **thousands of parallel cores**, but these cores are useless if they’re idle waiting for data.

For large models:

* Weights (tens of GBs) must be **repeatedly loaded**
* Data movement dominates runtime
* Inference is often **memory-bandwidth-bound**, not compute-bound

This is especially true for **autoregressive decoding** in LLMs.

---

## 3. GPU vs CPU Memory Technologies

### CPU Memory (DDR SDRAM)

* 2D structure
* Lower bandwidth
* Cheaper
* Typical bandwidth: **25–50 GB/s**

### GPU Memory (HBM / GDDR)

* 3D stacked (HBM)
* Much higher bandwidth
* Lower latency
* More expensive
* Bandwidth: **256 GB/s → 1.5+ TB/s**

This bandwidth difference alone explains why CPUs are terrible at LLM inference.

---

## 4. The GPU Memory Hierarchy

Accelerators interact with **three layers of memory**, each with different trade-offs:

### 1️⃣ CPU Memory (Host / DRAM)

* Shared with CPUs
* Lowest bandwidth
* High capacity (tens of GBs → TBs)
* Used when model doesn’t fit on GPU

### 2️⃣ GPU High-Bandwidth Memory (HBM)

* Dedicated GPU memory
* Holds model weights and KV cache
* Bandwidth: **hundreds of GB/s to TB/s**
* Size: **24–80 GB** typical

### 3️⃣ GPU On-Chip SRAM

* Extremely fast
* Includes:

  * Registers
  * L1/L2 caches
  * Shared memory
* Bandwidth: **10+ TB/s**
* Size: **tiny (≤ ~40 MB)**

> Most GPU optimization is about moving data **once**, reusing it from SRAM as much as possible.

---

## 5. Why GPU Memory Is Expensive

GPU memory costs more than CPU memory because:

* It must deliver **orders of magnitude more bandwidth**
* It must do so with **low latency**
* It uses advanced packaging (HBM is physically stacked near the chip)

This is also why:

* GPUs are expensive
* Memory bandwidth, not compute, is the limiting factor for inference

---

## 6. Why Low-Level GPU Programming Matters

High-level frameworks (PyTorch, TensorFlow):

* Prioritize correctness and flexibility
* Abstract away memory movement
* Do **not** allow fine-grained memory control

As a result, performance-focused teams use:

* **CUDA** (NVIDIA, proprietary)
* **Triton** (Python-based custom kernels)
* **ROCm** (AMD, open-source CUDA alternative)

These tools let engineers:

* Control memory layout
* Fuse operations
* Reduce memory reads/writes
* Improve MBU and MFU

---

## Core Mental Model (Very Important)

For LLM inference:

> **Compute gives you potential speed**
> **Memory bandwidth determines real speed**

That’s why:

* Quantization helps
* Smaller models decode faster
* Specialized inference hardware exists
* FLOP/s alone is a poor predictor of inference performance

---

## Interview-Ready Summary

> “While FLOP/s measures theoretical compute, LLM inference is typically memory-bandwidth-bound. Modern accelerators therefore prioritize low-precision arithmetic, high-bandwidth memory like HBM, and efficient use of on-chip SRAM. Real-world performance depends more on memory movement than raw compute.”

---
Below is a **clean, system-design–oriented explanation** of **Power Consumption and Accelerator Selection**, rewritten for *deep understanding*, *interview readiness*, and *real-world decision making*.

---

# Power Consumption in AI Accelerators

## 1. Why Accelerators Consume So Much Power

All chips perform computation using **transistors**.

* Each computation = transistors switching on/off
* Switching consumes **electrical energy**
* Modern GPUs contain **tens of billions of transistors**

Examples:

* **NVIDIA A100** → ~54 **billion** transistors
* **NVIDIA H100** → ~80 **billion** transistors

When a GPU is heavily utilized:

* Billions of transistors switch state **every second**
* This produces:

  * Large **energy consumption**
  * Significant **heat**

That heat must be removed using:

* Air cooling
* Liquid cooling
* Immersion cooling (in some advanced data centers)

👉 Cooling itself consumes electricity, meaning **power cost is more than just the chip**.

---

## 2. Environmental and Infrastructure Impact

Power has become one of the **hardest scaling bottlenecks** in AI.

### Real-world comparison

* **One NVIDIA H100 running at peak for one year ≈ 7,000 kWh**
* **Average US household per year ≈ 10,000 kWh**

That means:

> A single GPU uses ~70% of a household’s annual electricity.

Now scale that to:

* Thousands or tens of thousands of GPUs
* 24×7 operation
* Plus cooling and networking overhead

This is why:

* Data centers struggle to secure stable electricity
* Green data centers and renewable energy are strategic priorities
* Power availability limits how fast AI can scale

---

## 3. Electricity as a Deployment Constraint

When building large GPU clusters, companies must consider:

* **Electricity capacity** (not just cost)
* **Grid reliability**
* **Geography**

Trade-offs:

* Remote locations → cheaper electricity
* But:

  * Higher network latency
  * Reduced suitability for low-latency inference

⚠️ This is a real reason why some regions *cannot* host massive AI data centers—even if land is cheap.

---

## 4. Power Metrics: Max Power Draw vs TDP

Accelerators advertise power consumption using two related metrics:

### 1️⃣ Maximum Power Draw

* Absolute peak power a chip can consume
* Happens under extreme workloads
* Rarely sustained for long periods

### 2️⃣ TDP (Thermal Design Power)

* Amount of heat the cooling system must dissipate
* Proxy for *typical* power use
* Not exact, but practical for system design

For CPUs and GPUs:

* **Max power ≈ 1.1× to 1.5× TDP**
* Depends on architecture and workload characteristics

---

## 5. Why Power Still Matters in the Cloud

In cloud setups:

* You don’t manage cooling or electricity directly
* But **power is baked into pricing**

Higher power usage leads to:

* Higher instance costs
* Lower availability
* Regional constraints
* Carbon impact reporting

Understanding power helps you:

* Estimate long-term cost
* Compare accelerators realistically
* Make sustainability-aware decisions

---

# Selecting Accelerators: How to Think About It

## The Core Principle

**Hardware choice depends entirely on your workload’s bottleneck.**

---

## 1. Compute-bound Workloads

Examples:

* Image generation (e.g., diffusion)
* Dense prefill-heavy transformer workloads

What matters most:

* High **FLOP/s**
* Tensor core performance
* Precision support (FP16, FP8)

👉 Prioritize compute-heavy accelerators

---

## 2. Memory-bandwidth-bound Workloads

Examples:

* Autoregressive LLM decoding
* Long-context inference
* Large models with KV cache pressure

What matters most:

* **Memory bandwidth**
* Memory size (HBM capacity)
* Efficient low-precision support

👉 Paying for more FLOP/s won’t help if bandwidth is the bottleneck

---

## 3. The Three Questions to Ask When Choosing Hardware

### ✅ 1. Can it run my workload?

* Does the memory fit the model?
* Does it support required precision?
* Does the software stack support it?

### ⏱️ 2. How fast can it run?

* Throughput (tokens/sec)
* TTFT / TPOT impact
* Scaling behavior under load

### 💰 3. How much does it cost?

* Cloud pricing (hourly, per-token)
* Or:

  * Purchase price
  * Power consumption
  * Cooling/maintenance
  * Lifespan amortization

---

## 4. The “Big Three” Hardware Numbers

These answer the first two questions:

1. **FLOP/s** → How much compute is available
2. **Memory size** → Can the model + KV cache fit
3. **Memory bandwidth** → How fast decoding can run

The final decision is always:

> **Performance per dollar**, not peak specs

---

## Interview-Ready Summary

> “Accelerator power consumption is driven by transistor switching and memory movement. Modern GPUs consume thousands of kWh annually, making electricity a scaling bottleneck. Hardware selection should therefore match workload bottlenecks—compute-bound workloads favor high FLOP/s, while LLM inference is typically memory-bandwidth-bound, making memory capacity and bandwidth the dominant factors.”

---
# Inference Optimization — Model, Hardware, and Service Perspectives

## Core Idea
Inference optimization focuses on making AI models **faster and cheaper to serve**, while ideally preserving model quality. In practice, optimization almost always introduces **trade-offs**, and careless optimization can **degrade model behavior or output quality**.

---

## Three Levels of Inference Optimization

A useful mental model is the **archery analogy**:

### 1. Model-Level Optimization — *Crafting Better Arrows*
Optimization applied **to the model itself**.

**Goal**
- Reduce computation
- Reduce memory usage
- Speed up inference

**Typical Techniques**
- Quantization (FP16 → INT8 / FP8)
- Pruning
- Knowledge distillation
- Architectural changes (e.g., efficient attention variants)

**Trade-offs**
- Can negatively affect accuracy, reasoning depth, or robustness
- Requires careful evaluation after deployment

---

### 2. Hardware-Level Optimization — *Training a Stronger Archer*
Optimization achieved by using **better or more suitable hardware**.

**Goal**
- Execute the same model faster and cheaper

**Examples**
- More powerful GPUs (e.g., H100 vs A100)
- Specialized inference accelerators (Inferentia, LPU, TPU)
- Higher memory bandwidth or better tensor cores

**Key Point**
- Hardware optimization doesn’t change the model *itself*
- But hardware choice strongly determines cost, latency, and scalability

---

### 3. Service-Level Optimization — *Refining the Shooting Process*
Optimization at the **inference system / serving layer**.

**Includes**
- Request batching
- Caching
- Prefill–decode separation
- Load balancing
- Streaming responses
- Smart routing and scheduling

**Why It Matters**
- Most real-world inference gains come from **service-level design**
- Crucial for meeting **latency SLOs and cost targets**

---

## Model Quality Is Not Guaranteed to Stay the Same

> **Important reality:**  
> Inference optimization techniques can subtly change model outputs.

A real observation:
- The *same base model* (e.g., LLaMA) served by **different providers**
- Shows **measurable performance differences** across benchmarks

**Why this happens**
- Providers apply different:
  - Quantization strategies
  - Kernel fusions
  - Batching strategies
  - Prompt preprocessing
- These affect numerical precision and generation dynamics

**Implication**
- “Same model” ≠ “same behavior”
- Always re-evaluate models **after deployment optimization**

---

## Production Reality

- Optimizations are **never isolated**
- Real systems combine:
  - Model-level + service-level techniques
  - Hardware-aware serving strategies
- The challenge is not *maximizing optimization*, but **finding the optimal balance** between:
  - Latency
  - Cost
  - Throughput
  - Output quality

---

## Interview-Ready Key Takeaways

- Inference optimization operates at **three layers**: model, hardware, and service
- Most practical gains come from **service-level optimization**
- Faster inference often risks **model degradation**
- Different inference providers can change model behavior even with the same weights
- Always **measure post-optimization quality**, not just speed or cost

---

## Quick Mental Checklist (for System Design Interviews)

✅ What level am I optimizing at?  
✅ What bottleneck am I addressing (compute vs memory vs orchestration)?  
✅ What quality regressions might this introduce?  
✅ How will I detect post-deployment degradation?

---

> **Golden Rule:**  
> *Inference optimization is not just an engineering problem — it’s a product-quality decision.*

### Model Optimization (Model-Level Inference Optimization)

Model-level optimization aims to **make a model cheaper and faster at inference time** by modifying the model itself. Because this directly changes the model’s numerical representation or structure, it **can alter model behavior** and must be evaluated carefully.

Most modern foundation models share three properties that make inference expensive:

1. **Large model size** (many parameters)
2. **Autoregressive decoding** (token-by-token generation)
3. **Attention mechanism** (quadratic cost in sequence length)

This section focuses on **addressing model size**, mainly through **model compression**.

---

## Model Compression

**Model compression** reduces the storage and/or compute requirements of a model. Smaller models:

* Fit in less memory
* Move fewer bytes during inference
* Often generate tokens faster

### 1. Quantization ✅ (Most Used in Practice)

**What it is**

* Reduce numerical precision of model parameters (and sometimes activations)
* Examples: FP32 → FP16 → INT8 → FP8

**Why it works**

* Fewer bits per parameter:

  * Less memory bandwidth usage
  * Higher throughput
* Especially effective for memory-bandwidth-bound inference

**Why it dominates in practice**

* Easy to apply
* Works out of the box for many models
* Supported by modern hardware (Tensor Cores)
* Weight-only quantization often needs **no retraining**

**Limit**

* We’re approaching the lower bound (1 bit/value)
* Aggressive quantization can degrade reasoning, math, and long-context quality

---

### 2. Model Distillation ✅✅

**What it is**

* Train a smaller **student** model to mimic a larger **teacher** model

**Why it’s powerful**

* The student can retain *task-relevant behavior* with far fewer parameters
* Often ideal when:

  * You care about a specific task or domain
  * You want predictable latency and cost

**Key insight**

> Large models contain **redundant capacity** for many tasks.

**Trade-offs**

* Requires training effort
* Student model may generalize worse outside target tasks

---

### 3. Pruning ⚠️ (Promising but Less Common)

Pruning is inspired by a bold question:

> *Is there a small subset of parameters inside a large model that captures most of its behavior?*

There are **two meanings of pruning**:

#### A. Structured Pruning (Architecture-Changing)

* Remove entire neurons, layers, or attention heads
* Actually reduces parameter count
* Changes the architecture

#### B. Unstructured Pruning (Sparsity-Based)

* Identify “unimportant” parameters
* Set them to zero
* Total parameter count stays the same, but many are zero → **sparse model**

**What research shows**

* Up to **90% of parameters** can sometimes be removed with little accuracy loss
  (Frankle & Carbin, 2019 – *Lottery Ticket Hypothesis*)

**Why pruning is rare in production**

* Sparse models require hardware that can exploit sparsity
* Many GPUs are optimized for **dense** matrix ops
* Engineering complexity is high
* Gains are usually smaller than quantization gains

**Where pruning is useful**

* Architecture search
* Research
* When combined with retraining or distillation

---

## Practical Reality (as of today)

| Technique    | Popularity | Reason                                      |
| ------------ | ---------- | ------------------------------------------- |
| Quantization | ⭐⭐⭐⭐⭐      | Simple, effective, hardware-friendly        |
| Distillation | ⭐⭐⭐⭐       | Big wins for task-specific models           |
| Pruning      | ⭐⭐         | Hard to implement, limited hardware benefit |

**Key takeaway**

> **Weight-only quantization is the default choice** for inference optimization today.

---

## Interview-Grade Summary

* Model-level optimization trades **model fidelity** for **speed and cost**
* Compression methods target model size:

  * **Quantization** → easiest and most impactful
  * **Distillation** → best for task-specific deployment
  * **Pruning** → theoretically powerful, practically limited
* Sparse models don’t help unless hardware can exploit sparsity
* Always evaluate optimized models for **behavior drift**, not just accuracy

---

## One-Line Rule of Thumb

> *If you want fast wins → quantize.
> If you want controlled behavior → distill.
> If you want to publish papers → prune.*

### Overcoming the Autoregressive Decoding Bottleneck

Autoregressive decoding is one of the **core performance limits** of today’s LLMs.

#### Why decoding is such a bottleneck

* Models generate **one token at a time**
* Each token:

  * Requires loading large model weights from memory
  * Is **memory bandwidth–bound**
* Cost impact:

  * One output token ≈ **100 input tokens** in latency impact
  * Output tokens cost **2–4× more** than input tokens

So even small improvements in decoding speed lead to **large gains in latency, cost, and UX**.

---

## Speculative Decoding (Speculative Sampling)

Speculative decoding tackles the problem by **parallelizing what was previously sequential**, without changing the final model’s output distribution.

### Core Idea

Use a **small, fast model** to *guess* multiple future tokens, then let the **large target model** *verify* them in parallel.

* **Target model**: the model whose quality you care about
* **Draft (proposal) model**: smaller, faster, less accurate

---

## Step-by-Step Process

Let the input tokens be:

```
x₁, x₂, …, xₜ
```

### 1. Draft generation (fast, sequential)

The draft model generates **K tokens**:

```
xₜ₊₁, xₜ₊₂, …, xₜ₊ₖ
```

### 2. Parallel verification (slow model, but parallel)

The target model **verifies all K tokens at once**.

### 3. Acceptance

* The target model accepts the **longest prefix** it agrees with:

```
xₜ₊₁, …, xₜ₊ⱼ   (0 ≤ j ≤ K)
```

### 4. One authoritative token

* The target model then generates **one new token**:

```
xₜ₊ⱼ₊₁
```

### 5. Loop

Repeat the process, conditioning on all accepted tokens.

---

## What This Achieves

| Scenario           | Tokens produced per loop          |
| ------------------ | --------------------------------- |
| All rejected       | 1 token (same as normal decoding) |
| Partial acceptance | >1 tokens                         |
| All accepted       | **K + 1 tokens**                  |

Even moderate acceptance rates lead to **significant speedups**.

---

## Why This Works (Key Insights)

### 1. Verification is cheaper than generation

* Verifying K tokens is **parallel** → similar to prefilling
* Generation is **sequential**
* Speculative decoding converts decoding into a *prefill-like* workload

### 2. Not all tokens are equally hard

* Many tokens are **easy to predict**
* Smaller models can often guess structure-heavy text correctly:

  * Code
  * JSON
  * SQL
  * Boilerplate language

### 3. Decode is bandwidth-bound, not compute-bound

* There are usually **idle FLOPs**
* Verification “fills in” unused compute
* Works best when MFU is not already maxed out

---

## Choosing Parameters

### Draft model

* Smaller version of target model (same tokenizer preferred)
* Can be:

  * Distilled
  * Independently trained
  * Off-the-shelf weaker model

### K (draft length)

* Larger K:

  * Fewer verification rounds
  * Lower acceptance rate
* Smaller K:

  * Higher acceptance
  * More target-model calls

Optimal values are **domain-specific**.

---

## Real-World Results

* **DeepMind (Chinchilla-70B)**

  * Draft model: 4B parameters
  * Token speed:

    * Draft: **1.8 ms/token**
    * Target: **14.1 ms/token**
  * Result: **>2× latency reduction**, no quality loss

* Similar gains shown for **T5-XXL**

---

## Practical Adoption

Speculative decoding:

* ✅ Preserves output quality
* ✅ Relatively easy to implement (~50 lines)
* ✅ Widely supported

Available in:

* **vLLM**
* **TensorRT-LLM**
* **llama.cpp**

---

## When Speculative Decoding Makes Sense

✅ Best fit:

* Latency-sensitive applications
* Code / structured text
* When decode is memory-bound
* When MFU is not already saturated

⚠️ Less effective:

* Very creative, unpredictable text
* Systems already compute-bound
* Extremely small models

---

## One-Sentence Summary

> **Speculative decoding speeds up LLMs by letting a small model guess ahead and a big model verify in parallel—turning slow, sequential decoding into fast, parallel work without degrading quality.**
## Inference with Reference (Copy-Based Decoding)

![Image](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41586-023-06291-2/MediaObjects/41586_2023_6291_Fig1_HTML.png?utm_source=chatgpt.com)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2ALnKuukJKHIJX1fRKQXnoLA.png?utm_source=chatgpt.com)

### Core idea

When the output **largely overlaps with the input**, don’t generate those tokens again—**copy them directly from the prompt/context**. This avoids unnecessary autoregressive decoding steps.

### Typical scenarios

* **Document QA / RAG**: quoting passages verbatim
* **Code editing**: reusing most of the original code with small fixes
* **Multi-turn chat**: repeating earlier entities, constraints, or phrasing

### How it works

* At each decoding step, the system **detects a relevant span in the input** (often via exact/near-exact matching against the current prefix).
* Instead of sampling the next token, it **copies a contiguous span** from the input.
* If no suitable span is found, it falls back to normal generation.

### Why it’s appealing

* ✅ **Lossless** (no quality change)
* ✅ **No extra model** required (unlike speculative decoding)
* ✅ Big wins **when overlap is high**

Yang et al. (2023) report **~2× generation speedup** on overlap-heavy workloads.

### Key limitation

* Only helpful when **context–output overlap is substantial**. For creative or low-overlap tasks, benefits fade.

---

## Parallel Decoding (Breaking Sequential Dependency)

![Image](https://hao-ai-lab.github.io/blogs/cllm/img/jacobi_objective.png?utm_source=chatgpt.com)

![Image](https://cdn.prod.website-files.com/650c3b59079d92475f37b68f/652fff7890148f433e4eea14_Medusa.drawio.png?utm_source=chatgpt.com)

### Core idea

Generate **multiple future tokens at once**—even before knowing the exact previous token—then **verify and reconcile** them.

This directly attacks the autoregressive bottleneck.

---

## Two Main Families

### 1) Lookahead / Jacobi Decoding

* **Generate K future tokens in parallel**
* **Verify coherence** with context
* **Repair only failed tokens**, not the whole sequence (Jacobi-style iteration)
* Repeat until all tokens pass verification

**Pros**

* Works with a single decoder
* Gradually refines guesses

**Cons**

* Verification complexity
* Gains depend on easy-to-predict continuations

---

### 2) Medusa (Multi-Head Parallel Decoding)

* Add **multiple lightweight decoding heads** to a frozen base model
* Head *k* predicts token at position *t + k*
* Each head proposes **multiple options**
* A **tree-based attention mechanism** selects the best coherent path

**Results**

* NVIDIA reports **up to 1.9× TPS** for Llama 3.1 on HGX H200 GPUs

**Trade-offs**

* ⚠️ Architectural changes
* ⚠️ Non-trivial training + integration pipeline

---

## How These Compare

| Technique                  | Extra model? | Overlap dependent | Model changes | Typical speedup     |
| -------------------------- | ------------ | ----------------- | ------------- | ------------------- |
| Speculative decoding       | ✅            | ❌                 | ❌             | 1.5–2×              |
| Inference with reference   | ❌            | ✅                 | ❌             | ~2× (overlap cases) |
| Parallel / Jacobi decoding | ❌            | ❌                 | ❌             | workload-dependent  |
| Medusa                     | ❌            | ❌                 | ✅             | ~1.9×               |

---

## When to Use What

* **RAG, editing, refactoring** → *Inference with reference*
* **Structured outputs (code, JSON)** → *Speculative decoding*
* **Ultra-low latency decoding** → *Parallel decoding / Medusa*
* **Production simplicity** → *Speculative + reference (composable)*

---

### One-line takeaway

> **Inference with reference skips generation by copying; parallel decoding skips waiting by predicting ahead—both attack autoregressive latency from different directions.**
``
# Attention Mechanism Optimization (Inference-Focused Notes)

These notes summarize **why attention is a major inference bottleneck**, how the **KV cache works**, and the **main optimization strategies** used in modern LLM systems. This is **high-yield material** for both **system design interviews** and **production inference discussions**.

---

## 1. Why Attention Is Expensive During Inference

For autoregressive transformers:

- To generate token **xₜ**, the model needs **keys and values for all previous tokens**:
```

x₁, x₂, …, xₜ₋₁

```
- To generate **xₜ₊₁**, it needs:
```

x₁, x₂, …, xₜ

```

If we naively recomputed attention at every step:
- Attention computation would be **O(n²)** per token
- Generation would be prohibitively slow

---

## 2. KV Cache (Key–Value Cache)

### What it does
- **Stores key and value vectors** from previous tokens
- Avoids recomputing them during decoding
- Only **new token’s K & V** are computed and appended

### Important properties
- ✅ **Used only during inference**
- ❌ **Not used during training** (training sees full sequences at once)
- Grows **linearly with sequence length**
- Grows **linearly with batch size**

### Why it becomes a bottleneck
- Large KV cache:
- Consumes massive GPU memory
- Slows memory transfers
- Limits context length and batch size
- KV cache can be **larger than model weights**

---

## 3. KV Cache Memory Calculation (Interview Formula ⭐)

Unoptimized KV cache size:

```

2 × B × S × L × H × M

```

Where:
- **B** = batch size  
- **S** = sequence length  
- **L** = number of transformer layers  
- **H** = hidden/model dimension  
- **M** = bytes per value (e.g., FP16 = 2 bytes)  
- `2` = key + value  

### Example (LLaMA-2 13B):
- L = 40, H = 5,120
- B = 32, S = 2,048
- FP16 → 2 bytes

➡ **KV cache ≈ 54 GB**

> 🔔 This alone exceeds the memory of many GPUs.

---

## 4. Why Long Context Is Hard

- Attention computation: **O(n²)**
- KV cache growth: **O(n)**
- With large batches + long context → **memory bandwidth & capacity bottlenecks**
- Google reports a **3 TB KV cache** for:
  - 500B+ model
  - Batch size = 512
  - Context length = 2048

---

## 5. Three Major Optimization Strategies

### Bucket 1: Redesigning the Attention Mechanism  
*(Requires training or finetuning — changes architecture)*

#### 5.1 Local Windowed Attention
- Attend only to a **sliding window** (e.g., last 1K tokens)
- Reduces:
  - Attention compute
  - KV cache size
- Often combined with **global attention** for important tokens

📌 If avg sequence = 10K tokens and window = 1K → **10× memory reduction**

#### 5.2 Cross-Layer Attention
- **Share KV cache across multiple layers**
- If 3 layers share KV:
  - KV cache reduced by **3×**
- Very effective for deep models

#### 5.3 Multi-Query Attention (MQA)
- All query heads share **one KV pair**
- Big reduction in KV cache size
- Trade-off: slightly reduced expressiveness

#### 5.4 Grouped-Query Attention (GQA)
- Middle ground between:
  - Multi-head attention
  - Multi-query attention
- Query heads grouped; KV shared **within each group**
- Used in LLaMA-2/3

✅ Allows better **KV/memory vs quality trade-off**

---

### Bucket 2: KV Cache Optimization  
*(No architecture change; inference-time techniques)*

Examples (covered elsewhere in chapter):
- KV cache **compression**
- **KV paging / offloading**
- **Selective eviction**
- **Quantized KV cache**

These reduce:
- Memory footprint
- Memory bandwidth pressure

---

### Bucket 3: Optimized Attention Kernels  
*(System & kernel-level optimizations)*

- FlashAttention-style kernels
- Better memory layout
- Fewer memory reads/writes
- Exploits GPU SRAM and tensor cores efficiently

📌 Does **not change model behavior**, only performance

---

## 6. Real-World Case: Character.AI

- Average conversation history: **~180 messages**
- Primary inference bottleneck: **KV cache size**, not FLOPs
- Applied:
  - Multi-query attention
  - Local + global attention
  - Cross-layer attention

✅ Result:
- **>20× KV cache reduction**
- Memory no longer the bottleneck
- Enables **large batch serving**

---

## 7. Key Interview Takeaways ✅

- **KV cache is the main limiter** for long-context inference  
- Attention bottleneck = memory, not compute  
- Long context ≠ just bigger GPUs → needs architectural changes  
- Optimizations fall into:
  1. Attention redesign (training-time)
  2. KV cache management (inference-time)
  3. Kernel-level efficiency  

> **Strong system design answer:**  
> “For long-context LLM inference, the dominant bottleneck is KV cache memory and bandwidth, not FLOPs. Modern systems mitigate this using GQA/MQA, cross-layer KV sharing, and optimized attention kernels.”

---

## 8. One-Line Summary (Revision Ready)

**Attention inference bottlenecks come from KV cache growth; modern LLMs reduce it via smarter attention designs (GQA, MQA, local attention), KV cache optimizations, and highly optimized attention kernels.**
``
# Optimizing the KV Cache & Attention Computation

This section covers **practical inference-time techniques** used in modern LLM
serving systems to overcome **memory bottlenecks** and improve **throughput and
batch size**, without changing model quality.

---

## 1. Why KV Cache Optimization Matters

The **KV (Key–Value) cache** is often the dominant bottleneck during LLM inference,
especially for:
- Long-context applications
- Multi-turn conversations
- Large batch sizes

Problems caused by an unoptimized KV cache:
- Excessive GPU memory usage
- Memory fragmentation
- Limited batch size
- Higher latency due to memory movement

Modern inference optimization focuses heavily on **efficient KV cache management**.

---

## 2. KV Cache Optimization Techniques

### 2.1 PagedAttention (vLLM)

**PagedAttention** (Kwon et al., 2023), introduced by **vLLM**, is one of the most
influential KV cache innovations.

#### Core ideas:
- Split KV cache into **fixed-size memory blocks (pages)**
- Pages are **non-contiguous**
- Enable:
  - Flexible memory allocation
  - Reduced fragmentation
  - Memory sharing across requests

#### Benefits:
- Much higher GPU memory utilization
- Supports **large batch sizes**
- Essential for serving **long-context LLMs**

✅ Primary reason for vLLM’s rapid adoption in production

---

### 2.2 KV Cache Quantization

Reduce the **precision** used to store KV values:
- FP16 → INT8 or INT4

#### Benefits:
- KV cache size reduced by **2–4×**
- Memory bandwidth usage reduced
- Enables longer context or larger batch size

#### Trade-off:
- Potential (usually small) accuracy degradation

References:
- Hooper et al. (2024)
- Kang et al. (2024)

---

### 2.3 Adaptive KV Cache Compression

Instead of storing all KV vectors with the same fidelity:
- Compress earlier or less-relevant tokens more aggressively
- Preserve higher precision for recent or important tokens

📌 Observation:
Later tokens typically matter more for next-token prediction.

Reference:
- Ge et al. (2023)

---

### 2.4 Selective KV Cache

Only retain **useful tokens** in the KV cache:
- Drop or compress tokens unlikely to be attended to
- Especially useful for:
  - Retrieval-augmented systems
  - Structured or hierarchical contexts

Reference:
- Liu et al. (2024)

---

## 3. Writing Efficient Attention Kernels

Instead of modifying **what attention computes**, kernel optimization focuses on
**how it is computed**.

### Key idea:
Hardware-aware kernels significantly reduce:
- Memory reads/writes
- Kernel launch overhead
- Latency

---

## 4. FlashAttention

**FlashAttention** (Dao et al., 2022) is the most famous example.

### What it does:
- Fuses multiple attention operations:
  - QKᵀ
  - Softmax
  - Attention-weighted value multiplication
- Avoids materializing large intermediate matrices
- Uses GPU SRAM efficiently

### Benefits:
- Large speedups (especially for long sequences)
- Reduced memory footprint
- Better GPU utilization

Variants:
- FlashAttention → A100
- FlashAttention-3 → H100 (Shah et al., 2024)

✅ Widely used in production LLM stacks

---

## 5. Kernels and Compilers

### 5.1 What is a Kernel?

A **kernel** is:
- A specialized, hardware-optimized function
- Designed for repetitive, parallel operations
- Examples:
  - Matrix multiplication
  - Attention computation
  - Convolutions

Kernels are highly hardware-specific.

---

### 5.2 Kernel Programming Languages

Common tools:
- **CUDA** – NVIDIA GPUs
- **Triton** – OpenAI (simplifies kernel writing)
- **ROCm** – AMD GPUs

These languages give:
- Fine-grained control over memory and threads
- Higher performance
- Steep learning curve

---

## 6. Four Core Kernel Optimization Techniques

### 6.1 Vectorization
Process multiple contiguous data elements at once:
- Reduces memory I/O
- Exploits SIMD / tensor cores

---

### 6.2 Parallelization
Split work across:
- Threads
- GPU cores
- Streaming multiprocessors

---

### 6.3 Loop Tiling
Optimize data access patterns to:
- Maximize cache reuse
- Minimize global memory access

⚠️ Hardware-dependent (CPU ≠ GPU)

---

### 6.4 Operator Fusion
Combine multiple operations into a single kernel:
- Reduces memory reads/writes
- Removes intermediate buffers

📌 Critical for performance gains in transformers

---

## 7. Hardware-Specific Kernels

Because kernels are hardware-aware:
- New GPUs → new kernels
- Example:
  - FlashAttention → A100
  - FlashAttention-3 → H100

This is why inference performance depends heavily on:
- GPU generation
- Kernel availability

---

## 8. Compilers: Bridging Models and Hardware

### What compilers do:
- Convert high-level model code → hardware-executable kernels
- Optimize execution during **lowering**

### Popular compilers:
- **torch.compile** (PyTorch)
- **XLA / OpenXLA**
- **TensorRT**
- **Apache TVM**
- **MLIR**

Many frameworks integrate compilers directly.

---

## 9. PyTorch Inference Optimization Case Study

(PyTorch, 2023 – LLaMA-7B on A100 80GB)

Optimization steps:
1. `torch.compile`
2. INT8 quantization
3. INT4 quantization
4. Speculative decoding

✅ Result:
- Significant throughput improvement
- Combined optimizations stack multiplicatively

⚠️ Impact on output quality was not disclosed

---

## 10. Key Takeaways (Interview-Ready ✅)

- **KV cache is the dominant inference bottleneck**
- Memory management > raw FLOPs
- vLLM’s PagedAttention is a major breakthrough
- Kernel-level optimizations (FlashAttention) deliver massive gains
- Compilers are essential for production inference performance
- Inference performance depends on:
  - Hardware
  - Kernel availability
  - Compiler maturity

---

## One-Line Summary

> Modern LLM inference performance is driven by KV cache optimization, hardware-aware
attention kernels, and compilers that fuse and lower operations efficiently to GPUs.
``
## Inference Service Optimization (Service-Level)

Service-level optimization focuses on **resource management**, not model behavior.

**Goal:**
Given:

* Fixed compute & memory (GPUs, VRAM)
* Dynamic workloads (variable user requests)

Optimize for:

* ✅ Lower latency
* ✅ Higher throughput
* ✅ Lower cost

📌 Unlike model-level optimization, **service-level techniques do not change model outputs**.

---

## Batching: The Most Important Service-Level Optimization

### Why batching works

Processing requests individually:

* Repeats the same GPU kernel launches
* Underutilizes GPU parallelism
* Increases cost per request

Batching:

* Executes multiple requests together
* Shares computation across requests
* Increases throughput → lowers cost

**Analogy**

* ❌ Individual requests = everyone drives their own car
* ✅ Batching = people use a bus
  More efficient, but timing must be managed.

---

## 1. Static Batching

### How it works

* Server waits until a **fixed batch size** is filled
* Then processes all requests together

### Characteristics

* ✅ Maximal GPU efficiency
* ❌ Worst latency

### Problem

The **first request waits the longest**, even if it arrived much earlier.

**Example**

* Batch size = 8
* 1st request arrives → waits for 7 more
* Latency can explode under low traffic

📌 Rarely used in latency-sensitive systems

---

## 2. Dynamic Batching (Most Common)

### How it works

* Two conditions trigger execution:

  * Batch reaches max size **OR**
  * Max wait time (time window) expires

**Example**

```
Batch size = 8
Time window = 100 ms
→ Execute when either condition is met
```

### Characteristics

* ✅ Controls latency
* ✅ Works well under variable traffic
* ❌ Batch may be partially filled → some wasted compute

### Trade-off

| Aspect     | Result                     |
| ---------- | -------------------------- |
| Latency    | Controlled                 |
| Throughput | Slightly lower than static |
| Cost       | Efficient                  |

📌 Standard approach for most online LLM APIs

---

## 3. Continuous Batching (In-Flight Batching)

> **Critical for LLM serving**

Introduced in **ORCA (Yu et al., 2022)**
Used by **vLLM, TensorRT-LLM, OpenAI-style servers**

---

### The LLM Problem with Naive Batching

In naive batching:

* All requests must finish before **any response is returned**

🚨 But LLM requests vary wildly:

| Request | Output Tokens |
| ------- | ------------- |
| A       | 10            |
| B       | 1,000         |

Request A finishes early but **waits for B** → unnecessary latency.

---

### Continuous Batching Solution

**Key idea:**

* Return each response **as soon as it finishes**
* Do not wait for the entire batch

### How it works

1. Requests are batched for decoding
2. When one request completes:

   * Its response is returned immediately
   * A **new request enters the batch**
3. GPU stays busy continuously

**Analogy**

* A bus that:

  * Drops off passengers
  * Immediately picks up new ones
  * Never runs half-empty

---

### Benefits

* ✅ Low latency for short responses
* ✅ High GPU utilization
* ✅ Higher throughput
* ✅ Essential for streaming & chat systems

📌 **This is the dominant batching strategy in modern LLM serving**

---

## Comparison Summary

| Technique           | Latency     | Throughput | GPU Utilization | Used in Practice |
| ------------------- | ----------- | ---------- | --------------- | ---------------- |
| Static batching     | ❌ Very high | ✅✅         | ✅✅✅             | ❌ Rare           |
| Dynamic batching    | ✅ Medium    | ✅          | ✅               | ✅ Common         |
| Continuous batching | ✅✅ Best     | ✅✅         | ✅✅✅             | ✅✅ Standard      |

---

## Interview-Ready One-Liner

> *“Continuous batching allows LLM inference servers to return responses as soon as they complete while dynamically refilling the batch, maximizing GPU utilization and minimizing latency.”*

---

## Key Takeaways

* Batching is **the biggest cost lever** in inference
* Static batching = high efficiency, terrible latency
* Dynamic batching = good balance
* **Continuous batching is essential for LLMs**
* vLLM, TensorRT-LLM, and OpenAI-like systems rely on it heavily

---

# Inference Service Optimization (Advanced)

## Decoupling Prefill and Decode

### Why this matters

LLM inference has **two fundamentally different phases**:

| Phase       | Behavior                           | Bottleneck                 |
| ----------- | ---------------------------------- | -------------------------- |
| **Prefill** | Processes input tokens in parallel | **Compute-bound**          |
| **Decode**  | Generates tokens one-by-one        | **Memory bandwidth-bound** |

Running both on the **same GPU causes resource contention**:

* Prefill consumes compute and starves decoding jobs
* Decode slows → **higher TPOT**
* Adding a new request hurts **existing in-flight requests**

---

### Key Insight

> **Compute-bound and memory-bound workloads should not fight for the same hardware.**

---

### Decoupling Strategy (Disaggregation)

**Assign prefill and decode to different machines (GPUs):**

* Prefill GPUs → optimized for compute
* Decode GPUs → optimized for memory bandwidth

This idea is validated by:

* **DistServe** (Zhong et al., 2024)
* **Inference Without Interference** (Hu et al., 2024)

#### Results

* ✅ Higher request throughput
* ✅ Stable TTFT and TPOT under load
* ✅ Predictable latency SLOs

📌 **Communication overhead is acceptable**
Intermediate states (e.g., KV cache) are transferred once and efficiently using:

* NVLink (intra-node)
* High-speed interconnects (inter-node)

---

### Prefill : Decode GPU Ratio

The optimal ratio is **workload-dependent**:

| Scenario     | Goal          | Typical Ratio      |
| ------------ | ------------- | ------------------ |
| Long inputs  | Minimize TTFT | **2–4 : 1**        |
| Short inputs | Minimize TPOT | **1 : 1 or 1 : 2** |

📌 Used by large-scale deployments (e.g., Meta Llama inference).

---

### Interview One-Liner

> *“Decoupling prefill and decode isolates compute-bound and memory-bound workloads, preventing resource contention and significantly improving throughput while preserving latency SLOs.”*

---

## Prompt Caching (Context / Prefix Cache)

### Motivation

Many production prompts share **large overlapping prefixes**, such as:

* System prompts
* Long documents
* Conversation histories
* Codebases or books

Without caching, the model **repeatedly re-processes the same tokens**, wasting:

* Compute
* Memory bandwidth
* Money

---

### How Prompt Caching Works

* Detect shared **prefix segments**
* Compute and store them once
* Reuse their intermediate representations for future queries

Also called:

* **Context cache**
* **Prefix cache**

---

### Where Prompt Caching Shines

| Use Case                  | Benefit                       |
| ------------------------- | ----------------------------- |
| Long system prompts       | Massive cost & TTFT reduction |
| Chat with documents/books | Huge TTFT reduction           |
| Multi-turn conversations  | Faster response & lower cost  |
| Many-shot prompting       | Reduced prompt inflation cost |

---

### Impact: Real Numbers (Anthropic, 2024)

| Use Case                     | TTFT ↓   | Cost ↓   |
| ---------------------------- | -------- | -------- |
| Chat with book (100k tokens) | **–79%** | **–86%** |
| 10k many-shot prompt         | **–31%** | —        |
| Long multi-turn chat         | **–75%** | **–53%** |

📌 A 1,000-token system prompt used in **1M daily queries** → **1B tokens saved/day**

---

### Cost Trade-off

Prompt caching is **not free**:

* Cache consumes memory (similar to KV cache)
* Storage has a cost (e.g., per-token per-hour pricing)

Example offerings:

* **Google Gemini**: ~75% input-token discount, storage cost applies
* **Anthropic**: up to 90% cost reduction, 75% latency reduction

---

### Engineering Complexity

* Non-trivial to implement from scratch
* Requires cache invalidation strategies
* Easier when supported natively by model APIs

---

### Interview One-Liner

> *“Prompt caching avoids redundant prefilling by reusing shared prompt prefixes, dramatically reducing TTFT and inference cost for long-context applications.”*

---

## Final Mental Model (Quick Recall)

✅ **Decouple prefill & decode**
→ Fix resource contention
→ Boost throughput without hurting latency

✅ **Use prompt caching**
→ Stop paying for repeated tokens
→ Essential for long prompts & conversations

🚨 Both are **service-level optimizations**:

* Do **not change model behavior**
* Produce **immediate ROI**

---

# Parallelism in LLM Inference

## Why Parallelism Matters

Accelerators (GPUs, TPUs) are built for **massive parallelism**. Efficient inference at scale is fundamentally about:

* Exploiting parallel compute
* Managing memory constraints
* Balancing **latency vs throughput vs cost**

Most production systems combine **multiple parallelism strategies** rather than relying on just one.

---

## High-Level Taxonomy

| Parallelism Family             | Scope          | Primary Goal                 |
| ------------------------------ | -------------- | ---------------------------- |
| **Replica (Data) Parallelism** | Whole model    | Higher throughput            |
| **Model Parallelism**          | Within a model | Fit & speed up large models  |
| **Context Parallelism**        | Long inputs    | Faster long-context handling |
| **Sequence Parallelism**       | Operator-level | Balance compute stages       |

---

## 1. Replica Parallelism (Data Parallelism in Training)

### What it is

* Deploy **multiple identical copies (replicas)** of the same model
* Each replica handles different requests independently

### Benefits

* ✅ Easiest to implement
* ✅ Scales request throughput linearly
* ✅ No model changes required

### Trade-offs

* ❌ High memory usage (each replica stores full model weights)
* ❌ Chip assignment becomes a **bin-packing problem**

### Real-world complexity

Given:

* Models: 8B, 13B, 34B, 70B
* GPUs: 24 GB, 40 GB, 48 GB, 80 GB

Questions arise:

* Should a 40 GB GPU run **one 34B model** or **three 13B replicas**?
* How do we balance utilization, cost, and demand?

📌 This problem grows exponentially with:

* More models
* More replicas
* Heterogeneous hardware

---

### Interview One-Liner

> *“Replica parallelism scales throughput by duplicating models, but it’s memory-expensive and leads to complex GPU bin-packing decisions.”*

---

## 2. Model Parallelism

Used **when the model does not fit on a single device** or needs latency reduction.

---

### 2.1 Tensor Parallelism (Intra-operator Parallelism)

#### What it is

* Split tensors within a single operator (e.g., matrix multiply) across devices
* Most common inference-time model parallelism

Example:

* Split weight matrix column-wise
* Each GPU computes part of the output
* Results are merged

#### Benefits

* ✅ Enables serving very large models
* ✅ Reduces per-request latency

#### Costs

* ❌ Collective communication (e.g., all-reduce)
* ❌ Diminishing returns at large scale

---

### Interview One-Liner

> *“Tensor parallelism partitions model tensors across GPUs to both fit large models and reduce latency, at the cost of inter-device communication.”*

---

### 2.2 Pipeline Parallelism

#### What it is

* Split the model into **sequential stages**
* Each stage runs on a different device
* Requests flow stage by stage

#### How it works

* Input batch → split into micro-batches
* While Stage 2 processes micro-batch 1, Stage 1 processes micro-batch 2
* Computation overlaps

#### Benefits

* ✅ Enables very large models
* ✅ High throughput when batch sizes are large

#### Drawbacks

* ❌ Increases end-to-end latency
* ❌ Pipeline bubbles for small batches
* ❌ Communication overhead

✅ **Common in training**
❌ **Usually avoided for low-latency inference**

---

### Interview One-Liner

> *“Pipeline parallelism increases throughput for large models but adds latency, making it rarely suitable for latency-sensitive inference.”*

---

## 3. Context Parallelism (Long Context Optimization)

### What it is

* Split the **input sequence itself** across devices
* Example:

  * Tokens 1–50k → GPU 1
  * Tokens 50k–100k → GPU 2

### Why it exists

* Long-context inference is **memory-heavy (KV cache explosion)**
* Parallelizing context reduces per-device memory pressure

### Limitations

* ❌ Complex synchronization
* ❌ Attention still requires cross-device interaction
* ❌ Primarily useful for **very long contexts**

---

### Interview One-Liner

> *“Context parallelism partitions long input sequences across devices to make long-context inference tractable.”*

---

## 4. Sequence Parallelism (Operator-Level Parallelism)

### What it is

* Split **different operators** across devices
* Example:

  * Attention → GPU 1
  * Feedforward → GPU 2

### Motivation

* Different operators stress different resources:

  * Attention → memory bandwidth
  * MLP → compute

### Characteristics

* Specialized
* Hardware-aware
* Less commonly used standalone

---

### Interview One-Liner

> *“Sequence parallelism distributes different operators across devices to balance compute and memory bottlenecks.”*

---

## Putting It All Together (Production Reality)

Modern inference systems commonly use **hybrid parallelism**:

Example:

* Replica parallelism → scale users
* Tensor parallelism → fit large models
* Context parallelism → handle long prompts
* Continuous batching → maximize utilization
* Prefill/decode disaggregation → stabilize latency

---

## Mental Model (Quick Recall)

| Goal                   | Strategy                       |
| ---------------------- | ------------------------------ |
| More users             | Replica parallelism            |
| Bigger models          | Tensor / pipeline parallelism  |
| Long context           | Context parallelism            |
| Balance compute stages | Sequence parallelism           |
| Low latency            | Avoid pipeline, favor replicas |

---

## Final Interview Summary

> *“Efficient LLM inference relies on combining replica parallelism for throughput, tensor parallelism for model size and latency, and specialized context or sequence parallelism for long-context and operator-level optimization.”*

---











