# Session Notes — AI/ML Certification Course (Introductory Session)
**Instructor:** Prof. Durga Toshniwal
**Session 1:** Introductory Concepts + start of Data Preprocessing

---

## 1. Cognitive Science — The Starting Point

**Cognitive science** = the study of how humans *perceive*, *understand*, and *come to know* things.

- It is an **interdisciplinary** field drawing on:
  - Computer Science
  - Philosophy
  - Linguistics
  - Neuroscience
  - Psychology/Sociology
- Focus: how the **human nervous system represents, processes and transforms information**.
- **AI is generally considered a part of cognitive science.**

```
                    COGNITIVE SCIENCE
    ┌──────────────────────────────────────────────┐
    │  Computer Sc. │ Philosophy │ Linguistics     │
    │  Neuroscience │ Psychology │ ...             │
    │            ┌───────────────┐                 │
    │            │      AI       │                 │
    │            └───────────────┘                 │
    └──────────────────────────────────────────────┘
```

---

## 2. What is AI? (Multiple valid descriptions)

AI can be described in several ways — all are acceptable:

1. Making **computers think**.
2. **Automating activities** associated with human thinking (e.g., decision-making).
3. Creating **machines that perform functions requiring human-like intelligence**.
4. A field that tries to **emulate human intelligence** + automation.

### Breaking the term into two words

| Word | Meaning |
|---|---|
| **Artificial** | Something **created by explicit human effort** — *not occurring naturally* |
| **Intelligence** | The ability to **acquire knowledge and utilize it** |

> **Definition:** AI is that part of computer science focused on the **design of computer systems that exhibit human-like intelligence in multiple ways.**

### So AI involves TWO things:
1. **Studying** the intelligence concerned with humans.
2. **Representing** it using techniques/actions that **utilize computers**.

### AI is itself multidisciplinary
Computer Science · Statistics · Mathematics · Sociology · Linguistics · etc.

---

## 3. Relationship: AI → ML → DL → GenAI → Agentic AI (+ NLP)

This was a key diagram, because people commonly get confused between these terms.

```
┌───────────────────────────────────────────────────────────────┐
│  ARTIFICIAL INTELLIGENCE  (superset)                          │
│                                                               │
│   ┌─────────────────────────────────────────────────┐         │
│   │  MACHINE LEARNING                               │         │
│   │                                                 │         │
│   │    ┌───────────────────────────────────┐        │         │
│   │    │  DEEP LEARNING                    │        │         │
│   │    │                                   │        │         │
│   │    │   ┌────────────────────────┐      │        │         │
│   │    │   │  GENERATIVE AI         │      │        │         │
│   │    │   │                        │      │        │         │
│   │    │   │   ┌──────────────┐     │      │        │         │
│   │    │   │   │ AGENTIC AI   │     │      │        │         │
│   │    │   │   └──────────────┘     │      │        │         │
│   │    │   └────────────────────────┘      │        │         │
│   │    └───────────────────────────────────┘        │         │
│   └─────────────────────────────────────────────────┘         │
│                                                               │
│   ╔═══════════════════════════════════════════════════════╗   │
│   ║  NLP  — a subset of AI that CUTS ACROSS / intersects  ║   │
│   ║  with ML, DL, Generative AI and Agentic AI            ║   │
│   ╚═══════════════════════════════════════════════════════╝   │
└───────────────────────────────────────────────────────────────┘
```

**Key points:**
- AI ⊃ ML ⊃ DL ⊃ GenAI ⊃ Agentic AI (each is a **subset** of the previous)
- **NLP** is a subset of AI, but it **draws concepts from ML, DL, GenAI and Agentic AI** — so it has an **intersection with all of them**.

---

## 4. Types of Tasks in AI

- Knowledge representation and reasoning
- **Learning**  ← *primary focus of this course*
- Natural Language Processing (NLP)
- Searching
- Path planning
- Expert systems

> Since **machine learning involves learning**, the course focus will be on **learning** (which will also involve NLP concepts).

### What does "learning" mean here?
A system exhibits learning if it can **understand a situation and change its course of action based on experience.**

Learning requires the ability to:
1. **Generate new facts from old ones**
2. **Generate completely new concepts**
3. **Distinguish among different types of environments and situations**

Machine learning = making machines intelligent in some way.

---

## 5. Major AI Techniques (Taxonomy)

```
                    MAJOR AI TECHNIQUES
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
  SUPERVISED          UNSUPERVISED         REGRESSION
   LEARNING             LEARNING          (value predn.)
        │                   │
   ┌────┴────┐          CLUSTERING
   │         │
CLASSIFI-  PREDICTION
 CATION

  [ Generative AI & Agentic AI are NOT included here —
    they are different and do not fall in this category ]
```

---

## 6. Supervised Learning — CLASSIFICATION

### Terminology (all mean the SAME thing)
| Term | Meaning |
|---|---|
| Records / Tuples / Instances / Samples | **Rows** in a table |
| Attributes / Fields / Features | **Columns** in a table |

### What is Classification?

We are always given a special dataset called the **TRAINING DATASET**.

- It has a set of **attributes**.
- **One attribute is special** — called the **classifying attribute** or **class label**.
- The class label **is given to us** in the training data.

**The task:** map **input → output**
- Input = the set of attributes
- Output = the class label

$$Y = F(X)$$

where:
- **X** = set of input attributes (A1, A2, A3, …, An)
- **Y** = the class label
- **F** = the function/model — **finding F is the task of classification**

```
   TRAINING DATA (class label KNOWN)
   ┌─────┬────┬────┬────┬─────────┐
   │ Rec │ A1 │ A2 │ A3 │ CLASS   │
   ├─────┼────┼────┼────┼─────────┤
   │ R1  │ .. │ .. │ .. │  Yes    │
   │ R2  │ .. │ .. │ .. │  No     │
   └─────┴────┴────┴────┴─────────┘
                 │
                 ▼   learn / build model
        ┌────────────────────┐
        │   Y = F(X)         │   ← the CLASSIFIER
        └────────────────────┘
                 │
                 ▼   evaluate
   TEST DATA (class label ALSO KNOWN — special data)
                 │
      performance acceptable?
                 │ YES
                 ▼
   UNSEEN DATA (class label NOT known)  →  PREDICT class label
```

### The three datasets
| Dataset | Class label available? | Purpose |
|---|---|---|
| **Training data** | ✅ Yes | Build / learn the model |
| **Test data** | ✅ Yes (also special) | Test the **performance** of the model |
| **Unseen data** | ❌ No | The **actual goal** — predict its class label |

> **Primary goal** of learning a classifier from training data = **make predictions on unseen data in the future.**

---

## 7. Applications of Classification

### (a) Targeting consumers likely to buy a new product
- Take the existing set of customers.
- Classify them into → **will buy** / **will not buy**, based on attribute values.
- Attributes used: **salary, lifestyle, demographics**, and many others.

### (b) Fraud identification / Fraud detection
Example: identifying fraudulent credit card transactions.
- The classifier **learns from previously available training data** how fraudulent cases can be identified.
- Signals a classifier can learn:
  - Transaction amount **much higher than the average spend** on the card
  - Transaction in a **foreign currency**
  - Purchase from a **website not used earlier** by the customer
- This is how financial institutions actually work — hence the verification calls we all receive: *"Did you make this transaction?"*

---

## 8. Q&A — Types of Classification
**(Question raised by Gunjan Bhaiya: "Can classification be multi-class, not just two?")**

Yes. Classification is of more than two types:

| Type | Description | Example |
|---|---|---|
| **Binary classification** | Only **two** class labels | Yes/No, Buy/Don't-Buy |
| **Multi-class classification** | More than two (e.g., 5) class labels | Instead of just fraud/non-fraud → **5 types of fraudulent activity** |
| **One-vs-Rest (OvR)** | Multi-class problem, but interested in **only ONE** class label. Identify that one class; **lump all the rest into a single label** | Pick class C3; everything else → "Rest" |

```
   MULTI-CLASS                      ONE vs REST
   ┌───┬───┬───┬───┬───┐            ┌───┬───────────────┐
   │C1 │C2 │C3 │C4 │C5 │    ──►     │C3 │     REST      │
   └───┴───┴───┴───┴───┘            └───┴───────────────┘
                                     (C1,C2,C4,C5 merged)
```

---

## 9. Discriminative AI Models

**Definition:** Models that **learn a decision boundary**, which helps **categorize the different classes** in the given data.

- Idea: **only to distinguish / identify** the different classes or categories in the data.
- They **map the input attributes** and **model the probability** of what the class label will be for the given data.
- They **discriminate / distinguish** between the classes available in the data.

### Key characteristics of discriminative models
| # | Characteristic |
|---|---|
| 1 | **Goal** → classify or predict the class label from input data |
| 2 | **How** → build a **decision boundary** that separates classes in the data space |
| 3 | **Output** → the **probability** for the given class label (e.g., P(Yes), P(No)) |

```
      Decision Boundary (discriminative model)
      A2 ▲
         │   ○ ○    ╲
         │  ○ ○ ○    ╲      ×  ×
         │   ○ ○      ╲   ×  × ×
         │  ○          ╲   × ×  ×
         └──────────────╲──────────► A1
                    decision boundary
         ○ = Class "No"        × = Class "Yes"
```

### Examples of discriminative models (a.k.a. **classifiers**)
- Logistic Regression
- Support Vector Machines (SVM)
- Decision Trees
- Random Forest
- Naïve Bayes classifier
- *(each to be covered in detail later)*

> **Why is classification called "supervised" learning?**
> Because **class labels are available** in the given training data (attribute values **+** class labels are given to us).

---

## 10. Unsupervised Learning — CLUSTERING

**Definition:** We have **no information about class labels** and **no group information** about the data — we **arrive at the grouping at the END of the process.**

- Unsupervised learning is also called **clustering**.
- Given: a set of points with a set of attributes.
- Use a **similarity measure** to identify **groupings (clusters)** in the data.

### The core principle
> Points belonging to **one cluster** are **much more similar to each other** than to points belonging to **different/separate clusters.**

- **Points inside a cluster** → very **similar**
- **Points across clusters** → **less similar**

### The two objectives
```
   ┌─────────────────────────────────────────────────────┐
   │                                                     │
   │    ┌───────────┐                                    │
   │    │  ○ ○ ○ ○  │  ◄── INTRA-cluster distance        │
   │    │ ○ ○ ○ ○ ○ │      (within a cluster)            │
   │    │  ○ ○ ○ ○  │      ►  MINIMIZE  ◄                │
   │    └───────────┘                                    │
   │          ╲                                          │
   │           ╲  INTER-cluster distance                 │
   │            ╲ (across clusters)                      │
   │             ╲  ►  MAXIMIZE  ◄                       │
   │              ╲          ┌───────────┐               │
   │               ╲         │  × × × ×  │               │
   │                ╲────────│ × × × × × │               │
   │                         │  × × × ×  │               │
   │                         └───────────┘               │
   └─────────────────────────────────────────────────────┘
```

| Distance | Meaning | Objective |
|---|---|---|
| **Intra-cluster distance** | Distance **among points within** the same cluster | **MINIMIZE** |
| **Inter-cluster distance** | Distance **across** clusters | **MAXIMIZE** |

*(Instructor initially mis-spoke "inter" and corrected it to **intra**-cluster for the "within" case.)*

### Similarity measures
- **Euclidean distance** and others — all important ones will be covered later.

### 3-D visual example shown in class
A three-dimensional data space where data points are visibly **crowded into three regions** — shown in **light blue, dark blue, and red**. These are the natural clusters.

> **The broad idea:** identify the **naturally occurring groupings** available in the data — *not* create clusters just for the sake of it.

---

## 11. How Clustering Actually Works (whiteboard explanation)

Given **n** points, compute the **pairwise distance matrix**:

```
          1     2     3    ...    n
      ┌─────┬─────┬─────┬─────┬─────┐
   1  │  0  │d12  │d13  │ ... │d1n  │
      ├─────┼─────┼─────┼─────┼─────┤
   2  │d21  │  0  │d23  │ ... │d2n  │
      ├─────┼─────┼─────┼─────┼─────┤
   3  │d31  │d32  │  0  │ ... │d3n  │
      ├─────┼─────┼─────┼─────┼─────┤
  ... │ ... │ ... │ ... │ ... │ ... │
      ├─────┼─────┼─────┼─────┼─────┤
   n  │dn1  │dn2  │dn3  │ ... │  0  │
      └─────┴─────┴─────┴─────┴─────┘
```
1. Find the distance of **all points w.r.t. all other points** (pairwise).
2. Put points that are **very close to each other** into a **single cluster**.
3. Points **across clusters** end up **far away**.
4. A **similarity measure** is needed to compute these distances.

### Data format in clustering (vs classification)
```
   CLUSTERING input                    CLASSIFICATION input
   ┌────┬──────┬──────┬─────┐          ┌────┬──────┬──────┬───────┐
   │Pt  │ Att1 │ Att2 │ ... │          │Rec │ A1   │ A2   │ CLASS │
   ├────┼──────┼──────┼─────┤          ├────┼──────┼──────┼───────┤
   │P1  │  1   │  0   │ ... │          │R1  │  1   │  0   │  C1   │
   │P2  │ ...  │ ...  │ ... │          │R2  │ ...  │ ...  │  C2   │
   └────┴──────┴──────┴─────┘          └────┴──────┴──────┴───────┘
     ✗ NO group information              ✓ Class label GIVEN
       (derived at the END)                 (special data)
```

---

## 12. Numerical Intuition for Similarity (2-D example)

```
   Y ▲
  10 │                    ● C (10,10)
     │
     │
     │
   2 │  ● B (2,2)
   1 │ ● A (1,1)
     └──┴──┴──────────────┴────► X
        1  2             10

   dist(A,B) = SMALL   →  A and B are SIMILAR
   dist(A,C) = LARGE   →  dissimilar
   dist(B,C) = LARGE   →  dissimilar
```

**Intuition:** similar points are **close to each other**; dissimilar points are **far away**.
- Points **within** a cluster **must be similar → close**
- Points **across** clusters **must be dissimilar → distance high**

### Why must inter-cluster distance be maximized? (Hitesh Suryawanshi's question)
- The idea of clustering is to **put similar points together**.
- If there are **large distances within** a cluster, **it is not a cluster at all**. Example: an elongated blob where some points are close but the two ends are far apart — that's not a valid cluster.
- We want to find **regions of the data space that are full of points** (densely crowded).
  - Dense region ⇒ points closely packed ⇒ **intra-cluster distance minimized**.
- **Maximize inter-cluster distance** so that **dissimilar points are NOT put together**.

---

## 13. Clustering Example from the Class Itself

The instructor clustered **participants' first names by their starting alphabet**, shown as **histograms**:

```
  Freq
   ▲
   │  ███
   │  ███  ███
   │  ███  ███  ███
   │  ███  ███  ███  ███
   │  ███  ███  ███  ███  ███  ███  ███
   └──────────────────────────────────────►
       S    A    M    V    R    P    D   ...
```

**Order of frequency (highest → lower):** **S → A → M → V → R → P → D → …**

Earlier, the instructor had also shown **clusters of participants based on their locations on the map of India.**

**Case-study use of location clusters:**
- Given a location → predict the **number of participants**.
- Given a course → predict the **probable number of participants at different locations**.

---

## 14. Applications of Clustering

### (a) Market Segmentation
Divide customers into subsets, e.g.:
- **Budget buyers** — spend economically
- **High-spending customers** — buy niche/costly products not meant for the common person

Segmentation can be based on: **buying habits, geographical location, lifestyle**, or anything else. Goal: identify customers **similar in some way**.

### (b) Document Grouping / Document Clustering
- Form groups from a set of documents so that **documents on similar topics come together**.
- **Why:** with thousands of documents, it's impractical to open each one to see its topic.
- Once clusters exist → a **new document** can be assigned to a cluster → its topic is immediately known.
- Benefit: **quick access** to documents on a particular topic or set of topics.
- We already do this manually in daily life — keeping news articles in one folder, AI documents in another, etc.

### (c) Stock Movement Behaviour
Identify clusters of behaviour — e.g., a cluster of stocks that are **going down**, another **going up**, etc.

---

## 15. Classification vs Clustering — The Critical Distinction

> In plain English, both "classification" and "clustering" mean **grouping of data**. But technically they are **very different**.

| Aspect | **Classification** | **Clustering** |
|---|---|---|
| Learning type | **Supervised** | **Unsupervised** |
| Group info (class label) | **GIVEN** in training data | **NOT given** |
| When is grouping known | **Before** — at the start | **At the END** of the process |
| Mechanism | Learn `Y = F(X)`, build decision boundary | Compute similarity/distance, group similar points |
| Output | Predicted class label for unseen data | Clusters / groups |

### Q&A — Shivansh Sharma: *"What is the model learning in clustering, in the absence of labels?"*
- In clustering, you only have **P1, P2, … with Att1, Att2, …** — **no group identification**.
- From these **values**, you identify **which points are close to each other**, and close points go into a cluster.
- The **group information is not given** — but **through the process of clustering you discover** what groups exist in the data.

### Follow-up: Using clustering to *enable* classification
> **Important practical workflow.** If you collect new data and don't know what the classes are:

```
   New unlabelled data
          │
          ▼
   Step 1: CLUSTER the data  (based on similarity between points)
          │
          ▼
   Step 2: STUDY the clusters — examine points within each cluster,
           look at their attributes / behaviour
          │
          ▼
   Step 3: ASSIGN / derive a LABEL for each cluster
          │
          ▼
   Step 4: Now you have class labels → BUILD A CLASSIFIER
```

- **Clustering is often used along with classification when class information is not available.**
- To build a classifier we **need** class labels — clustering is how we obtain them.

---

## 16. Other Clustering Q&A

### (a) Abhishek Dagar: *"How do we set the cutoff for distance — e.g., if the distance between two entries is almost equal to one in a different cluster?"*
→ **To be covered in later lectures.** There are many ways to decide how to cluster points and whether points are similar or not.

### (b) On centroids (chat comment)
- There are **different algorithms** to perform clustering.
- **In one of these techniques, centroids are used** to form clusters — but **NOT all techniques use centroids.** Other techniques exist and will be covered.

### (c) On hyperparameter tuning (chat comment)
- **Hyperparameter tuning is very different** from finding similarity between points and grouping them into clusters. **Do not conflate the two.**

### (d) Dwarakesh T P: *"What if the given data does not readily form clusters?"*
- Yes — sometimes data is organised such that **in raw form it is difficult to form clusters**.
- Example: **very high-dimensional data**, where points are **generally very far away** from each other, so **all of them look dissimilar**.
- **Solution:** perform **transformations on the data first**, then use the transformed data for clustering.
- High-dimensional data + these techniques → covered later.

---

## 17. Regression and Forecasting

| Term | Meaning |
|---|---|
| **Forecasting** | **Value prediction** — whenever we predict *values*, we're doing forecasting |
| **Regression** | Can be **used for** forecasting values; in that case regression performs the task of forecasting |
| **Predictive analytics** | Analyze **current + historical** data → find **predictions about future (unseen) data** |

### Applications of forecasting
- Weather forecast / forecasting **rain**
- Forecasting **stock price**
- Forecasting **performance**
- Prediction used by **doctors** (e.g., identifying the healthy patient)
- Used "pretty much in anything"

---

## 18. Anomaly Detection

**Definition:** Anything that is **very different from the normal behaviour** is an anomaly.

- Abnormal behaviour is called an **anomaly** or **outlier**.
- In certain applications it is **more important to identify abnormal behaviour than the normal one** — e.g., **medical applications**.

### Both approaches work
```
   UNSUPERVISED approach (clustering)          SUPERVISED approach
   ┌───────────────────────────────┐
   │   ┌─────┐      ┌─────┐        │           Learn from labelled
   │   │○ ○ ○│      │× × ×│        │           normal vs anomalous
   │   │○ ○ ○│      │× × ×│        │           examples
   │   └─────┘      └─────┘        │
   │                               │
   │              ★  ◄── ANOMALY   │
   │           (far away from      │
   │            normal groups)     │
   └───────────────────────────────┘
```
- **Unsupervised:** cluster the data → **anomalies lie very far away from the normal groups**.
- **Supervised:** anomalous behaviour can also be identified using supervised methods.

### Q&A — Muni Prakash Ganji
*"Will we get this material in email?"* → **Yes, slides will be shared after the class.**

---

## 19. Deep Learning

**Definition:** A **subset of machine learning** (which is a subset of AI) in which **neural networks** are used.

- Neural networks are organised in the form of **LAYERS** — these layers make the model **"deep"**.
- **When to use:** when data is **very complex** and it is **very difficult to easily identify the patterns or relationships** in the data.

### Neural networks
- Comprised of **neurons**, based on the **human brain**.
- Just as neurons in the brain help to understand/identify/learn, **nodes in a neural network perform the learning**.
- Especially good for data otherwise difficult to handle: **text, images, audio**.

```
   Input      Hidden Layer 1   Hidden Layer 2    Output
   Layer      (what makes it "deep")
    ○ ───────────► ○ ───────────► ○ ───────────► ○
    ○ ───────────► ○ ───────────► ○ ───────────► ○
    ○ ───────────► ○ ───────────► ○
    ○ ───────────► ○ ───────────► ○
```

### ⭐ Key difference: ML vs Deep Learning
| Machine Learning | Deep Learning |
|---|---|
| Features are **manually / handcrafted** extracted from the data | **Feature extraction is AUTOMATIC** — not handcrafted |

---

## 20. Applications of Deep Learning

| Domain | Applications |
|---|---|
| **NLP** | Handling language **as it is spoken by humans** — called "natural" because humans naturally use it. English, Hindi, or any regional language |
| **Computer Vision** | Identifying things with the help of **images**. **Autonomous / semi-autonomous vehicles**: cameras in different directions for a **360° view** of surrounding vehicles; **driver-sleep identification + alarm**; **automatic braking** |
| **Speech Recognition** | Identify speech data, derive features. Used in **voice assistants like Alexa**, **transcription tools**, **AI summarization / AI meeting notes** |
| **Healthcare** | Identifying certain conditions in persons, **drug discovery**, **recommending treatments / plans** |
| **Finance** | **Algorithmic trading**, **fraud identification**, **credit risk modeling** |
| **Forecasting** | Weather, stock price |

Other domains where AI is used: vision, **social media**, **data analytics**, NLP.

> The **primary focus of this certification course** is **Generative AI and Agentic AI.**

---

## 21. Generative AI

**Definition:** A category of algorithms / AI models that help **generate NEW content**.

- New content can be in **any form**: **text, images, code, audio** — anything.
- The generated content is **very similar to real-world examples**.

### How it works
1. **Understand the context** of the given data (image, text, whatever it is).
2. Use the **context + given style + structure**.
3. **Produce content that looks natural / humanly generated.**
4. Learning is based on **patterns learned from vast amounts of data** — those patterns are then used to generate **completely new kinds of outputs**.

### ⭐ Discriminative vs Generative
| Traditional AI systems | Generative AI |
|---|---|
| **Discriminative** in nature | **Generative** in nature |
| Meant to **classify or predict** | Meant to **generate** |

### Generative AI models to be covered in the course
**BERT · GPT · T5 · GANs · LLMs** — and others.

---

## 22. Agentic AI

### The evolution sequence
```
  Traditional/simple      Deep Learning        Generative AI          AGENTIC AI
  ML algorithms      ──►  techniques      ──►  (not just         ──►  (not just generate —
  (classical)             (diversified)        discriminative,        decide, plan, adapt,
                                               but generative)        act autonomously)
```

### What Agentic AI systems do
- Behave like **partially or fully autonomous agents**.
- They **don't just generate** — they can:
  - **Take decisions**
  - **Learn the situation**
  - **Take feedback and adapt accordingly**
  - **Plan** and **act in dynamic environments**
  - **Set goals**, refine, or stop a task
- **Do NOT require human intervention all the time.**

### The gaming analogy
- Players take up different roles/avatars, decide what action to take.
- Based on that action, **the environment dynamically changes**.
- Every action carries a **reward or penalty** (e.g., improvement or degradation in performance).
- Agents then take **corrective action**, **plan the task**, **set goals**, and **adapt to dynamic environments**.

```
   ┌────────────┐   action    ┌──────────────┐
   │   AGENT    │────────────►│ ENVIRONMENT  │
   │            │             │  (dynamic)   │
   │  plan      │◄────────────│              │
   │  decide    │  reward /   └──────────────┘
   │  adapt     │  penalty
   └────────────┘  + feedback
```

### Applications of Agentic AI
| Application | Details |
|---|---|
| **Autonomous robotics** | For **safety-critical situations** where it isn't good/safe for a human to go. Robots go to the risky location, perform the task, and return. Currently used in **manufacturing** and **cleaning of hazardous materials** |
| **Software development** | — |
| **Personal AI assistants** | Virtual agents that **manage calendars, send emails, produce reports, make bookings**, and **adapt to preferences** |
| **Scientific discovery** | Deciding **how many experimental runs** to go for, executing those runs, **interpreting the data**, then **refining the experiments and re-running**, and **making inferences** |

---

## 23. Q&A on Agents, LLMs and Agentic AI

### (a) Sunil Saini: *"Fundamental difference between an agent and an LLM? We give the same kind of instructions as a prompt to an LLM and to agent code."*
- **LLMs can be used in Agentic AI directly — they can be used AS agents.**
- LLMs don't only do the required NLP processing; **they can serve as agents too.**

### (b) Sunil Saini follow-up: *"Can we have agents WITHOUT an LLM?"*
→ **Yes, of course.**

### (c) ⭐ Gunjan Bhaiya: *"What's the thin line between an AI Agent and Agentic AI?"*

| **AI Agent** (traditional) | **Agentic AI** |
|---|---|
| **Not smart** agents | **Autonomous** — may or may not rely on any trigger |
| Meant to do **one specific task**, do it, and finish | Can **plan, execute, refine, stop** a task, and **set goals** |
| Do **not work autonomously** | Works autonomously; **dynamically changes** |
| Do **not take decisions** the way Agentic AI does | **Takes decisions** |
| Example: a software agent that books a train / bus / taxi ticket via one particular mode — does it and finishes | Example below ↓ |

#### The travel example (Agentic AI in action)
> **Task given:** "Travel from point A to point B, minimizing travel time (and/or optimizing cost)."

```
   Task: A ──► B   Constraints: minimum time + optimal cost
        │
        ▼
   1. Agentic AI TAKES UP the task
   2. Looks at ALL different MODES of transport available
   3. Evaluates the PERMUTATIONS / combinations
   4. Decides which takes the LEAST time AND is optimal on COST
   5. Takes an OPTIMAL COMBINATION of the two constraints given
   6. Returns a DECISION:
        e.g., A ──► B ──► C , with the modes to use for each leg
```
A **simple AI agent** would only have executed the single booking it was assigned.

### (d) Deepesh Verma: *"So a rule-based simple agent doesn't use an LLM — it just applies rules and predicts. But an AI agent WITH an LLM can take other decisions too, not rule-based?"*
→ **Yes, that's correct.**

### (e) Pallavi Chakravarty: *"How can Agentic AI exist without an LLM? Example?"*
- The reference to LLMs was specifically about **language** models.
- There are also **vision models** — **vision-based AI** — and you can have **Agentic AI for those vision models** too.
- ⭐ **"There could be no Agentic AI without Generative AI."**
- On prompts: the prompt/instruction will be **fixed based on some task**, but it **could also be dynamically changing**.

### (f) Sacheen Adavinavar: *"Besides LLMs/VLMs, what models can agents use?"*
→ Deferred — **this will be covered at length throughout the entire course**; too early to cover now. Specific questions welcome.

---

# PART 2 — DATA PREPROCESSING
*(Started a day early — was scheduled for the next session)*

## 24. Why Data Preprocessing?

> Data is **not of much use** until and unless it is **in a condition that makes it useful** for us. To make it useful, we must bring it to a **form/state where it is rendered useful.**

- **To get quality results from data, the data itself must be of good quality** and **up to the mark**.

### Q&A — Deepesh Verma: *"So this means unstructured data → structured data?"*
→ **Not necessarily.** It means data that is **not in good shape**, which we **pre-process and bring up to the mark**.

---

## 25. The 5 Major Tasks of Data Preprocessing

```
                    DATA PREPROCESSING
                            │
   ┌──────────┬─────────────┼─────────────┬──────────────┐
   │          │             │             │              │
   ▼          ▼             ▼             ▼              ▼
 DATA       DATA          DATA          DATA           DATA
CLEANING  INTEGRATION  TRANSFORMATION REDUCTION   DISCRETIZATION
```

| Task | What it involves |
|---|---|
| **1. Data Cleaning** | **Filling missing values**, **smoothing noisy data**, **resolving inconsistencies** |
| **2. Data Integration** | Data may come from **multiple sources** → integrate into a **single warehouse / single structure** so it becomes useful |
| **3. Data Transformation** | Certain **operations performed on the data** so it is brought up to the **quality/mark** we want |
| **4. Data Reduction** | Useful when there is a **huge quantity of data** that is hard to handle due to **compute or other constraints** |
| **5. Data Discretization** | **Putting the data into BINS** |

*(These are the important ones always included; there are many others.)*

---

## 26. Data Cleaning — Handling Missing Values

The **first thing** done in data cleaning is **filling in missing values**.

### The example table shown in class

```
   ┌──────┬────┬────┬────┬────┬────┬───────┐
   │ Rec  │ A1 │ A2 │ A3 │ A4 │ A5 │  X    │  ← X = class label
   ├──────┼────┼────┼────┼────┼────┼───────┤
   │ R1   │ .. │ .. │ .. │ .. │ .. │  C1   │
   │ R2   │ .. │ .. │ .. │ .. │ .. │  C2   │
   │ R3   │ .. │ ██ │ .. │ .. │ .. │  C2   │  ██ = MISSING
   │ R4   │ .. │ .. │ .. │ .. │ .. │  C1   │
   │ R5   │ .. │ .. │ ██ │ .. │ .. │  C2   │
   │ R6   │ .. │ .. │ .. │ .. │ .. │  C2   │
   │ R7   │ .. │ .. │ .. │ .. │ .. │  C1   │
   │ R8   │ .. │ .. │ .. │ .. │ .. │  C2   │
   │ R9   │ .. │ ██ │ .. │ .. │ .. │  C2   │
   │ R10  │ .. │ .. │ .. │ .. │ .. │  ??   │  ← class label UNKNOWN
   └──────┴────┴────┴────┴────┴────┴───────┘
```

### Facts stated about this dataset
| Fact | Value |
|---|---|
| Total records | **10** |
| Records of class **C2** (majority class) | **6** |
| Records of class **C1** | **3** |
| Records with **unknown class label** | **1** (hence 6 + 3 = 9 known) |
| **Cells with missing attribute values** | **3** |
| **Proportion of data with missing values** | **~5%** |
| Attribute range observed | values spread roughly over **1 to 6** |

---

## 27. Suggested Approaches for Handling Missing Values (from participants)

| # | Who | Suggestion |
|---|---|---|
| 1 | **Gunjan Bhaiya** | Take the **MEAN / average over the COLUMN** (e.g., column A2: 2, 4, 2, 2 → average) and fill it in (e.g., into R9). Rationale: *"it's going to be a nearby value — not going to be discrete much."* Also: since X appears to be **derived from A1–A5**, put a **formula** instead of a raw value; and use **rule-based class prediction** — e.g., form clusters: if the value is **≤ 10 (or 15) → C1**, **above that → C2** — to handle the missing class label |
| 2 | **Swagat Kumar Pattnaik** | **Replace with 0 or −1** |
| 3 | **Nikhil Chinta** | Since values are **spread across a range of 1 to 6**, take the **MEDIAN**. For column **X (the class label)**, take the **MODE** — the value repeating the most number of times |
| 4 | **Hitesh Suryawanshi** | Replace with the **MOST FREQUENT VALUE** (i.e., the **MODE**) |

### Summary of answers collected
```
   MISSING ATTRIBUTE VALUES          MISSING CLASS LABEL
   ─────────────────────────         ───────────────────────
   • MEAN of the column              • Rule-based inference
   • MEDIAN of the column            • MODE (most frequent
   • Replace with 0 / −1               class label)
   • MODE (most frequent value)
```

> Instructor's closing remark: **"There are many different ways in which we can handle this."** The simplest solution was being introduced when the session extract ends — with 10 samples, of which 3 contain missing/class-related issues.

---

## 28. Consolidated Summary Table — All Techniques Introduced

| Technique | Learning type | Labels given? | Output | Key examples/models |
|---|---|---|---|---|
| **Classification** | Supervised | ✅ Yes | Class label | Logistic Reg., SVM, Decision Tree, Random Forest, Naïve Bayes |
| **Clustering** | Unsupervised | ❌ No | Clusters/groups | Distance-based (Euclidean); some use centroids |
| **Regression / Forecasting** | Supervised | ✅ Yes | **Value** | — |
| **Anomaly Detection** | Supervised **or** Unsupervised | Either | Outlier/anomaly | — |
| **Deep Learning** | Sub-field of ML | Depends | Complex patterns | Neural networks (layered) |
| **Generative AI** | Sub-field of DL | — | **New content** | BERT, GPT, T5, GANs, LLMs |
| **Agentic AI** | Sub-field of GenAI | — | **Autonomous decisions & actions** | LLM-based agents, vision-model agents |

---

## 29. Administrative / Housekeeping Points

- Slides **will be shared by email after each class** (confirmed to Muni Prakash Ganji).
- Today was **purely introductory** — every concept will be revisited **in detail** in dedicated sessions.
- **Data preprocessing** was originally scheduled for the next session but was **started today**.
- Chat had to be enabled by the co-host; instructor switched to **pen/annotation** mode to draw the intra/inter-cluster explanation.

---

## 30. Topics Explicitly Deferred to Later Sessions

1. Detailed **classification algorithms** (each classifier individually)
2. Detailed **clustering** — algorithms, **how to set distance cutoffs**, centroid vs non-centroid methods
3. All important **similarity measures**
4. **High-dimensional data** and the **transformations** needed before clustering
5. Several full sessions on **Deep Learning** techniques
6. **Generative AI** models in depth (BERT, GPT, T5, GANs, LLMs)
7. **Agentic AI** in depth — how agents work, which models they use
8. Remainder of **data preprocessing**: full missing-value treatment, noisy-data smoothing, integration, transformation, reduction, discretization
