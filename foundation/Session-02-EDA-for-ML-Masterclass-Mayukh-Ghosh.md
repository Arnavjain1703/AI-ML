# Session Notes — AI/ML Certification Course (Masterclass: EDA for Machine Learning)
**Instructor:** Mayukh Ghosh
**Session 2:** Exploratory Data Analysis from an ML/industry perspective + the road to the first model

---

## 1. Where This Session Fits — EDA vs Data Preprocessing

This was a **one-off guest masterclass**, not a change of instructor. Prof. Durga Toshniwal continues the main syllabus from Session 3 (confirmed to Rashmi, Chandrasekhar and Jithu).

### Q&A — Jithu: *"Isn't preprocessing different from EDA?"*
**The correction was accepted as technically right.**

| Term | Meaning |
|---|---|
| **EDA** | **Exploration** — understanding what the data is |
| **Data Preprocessing** | **Preparation** — putting the data into a usable state |

One follows the other. **But:** *"the industry these days has clubbed everything into EDA"* — so in practice they are interleaved, and this session treats them together.

```
   Strictly speaking:   EDA (explore) ──► Preprocessing (prepare) ──► Model

   Industry usage:      └─────────── all called "EDA" ───────────┘
```

### It was also not a "theory vs practical" split
Asked whether this session was the practical counterpart to Session 1's theory → **no**. The framing was *"how these ideas are actually used."*

---

## 2. Instructor Background — Mayukh Ghosh

```
Mayukh Ghosh
├── Base: Calcutta (Kolkata) → Bangalore, 10+ years
├── Core work: TRAINING — 11–12 years total
│     ├── B2C space (individual learners)
│     ├── B2B space (organisation-facing)
│     └── Corporate training (heavy volume)
├── Program leadership
│     ├── Headed a program for one of the IIMs
│     └── Headed a University of Chicago collaborative program (~6–7 yrs ago)
└── Consulting (secondary — "a bit of consulting… let me be very honest")
      ├── Aditya Birla Group
      ├── Adani Group
      └── 1–2 multinationals + "many other places"
```

**Program name confirmed on the call:** *"a progression to generative AI"* — the name signals a **staged progression** that builds up *to* GenAI rather than starting there.

> **A trait worth noting:** he flags his own honesty markers (*"let me be very honest"*, *"to be honest"*) precisely when giving an unvarnished opinion rather than the standard narrative — the scope of his consulting, the agentic-AI reliability verdict (§30), and the *"not a resounding universal yes"* on data reduction (§29). **He systematically resists hype.**

---

## 3. Why Statistics is the Gate, Not the Garnish

> *"Doing EDA is hazardous, it's risky, if you're vague about normal distribution, correlation, skewness. You will misinterpret a lot of things."*

> *"EDA is not about drawing diagrams or finding some reasons to impute missing values and outliers. It's much more comprehensive than that."*

### Why EDA consumes 60–70% of a project's time
The cost of EDA scales with **dimensionality (number of columns)**, not just rows. Per-column work multiplies: plots, missing-value logic, outlier logic, transformation decisions. Scaling is the only "one-shot" operation; everything else is per-variable.

```
    1 column   →  1× (plot + missing + outlier + transform)

   50 columns  →  50× everything
                  + 49 bivariate plots against the target   ← this is where the time goes
```

> **On people who claim statistics is obsolete now that transformers exist:** *"use both your ears — enter from here, exit from here. That's absolute rubbish."*

> *"If you don't know stats, you don't know data science. Full stop."*

---

## 4. The Five Pillars of Inferential Statistics

The required sequence, in order — each one feeds the next.

```
                    INFERENTIAL STATISTICS
                            │
                            ▼
              ┌───────────────────────────┐
              │  1. PROBABILITY           │  basic — not axiomatic
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │  2. PROBABILITY           │  binomial (discrete)
              │     DISTRIBUTIONS         │  normal + standard normal
              └─────────────┬─────────────┘     (continuous)
                            ▼
              ┌───────────────────────────┐
              │  3. SAMPLING              │  ways of drawing samples
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │  4. ESTIMATION            │  point · interval · MLE
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │  5. HYPOTHESIS TESTING    │  ← regression & time series
              └───────────────────────────┘        are built on this
```

### What breaks downstream without each one

| Stats topic | What it is consumed by |
|---|---|
| **Binomial distribution + odds** | **Logistic regression** — logistic *is* a binomial-family model; odds and probability are *"twin brothers"* |
| **Normal + standard normal** | All of EDA — outlier rules, scaling interpretation, skewness judgement |
| **Estimation, especially MLE** | Logistic regression's fitting algorithm **is** maximum likelihood estimation |
| **Hypothesis testing** | **The most important parts of regression and time series** — *"it is all built on hypothesis testing"* |

---

## 5. How to Re-Learn Statistics at This Level

### Q&A — Vinit: *"Which topics should I concentrate on, and how?"*

**Not the college way** (definitions plus formulas — *"no one cares, everyone can Google it"*). For **every** concept, ask two questions instead:

1. **Where does this fit in my ML journey?** (which model or step consumes it)
2. **How is it connected to the neighbouring concepts — and how is it *dis*connected?**

> *"A subject has been built over 200 years with certain connections. We can unwrap them as we wish, but there are connectivities we can't avoid — if we do that, that's disrespecting the subject."*

> *"There are 3 topics you should concentrate on: one is statistics, two is statistics, three is statistics."*

---

## 6. Descriptive Statistics — Reading a `.describe()` Output

The measures that come out of `df.describe()`, and what each one is for.

| Measure | Family | What it tells you |
|---|---|---|
| **Mean** | Central tendency | The average — **but pulled by outliers** |
| **Median** | Central tendency | The middle value — **resistant to outliers** |
| **Mode** | Central tendency | The most frequent value |
| **Min / Max** | Range | The extremes — first place to spot an absurd value |
| **Standard deviation** | Dispersion | Typical distance from the mean |
| **Variance** | Dispersion | SD squared |
| **Quartiles (Q1, Q2, Q3)** | Position | The 25th / 50th / 75th percentiles |
| **Percentiles** | Position | Used directly in outlier capping (§20) |
| **Skewness** | Shape | Symmetry — see §9 |
| **Kurtosis** | Shape | Peakedness — see §10 |

### ⭐ The mean/median comparison is a free normality test
```
   mean ≈ median ≈ mode   →  symmetric / near-normal
   mean >  median          →  right-skewed (positive)  — a few large values pulling the mean up
   mean <  median          →  left-skewed  (negative)
```
Theoretically the three are equal in a normal distribution; in practice *"within 5% of each other and we're pretty happy."*

---

## 7. The Excel Demo — What ONE Outlier Does to Everything

A sheet of **20 rows and 2 columns**: `Age` and `Salary` (₹ lakhs, CTC). Every statistic Python would give you, computed by hand.

### The planted anomaly

```
   ┌──────┬──────┬────────────┐
   │ Row  │ Age  │ Salary(₹L) │
   ├──────┼──────┼────────────┤
   │ ...  │  28  │     14     │
   │ ...  │  32  │     19     │
   │ ...  │  35  │     20     │
   │  18  │  30  │  ██ 75 ██  │  ← nobody near age 30 earns anywhere close
   └──────┴──────┴────────────┘

   That is 1 row out of 20 = 5% of the data.
```

### Before and after changing 75 → 25

| Statistic | With the outlier (75) | After fixing it to 25 | Effect |
|---|---|---|---|
| **Correlation (salary ~ age)** | **≈ 0.46** — *"no real relationship"* | **≈ 0.83** — strong positive | **+37 points**, nearly doubled |
| **Skewness** | **+4.11** — severely right-skewed | **≈ 1.1** | **−75%** |
| **Standard deviation** | **≈ 13** | **≈ 3** | collapsed |
| **Mean / Median** | 16–17 / 13 | **14 / 13.5** → now ≈ equal | became **symmetric / near-normal** |
| **Kurtosis** | high | normalised | (not important — see §10) |

```
        BEFORE                              AFTER
   corr = 0.46  "no relationship"      corr = 0.83  "strong positive"
   skew = 4.11  "badly skewed"         skew = 1.1   "under control"
   sd   = 13                           sd   = 3
   mean ≠ median                       mean ≈ median  → normal
          ▲
          └── ONE row out of 20 caused ALL of this
```

> *"One number geopoliticises the entire thing."*

Because that single value distorts correlation and dispersion here, it will distort a **regression** built on the same data in exactly the same way.

### Q&A — Yogan Naik: *"Aren't we just playing with the data to get a better number?"*
> *"I am **NOT** doing it because I want correlation and skewness to be better. I'm doing it because 75 is an extreme outlier that makes no sense in this cohort. I get a better result as a by-product — that is not the reason."*

This distinction is the whole subject of §11.

---

## 8. Distributions — Symmetric vs Asymmetric

```
                        ALL DISTRIBUTIONS
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
         SYMMETRIC                        ASYMMETRIC (= SKEWED)
              │                                   │
       Normal / bell curve              ┌─────────┴─────────┐
       mean = median = mode             │                   │
       skewness ≈ 0              LEFT / NEGATIVE      RIGHT / POSITIVE
                                 long left tail       long right tail
                                 skew < 0             skew > 0
```

### The normal distribution
- **Bell-shaped and symmetric** — cut it at the middle and you get **50% on the left, 50% on the right**.
- **mean ≈ median ≈ mode.**
- Different normals can look very different (different height, different spread) and **still all be normal**. What makes a distribution normal is the **symmetry and the centred mean/median/mode**, not the steepness of the curve.

```
   NORMAL (symmetric)                  RIGHT-SKEWED (positive)

  Freq                                Freq
   ▲                                   ▲
   │        ███                         │  ███
   │     ██████████                     │  ██████
   │  ████████████████                  │  █████████
   │  ████████████████                  │  ██████████████
   └──────────┬──────────►              └──────────────────────────►
              │                            ▲              ▲
         50%  │  50%                       │              │
   mean = median = mode                  bulk of      long right tail
                                         values       (the ₹75 L salary)

                                       mode < median < mean
                                       e.g. salary column: skew = +4.11
```

---

## 9. Skewness

| Skewness value | Reading |
|---|---|
| **≈ 0** | Symmetric / normal — *"data is under control, range is lower, values well distributed"* |
| **> +1** | Positively skewed — a few large values stretching the right tail (the 75) |
| **< −1** | Negatively skewed — a few very small values stretching the left tail |

### Industry reality check
> **~80% of real field variables are skewed.** Ad spend, marketing spend, sales — *"either I've done less or more."* Expect **|skew| > 1**. Perfectly normal real data is rare.

### Q&A — Gunjan: *"So lower skewness is better?"*
→ **Confirmed. Lower |skewness| = healthier data = less work for you.**

---

## 10. Kurtosis

**Kurtosis = the peakedness of the distribution.**

| Type | Sign | Shape |
|---|---|---|
| **Leptokurtic** | Positive | Very sharp peak, narrow spread — data crammed centrally, falls off steeply |
| **Mesokurtic** | ≈ 0 | The normal distribution — no excess peakedness, uniform slope |
| **Platykurtic** | Negative | Flat, plateau-like |

```
   LEPTOKURTIC             MESOKURTIC (normal)        PLATYKURTIC
   sharp peak              the reference shape        flat / plateau

  Freq                     Freq                      Freq
   ▲                        ▲                          ▲
   │      ███               │        ███               │
   │      ███               │     ██████████           │  ██████████████
   │      ███               │  ████████████████        │  ██████████████
   │   █████████            │  ████████████████        │  ██████████████
   └───────────────►        └──────────────────►       └──────────────────►

   kurtosis > 0             kurtosis ≈ 0               kurtosis < 0
   narrow spread,           uniform slope              wide spread,
   falls off steeply                                   no real peak
```

### Q&A — Nirav: *"How important is kurtosis for us?"*
→ **Low priority.** Kurtosis belongs to **SQC (Statistical Quality Control)** and **Design of Experiments**, not to your ML workflow.

> *"In your machine learning you will not need that depth."*

Know the three words; move on.

---

## 11. Manipulation vs Tampering

### Q&A — Swagat Kumar Pattnaik: *"How do we know we're tampering rather than manipulating?"*

> *"There is a very fine line between manipulating and tampering. In your initial few projects, you will actually tread over the line — in order to get high accuracy, because that's what plays in our mind."*

### The two golden rules — check the distribution BEFORE and AFTER every single operation

```
      ANY EDA operation  (impute / cap / transform / encode)
                        │
             ┌──────────┴──────────┐
          BEFORE                 AFTER
       record the dist.      record the dist.
             └──────────┬──────────┘
                        ▼
              compare, then judge:
   ┌───────────────────────────────────────────────────┐
   │  RULE 1: Don't change the distribution by much    │  (~20–30% shift max)
   │  RULE 2: Any change must be TOWARD normality      │
   └───────────────────────────────────────────────────┘

   Changed a LOT              →  TAMPERING
   Moved AWAY from normal     →  your LOGIC / analysis is wrong
   Small move toward normal   →  legitimate manipulation ✓
```

### The decisive test
The **reason** must be logic about the data — never the desirability of the output.

> *"If you have the right logic to do it, and in the process you get a better result — fine. If you manipulate data to serve your own nice/short work — that is tampering."*

### ⭐ The temptation to force normality
Everyone wants normal variables, because then outlier detection becomes trivial (see §19). So people log-transform everything.

> *"That is something you should be very, very wary of."*
> *"It's a temptation of having biryani every day. Not good for your health."*

**If a variable is already normal — great. If it isn't, don't over-engineer it into normality.**

---

## 12. What Happens if You Cross the Line — The Production Death Spiral

```
   Biased / over-engineered EDA
            │
            ▼
   Great in-sample accuracy (95%+)
            │
            ▼
   30-slide PPT to a non-statistical business stakeholder → they love the numbers
            │
            ▼
   Deployed to production  (CI/CD, Airflow monitoring, Azure / AWS / GCP / Databricks costs)
            │
            ▼
   NEW DATA ARRIVES  →  MODEL FAILS
            │
            ▼
   "Give me a month" → rebuild → fails again after 2 runs
            │
            ▼
   Client: "I've spent ₹15–20 lakhs, no ROI. We are scrapping the model."
            │
            ▼
   Model dead in 3–6 months
```

> *"A very good result doesn't necessarily mean a right result. In data science, a **correct** result is more important than a **good** result."*

> *"Give me 75% accuracy sustainable over 6 months in production — I'm happy. Give me 95% sustainable for 7 days — I don't care, I will not take it, I will not make it billable."*

---

## 13. Correlation

**Definition:** not just "a relationship" — **the degree of association between two variables.**

### The classic example — height vs weight of 100 people
> *"I am not interested in your height. Neither am I interested in your weight. You are fat, thin, tall, short — I simply don't care. You are just one data point, one entry of the hypothesis I'm trying to prove."*

The hypothesis being tested is **only**: do height and weight move together?

*(Aside: the older textbook example was height vs IQ. Short people protested, and the next edition changed it. He agreed it was a bad example.)*

### The scale

```
   −1 ─────────── −0.7 ───────── 0 ───────── +0.7 ─────────── +1
 perfect          strong      NO / weak      strong         perfect
 inverse          inverse    correlation    positive        positive
                            (−0.3 … +0.3)
```

| Range | Reading |
|---|---|
| **0.8 to 1.0** | Very strong positive |
| **−0.8 to −1.0** | Very strong inverse |
| **≈ −0.3 to +0.3** | **Weak / no correlation** |

### Examples given
| Direction | Example |
|---|---|
| **Positive** | Height ↑ → weight ↑ (taller people carry more bone and body mass) |
| **Negative** | Delhi **air quality vs pollution** — pollution ↑ → air quality ↓ → strong negative expected |

### Q&A — Sridutt: *"How do we know we've got wrong data?"*
The correlation number must **agree with what the variables mean**.

> *"If the value says something and the variables mean something else — there is something wrong."*

That is a **biased / treacherous data** signal. (He returns to this question more fully in §22.)

### Underneath correlation
Correlation is derived from **covariance** — flagged as worth looking up, deliberately skipped to avoid turning the masterclass into a statistics class.

---

## 14. Spurious Correlation — The Yale Ships Story

**"Spurious" = false.** Spurious correlation happens when **you** are subjectively or subconsciously biased *before* you build the model.

The story — a PhD thesis defence at Yale, roughly 10–12 years ago:

```
   THE CANDIDATE'S THESIS
   Target:   did a ship go down in the Atlantic?  (yes/no → a logistic regression problem)
   Features: weather variables — wind pressure, air pressure, sea conditions
   Method:   pulled 40–50 years of data for the days ships WENT DOWN
   Finding:  on those days the weather was bad
             →  "strong relationship: ships sink in bad weather"
                        │
                        ▼
   THE CHALLENGER  (a panel member): "Give me 15 days, I'll find a contrary result."
   Method:   looked ONLY at the days when NO ship went down — every ship sailed fine
   Finding:  on 46% of those days the weather was WORSE
             by the candidate's own parameters
                        │
                        ▼
   CONCLUSION: the original finding is NOT CONCLUSIVE.
               The first analysis was wrong — not because the maths was wrong,
               but because the SAMPLE was biased by design.
```

### Q&A — Deepak Bobade: *"Which of the two was correct?"*
→ **Neither is a "correct" answer.** The first one is simply **invalid**, because it only ever looked at one side of the outcome.

### Practical rule
If a correlation looks **very strong but unlikely → double check it.**

---

## 15. You Are the Biggest Source of Error

> *"That is the biggest problem we data scientists suffer from, because we are human beings, we are not machines. We will all have preconceived notions and biases."*

> *"Even if you know your sales are driven by advertisement and marketing — you cannot bring that prior knowledge in when you're building the model. Because then, subconsciously, you will make the EDA drive towards your beliefs."*

### Q&A — Meet: *"Can we quantify our own bias?"*
> *"No. Not even agentic AI has been able to do that. It's not a stupid question — the question just doesn't have an answer."*

### The consequence
```
   Human judgement can never be bias-free or error-free
                        │
                        ▼
   100% accuracy is NOT achievable
                        │
                        ▼
   Everything unquantifiable gets dumped into the ERROR TERM
                        │
                        ▼
              y = m₁x₁ + m₂x₂ + … + c + e
                                          ▲
                                          └── everything we cannot explain
```

> *"And that is the beauty of it. If we could quantify it, it would become very dull — the thrill goes off."*

---

## 16. Bivariate Outliers

**"Bivariate outlier" is the instructor's own coined term.** This is the centrepiece of the session.

### The interview question he asks candidates
> *"If I have already drawn a **heatmap** during EDA, do I still need a **scatterplot**? Does the scatterplot give me anything extra?"*

**Answer: YES.** The heatmap gives you the correlation **number**; only the scatterplot shows you **which individual points** are misbehaving jointly.

### The demonstration
He restored the 75 and appended two legitimate rows:

| Age | Salary (₹L) | Univariate verdict |
|---|---|---|
| 50 | 50 | Salary looks extreme… but **plausible at age 50** |
| 50 | 62 | Salary looks extreme… but **plausible at age 50** |
| **30** | **75** | Salary extreme **and impossible at 30** ← the real problem |

| View | What you see |
|---|---|
| **`Salary` alone** | The class flagged **three** outliers: 50, 62, 75 (mean ≈ 18.9, so all are 2–3× away) |
| **`Age` alone** | **Zero** outliers. Every age is plausible. The 30-year-old will *never* be an outlier on age |
| **(Age, Salary) jointly** | Only **one** row is anomalous — age 30 with ₹75 L. The two age-50 rows are fine |

```
   Salary
   (₹L) ▲
        │
     75 │        ★  ◄── BIVARIATE OUTLIER (30, 75)
        │            invisible to BOTH box plots
        │            an odd COMBINATION, not an odd VALUE
     62 │                                    ○  (50, 62)  ← fine
        │
     50 │                                    ○  (50, 50)  ← fine
        │
        │              ●  ●
     20 │        ●  ●
     14 │  ●  ●
        └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──► Age
           25    30    35    40    45    50

   ● = the normal cohort      ○ = high salary, high age → VALID
   ★ = high salary, LOW age → the only genuine anomaly
```

### The rules that fall out of it

| Rule | Detail |
|---|---|
| **Bivariate outliers matter MORE than univariate ones** | Univariate methods (box plot, IQR, distance-from-mean, binning) **structurally cannot see them** |
| **Always plot each X against the target Y** | Anything sitting in the **top-left or bottom-right** — the "wrong" corners — behaves differently from the rest |
| **High-X with high-Y is fine** | *"Then it is okay — it's a problem when there's a problem in between them"*, i.e. when the combination contradicts the trend (confirmed to Deepan) |
| **Cost of keeping them** | They drag model efficiency down, and they are **the main source of heteroscedasticity** in linear regression |

---

## 17. The Minimum-Number-of-Plots Rule

### Setup — Sathish's and Pallavi's exchange
**50 variables**: 1 is the target `Y`, the other **49 are `X`s**.

### The linear regression skeleton
$$y = mx + c$$

$$y = m_1x_1 + m_2x_2 + \dots + m_{49}x_{49} + c + e$$

Add the error term `e` to the school straight line and it becomes linear regression.

| | Focus |
|---|---|
| **Correlation** | The association between **any two** variables |
| **Regression** | **Exactly one** variable — **Y** |

> *"I don't care about the other 49. Why do I have them? Because I think they will potentially explain my target."*

### The question: what is the MINIMUM number of bivariate plots you must draw?
**Pallavi's answer — 49. Correct.**

```
   50 variables  =  1 target Y  +  49 predictors X

   MINIMUM bivariate plots  =  49        (every X against Y, one by one)

   ── skip any of them and you will miss bivariate outliers ──

   MAXIMUM?  Unbounded (X-vs-X pairs, pair plots) — "it depends"
```

> *"It's not about whether you give the correct answer, it's about how you think through it."*

---

## 18. Order of Operations — Outliers First, or Missing Values First?

He called this **"a million-dollar question in the industry."**

### Two answers from the room, both accepted
| Who | Answer |
|---|---|
| **Meet** | Outliers first — so the outliers don't contaminate however you fill the missing data |
| **Deepak Katara** | Analyse first — the percentages and deviations should dictate the approach |

### The decision rule

```
              Are you imputing missing values with a
              STATISTICAL MEASURE?  (mean especially, or median)
                             │
              ┌──────────────┴──────────────┐
             YES                            NO
              │                   (binning / categorising /
              ▼                    interpolation / MICE / advanced)
   ── OUTLIERS FIRST, ALWAYS ──                │
   otherwise your mean is                      ▼
   already wrong before you use it     Order doesn't matter →
                                       do whichever problem is BIGGER first
                                       (higher % / graver issue)
```

### Why "bigger problem first" when the order is free
> *"You want to have control of the data as soon as possible. You don't want to keep control away from you for longer by doing other things first — you need to see the real figures, the right figures, as soon as possible."*

---

## 19. The 3-Sigma Rule and Where 1.5 × IQR Comes From

### Q&A — Jithu: *"Where does the 1.5 in the box-plot formula come from?"*
→ **Accepted as correct in spirit:** it is tied to the **3-sigma** idea.

### The rule
```
  Freq
   ▲
   │                    ███
   │                 ██████████
   │              ████████████████
   │           ██████████████████████
   │        ████████████████████████████
   └────────┼───────────────┼───────────────┼────────►
        mean − 3·SD       mean         mean + 3·SD
            │                               │
    ◄───────┤                               ├───────►
   OUTLIERS │◄── 99.73% of observations ───►│ OUTLIERS
            │                               │
            └── 0.27% total, split across both tails ──┘
```

Worked example: mean = 50, SD = 10 → the "normal" range is **20 to 80**. Anything outside → **potential outlier**.

> **Precondition, stated explicitly:** this only holds when the distribution is **normal / symmetric**. For a skewed distribution it does **not** hold.

### Box-plot fences
$$\text{Lower fence} = Q1 - 1.5 \times IQR \qquad \text{Upper fence} = Q3 + 1.5 \times IQR$$

where $IQR = Q3 - Q1$.

**Why 1.5 and not 0.5 or 2.5?** Because it is calibrated to reproduce roughly the same tail cut-off as ±3σ under normality.

> **The actual arithmetic behind the "magic" 1.5** (worth knowing for an exam): for a normal distribution, `Q3 + 1.5·IQR ≈ mean + 2.698σ`, which leaves ≈ **0.7%** outside — deliberately tuned to approximate the 3-sigma / 99.7% cut-off.

### ⚠️ One naming correction to carry
He called this **"Chebyshev's inequality."** An exam will separate the two:

| | Applies to | Guarantee within mean ± 3·SD |
|---|---|---|
| **Empirical rule (68–95–99.7)** | **Normal** distributions only | **≈ 99.73%** ← the figure he used |
| **Chebyshev's inequality** | **Any** distribution | **At least 88.9%** (1 − 1/k², k = 3) |

The 99.73% figure plus the "only if normal" caveat he gave together describe the **empirical rule**. His caveat was right even though the name was loose.

### Why this creates the normality temptation
> *"If I can make it normal, finding outliers is easy — I don't have to draw box plots and do all that drilling. I just use this formula, I'm done. A lot less work."*

**And that is exactly why you must resist it — see §11.**

---

## 20. Outlier Treatment — Capping Off and Winsorization

The class had learned how to **find** outliers (box plot, IQR, distance from mean, binning) but **not how to treat** them.

### The worked example — `Age` in a banking dataset
```
   min = 18          max = 105
   mean = 45         99th percentile = 85
   n = 100  (later 500)
```

### Step 1 — Is it even non-normal? You can tell without plotting anything
min 18, max 105, mean 45 → the max is far further from the mean than the min is → **right-skewed.**

> *"A smart data scientist doesn't write 100 lines of code. They write 20 lines, but draw 5 inferences from one output."*

### Step 2 — ⭐ Ask WHO these people are, before any formula

| Population | Is age 105 plausible? |
|---|---|
| **India** | **No.** Average life expectancy ≈ **70**; 105 is ≈ 1.5× that. **Absurd** — no Indian bank would entertain a walk-in claiming 105 |
| **Japan / New Zealand / Scandinavia** | **Yes.** Average life expectancy ≈ **92–94**; 105 is only ~10 above average. He knows someone in the UK who did their own banking and post-office work at **101** |

> *"So I need to know: is this person from Tokyo, Copenhagen, or Mumbai? I need to know how **improbable** this value is."*

**Demographic and domain context decides whether a value is an error or a fact. Do this before any formula.**

### Step 3 — The treatment

| Term | Meaning |
|---|---|
| **Capping off** (older term) | Replace the outlier with the **nearest sensible boundary value** — e.g. the 99th percentile |
| **Winsorization** | The modern name for the same idea; **widely used in industry** |

```
   105  ──cap──────────────►  85   (the 99th percentile)     ✓ CORRECT
   105  ──impute with mean─►  45   (the central value)       ✗ WRONG
```

> *"Imputing an outlier with a central value doesn't make sense. The central value is somewhere in Delhi, the outlier is in London."*

Imputing an outlier with the mean causes a wide-ranging distribution change → **tampering, not manipulation** (§11).

---

## 21. Choose Your Battles — Is the Treatment Even Worth Doing?

The step most people skip sits **between** *"I found outliers"* and *"I treated outliers"*: **is it worth treating?**

### The arithmetic
n = **500** observations, capped at **85**. The top 1 percentile = **5 people**:

| Value | Capped to | Deviation |
|---|---|---|
| 87 | 85 | 2 |
| 90 | 85 | 5 |
| 92 | 85 | 7 |
| 95 | 85 | 10 |
| 105 | 85 | 20 |
| | **Total** | **44** |

```
   Average distortion caused by this "groundbreaking" outlier treatment

        =  44 / 500  =  0.088        ← not even 0.1
```

**Will this change whether `Age` becomes an important variable in the model? No.**

Compare with the salary demo in §7, where **one** value swung correlation by 37 points. *That* was worth treating.

### The rule
```
   ┌────────────────────────────────────────────────────────────────┐
   │  BORDERLINE outliers  →  do NOTHING. You are wasting time.     │
   │                                                                │
   │  EXTREME outliers     →  treat them: winsorize / cap /          │
   │                          categorise / dummy variable / drop     │
   └────────────────────────────────────────────────────────────────┘
```

> *"There is no brownie point or Nobel Prize for treating outliers you found."*
> *"Don't be a process-driven data scientist. Be a **thinking** data scientist."*

---

# PART 2 — FROM EDA TO THE FIRST MODEL

## 22. Framing the Problem Statement and Picking the Target

He opened a real dataset — an **online retailer's customer data** — and made the class pick the target variable.

### The dataset

| Column | Meaning |
|---|---|
| `CustomerID` | Identifier — not useful for modelling |
| `Age` | Old / middle / young |
| `Gender` | — |
| `OwnHome` | Owns a home or not |
| `Location` | **Near / far from a competitor's PHYSICAL store** |
| `Married` | Marital status |
| `Salary` | The customer's salary |
| `Children` | Number of children in the household |
| `History` | Past buying frequency (last year) |
| `Catalogs` | Number of catalogs **the business sent** them |
| `AmountSpent` | How much they have spent on your product (USD) |

**Why `Location` matters:** if a furniture shop sits below your customer's apartment, they will walk down and look at it in person. If the nearest shop is 3–4 km away needing a car, auto or bus, they will buy online instead. → **Near a competitor = worse for an online seller.**

### The elimination logic

```
   Columns 1–7  (Age, Gender, OwnHome, Married, Location, Salary, Children)
        Q: as the business owner, is PREDICTING any of these useful to you?
           "Can you give this customer a higher salary?"       → No
           "Can you turn an old customer into a young one?"    → No
        → ZERO control  →  these can only ever be PREDICTORS (X), never the target

   Catalogs
        → YOU decide how many to send.  100% control  →  not a target either

   History, AmountSpent
        → PARTIAL control: they have bought before, they will probably keep buying,
          and you WANT to increase it  →  candidate targets

        → AmountSpent WINS: it is revenue, and it is the thing you want to grow.
          "He spends $100 today; tomorrow I want him to buy for $200."
```

### ⭐ The rule to memorise

```
   ┌────────────────────────────────────────────────────────────────────┐
   │  Your TARGET variable is never something you control 0% of,        │
   │  and never something you control 100% of.                          │
   │                                                                    │
   │  It is always the variable you have SOME control over and          │
   │  WANT TO IMPROVE your control over.                                │
   └────────────────────────────────────────────────────────────────────┘
```

> *"Because business wants results about only those things. The rest, they don't care."*

### Why you need this skill
Clients hand you a data dump and say *"do some analysis"* or *"predictive modelling karke do."* They will **not** tell you the target.

> *"Do some analysis — I'll draw a bar graph and send it back to you. That is also analysis, right?"*

### Answering Sridutt's "wrong data" question properly (cf. §13)
Before touching Python you must establish:

```
   1. The PROBLEM STATEMENT
   2. The TARGET VARIABLE
   3. Whether the other variables plausibly explain it at all
   4. Whether it is even WORTH investing the time
   5. POC  →  stakeholder approval  →  sometimes a pilot
                        │
                        ▼
   … and ONLY THEN your first line of  pd.read_csv()
```

> *"It's not about wrong or right data. It's about whether what you're trying to do is possible on this data or not."*

---

## 23. Train / Test Split

Belongs to feature engineering / extended preprocessing, before modelling. It exists because you cannot tell a client *"I built a model, seems like it'll work next time."* You need **validation**.

```
   Full data (1000 rows)
        │   random, ROW-WISE split
        │   (confirmed to Sumojit — nothing to do with columns)
        │
        ├────────────── 70% = 700 rows ──────────►  TRAIN → build the model
        │
        └────────────── 30% = 300 rows ──────────►  TEST  → held out, unseen

   Compare the error on TEST against the error on TRAIN:
        similar  →  the model generalises to unseen data          ✓
        worse    →  go back, fine-tune, or restart from EDA       ✗
```

| Question | Answer |
|---|---|
| **Why 70–30?** | Build on the bigger chunk, but don't shrink the validation set so far that validation becomes meaningless — *"strike a middle ground"* |
| **Why random?** | *"If I make a biased split on my own, I will end up with spurious correlation"* (§14) |

> *"Model building is a tedious process. You go to the end, find something isn't working, and sometimes have to start from scratch."*

---

## 24. Data Leakage

> *"Data leakage is a very, very, very critical thing. It comes from doing certain parts of EDA before splitting and certain parts after."*

Standard scaling is $z = (X - \mu)/\sigma$ — it **uses the mean and SD of whatever data you feed it**.

### Q&A — Meet (called *"a fantastic answer"*)
If you normalise on all the data, information about the test rows flows into the transformation, so the model has effectively already seen part of the test set.

```
   ✗ WRONG                                  ✓ RIGHT
   ┌─────────────────────────┐              ┌─────────────────────────┐
   │  scale the WHOLE dataset│              │  SPLIT  70 / 30         │
   │  one μ, one σ from all  │              └────────────┬────────────┘
   │  1000 rows              │                           │
   └────────────┬────────────┘               scaler.fit(TRAIN)
                │                            μ, σ from TRAIN ONLY
          SPLIT 70 / 30                                  │
                │                            ┌───────────┴───────────┐
   test rows already carry μ,σ               ▼                       ▼
   influenced by the train rows      transform(TRAIN)        transform(TEST)
                │                              (the same μ, σ)
                ▼                                       │
   test set is NOT unseen                               ▼
   →  INVALID validation                     test set is genuinely unseen
   →  "outputs better than I deserve"        →  HONEST validation
```

### The fit / transform rule
```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit ONCE, on train only
X_test_s  = scaler.transform(X_test)        # transform only — never fit again
```

> *"I will fit only once on train. With that result, I will transform both train and test — NOT fit separately and transform separately. **90% of people make a mistake there.**"*

### Homework he set deliberately
**Why must the fit happen on train only?** Think it through after the next couple of sessions — he withheld the full answer on purpose.

*(Working answer to check yourself against: `transform` must be the identical function for both sets, and it must be derived only from information the model is allowed to have. Re-fitting on test both leaks test statistics and applies a different mapping, so train and test values would no longer be comparable.)*

---

## 25. Scaling vs Transformation

### Q&A — Deepak Bobade: *"So we take a log of it?"*
→ **Corrected: that is transformation, not scaling.** They are different operations with different purposes.

| | Operations | Effect on the distribution |
|---|---|---|
| **Scaling** | Standard scaler, min-max | **Shrinks the range; the distribution shape does NOT change** |
| **Transformation** | Log, reciprocal, square root, **Box-Cox** | **Changes the distribution** — that is the entire point |

### The two scalers

| Method | Formula | Output range |
|---|---|---|
| **StandardScaler** | $(X - \mu)/\sigma$ | mean 0, SD 1 → typically **−3 … +3** for clean normal data |
| **MinMaxScaler** | $(X - X_{min})/(X_{max} - X_{min})$ | **0 … 1** |

> ⚠️ **Correction:** he said min-max gives **−1 to +1**. The default `MinMaxScaler` maps to **[0, 1]** (Lokesh actually said "0 to 1" in the class). You only get `[−1, 1]` by explicitly setting `feature_range=(-1,1)`.

---

## 26. Using Scaling to Audit Your Own Outlier Treatment

**This is the "connect the concepts" showpiece of the session.**

### Step 1 — The standard normal distribution
$$z = \frac{X - \mu}{\sigma}$$

That is the standard scaler formula. `z` follows the **standard normal distribution: mean = 0, SD = 1.**

### Step 2 — Plug the 3-sigma rule into it
$$\text{mean} \pm 3 \cdot SD \;\longrightarrow\; 0 \pm 3 \cdot 1 \;\longrightarrow\; [-3, +3]$$

### Step 3 — The diagnostic (Meet got this: *"plus 3, minus 3"*)

> **For a variable that is (a) normally distributed and (b) already outlier-treated, all post-standard-scaling values must lie between −3 and +3.**
>
> **If you see values outside ±3, your outlier treatment was wrong or incomplete. Go back and use a better method.**

```
   Normal variable + outlier treatment done
                    │
                    ▼
            standard-scale it
                    │
                    ▼
   ┌──────────────────────────────────────────────────┐
   │  all values within −3 … +3   →  treatment sound ✓ │
   │  any value outside −3 … +3   →  treatment wrong ✗ │
   └──────────────────────────────────────────────────┘

   The logic: the distribution was GIVEN to you — you cannot change that.
   The ONLY thing YOU did was the outlier treatment.
   So a violation can only be your treatment's fault.
```

### The two jobs scaling does beyond rescaling
| # | Job |
|---|---|
| 1 | Correctly sequenced, it is where **data leakage** is avoided (§24) |
| 2 | It **back-validates** your outlier and missing-value treatment (this section) |

> *"This is how we connect concepts. Scaling is not a process limited to itself — it can help you assess other things in your EDA, which makes your EDA more robust."*

---

## 27. The Complete End-to-End Pipeline

### Q&A — Abhishek Diwan: *"Can you just take us through the journey of how you would go about, from the raw data, and then check all the steps to come up with the model?"*

```
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  0. GET THE DATA                                                          │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  1. FRAME THE PROBLEM STATEMENT + IDENTIFY THE TARGET VARIABLE     (§22)   │
 │     → in discussion with stakeholders. "You cannot do it alone."          │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  2. df.describe()  →  central tendency, dispersion, skewness, quartiles,   │
 │     percentiles. Read the DISTRIBUTIONS. How good or bad is each variable? │
 │     KEEP A PEN AND PAPER. Number every insight 1, 2, 3, 4 as you go.      │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  3. UNIVARIATE ANALYSIS — histograms, box plots, crosstabs (categorical)   │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  4. BIVARIATE → MULTIVARIATE — scatterplots (min 49 for 50 vars, §17),     │
 │     heatmaps, pair plots if needed.                                       │
 │     CROSS-CHECK these insights against your step-2 describe insights.     │
 │     Matching → good. Not matching → re-look.                              │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  5. OUTLIERS + MISSING VALUES — find, judge worth (§21), treat.            │
 │     Order per §18. If not worth treating, leave them.                     │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  6. TRANSFORMATION / FEATURE ENGINEERING — only IF required.               │
 │     "Not mandatory. Every time you will not require this."                │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  7. DUMMY VARIABLES / ONE-HOT ENCODING — categorical & string → numeric    │
 │     (mandatory: you cannot model on objects or strings)                   │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  8. TRAIN / TEST SPLIT   (70 / 30, random, row-wise)               (§23)   │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  9. SCALE — numeric columns only, AFTER the split.                 (§24)   │
 │     fit on TRAIN → transform TRAIN and TEST.                             │
 │     Then CONCATENATE scaled-numeric + categorical-dummies,               │
 │     separately for train and separately for test.                        │
 ├───────────────────────────────────────────────────────────────────────────┤
 │ 10. → This concatenated TRAIN set is "the data which is ready for your     │
 │      FIRST ITERATION of the model."                                       │
 └───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        "That's the First World War. The Second World War starts after that."
                                    │
                                    ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ 11. Run the FULL model on TRAIN. Feature selection. Check every yardstick  │
 │     the model exposes.                                                    │
 │ 12. Loop: model ⇄ EDA, up and down, MANY times.                          │
 │ 13. ONLY once the trained model is FINAL and its variables are FIXED       │
 │     → compare against TEST.      ⚠️ NOT earlier. Not initially.           │
 └───────────────────────────────────────────────────────────────────────────┘
```

### ⭐ The ordering trap
**Encoding and dummies come BEFORE the split. Scaling comes AFTER.** And prediction on test is the **last** thing you do, not something you peek at while tuning — he corrected Deepak Bobade on exactly this point.

---

## 28. When is the Model Good Enough?

### Q&A — Sai Gowtham: *"How many iterations until the model is good?"*
- Not a fixed number of iterations. You comply with **yardsticks**: a target accuracy level, an error you must get below.
- **Those yardsticks are not fixed — they change from model to model.**

> *"That's the beauty of data science. The model you build today, you will never build again in your life. Everything is unique — so every yardstick is unique."*

---

## 29. Practical Constraints Raised in Q&A

### (a) Jithu: *"My data is millions of rows and Colab can't load it."*
| Option | Verdict |
|---|---|
| **Colab paid tier** (~₹1000/month) | Gives GPU and parallel compute — his first suggestion |
| **Sample down to 10k–50k rows** | ✓ Allowed — **but the sample must represent the population's distribution across ALL variables.** Otherwise pointless |
| **Chunking / EDA on a subset** | ✓ Fine — **but it is a PILOT.** You cannot claim the full job is done |
| **The full job** | Needs a **GPU or cloud support** |

### (b) Swagat Kumar Pattnaik: *"Can we use data reduction techniques?"*
> *"Short answer is yes — but with a lot of caveats. It's **not a resounding universal yes**."*

It has its own plus and minus; a 30-minute discussion he didn't have time for. **Parked.**

### (c) ⚠️ Sridutt — never upload client data to public LLMs
> *"You upload your data on ChatGPT and all. **Please don't.**"*

- He cited an **ongoing case against HSBC** after an employee did this, with a European-region fine quoted at **€20 million**. *(Treat the specifics as an unverified anecdote; the principle is not in doubt.)*
- This is the domain of **Responsible AI / AI ethics**.

> *"Thousands of people use ChatGPT because it's free. How many know how to use it? 1%. That's the entire problem."*

---

## 30. How the Field Got Here

### Q&A — Sridutt: *"Where is AI still lagging?"*

```
  ~1900s   t-tests, z-tests — two-sample comparisons (Galton, Fisher era)
     │     "who is better, who is comparable"
     ▼
  ~1910s   ANOVA (Analysis of Variance) → multiple ANOVA
     │     more variables, more storage
     ▼
  WW I     Need to OPTIMISE (rationing in Europe — how much meat/rice per family)
     │     → REGRESSION: find Y with respect to X
     ▼
  WW II    the same trajectory continues
     ▼
  1973     Oil shock — Middle East embargo on the US
     │     → monetary policy work; SAS (Statistical Analysis Software)
     │       — "the precursor of Python"
     ▼
  1990s    DATA STORAGE BECOMES CHEAP   ← the real unlock
     │     → proper ML: decision trees, boosting, bagging, clustering, forecasting
     ▼
  2000s    Deep neural networks, NLP
     │     Social media (Facebook / Twitter / WhatsApp) + web scraping → data abundance
     ▼
  2011     NVIDIA GPUs → complex DL on non-numeric data: text, image, video, speech
     ▼
  2017     "Attention Is All You Need" (Google Brain) → TRANSFORMERS
     │     ⭐ "If you don't understand transformers, you will not understand LLMs
     │        at all. That is the gateway to large language models."
     ▼
  2025     Pre-trained LLMs. Nobody builds their own — costs ₹40–50 crore.
     │     Only Google / Microsoft / Amazon / Apple build; everyone else adapts.
     ▼
  2024→    AGENTS. "Still experimental. Failing most of the times, then passing.
           Still a thing developing. It's still not there — but it started."
```

**The storage-cost anecdote:** in 1990, storing **1 GB** cost roughly **half the price of a Beetle car in New York**. Today a phone holds far more. *That* is why ML happened when it did, and not earlier.

### ⚠️ Historical accuracy flags
The pedagogical arc is fine; several details are not. Don't repeat these in an exam or interview:

| Claim as told | Correct version |
|---|---|
| SAS was created after the 1973 oil shock as a US policy response | **SAS Institute: 1976, North Carolina State University** (Barr, Goodnight, Sall, Helwig), out of an agricultural-statistics project |
| **Bill Gates and Steve Jobs interned at SAS** | **Never happened** |
| **Reagan** was president in 1973 | **Nixon** was. Reagan: 1981–89 |
| **Milton Friedman** chaired the Fed | He was an academic economist (Chicago). **Arthur Burns** chaired the Fed then |
| **"Irving Fisher"** doing t-tests | **Ronald A. Fisher**, the statistician. Irving Fisher was an economist |
| GPUs unlocked DL in 2011 | Usually dated to **CUDA (2006)** and **AlexNet (2012)** |

---

## 31. The Building Metaphor — What to Actually Be Good At

```
        ┌────────────────────────────────────────────────┐
        │  AGENTIC AI                                    │  ← still under
        │                                                │    construction
        ├────────────────────────────────────────────────┤
        │  GENERATIVE AI / LLMs                          │  ← gated by
        │                                                │    TRANSFORMERS
        ├────────────────────────────────────────────────┤
        │  DEEP LEARNING   +   NLP        FIRST FLOOR    │
        │                                                │
        ├────────────────────────────────────────────────┤
        │  MACHINE LEARNING               GROUND FLOOR   │  ⭐
        │                                                │
        ╞════════════════════════════════════════════════╡
        │  STATISTICS  →  EDA             FOUNDATION     │  ⭐⭐ this session
        └────────────────────────────────────────────────┘
        ////////////////////////////////////////////////////
```

> *"These three are the blocks — especially **ML is your ground floor**, **DL is your first floor**, along with **NLP**."*

> *"If your ground floor and first floor are weak, no matter how tall a building you build, it will fall off."*

> *"You need to be **very good with** your machine learning, you need to be **very good with** your deep learning and your NLP, if you have to **survive and thrive and be successful** in what we call as data science and AI today."*

Note the bar: not "familiar with" but **"very good with"** — said twice. The GenAI layer at the top is **not a shortcut**; the ML and DL/NLP floors are where the actual competence lives.

---

## 32. Consolidated Summary Table — Every Technique Introduced

| Technique | Purpose | Changes the distribution? | When in the pipeline | Key caveat |
|---|---|---|---|---|
| **`describe()` / descriptive stats** | Understand each variable | No | Step 2 | Compare mean vs median for free normality info |
| **Univariate plots** (histogram, box plot) | Find single-variable outliers | No | Step 3 | **Cannot see bivariate outliers** |
| **Heatmap** | Correlation **numbers** | No | Step 4 | Doesn't show *which* points misbehave |
| **Scatterplot** | Find **bivariate outliers** | No | Step 4 | **Minimum N−1 plots** (49 for 50 vars) |
| **Missing-value imputation** (mean/median) | Fill gaps | Yes — check before/after | Step 5 | **Treat outliers FIRST** if using mean/median |
| **Binning / categorising / MICE** | Fill gaps | Yes — check before/after | Step 5 | Order vs outliers is then irrelevant |
| **Winsorization / capping** | Treat outliers | Yes — check before/after | Step 5 | Cap to the **percentile**, never the mean |
| **Transformation** (log, reciprocal, sqrt, Box-Cox) | Reshape a variable | **YES — that's the point** | Step 6, optional | Don't force normality (§11) |
| **One-hot encoding / dummies** | Categorical → numeric | n/a | Step 7, **before split** | Mandatory — can't model on strings |
| **Train/test split** | Enable validation | No | Step 8 | Random, **row-wise**, 70/30 |
| **StandardScaler** | Rescale to mean 0, SD 1 | **NO** | Step 9, **after split** | fit on TRAIN only; also **audits outlier treatment** via ±3 |
| **MinMaxScaler** | Rescale to 0–1 | **NO** | Step 9, **after split** | Default range is **0 to 1** |

---

## 33. Study Roadmap and Habits

### Study priorities, in his explicit order
| # | Topic | Why |
|---|---|---|
| **1** | **Hypothesis testing** | *"The most important parts of regression and time series are all built on hypothesis testing"* |
| **2** | **Probability distributions** — binomial (discrete), normal + **standard normal** (continuous) | Binomial + **odds** → logistic regression. Normal/standard normal → all of EDA, outlier rules, scaling interpretation |
| **3** | **Theory of estimation** — point, interval, **Maximum Likelihood Estimation** | *"Logistic regression uses MLE. If you don't understand that, you don't understand logistic"* |
| **+** | Basic probability (not axiomatic), sampling, **covariance** | Feeds 1–3; covariance underpins the correlation formula |

**Later, and hardest: time series forecasting** — needs both statistics *and* linear regression to be top-notch. *"It takes years for people to master it."*

> *"I learn statistics today and in 6 months I'm building an LLM — it doesn't happen. You can't. It's a serious subject. Give it time."*

### Habits to adopt starting now
```
   1. PEN AND PAPER beside you during EDA. Number every insight.
      "If you don't write it down, later it's all a fish market —
       you won't remember what connects to whom."

   2. BEFORE / AFTER distribution check on every single operation.        (§11)

   3. PLOT every predictor against the target — minimum N−1 scatterplots. (§17)

   4. ASK "is this outlier worth treating?" BEFORE treating it.           (§21)

   5. ASK "who are these people?" (domain context) before calling a
      value wrong.                                                       (§20)

   6. NEVER upload client data to a public LLM.                          (§29c)

   7. Aim for 5 INFERENCES PER OUTPUT, not 1 inference per 10 lines of code.
```

---

## 34. Verbatim Quote Bank

Kept for accuracy — these are the lines that carry the actual teaching.

1. *"70% of a project's time typically goes to EDA."*
2. *"There are no shortcuts in data science. Don't take any of them — it doesn't lead you anywhere."*
3. *"There are hundreds and thousands of people doing machine learning in the industry. Very few do it **properly**."*
4. *"There is a very fine line between manipulating and tampering."*
5. *"A very good result doesn't necessarily mean a right result. In data science, a **correct** result is more important than a **good** result."*
6. *"Give me 75% accuracy sustainable over 6 months in production, I'm happy. 95% sustainable for 7 days — I will not take it, I will not make it billable."*
7. *"Spurious correlation happens when you yourself are subconsciously biased even before you build a model. And that is the biggest problem we data scientists suffer from — because we are human beings, we are not machines."*
8. *"A smart data scientist doesn't write 100 lines of code. They write 20 lines, but draw 5 inferences from one output."*
9. *"Don't be a process-driven data scientist. Be a **thinking** data scientist."*
10. *"You only become a good data scientist if you are a perfectionist."*
11. *"There is no brownie point or Nobel Prize for treating outliers you found."*
12. *"You have to choose your battles. Don't fight every battle."*
13. *"Finding bivariate outliers is more important than finding univariate outliers."*
14. *"Your target variable is never one you don't control at all, and never one you control fully. It's always the one you have some control over but want to improve your control over."*
15. *"If you don't know stats, you don't know data science. Full stop."*
16. *"One is statistics, two is statistics, three is statistics."*
17. *"Definition, formula — that is college. Done. Now ask: where does this fit in ML, and how is it connected — and disconnected — from everything else."*
18. *"This is how we connect concepts."*
19. *"If it fails, you have to solve it. And to solve it you need to know **where** it failed."*
20. *"I learn statistics today and in 6 months I'm building an LLM — it doesn't happen. You can't. It's a serious subject. Give it time."*
21. *"Coding has become a school kid's job. The make-or-break is not there — it's somewhere else."*
22. *"If you find outliers and see they are borderline outliers, don't do anything to them."*
23. *"The model you build today, you will never build again in your life. Everything is unique — so every yardstick is unique."*
24. *"If your ground floor and first floor are weak, no matter how tall a building you build, it will fall off."*
25. *"These three are the blocks — especially ML is your ground floor, DL is your first floor, along with NLP."*
26. *"Because that is **the gateway to large language text models**."* (why NLP sits on the same tier as DL)
27. *"Agents have been failing most of the times, then passing, to be honest. So it is still a thing developing. **It's still not there.** It started, right?"*
28. *"You need a fairly decent understanding of **statistics** in terms of getting through to **EDA**."*
29. *"It's a temptation of having biryani every day. Not good for your health."* (on forcing normality)
30. *"Imputing an outlier with a central value doesn't make sense. The central value is somewhere in Delhi, the outlier is in London."*
31. *"That's the First World War. The Second World War starts after that."* (on reaching the first model iteration)
32. *"What I tried is that… **for you to connect a few things**. More than that, what I hope is that you sort of **enjoyed** the part. That is most important for me."*

---

## 35. Administrative / Housekeeping Points

- **Materials he promised to share** (routed via the program team):
  - The **EDA notebooks** — he has 4–5 of them
  - A **PPT on missing-value and outlier best practices across 4–5 industries**
- He planned **2 hours**; the session ran **≈ 2h 21m** and he cut discussion points at the end for time.
- Session was in the **evening**; dated by the instructor as **2025** (*"as I speak in 2025"*).
- The `.describe()` walkthrough on **real Python output** was cut for time — *"maybe we'll catch up at some other time."*
- Jithu's Colab/big-data question was partly cut off by internet issues and should be re-raised.
- He stated his objective plainly: **connecting concepts and enjoying the session**, above completeness of coverage.

---

## 36. Topics Explicitly Deferred to Later Sessions

1. **Data reduction techniques** — the plusses and minuses (~30-minute discussion, parked)
2. **Heteroscedasticity** — bivariate outliers are its main source; deferred to the linear-regression session
3. **Covariance → correlation** derivation (self-study)
4. **Why fit the scaler on train only** — deliberate thinking homework, answer withheld (§24)
5. **One-hot encoding / dummy variables** — skipped in the demo, coming in feature engineering, *"maybe tomorrow itself"*
6. **`.describe()` on real Python output** — ran out of time
7. **Hypothesis testing, probability distributions, estimation theory / MLE** — self-study priorities (§33)
8. **Time series forecasting** — the hardest topic; needs statistics and linear regression both solid first
9. **Transformers** — the gateway to LLMs, and the single most important thing to understand later (§30)
