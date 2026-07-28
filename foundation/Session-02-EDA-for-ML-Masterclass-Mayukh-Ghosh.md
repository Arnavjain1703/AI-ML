# Session Notes — AI/ML Certification Course (Session 2)
**Instructor:** Mayukh Ghosh
**Session 2:** Exploratory Data Analysis (EDA) — from raw data to a model-ready dataset

> **How to read this file.** Every technical word is defined the first time it appears. Sections 1–8 build the vocabulary; sections 9–20 are the actual EDA workflow; section 21 puts the whole thing together as one ordered checklist. If you only read one section, read **§21**.

---

## 0. The Vocabulary You Need First

Everything in this session is built from these eight words. Learn these and the rest of the file reads easily.

| Term | Plain-English meaning | Example |
|---|---|---|
| **Variable** | One **column** in your data | `Age`, `Salary` |
| **Observation** | One **row** in your data | one customer, one employee |
| **Target variable (Y)** | The column you are **trying to predict** | `AmountSpent` |
| **Predictor / feature (X)** | A column you use **to make** that prediction | `Age`, `Salary`, `Location` |
| **Distribution** | The **shape** of a column — where its values pile up and how they spread out | "most salaries are ₹10–20 L, a few are much higher" |
| **Outlier** | A value that is **very far from the rest** | a ₹75 L salary among ₹14–20 L salaries |
| **Missing value** | An **empty cell** | a customer with no age recorded |
| **Univariate / Bivariate** | Looking at **one** column at a time / **two** columns **together** | histogram / scatterplot |

```
                     A DATASET
              ┌──────┬──────┬────────┬─────────────┐
              │ Age  │Salary│Location│ AmountSpent │  ← VARIABLES (columns)
              ├──────┼──────┼────────┼─────────────┤
   OBSERVA-   │  28  │  14  │  Far   │     100     │
   TIONS      │  32  │  19  │  Near  │      45     │
   (rows)     │  35  │  20  │  Far   │     130     │
              └──────┴──────┴────────┴─────────────┘
                └──── PREDICTORS (X) ───┘  └── TARGET (Y) ──┘
```

---

## 1. What is EDA, and How is it Different from Preprocessing?

### The two words

| Term | What it means | The question it answers |
|---|---|---|
| **EDA** (Exploratory Data Analysis) | **Exploring** the data — looking at it, plotting it, understanding it | *"What do I actually have?"* |
| **Data Preprocessing** | **Preparing** the data — fixing, filling, capping, rescaling | *"How do I make it usable?"* |

Strictly, exploration comes first and preparation follows. **But in industry the two words are used interchangeably** — people say "EDA" to mean both. This session treats them as one continuous activity, because in practice you loop between them.

```
   Strictly speaking:   EDA (explore) ──► Preprocessing (prepare) ──► Model

   Industry usage:      └─────────── all called "EDA" ───────────┘
```

*(This distinction came up as a question in class and the correction was accepted as technically right.)*

### Why this matters so much
EDA takes **60–70% of the total time** on a real project. Not because it's slow, but because **the work multiplies by the number of columns**:

```
    1 column   →  1× the work  (plot it + check missing + check outliers + decide transform)

   50 columns  →  50× that work
                  + 49 more plots, each column against the target   ← this is where time goes
```

Only one operation in the whole workflow is "do it once for everything" — scaling (§17). Everything else is **per column**.

---

## 2. Why You Cannot Skip Statistics

> *"Doing EDA is hazardous, it's risky, if you're vague about normal distribution, correlation, skewness. You will misinterpret a lot of things."*

The reason is simple: **EDA is a sequence of judgement calls**, and every one of those calls is a statistics question.

| The decision you must make | The statistics you need |
|---|---|
| "Is this value an outlier or a real extreme?" | Distributions, standard deviation |
| "Should I fill this gap with the mean or the median?" | Central tendency, skewness |
| "Do these two columns actually relate?" | Correlation |
| "Did my fix improve the data or damage it?" | Distributions, before/after comparison |

> *"EDA is not about drawing diagrams or finding some reasons to impute missing values and outliers. It's much more comprehensive than that."*

### The five topics, in the order they build on each other

```
                    INFERENTIAL STATISTICS
                            │
              ┌─────────────▼─────────────┐
              │  1. PROBABILITY           │  the basics — how likely is something
              └─────────────┬─────────────┘
              ┌─────────────▼─────────────┐
              │  2. PROBABILITY           │  binomial (for yes/no outcomes)
              │     DISTRIBUTIONS         │  normal (for continuous values)
              └─────────────┬─────────────┘
              ┌─────────────▼─────────────┐
              │  3. SAMPLING              │  how to pick a subset fairly
              └─────────────┬─────────────┘
              ┌─────────────▼─────────────┐
              │  4. ESTIMATION            │  guessing a population value
              └─────────────┬─────────────┘      from a sample (incl. MLE)
              ┌─────────────▼─────────────┐
              │  5. HYPOTHESIS TESTING    │  proving a claim is real,
              └───────────────────────────┘      not just luck
```

### What each one is used for later

| Topic | What it unlocks | Why |
|---|---|---|
| **Binomial distribution** + **odds** | **Logistic regression** | Logistic regression predicts yes/no outcomes, which is exactly what the binomial distribution describes. "Odds" and "probability" are two ways of saying the same thing |
| **Normal + standard normal** | **All of EDA** | Outlier rules, scaling, skewness judgement all assume you know this shape |
| **Estimation, especially MLE** (Maximum Likelihood Estimation — the method of picking the parameter values that make your observed data most likely) | **Logistic regression's engine** | It is literally how logistic regression is fitted |
| **Hypothesis testing** | **Regression and time series** | *"The most important parts of regression and time series are all built on hypothesis testing"* |

### How to study it (this was asked directly in class)
**Not** the college way — memorising a definition and a formula. *"No one cares, everyone can Google it."* Instead, for every concept ask two questions:

1. **Where does this fit in my ML journey?** (which model or step actually uses it)
2. **How is it connected to the concepts around it — and how is it *not*?**

> *"There are 3 topics you should concentrate on: one is statistics, two is statistics, three is statistics."*

---

## 3. Descriptive Statistics — Reading `df.describe()`

**Descriptive statistics** = numbers that summarise a column. In Python, `df.describe()` gives you most of them in one shot. This is always your **first look** at data.

### What each number tells you

| Measure | What it is | What it's for |
|---|---|---|
| **Mean** | The average | The centre — **but it gets dragged by outliers** |
| **Median** | The middle value when sorted | The centre — **ignores outliers**, so it's safer |
| **Mode** | The most frequently occurring value | The centre for categories |
| **Min / Max** | Smallest and largest values | **Fastest way to spot an absurd value** |
| **Standard deviation (SD)** | Typical distance of a value from the mean | How spread out the column is |
| **Variance** | SD squared | Same idea as SD, different units |
| **Quartiles Q1, Q2, Q3** | The values at 25%, 50%, 75% of the sorted data | Used to build box plots and outlier fences |
| **Percentile** | The value below which X% of data falls | The 99th percentile is used for capping (§16) |
| **Skewness** | How lopsided the shape is | See §5 |
| **Kurtosis** | How peaked the shape is | See §6 — low priority |

### ⭐ A free normality check hiding in `describe()`
You don't need a plot to get a first read on shape. Just compare two numbers:

```
   mean ≈ median   →  the column is SYMMETRIC (roughly normal)      ✓ healthy
   mean >  median   →  RIGHT-skewed — a few big values pulling the mean up
   mean <  median   →  LEFT-skewed  — a few small values pulling it down
```

**Why this works:** the median doesn't care about extreme values (it's just the middle one), but the mean does. So **the gap between them is a measure of how much the extremes are distorting your column.**

In a truly normal distribution mean = median = mode exactly. In real data, *"within 5% of each other and we're pretty happy."*

---

## 4. The Core Demonstration — What ONE Bad Value Does

This was the anchor example of the whole session. A tiny sheet: **20 rows, 2 columns** — `Age` and `Salary` (in ₹ lakhs).

### The planted problem

```
   ┌──────┬──────┬────────────┐
   │ Row  │ Age  │ Salary(₹L) │
   ├──────┼──────┼────────────┤
   │ ...  │  28  │     14     │
   │ ...  │  32  │     19     │
   │ ...  │  35  │     20     │
   │  18  │  30  │  ██ 75 ██  │  ← a 30-year-old earning ₹75 L,
   └──────┴──────┴────────────┘     when peers earn ₹14–20 L

   That is 1 row out of 20  =  5% of the data.
```

### What happened when 75 was changed to a sensible 25

| Statistic | With the outlier (75) | After fixing to 25 | What changed |
|---|---|---|---|
| **Correlation** (salary vs age) | **≈ 0.46** → reads as *"barely related"* | **≈ 0.83** → *"strongly related"* | **+37 points** — nearly doubled |
| **Skewness** | **+4.11** → severely lopsided | **≈ 1.1** → manageable | dropped ~75% |
| **Standard deviation** | **≈ 13** | **≈ 3** | collapsed |
| **Mean / Median** | 16–17 / 13 → far apart | **14 / 13.5** → nearly equal | became **symmetric** |

```
        BEFORE (with 75)                      AFTER (75 → 25)
   correlation = 0.46  "no relationship"   correlation = 0.83  "strong"
   skewness    = 4.11  "badly lopsided"    skewness    = 1.1   "fine"
   SD          = 13    "huge spread"       SD          = 3     "tight"
   mean ≠ median                           mean ≈ median → normal
              ▲
              └── ONE row out of 20 caused ALL FOUR of these
```

### The two lessons

**Lesson 1 — outliers don't distort one number, they distort everything.** Correlation, spread, and shape all moved together. And since a **regression model** (§12) is built from exactly these quantities, the model would be distorted by the same amount.

**Lesson 2 — and this is the important one.** A student asked: *"aren't we just playing with the data to get a better number?"* The answer:

> *"I am **NOT** doing it because I want correlation and skewness to be better. I'm doing it because 75 is an extreme outlier that makes no sense in this cohort. I get a better result as a by-product — that is not the reason."*

**The justification must come from the data, not from the result you want.** §8 makes this into a rule you can apply.

---

## 5. Distributions and Skewness

A **distribution** is just the shape you get when you plot how often each value occurs.

```
                        ALL DISTRIBUTIONS
                                │
              ┌─────────────────┴─────────────────┐
         SYMMETRIC                        ASYMMETRIC (= SKEWED)
              │                                   │
       Normal / bell curve              ┌─────────┴─────────┐
       mean = median = mode      LEFT / NEGATIVE      RIGHT / POSITIVE
       skewness ≈ 0              long left tail       long right tail
                                 skewness < 0         skewness > 0
```

### The normal distribution
The reference shape everything else is compared against.

- **Bell-shaped and symmetric** — split it down the middle and you get **50% each side**.
- **mean = median = mode**, all at the centre.
- **Important:** normal distributions can look very different from each other — taller, flatter, wider. What makes a distribution normal is the **symmetry**, not the steepness.

```
   NORMAL (symmetric)                  RIGHT-SKEWED (positive)

  Freq                                Freq
   ▲                                   ▲
   │        ███                        │  ███
   │     ██████████                    │  ██████
   │  ████████████████                 │  █████████
   │  ████████████████                 │  ██████████████
   └──────────┬──────────►             └──────────────────────────►
              │                           ▲              ▲
         50%  │  50%                      │              │
   mean = median = mode                 most of      LONG RIGHT TAIL
                                        the data     (the ₹75 L salary)

                                     mode < median < mean
                                     salary column: skewness = +4.11
```

### How to read a skewness number

| Skewness | What it means | What to do |
|---|---|---|
| **≈ 0** | Symmetric — *"data is under control, values well distributed"* | Nothing. You're lucky |
| **> +1** | **Right-skewed** — a few unusually large values | Investigate those large values |
| **< −1** | **Left-skewed** — a few unusually small values | Investigate those small values |

### ⚠️ Reality check before you panic
**About 80% of real-world columns are skewed.** Ad spend, marketing spend, sales — *"either I've done less or more."* Expect `|skewness| > 1` as normal life. **Perfectly normal real data is rare.**

Asked directly whether lower skewness is better: **yes — lower |skewness| means healthier data and less work for you.**

---

## 6. Kurtosis (know the words, then move on)

**Kurtosis = how peaked the distribution is.** Three names to recognise:

```
   LEPTOKURTIC             MESOKURTIC (normal)        PLATYKURTIC
   sharp, narrow peak      the reference shape        flat, no real peak

  Freq                     Freq                      Freq
   ▲                        ▲                          ▲
   │      ███               │        ███               │
   │      ███               │     ██████████           │  ██████████████
   │      ███               │  ████████████████        │  ██████████████
   │   █████████            │  ████████████████        │  ██████████████
   └───────────────►        └──────────────────►       └──────────────────►

   kurtosis > 0             kurtosis ≈ 0               kurtosis < 0
```

> **Priority: low.** Kurtosis belongs to **Statistical Quality Control** and **Design of Experiments**, not to your ML workflow. *"In your machine learning you will not need that depth."* Recognise the three words in an interview; don't spend study time here.

---

## 7. Correlation

**Correlation = the degree to which two variables move together.** Not "are they related" — *how strongly*.

### The mental model
Imagine measuring the height and weight of 100 people:

> *"I am not interested in your height. Neither am I interested in your weight. You are fat, thin, tall, short — I simply don't care. You are just one data point, one entry of the hypothesis I'm trying to prove."*

**The individual rows are not the point.** The only question is: *when height goes up, does weight tend to go up too?*

### The scale — always between −1 and +1

```
   −1 ─────────── −0.7 ───────── 0 ───────── +0.7 ─────────── +1
 perfect          strong      NO / weak      strong         perfect
 opposite         opposite    relationship   together       together
                            (−0.3 … +0.3)
```

| Value | Reading | Example |
|---|---|---|
| **+0.8 to +1.0** | Very strong, same direction | Height ↑ → weight ↑ (taller people carry more mass) |
| **−0.8 to −1.0** | Very strong, opposite directions | Pollution ↑ → air quality ↓ |
| **−0.3 to +0.3** | **Weak or none** | Two unrelated columns |

*(Underneath, correlation is calculated from **covariance** — a related measure of joint variation. Worth looking up separately; not needed to use correlation.)*

### ⭐ The interpretation duty
A correlation number is not automatically true. **It must agree with what the columns actually mean.**

> *"If the value says something and the variables mean something else — there is something wrong."*

If pollution and air quality come out *positively* correlated, you don't have a discovery — you have **bad or biased data**, and you go and check.

**And the practical version of this rule:** if a correlation is **very strong but unlikely, double check it.**

---

## 8. Spurious Correlation, and Why *You* Are the Risk

**Spurious = false.** A **spurious correlation** is a relationship that looks real in your output but isn't — usually because of how *you* set the analysis up.

### The story that makes it stick
A PhD defence at Yale:

```
   THE CANDIDATE'S THESIS
   Question:  did a ship sink in the Atlantic?  (a yes/no prediction problem)
   Data used: 40–50 years of weather records — but ONLY for the days ships SANK
   Finding:   on those days the weather was bad
              →  "ships sink in bad weather. Strong relationship."
                              │
                              ▼
   A PANEL MEMBER: "Give me 15 days, I'll find the opposite."
   Data used: ONLY the days when NO ship sank — every ship sailed fine
   Finding:   on 46% of those days the weather was WORSE,
              measured by the candidate's own parameters
                              │
                              ▼
   VERDICT: the original conclusion is NOT VALID.
            The maths was fine. The SAMPLE was biased by design —
            it only ever looked at ONE side of the outcome.
```

Asked which of the two was correct: **neither is the "right answer."** The first is simply **invalid**.

### The lesson: bias enters *before* you write any code

> *"Spurious correlation happens when you yourself are subconsciously biased even before you build a model. And that is the biggest problem we data scientists suffer from, because we are human beings, we are not machines."*

> *"Even if you know your sales are driven by advertisement and marketing — you cannot bring that prior knowledge in when you're building the model. Because then, subconsciously, you will make the EDA drive towards your beliefs."*

### Can you measure your own bias?
Asked in class. The answer was **no** — *"not even agentic AI has been able to do that. It's not a stupid question — the question just doesn't have an answer."*

This has a real consequence:

```
   Human judgement can never be fully bias-free
                    │
                    ▼
   100% accuracy is NOT achievable — ever
                    │
                    ▼
   Everything we cannot explain goes into the ERROR TERM

              y = m₁x₁ + m₂x₂ + … + c + e
                                          ▲
                                          └── the "everything else" bucket
```

---

## 9. ⭐ The Golden Rule: Manipulation vs Tampering

This is the single most important rule in the session. Everything you do in EDA changes the data. **How do you know whether you improved it or corrupted it?**

> *"There is a very fine line between manipulating and tampering. In your initial few projects, you will actually tread over the line — in order to get high accuracy, because that's what plays in our mind."*

### The procedure — check the distribution BEFORE and AFTER every single operation

```
      ANY EDA operation  (fill a gap / cap an outlier / transform / encode)
                              │
                 ┌────────────┴────────────┐
              BEFORE                     AFTER
        record the distribution    record the distribution
                 └────────────┬────────────┘
                              ▼
                    compare, then judge:
   ┌────────────────────────────────────────────────────────┐
   │  RULE 1:  Don't change the distribution by MUCH        │
   │           (rough guide: ≤ ~20–30% shift)               │
   │                                                        │
   │  RULE 2:  Any change must move TOWARD normality        │
   └────────────────────────────────────────────────────────┘

   Changed a LOT              →  you are TAMPERING
   Moved AWAY from normal     →  your LOGIC is wrong — rethink it
   Small move toward normal   →  legitimate manipulation ✓
```

### The test that decides it
The **reason** must be a fact about the data — never the result you were hoping for.

> *"If you have the right logic to do it, and in the process you get a better result — fine. If you manipulate data to serve your own nice/short work — that is tampering."*

### ⚠️ The trap: forcing normality
Normal columns are convenient, because then outlier detection is a one-line formula (§15). So people log-transform everything to force normality.

> *"That is something you should be very, very wary of."*
> *"It's a temptation of having biryani every day. Not good for your health."*

**If a column is already normal, great. If it isn't, don't torture it into being normal.**

### Why the stakes are real — what a tampered model does in production

```
   Over-engineered EDA
        │
        ▼
   Excellent accuracy on your own data (95%+)
        │
        ▼
   Business stakeholders love the slides
        │
        ▼
   Deployed to production (monitoring, cloud costs, pipelines)
        │
        ▼
   NEW DATA ARRIVES  →  THE MODEL FAILS
        │
        ▼
   Rebuild → fails again after 2 runs
        │
        ▼
   Client: "₹15–20 lakhs spent, no return. We're scrapping it."
        │
        ▼
   Model dead within 3–6 months
```

> *"A very good result doesn't necessarily mean a right result. In data science, a **correct** result is more important than a **good** result."*

> *"Give me 75% accuracy sustainable over 6 months in production — I'm happy. Give me 95% sustainable for 7 days — I will not take it, I will not make it billable."*

---

## 10. ⭐⭐ Univariate vs Bivariate Outliers

**This is the highest-value idea in the session, and it's one most courses skip.**

### Two kinds of outlier

| Type | Definition | How you find it |
|---|---|---|
| **Univariate outlier** | A value that is extreme **in its own column** | Box plot, IQR, distance from mean, binning |
| **Bivariate outlier** | A value that is **perfectly normal in its own column**, but **impossible in combination** with another column | **Only a scatterplot** |

*(The instructor coined the term "bivariate outlier" himself — you may not find it in textbooks under that name, but the phenomenon is real and is what causes heteroscedasticity, §10.4.)*

### The interview question he asks candidates
> *"If I have already drawn a **heatmap**, do I still need a **scatterplot**? Does the scatterplot give me anything extra?"*

**Answer: yes.** A **heatmap** shows you correlation *numbers* for every pair of columns. It cannot show you *which individual rows* are misbehaving. Only a scatterplot can.

### The demonstration
Two extra rows were added to the earlier sheet:

| Age | Salary (₹L) | Looking at salary alone | Looking at both together |
|---|---|---|---|
| 50 | 50 | Extreme! | **Fine** — ₹50 L at age 50 is plausible |
| 50 | 62 | Extreme! | **Fine** — ₹62 L at age 50 is plausible |
| **30** | **75** | Extreme! | **THE REAL PROBLEM** — ₹75 L at age 30 is not plausible |

| The view you take | What you conclude |
|---|---|
| **`Salary` column alone** | **Three** outliers: 50, 62, 75 (mean is ~18.9, so all are 2–3× away) |
| **`Age` column alone** | **Zero** outliers. Every age is plausible. Age 30 will *never* look extreme |
| **Both columns together** | **One** genuine problem: age 30 with ₹75 L |

```
   Salary
   (₹L) ▲
        │
     75 │        ★  ◄── BIVARIATE OUTLIER (30, 75)
        │            Invisible to BOTH box plots.
        │            An odd COMBINATION, not an odd VALUE.
     62 │                                    ○  (50, 62)  ← fine
        │
     50 │                                    ○  (50, 50)  ← fine
        │
        │              ●  ●
     20 │        ●  ●
     14 │  ●  ●
        └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──► Age
           25    30    35    40    45    50

   ●  = the normal cohort
   ○  = high salary WITH high age  →  follows the trend, VALID
   ★  = high salary with LOW age   →  contradicts the trend, ANOMALY
```

### The four rules that follow

| # | Rule |
|---|---|
| **1** | **Bivariate outliers matter more than univariate ones** — because every univariate method (box plot, IQR, distance-from-mean, binning) is **structurally blind** to them |
| **2** | **Plot every predictor against the target.** Anything in the **top-left or bottom-right** — the "wrong" corners — behaves differently from everything else |
| **3** | **High-X with high-Y is fine.** *"It's a problem when there's a problem in between them"* — i.e. only when the combination **contradicts the overall trend** |
| **4** | **Cost of ignoring them:** they drag model accuracy down, and they are the **main source of heteroscedasticity** — a condition where prediction errors get systematically bigger (or smaller) across the range, which breaks a core assumption of linear regression |

---

## 11. How Many Plots Do You Actually Have to Draw?

A question posed to the class, worth internalising.

### Setup
You have **50 columns**. One of them is the target `Y`. The other **49 are predictors `X`**.

### Why regression only cares about one column
The straight line you learned at school:

$$y = mx + c$$

Add more predictors and an error term, and you have **linear regression**:

$$y = m_1x_1 + m_2x_2 + \dots + m_{49}x_{49} + c + e$$

| | What it focuses on |
|---|---|
| **Correlation** | The relationship between **any two** columns |
| **Regression** | **One** column only — the target **Y** |

> *"I don't care about the other 49. Why do I have them? Because I think they will potentially explain my target."*

### The answer

```
   50 columns  =  1 target Y  +  49 predictors X

   MINIMUM number of bivariate plots  =  49
                                          (every X plotted against Y, one at a time)

   ── skip any of them and you will miss a bivariate outlier ──

   MAXIMUM?  Unbounded — you can also plot X against X, use pair plots, etc.
```

**General rule: minimum plots = (number of columns − 1).**

> *"It's not about whether you give the correct answer, it's about how you think through it."*

---

## 12. Which Comes First — Outliers or Missing Values?

Called *"a million-dollar question in the industry."* Two answers came from the class and **both were accepted** — because the right answer depends on *how* you plan to fill the gaps.

### The decision rule

```
        Are you filling missing values with a STATISTICAL MEASURE?
                    (the mean especially, or the median)
                              │
              ┌───────────────┴───────────────┐
             YES                              NO
              │                    (binning / categorising /
              ▼                     interpolation / MICE)
   ┌──────────────────────────┐               │
   │  OUTLIERS FIRST, ALWAYS  │               ▼
   └──────────────────────────┘   ┌────────────────────────────┐
   Because the outlier corrupts    │  Order doesn't matter.     │
   the mean BEFORE you use it      │  Do the BIGGER problem     │
   to fill the gaps.               │  first (higher %).         │
                                   └────────────────────────────┘
```

**Why "outliers first" when using the mean:** go back to §4. With the ₹75 L outlier present, the mean was 16–17. Without it, 14. If you fill empty salary cells with 16–17, **you have just injected the outlier's distortion into every row you filled.**

**Why "bigger problem first" when order is free:**
> *"You want to have control of the data as soon as possible. You don't want to keep control away from you for longer by doing other things first — you need to see the real figures, the right figures, as soon as possible."*

### Terms used above, defined
| Method | What it does |
|---|---|
| **Mean/median imputation** | Fill the gap with the column's average or middle value |
| **Binning / categorising** | Convert to ranges ("young / middle / old") so a missing value becomes just another category |
| **Interpolation** | Estimate the gap from neighbouring values (useful for time-ordered data) |
| **MICE** | Multivariate Imputation by Chained Equations — predict each missing value from the *other* columns, repeatedly. The advanced option |

---

## 13. Finding Outliers, Method 1 — The 3-Sigma Rule

**Sigma (σ) = standard deviation.** The rule says: in a normal distribution, almost everything sits within 3 standard deviations of the mean.

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
            └── only 0.27% total lies outside, split across both tails ──┘
```

### Worked example
```
   mean = 50,  SD = 10
   → normal range = 50 − 30  to  50 + 30  =  20 to 80
   → anything below 20 or above 80 is a POTENTIAL outlier
```

### ⚠️ The precondition that people forget
**This only works if the distribution is normal / symmetric.** On a skewed column it does not hold. Since ~80% of real columns are skewed (§5), you often cannot use this directly — which is why the next method exists.

### ⚠️ A naming correction worth carrying
In class this was called **"Chebyshev's inequality."** An exam will separate the two:

| | Applies to | Guarantee within mean ± 3·SD |
|---|---|---|
| **Empirical rule** (68–95–99.7) | **Normal** distributions only | **≈ 99.73%** ← the figure used in class |
| **Chebyshev's inequality** | **Any** distribution | **At least 88.9%** (1 − 1/k², with k = 3) |

The 99.73% figure plus the "only if normal" caveat together describe the **empirical rule**. The caveat given in class was right; the name was loose.

---

## 14. Finding Outliers, Method 2 — Box Plots and the 1.5 × IQR Fence

**IQR = Interquartile Range = Q3 − Q1** — the range covering the middle 50% of your data.

$$\text{Lower fence} = Q1 - 1.5 \times IQR \qquad \text{Upper fence} = Q3 + 1.5 \times IQR$$

Anything outside those fences is flagged as an outlier.

### Why 1.5? (asked in class, and the answer was accepted)
**Because 1.5 is calibrated to reproduce roughly the same cut-off as the 3-sigma rule** — but using quartiles, which don't require the column to be normal.

> **The actual arithmetic, if you want it:** for a normal distribution, `Q3 + 1.5·IQR ≈ mean + 2.698σ`, leaving ≈ 0.7% of data outside. So the "magic" 1.5 was deliberately chosen to approximate the 3-sigma / 99.7% boundary. It is not arbitrary.

### ⚠️ Both of these methods are univariate
Box plots, IQR, distance-from-mean, binning — **every one of them looks at a single column at a time**, so **none of them can see a bivariate outlier** (§10). You need scatterplots *in addition*.

---

## 15. Treating Outliers — Winsorization / Capping

You've found an outlier. Now what? The worked example: an `Age` column in a banking dataset.

```
   min = 18            max = 105
   mean = 45           99th percentile = 85
   n = 100  (later 500)
```

### Step 1 — Detect the skew without plotting anything
min is 27 below the mean; max is 60 above it. **The max is much further out than the min → right-skewed.** No code needed.

> *"A smart data scientist doesn't write 100 lines of code. They write 20 lines, but draw 5 inferences from one output."*

### Step 2 — ⭐ Ask who these people are, BEFORE any formula
Is age 105 an error or a fact? **It depends entirely on the population:**

| Population | Life expectancy | Verdict on age 105 |
|---|---|---|
| **India** | ≈ 70 | **Absurd.** 105 is ~1.5× the average. No Indian bank would entertain a walk-in claiming 105 |
| **Japan / New Zealand / Scandinavia** | ≈ 92–94 | **Plausible.** Only ~10 above average. (He knows someone in the UK doing their own banking at **101**) |

> *"So I need to know: is this person from Tokyo, Copenhagen, or Mumbai? I need to know how **improbable** this value is."*

**Domain context decides whether a number is an error. Ask this before reaching for any statistical rule.**

### Step 3 — Cap it to a sensible boundary

| Term | Meaning |
|---|---|
| **Capping off** | The older term: replace the outlier with the nearest **sensible boundary value** |
| **Winsorization** | The modern name for the same thing. **This is what industry uses** |

```
   105  ──cap to the 99th percentile──►  85     ✓ CORRECT
   105  ──replace with the mean────────►  45     ✗ WRONG
```

**Why replacing with the mean is wrong:**
> *"Imputing an outlier with a central value doesn't make sense. The central value is somewhere in Delhi, the outlier is in London."*

You'd be moving the value 60 years instead of 20 — a huge distribution change, which by §9's rules is **tampering**, not manipulation.

---

## 16. ⭐ Choose Your Battles — Is the Fix Even Worth Doing?

Most people go straight from *"I found outliers"* to *"I treated outliers."* **There is a step in between:** *is it worth it?*

### The arithmetic
n = **500** rows, capping at **85**. The top 1% is **5 people**:

| Value | Capped to | How far it moved |
|---|---|---|
| 87 | 85 | 2 |
| 90 | 85 | 5 |
| 92 | 85 | 7 |
| 95 | 85 | 10 |
| 105 | 85 | 20 |
| | **Total movement** | **44** |

```
   Average distortion introduced across the dataset

        =  total movement / number of rows
        =  44 / 500
        =  0.088        ← not even 0.1
```

**Will 0.088 change whether `Age` matters in your model? No.** So don't do it.

Now compare with §4, where **one** value swung correlation by **37 points**. *That* was worth treating.

### The rule
```
   ┌────────────────────────────────────────────────────────────────┐
   │  BORDERLINE outliers  →  DO NOTHING. You are wasting time.      │
   │                                                                │
   │  EXTREME outliers     →  treat them: winsorize / cap /          │
   │                          categorise / dummy variable / drop     │
   └────────────────────────────────────────────────────────────────┘
```

> *"There is no brownie point or Nobel Prize for treating outliers you found."*
> *"You have to choose your battles. Don't fight every battle."*
> *"Don't be a process-driven data scientist. Be a **thinking** data scientist."*

---

## 17. Scaling vs Transformation — Two Different Things

These get confused constantly. A student said *"so we take a log of it"* and was corrected: **that's transformation, not scaling.**

| | What it does | Effect on the distribution's SHAPE |
|---|---|---|
| **Scaling** | Shrinks the **range** of values | **NO change to the shape** |
| **Transformation** | Log, reciprocal, square root, Box-Cox | **Changes the shape — that's the entire point** |

**Why scaling exists:** if `Salary` runs 10–100 and `Age` runs 18–60, many algorithms will treat salary as "more important" purely because its numbers are bigger. Scaling puts every column on a comparable footing.

**Why transformation exists:** to pull a skewed column toward normality — subject to §9's warning about overdoing it.

### The two scalers

| Method | Formula | Output |
|---|---|---|
| **StandardScaler** | $(X - \mu)/\sigma$ | mean becomes **0**, SD becomes **1** |
| **MinMaxScaler** | $(X - X_{min})/(X_{max} - X_{min})$ | everything squeezed into **0 to 1** |

> ⚠️ **Correction:** min-max was described in class as giving −1 to +1. The default `MinMaxScaler` gives **0 to 1** (a student said this correctly). You only get −1 to +1 by explicitly setting `feature_range=(-1,1)`.

---

## 18. Train/Test Split, and the Trap Called Data Leakage

### Why splitting exists
You cannot tell a client *"I built a model, it seems like it'll work next time."* You need **evidence** that it works on data it has never seen. So you hide part of your data from the model and test on it afterwards.

```
   Full data (say 1000 rows)
        │
        │  split RANDOMLY, by ROW
        │  (nothing to do with columns — a question asked in class)
        │
        ├──────────── 70%  =  700 rows ────────►  TRAIN — build the model on this
        │
        └──────────── 30%  =  300 rows ────────►  TEST  — locked away, unseen

   Then compare the error on TEST against the error on TRAIN:

        similar errors  →  the model generalises ✓
        much worse on TEST  →  go back, fine-tune, or restart from EDA ✗
```

| Question | Answer |
|---|---|
| **Why 70/30?** | Build on the larger chunk, but keep the test set big enough for the test to mean something — *"strike a middle ground"* |
| **Why random?** | *"If I make a biased split on my own, I will end up with spurious correlation"* (§8) |

### ⭐⭐ Data leakage — the mistake 90% of people make

**Data leakage = information from the test set sneaking into the training process**, making your model look better than it really is.

Here's how it happens with scaling. `StandardScaler` computes $z = (X-\mu)/\sigma$ — it needs a mean and an SD, **calculated from whatever data you hand it**.

```
   ✗ WRONG                                  ✓ RIGHT
   ┌─────────────────────────┐              ┌─────────────────────────┐
   │ Scale the WHOLE dataset │              │  SPLIT first, 70 / 30   │
   │ One μ and σ computed    │              └────────────┬────────────┘
   │ from all 1000 rows      │                           │
   └────────────┬────────────┘                  scaler.fit(TRAIN)
                │                              μ and σ from TRAIN ONLY
          THEN split 70/30                                │
                │                            ┌───────────┴───────────┐
                ▼                            ▼                       ▼
   The test rows already influenced    transform(TRAIN)      transform(TEST)
   the μ and σ used on train                  └──── same μ, σ ────┘
                │                                        │
                ▼                                        ▼
   Test set is NOT truly unseen              Test set is genuinely unseen
   →  your validation is MEANINGLESS          →  HONEST validation
   →  "outputs better than I deserve"
```

### The rule in code
```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit ONCE, on train only
X_test_s  = scaler.transform(X_test)        # transform only — never fit again
```

**Two words to keep straight:**
- **`fit`** = *learn* the parameters (compute μ and σ)
- **`transform`** = *apply* them to the numbers

> *"I will fit only once on train. With that result, I will transform both train and test — NOT fit separately and transform separately. **90% of people make a mistake there.**"*

### 🏠 Homework he set deliberately
**Why must the fit happen on train only?** He withheld the full answer on purpose — think it through over the next couple of sessions.

*(A working answer to check yourself against: `transform` must be the exact same function applied to both sets, and it must be built only from information the model is allowed to see. Re-fitting on test does two bad things at once — it leaks test statistics, and it applies a *different* mapping, so a value of 30 in train and 30 in test would no longer become the same scaled number.)*

---

## 19. ⭐⭐ Using Scaling to Check Your Own Outlier Work

This was the payoff of the session — an example of **connecting two concepts to get something neither gives you alone.**

### Step 1 — What the standard scaler actually produces
$$z = \frac{X - \mu}{\sigma}$$

The output follows the **standard normal distribution: mean = 0, SD = 1.** (That's the definition of "standard" normal.)

### Step 2 — Apply the 3-sigma rule to it
From §13, almost all data lies within mean ± 3·SD. Substitute mean = 0 and SD = 1:

$$0 \pm 3 \times 1 \;\longrightarrow\; [-3, +3]$$

### Step 3 — The diagnostic

> **For a column that is (a) normally distributed and (b) already outlier-treated, every value after standard scaling must fall between −3 and +3.**
>
> **If any value falls outside ±3, your outlier treatment was wrong or incomplete. Go back and use a better method.**

```
   A normal column, outlier treatment already done
                    │
                    ▼
            standard-scale it
                    │
                    ▼
   ┌──────────────────────────────────────────────────────┐
   │  all values within  −3 … +3   →  treatment was sound ✓│
   │  any value outside  −3 … +3   →  treatment was wrong ✗│
   └──────────────────────────────────────────────────────┘

   WHY this pins the blame on you:
     The distribution was GIVEN to you — you didn't choose it.
     The ONLY thing YOU did was the outlier treatment.
     So a violation can only be your treatment's fault.
```

### So scaling does three jobs, not one
| # | Job |
|---|---|
| 1 | Put columns on a comparable numeric footing (the obvious one) |
| 2 | Sequenced correctly, it's where **data leakage** is avoided (§18) |
| 3 | It **back-checks** your outlier treatment (this section) |

> *"This is how we connect concepts. Scaling is not a process limited to itself — it can help you assess other things in your EDA, which makes your EDA more robust."*

---

## 20. Before Any of This — Framing the Problem and Picking the Target

Everything above assumes you know **what you're predicting**. Clients don't tell you. They hand you a file and say *"do some analysis."*

> *"Do some analysis — I'll draw a bar graph and send it back to you. That is also analysis, right?"*

### The dataset used in class — an online retailer's customers

| Column | Meaning |
|---|---|
| `CustomerID` | Just an identifier — useless for modelling |
| `Age` | Old / middle / young |
| `Gender` | — |
| `OwnHome` | Owns a home or not |
| `Married` | Marital status |
| `Location` | **Near or far from a competitor's physical store** |
| `Salary` | The customer's salary |
| `Children` | Number of children in the household |
| `History` | How often they bought last year |
| `Catalogs` | How many catalogs **the business sent them** |
| `AmountSpent` | How much they spent on your product (USD) |

**Why `Location` matters:** if a furniture shop is downstairs from your customer, they'll walk down and see it in person. If the nearest one is 3–4 km away needing a car or bus, they'll buy online. **Near a competitor = bad for an online seller.**

### How to eliminate candidates

```
   Age, Gender, OwnHome, Married, Location, Salary, Children
        Ask: as the business owner, would PREDICTING this help me?
             "Can you give this customer a higher salary?"      → No
             "Can you make an old customer young?"              → No
        →  ZERO control  →  these can only ever be PREDICTORS (X)

   Catalogs
        →  YOU decide how many to send.  100% control  →  not a target either

   History, AmountSpent
        →  PARTIAL control: they've bought before, they'll likely buy again,
           and you WANT that number to go up   →  candidate targets

        →  AmountSpent WINS. It's revenue, and it's what you want to grow:
           "He spends $100 today; tomorrow I want him to buy for $200."
```

### ⭐ The rule to memorise

```
   ┌──────────────────────────────────────────────────────────────────┐
   │  Your TARGET is never something you control 0% of,               │
   │  and never something you control 100% of.                        │
   │                                                                  │
   │  It is always the thing you have SOME control over               │
   │  and WANT MORE control over.                                     │
   └──────────────────────────────────────────────────────────────────┘
```

> *"Because business wants results about only those things. The rest, they don't care."*

### What must happen before your first line of code

```
   1. Agree the PROBLEM STATEMENT           ── with stakeholders,
   2. Identify the TARGET VARIABLE             "you cannot do it alone"
   3. Check the other columns could plausibly explain it
   4. Decide whether it's even WORTH the time
   5. Proof of concept  →  stakeholder approval  →  sometimes a pilot
                            │
                            ▼
   … and ONLY NOW your first  pd.read_csv()
```

> *"It's not about wrong or right data. It's about whether what you're trying to do is possible on this data or not."*

---

## 21. ⭐⭐⭐ The Complete Pipeline — Raw Data to First Model

This was the answer to a direct question in class: *"take us through the journey from the raw data to the model."* **This is your checklist.**

```
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  0. GET THE DATA                                                          │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  1. FRAME THE PROBLEM + IDENTIFY THE TARGET VARIABLE               (§20)   │
 │     With stakeholders. "You cannot do it alone."                          │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  2. df.describe()  →  centre, spread, skewness, quartiles, percentiles     │
 │     Read the DISTRIBUTIONS. How healthy is each column?            (§3)    │
 │     ✏️ KEEP PEN AND PAPER. Number every insight 1, 2, 3, 4…               │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  3. UNIVARIATE ANALYSIS — histograms, box plots, crosstabs          (§14)  │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  4. BIVARIATE → MULTIVARIATE — scatterplots (min 49 for 50 cols),   (§10)  │
 │     heatmaps, pair plots if needed.                                (§11)  │
 │     ✅ CROSS-CHECK against your step-2 insights.                          │
 │        Matching → good. Not matching → look again.                        │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  5. OUTLIERS + MISSING VALUES                                (§12–16)      │
 │     Find them → judge if treatment is worth it → treat.                   │
 │     Order per §12. If not worth treating, LEAVE THEM.                     │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  6. TRANSFORMATION / FEATURE ENGINEERING — only IF required.        (§17)  │
 │     "Not mandatory. Every time you will not require this."                │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  7. DUMMY VARIABLES / ONE-HOT ENCODING                                     │
 │     Turn text categories into numbers. MANDATORY — you cannot model       │
 │     on strings. ⚠️ This happens BEFORE the split.                        │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  8. TRAIN / TEST SPLIT — 70/30, random, row-wise                   (§18)  │
 ├───────────────────────────────────────────────────────────────────────────┤
 │  9. SCALE — numeric columns only, AFTER the split.                  (§18)  │
 │     fit on TRAIN → transform TRAIN and TEST.                             │
 │     Then CONCATENATE scaled-numeric + category-dummies,                  │
 │     separately for train and separately for test.                        │
 ├───────────────────────────────────────────────────────────────────────────┤
 │ 10. → "This is the data which is ready for your FIRST ITERATION            │
 │        of the model."                                                     │
 └───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        "That's the First World War. The Second World War starts after that."
                                    │
                                    ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │ 11. Run the full model on TRAIN. Feature selection. Check every measure    │
 │     the model gives you.                                                  │
 │ 12. Loop between model and EDA — up and down, MANY times.                │
 │ 13. ONLY when the model is FINAL and its variables are FIXED               │
 │     →  compare against TEST.       ⚠️ NOT earlier. Not "just to peek."    │
 └───────────────────────────────────────────────────────────────────────────┘
```

### ⭐ The two ordering traps
| Trap | The rule |
|---|---|
| **Encoding vs scaling** | **Encoding/dummies BEFORE the split. Scaling AFTER the split.** |
| **When to touch the test set** | **Last.** Not while tuning. A student was corrected on exactly this |

### When is the model good enough?
Asked in class. There is **no fixed number of iterations**. You work to **yardsticks** — a target accuracy, an error you must get under. And:

> *"The model you build today, you will never build again in your life. Everything is unique — so every yardstick is unique."*

---

## 22. Practical Constraints (Q&A)

### (a) "My data has millions of rows and Colab can't load it."
| Option | Verdict |
|---|---|
| **Paid Colab** (~₹1000/month) | Gives GPU and parallel compute — his first suggestion |
| **Sample down to 10k–50k rows** | ✓ Allowed — **but the sample must mirror the full data's distribution on every column.** Otherwise it's pointless |
| **Work on a subset / chunks** | ✓ Fine — **but call it a PILOT.** You cannot claim the job is done |
| **The real job** | Needs a **GPU or cloud support** |

### (b) "Can we use data reduction techniques?"
> *"Short answer is yes — but with a lot of caveats. It's **not a resounding universal yes**."*

It has real trade-offs; a 30-minute discussion there wasn't time for. **Parked for later.**

### (c) ⚠️ Never upload client data to a public LLM
> *"You upload your data on ChatGPT and all. **Please don't.**"*

An ongoing case against **HSBC** was cited, after an employee did this — with a European fine quoted at **€20 million**. *(Treat the specific numbers as an unverified anecdote; the principle is not in doubt.)* This falls under **Responsible AI / AI ethics.**

> *"Thousands of people use ChatGPT because it's free. How many know how to use it? 1%. That's the entire problem."*

---

## 23. What to Actually Be Good At

```
        ┌────────────────────────────────────────────────┐
        │  AGENTIC AI                                    │  ← still
        │                                                │    experimental
        ├────────────────────────────────────────────────┤
        │  GENERATIVE AI / LLMs                          │  ← gated by
        │                                                │    TRANSFORMERS
        ├────────────────────────────────────────────────┤
        │  DEEP LEARNING   +   NLP        FIRST FLOOR    │
        ├────────────────────────────────────────────────┤
        │  MACHINE LEARNING               GROUND FLOOR   │  ⭐
        ╞════════════════════════════════════════════════╡
        │  STATISTICS  →  EDA             FOUNDATION     │  ⭐⭐ this session
        └────────────────────────────────────────────────┘
        ////////////////////////////////////////////////////
```

> *"If your ground floor and first floor are weak, no matter how tall a building you build, it will fall off."*

> *"You need to be **very good with** your machine learning, **very good with** your deep learning and your NLP, if you have to survive and thrive and be successful in what we call as data science and AI today."*

Note the bar: not *"familiar with"* but **"very good with"** — said twice. **The GenAI layer at the top is not a shortcut.** And the gateway to LLMs is **transformers** — *"if you don't understand transformers, you will not understand LLMs at all."*

---

## 24. Consolidated Summary Table — Every Technique

| Technique | What it's for | Changes the distribution? | Where in the pipeline | The one thing to remember |
|---|---|---|---|---|
| **`describe()`** | Understand each column | No | Step 2 | mean vs median = free normality check |
| **Histogram / box plot** | Find **univariate** outliers | No | Step 3 | **Blind to bivariate outliers** |
| **Heatmap** | Correlation **numbers** | No | Step 4 | Doesn't show *which rows* misbehave |
| **Scatterplot** | Find **bivariate** outliers | No | Step 4 | **Minimum (N−1) plots** — 49 for 50 columns |
| **Mean/median imputation** | Fill gaps | Yes — check before/after | Step 5 | **Treat outliers FIRST** |
| **Binning / MICE / interpolation** | Fill gaps | Yes — check before/after | Step 5 | Then outlier order doesn't matter |
| **Winsorization / capping** | Treat outliers | Yes — check before/after | Step 5 | Cap to the **percentile**, never the mean |
| **Transformation** (log, sqrt, Box-Cox) | Reshape a column | **YES — that's the point** | Step 6, optional | Don't force normality (§9) |
| **One-hot encoding** | Text → numbers | n/a | Step 7, **before split** | Mandatory; can't model on strings |
| **Train/test split** | Enable honest validation | No | Step 8 | Random, **row-wise**, 70/30 |
| **StandardScaler** | mean 0, SD 1 | **No** | Step 9, **after split** | `fit` on TRAIN only; also **audits outliers** via ±3 |
| **MinMaxScaler** | Squeeze to 0–1 | **No** | Step 9, **after split** | Default range is **0 to 1** |

---

## 25. Your Study Plan and Working Habits

### Study priorities, in his explicit order
| # | Topic | Why it's on the list |
|---|---|---|
| **1** | **Hypothesis testing** | *"The most important parts of regression and time series are all built on hypothesis testing"* |
| **2** | **Probability distributions** — binomial, normal, **standard normal** | Binomial + odds → logistic regression. Normal → all of EDA and outlier rules |
| **3** | **Estimation theory** — point, interval, **MLE** | *"Logistic regression uses MLE. If you don't understand that, you don't understand logistic"* |
| **+** | Basic probability, sampling, **covariance** | Feeds 1–3; covariance sits under the correlation formula |

**Hardest, save for later: time series forecasting.** Needs both statistics *and* linear regression to be solid. *"It takes years for people to master it."*

> *"I learn statistics today and in 6 months I'm building an LLM — it doesn't happen. You can't. It's a serious subject. Give it time."*

### Seven habits to start now
```
   1. PEN AND PAPER beside you during EDA. Number every insight.
      "If you don't write it down, later it's all a fish market —
       you won't remember what connects to whom."

   2. CHECK the distribution BEFORE and AFTER every operation.        (§9)

   3. PLOT every predictor against the target — minimum (N−1) plots.  (§11)

   4. ASK "is this outlier worth treating?" before treating it.       (§16)

   5. ASK "who are these people?" before calling a value wrong.       (§15)

   6. NEVER upload client data to a public LLM.                       (§22c)

   7. Aim for 5 INFERENCES PER OUTPUT — not 1 inference per 10 lines of code.
```

---

## 26. The Ten Lines Worth Memorising

1. *"There are no shortcuts in data science. Don't take any of them — it doesn't lead you anywhere."*
2. *"There is a very fine line between manipulating and tampering."*
3. *"A very good result doesn't necessarily mean a right result. In data science, a **correct** result is more important than a **good** result."*
4. *"Give me 75% accuracy sustainable over 6 months in production, I'm happy. 95% sustainable for 7 days — I will not take it."*
5. *"Spurious correlation happens when you yourself are subconsciously biased even before you build a model."*
6. *"A smart data scientist doesn't write 100 lines of code. They write 20 lines, but draw 5 inferences from one output."*
7. *"Don't be a process-driven data scientist. Be a **thinking** data scientist."*
8. *"There is no brownie point or Nobel Prize for treating outliers you found."*
9. *"Finding bivariate outliers is more important than finding univariate outliers."*
10. *"This is how we connect concepts."*

---

## 27. Corrections to Carry Forward

Four things stated in class that an exam or interview would mark differently:

| # | As said in class | The precise version |
|---|---|---|
| **1** | mean ± 3σ covering 99.73% is **"Chebyshev's inequality"** | That's the **empirical rule** (normal distributions only). **Chebyshev** gives **≥ 88.9%** for *any* distribution. The caveat given was right; the name was loose (§13) |
| **2** | **MinMaxScaler** outputs **−1 to +1** | Default is **0 to 1**. `feature_range=(-1,1)` is opt-in (§17) |
| **3** | **SAS** was created as a response to the 1973 oil shock, and **Gates/Jobs interned there** | **SAS Institute: 1976, North Carolina State University**, out of an agricultural-statistics project. Gates and Jobs never interned there |
| **4** | **"Irving Fisher"** developed the t-test | **Ronald A. Fisher**, the statistician. Irving Fisher was an economist — different person |

---

## 28. Topics Deferred to Later Sessions

1. **Why we fit the scaler on train only** — 🏠 deliberate thinking homework, answer withheld (§18)
2. **One-hot encoding / dummy variables** — skipped in the demo; coming in feature engineering
3. **Heteroscedasticity** — bivariate outliers are its main cause; deferred to linear regression (§10)
4. **Covariance → correlation** derivation — self-study (§7)
5. **Data reduction techniques** — parked, ~30-minute discussion (§22b)
6. **`.describe()` on real Python output** — ran out of time
7. **Hypothesis testing, probability distributions, estimation/MLE** — self-study priorities (§25)
8. **Time series forecasting** — hardest topic; needs statistics + regression solid first
9. **Transformers** — the gateway to LLMs, the most important thing to understand later (§23)

### Materials promised
- The **EDA notebooks** (he has 4–5)
- A **PPT on missing-value and outlier best practices across 4–5 industries**

Both to be shared via the program team.
