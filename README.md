# Customer Lifestyle Segmentation Model

An end-to-end unsupervised machine learning pipeline that 
classifies retail banking customers into behavioural lifestyle 
segments using 12 months of transaction history.

Built to reflect production-scale deployment patterns in 
Nigerian retail banking.

---

## What This Project Does

Most banks segment customers by demographics: age, income 
bracket, account type. This model takes a different approach. 
It uses actual transaction behaviour like what customers spend on, 
how they receive income, how they use digital channels e.t.c. to 
group customers into meaningful lifestyle profiles.

The output powers cross-selling, product targeting, churn 
prediction, and relationship management decisions.

---

## Segment Taxonomy

| Segment | Primary Signal |
|---------|----------------|
| STRATEGIST | High savings rate, insurance, professional services |
| WEALTH BUILDER | Savings + cross-bank self-transfers |
| FAMILY BUILDER | Education, healthcare, household spending |
| FINANCIALLY COMMITTED | High loan repayment burden |
| URBAN YOUNG PROFESSIONAL | Entertainment, food, lifestyle spending |
| INFORMAL / TRADER | High peer transfers, business expenses |
| ACTIVE BETTOR | Significant betting spend |
| CASH DEPENDENT | ATM, Branch withdrawals dominate spend |
| MINIMAL ACTIVITY | Very low transaction volume |
| GENERAL RETAIL CUSTOMER | Residual |

---

## Pipeline Architecture

```
Section 1 Import libraries
Section 2 Synthetic data generation
└── Saves monthly parquets to data/cache/
├── df_debit_{month}.parquet
├── df_pos_{month}.parquet
└── df_credit_{month}.parquet
Section 3 Classification engine
├── Builds terminal_profile.pkl from POS history
├── Classifies debit (Tiers 0, 2, 2.8, 2.9, 4, 5)
├── Classifies card/POS (T1 through T6 + cashback)
└── Classifies credit (income source detection)
Section 4 Feature engineering
└── Streams classified parquets month by month
(never loads full dataset into memory)
Section 5 Segmentation model
├── MiniBatchKMeans (k=9)
└── 10-rule priority segment name engine
Section 6 Model evaluation
├── Silhouette score
├── Segment distribution charts
└── Classification tier coverage
Section 7 Interactive widgets
├── Widget A: Customer lifestyle profile
└── Widget B: Transaction drill-down
Section 8 Export
├── Drift detection (prior 6M vs recent 6M)
├── TOP5 spend categories per customer
├── TOP5 credit income sources
└── Saves to outputs/customer_lifestyle_segments.csv
```

---

## Classification Tiers

### Debit
| Tier | Method |
|------|--------|
| 0 | Structural rules (ATM channel, fee codes) |
| 2 | Keyword matching on narration text |
| 2.8 | Beneficiary name keyword matching |
| 2.9 | Cross-bank self-transfer name detection |
| 4 | Transfer-to-others structural rule |
| 5 | Forced channel label, never NULL |

### Card (POS/ATM)
| Tier | Method |
|------|--------|
| T1 | TXN_NATURE flag (cash withdrawal, refund) |
| T2 | Terminal profile model (built from MCC history) |
| T3 | Merchant name keyword matching |
| T4 | MCC direct lookup table |
| T5 | MCC numeric range fallback |
| T6 | Catch-all |

---

## Installation

```bash
pip install -r requirements.txt
```

For interactive widgets:

```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

---

## Usage

```bash
jupyter notebook customer_lifestyle_segmentation.ipynb
```

Run all cells from top to bottom.

First run generates all data and cache files automatically.
Subsequent runs skip already-generated files.

---

## Key Design Decisions

**Why MiniBatchKMeans?**
Standard KMeans loads the full feature matrix into BLAS on 
each iteration. At 1M+ customers this causes memory failure. 
MiniBatchKMeans processes random subsets per iteration and 
scales to any dataset size.

**Why monthly parquets?**
According to the real dataset used in the real-life project,
loading 12 months of classified transactions simultaneously 
requires 15-20 GB of RAM. The monthly cache architecture 
allows feature engineering through streaming aggregation 
with constant memory footprint.

**Why rule-based segment naming over cluster labels?**
Cluster IDs are arbitrary numbers. Business teams need named 
segments with interpretable profiles. The 10-rule priority 
engine maps each customer to a named segment based on their 
dominant spending behaviour, independent of which cluster 
the model assigned them to.

**Why synthetic data?**
This notebook mirrors a production pipeline deployed at a 
Nigerian bank. Real customer transaction data cannot be 
shared publicly. The synthetic generation produces 
statistically realistic transaction patterns using similar
narration keywords and MCC codes as the production 
classifier, so the classification tiers produce realistic 
distributions.

---

## Output Schema

| Column | Description |
|--------|-------------|
| CUSTOMER_NO | Customer identifier |
| SEGMENT_12M | Lifestyle segment (full 12 months) |
| SEGMENT_6M_PRIOR | Segment for months 1-6 |
| SEGMENT_6M_RECENT | Segment for months 7-12 |
| SEGMENT_DRIFTED | Y/N - did segment change between windows |
| DRIFT_DIRECTION | e.g. WEALTH BUILDER -> FAMILY BUILDER |
| TOP1-5_SPEND_CAT | Top 5 spending categories |
| TOP1-5_SPEND_PCT | Share of total spend |
| TOP1-5_CREDIT_CAT | Top 5 income sources |
| PRIMARY_CHANNEL | Most used transaction channel |
| DIGITAL_RATIO | % of transactions through digital channels |
| RECOMMENDED_PRODUCT | Product suggestion by segment |
| REPORT_DATE | Analysis end date |

---

## Author

**Gbemileke Falade**

Senior Data Analyst | AI/ML Practitioner | Data Consultant

Lagos, Nigeria

https://www.linkedin.com/in/gbemileke-falade
