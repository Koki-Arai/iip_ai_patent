# Variable Dictionary

This document defines all variables used in the analysis. Variable names follow IIP Patent Database conventions where applicable.

## AI Indicators

| Variable | Definition | Source |
|---|---|---|
| `ai_core` | 1 if `class1 == "G06N"` or `class2 == "G06N"`; 0 otherwise. **Baseline AI measure.** | Constructed from IIP application table |
| `ai_core_primary` | 1 if `class1 == "G06N"`; 0 otherwise. Primary-class-only definition. | Constructed |
| `ai_broad` | 1 if any of {`class1`, `class2`, `class3`} is in {G06N, G06F, G06Q, G06T, G10L, H04N}; 0 otherwise. Broad AI/software measure. | Constructed |
| `ai_broad_noncore` | `ai_broad == 1` AND `ai_core == 0`. Broad-but-not-core AI. | Constructed |

## Outcome Variables (Application-Level)

| Variable | Definition | Notes |
|---|---|---|
| `grant` | 1 if application was granted (`idr` is not null); 0 otherwise. | Binary |
| `grant_delay_days` | Days between application date and grant date. | Conditional on grant; censored otherwise. |
| `claim1` | Number of claims at filing. | From IIP application table |
| `claim3` | Number of claims at grant. | Conditional on grant |
| `claim_reduction` | `claim1 - claim3` | Conditional on grant |
| `claim_reduction_rate` | `(claim1 - claim3) / claim1` | Conditional on grant; capped at [0, 1] |
| `backward_cites` | Total examiner citations cited by this application. | From IIP citation table |
| `reject_cites` | Citations made for rejection reasons (codes 19 and 89). | From IIP citation table |
| `has_reject_cite` | 1 if `reject_cites > 0`; 0 otherwise. | Binary |
| `forward_cites` | Total times this application is cited by later applications (within sample window). | From IIP citation table |
| `has_forward_cite` | 1 if `forward_cites > 0`; 0 otherwise. | Binary |
| `inventor_count` | Number of unique inventors listed on the application. | From IIP inventor table |
| `applicant_count` | Number of unique applicants listed on the application. | From IIP applicant table |

## Citation Decomposition (Section 5.6)

| Variable | Definition | Reason Codes |
|---|---|---|
| `refusal_related_cites` | Examiner citations for rejection reasons. | Codes 19, 89 |
| `post_grant_ref_cites` | Post-grant references. | Code 31 |
| `trial_appeal_cites` | Trial- and appeal-stage citations. | Various codes |
| `self_cites` | Citations where citing application's first applicant matches cited application's first applicant. | Constructed via `idname` join |
| `external_cites` | `backward_cites - self_cites` | Constructed |
| `self_cite_share` | `self_cites / backward_cites` (conditional on `backward_cites > 0`). | Conditional |

## Applicant Characteristics

| Variable | Definition | Notes |
|---|---|---|
| `first_applicant_id` | IIP `idname` of the first listed applicant. | Used for applicant FE |
| `corporate` | 1 if first applicant is a corporation (per IIP `kojin_kbn` field); 0 if individual. | Binary |
| `domestic` | 1 if first applicant is a Japanese resident; 0 if foreign. | From `country_jp` field |
| `top50` | 1 if first applicant is among the top 50 patent filers in 2010–2018; 0 otherwise. | Constructed |

## Technology Field Variables

| Variable | Definition | Granularity |
|---|---|---|
| `class1` | First-listed IPC subclass (4-character: e.g., G06N). | 4-digit |
| `group1` | First-listed IPC subgroup (7-character: e.g., G06N3/02). | 7-digit |
| `app_year` | Application filing year (calendar year). | Annual |

## Aggregated Field-Year Panel Variables

For the density analysis (Sections 4.5 and 5.7):

| Variable | Definition | Level |
|---|---|---|
| `ai_core_density` | Share of `ai_core == 1` applications in the (field, year) cell. | 4-digit class or 7-digit subgroup |
| `total_apps` | Number of applications in the (field, year) cell. | Same |
| `follow_on_apps_t1` | Number of applications in the same field in `year + 1`. | Same |
| `hhi` | Herfindahl-Hirschman Index of applicant shares within the cell. | Same |
| `top5_share` | Sum of shares of the top 5 applicants within the cell. | Same |

## Control Variables

The standard control set (referred to as `Z` in the paper) includes:
- `log1p(claim1)`: log of claims at filing
- `log1p(inventor_count)`
- `log1p(applicant_count)`
- `corporate` indicator
- `domestic` indicator

For outcomes that are themselves controls (e.g., when `inventor_count` is the dependent variable), the corresponding control is removed from `Z`.

## Fixed Effects

| Notation | Description |
|---|---|
| `class1 FE` | 4-digit IPC subclass fixed effects (~700 levels) |
| `app_year FE` | Application-year fixed effects (9 levels for 2010–2018) |
| `first_applicant_id FE` | Applicant fixed effects (Section 5.6 only) |
| `group1 FE` | 7-digit IPC subgroup fixed effects (Section 5.7 only; ~14,000 levels) |

## Sample

- **Main sample (application-level)**: 2,637,795 applications filed 2010–2018 in IIP database
- **Granted-only sub-sample**: 1,656,342 applications
- **Two-applicant-or-more sub-sample (for applicant FE)**: 2,549,296 applications
- **4-digit IPC class-year panel**: 12,272 (class, year) cells
- **7-digit IPC subgroup-year panel**: 94,413 (group1, year) cells with ≥5 applications
