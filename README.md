# Replication Package: Core AI Patents and Patent Examination in Japan

This repository contains the replication code and reference results for:

> Arai, K. (2026). "Core AI Patents and Patent Examination in Japan." Working paper, Faculty of Business Studies, Kyoritsu Women's University.

## Abstract

This paper analyzes core AI patents in Japan using the IIP Patent Database for 2010–2018, defining core AI by IPC class G06N. Core AI patents are more likely to be granted (β = 0.304, Oster |δ| = 44.6), command a substantial forward-citation premium that has grown over time (β = 0.700 in Poisson PML), and exhibit fewer rejection-related citations. They involve more inventors but fewer co-applicants, indicating team-based invention with concentrated ownership. The grant probability premium is more than twice as large for individual as for corporate applicants, and the forward-citation premium is concentrated among domestic applicants. At the 7-digit IPC subgroup level, AI density and follow-on innovation exhibit an inverted-U relationship (turning point ≈ 44 percent) that the 4-digit class-level analysis missed, consistent with prospect-like spillovers at low density and anti-commons-like attenuation at high density. These conclusions are limited to patents identified as core AI by IPC class G06N.

**Keywords:** Artificial intelligence; Patent examination; Forward citations; Anti-commons.

**JEL classification:** O34, O33, K11, L40.

## Data

This analysis uses the **IIP Patent Database** (Institute of Intellectual Property 2024), which is a licensed database and is not redistributed here. Researchers can obtain access from the Institute of Intellectual Property of Japan:

> https://www.iip.or.jp/patentdb/

The replication code assumes the IIP data is mounted at:

```
DATA_ROOT/
├── application.csv
├── applicant.csv
├── inventor.csv
├── right_holder.csv
└── citation.csv
```

The default `DATA_ROOT` is `/content/drive/MyDrive/iip_patent_ai/data/` (Google Colab convention). Users can override this via the `DATA_ROOT` environment variable or by editing `code/config.py`.

## Requirements

- Python 3.10 or later
- See `requirements.txt` for full dependency list
- Approximately 32 GB RAM recommended for the full main sample (~2.6M observations)

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Replication Workflow

The analysis is organized into 11 Python scripts that should be run in numerical order:

| Script | Section | Description | Output Tables/Figures |
|---|---|---|---|
| `01_data_prep.py` | Section 3.1–3.2 | Load IIP data, construct variables, define AI indicators | — |
| `02_descriptive.py` | Section 4.1 | Descriptive statistics, AI share by year | Figure 1 |
| `03_main_regressions.py` | Section 4.2–4.3 | Main 9 regressions: grant, delay, claims, citations, organization | Table 1 |
| `04_forward_citations.py` | Section 4.4 | Forward citations: log-OLS, PPML, has-forward-cite, period split | Table 2 |
| `05_heterogeneity.py` | Section 4.5 | Sub-sample regressions by applicant type | Table 3 |
| `06_density_followon.py` | Section 4.5 | 4-digit IPC class density (linear and quadratic) | (in text) |
| `07_period_split.py` | Section 5.2 | Pre-/post-AI-boom period split | Table 4 |
| `08_did_2018.py` | Section 5.3 | DiD around 2018 JPO update + event study | Table 5, Figure 2 |
| `09_robustness.py` | Section 5.4–5.5 | Functional form, multi-way clustering, multiple testing, Oster δ | Tables 6, 7 |
| `10_citation_decomp_applicant_fe.py` | Section 5.6 | Citation decomposition (reason codes, self/external) and applicant fixed effects | Tables 8, 9 |
| `11_subgroup_density.py` | Section 5.7 | 7-digit IPC subgroup density (linear, quadratic, HHI, top-5 share) | Table 10, Figure 3 |
| `99_generate_figures.py` | — | Generate all 3 figures from saved data | Figures 1–3 |

Run from the repository root:

```bash
cd code/
python 01_data_prep.py
python 02_descriptive.py
# ... etc
python 99_generate_figures.py
```

Each script is self-contained: it loads required intermediate data, runs its analysis, and writes results to `results/tables/` and figures to `results/figures/`.

## Outputs

### Tables (10 total, in `results/tables/`)

| Table | Filename | Section |
|---|---|---|
| 1 | `table01_main_results.csv` | 4.2 |
| 2 | `table02_forward_citations.csv` | 4.4 |
| 3 | `table03_heterogeneity.csv` | 4.5 |
| 4 | `table04_period_split.csv` | 5.2 |
| 5 | `table05_did_2018.csv` | 5.3 |
| 6 | `table06_functional_form.csv` | 5.4 |
| 7 | `table07_oster_delta.csv` | 5.5 |
| 8 | `table08_citation_decomp.csv` | 5.6 |
| 9 | `table09_applicant_fe.csv` | 5.6 |
| 10 | `table10_subgroup_density.csv` | 5.7 |

### Figures (3 total, in `results/figures/`)

| Figure | Filename | Section |
|---|---|---|
| 1 | `Figure1_AI_shares.png` | 4.1 |
| 2 | `Figure2_event_study.png` | 5.3 |
| 3 | `Figure3_inverted_U.png` | 5.7 |

### Reference Outputs

The `results/tables/` directory also contains the original CSV outputs from the author's run (with prefix `ref_`). These can be compared with your replication outputs to verify consistency.

## Variable Dictionary

See `docs/variable_dictionary.md` for definitions of all variables used in the analysis.

## Computational Notes

- Two-way and applicant fixed effects regressions on the 2.5M+ row sample require approximately 32 GB RAM; consider running on a high-memory cloud instance (e.g., Google Colab Pro+).
- Poisson PML estimation (PPML) using `pyfixest.fepois()` may take 5–10 minutes per outcome.
- The 7-digit IPC subgroup analysis (Section 5.7) operates on ~94,000 (subgroup-year) cells and runs quickly.

## License

Code in this repository is licensed under the MIT License (see `LICENSE`).

The IIP Patent Database is the property of the Institute of Intellectual Property of Japan and is governed by their licensing terms.

## Acknowledgments

This research is funded by JSPS KAKENHI Grant Number 23K01404.

The author thanks colleagues at Kyoritsu Women's University and the Institute of Intellectual Property for helpful discussions, and acknowledges that the empirical analysis was conducted using the IIP Patent Database.

## Citation

If you use this code or build on this analysis, please cite:

```bibtex
@article{arai2026coreai,
  title   = {Core AI Patents and Patent Examination in Japan},
  author  = {Arai, Koki},
  year    = {2026},
  journal = {Working paper, Kyoritsu Women's University},
}
```

## Contact

Koki Arai
Faculty of Business Studies, Kyoritsu Women's University
Email: [author email]
