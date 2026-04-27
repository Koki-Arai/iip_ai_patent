# IIP AI Patent Estimation

Reproducible Python code for estimating empirical patterns of AI-related patent applications in Japan using the IIP Patent Database. The workflow is designed for Google Colab and large decade-level text files. It uses checkpointing so that processing can resume after runtime interruption.

## Repository structure

```text
src/        Core Python modules
scripts/    Entry-point scripts for Colab or local execution
docs/       Data and empirical-design notes
output/     Generated checkpoints, tables, figures, and model outputs
```

## Data

Raw IIP data are not included. Place the following files in `data/`:

```text
ap_1990s.txt, applicant_1990s.txt, inventor_1990s.txt, cc_1990s.txt
ap_2000s.txt, applicant_2000s.txt, inventor_2000s.txt, cc_2000s.txt
ap_2010s.txt, applicant_2010s.txt, inventor_2010s.txt, cc_2010s.txt
ap_2020s.txt, applicant_2020s.txt, inventor_2020s.txt, cc_2020s.txt
```

## Colab quick start

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/iip_patent_ai')

!pip install -r requirements.txt

%run scripts/run_01_build_dataset.py
%run scripts/run_02_descriptive.py
%run scripts/run_03_regressions.py
%run scripts/run_04_robustness.py
%run scripts/run_05_extract_results.py
```

## Memory-safe workflow

The builder saves decade-level analysis files instead of creating one massive all-period file:

```text
output/checkpoints/analysis_dataset_1990s.parquet
output/checkpoints/analysis_dataset_2000s.parquet
output/checkpoints/analysis_dataset_2010s.parquet
output/checkpoints/analysis_dataset_2020s.parquet
output/checkpoints/analysis_dataset_main_2010_2018.parquet
```

If Colab crashes, rerun the same script. Existing checkpoints are reused automatically.

To restrict the workflow to the main 2010s sample:

```python
import os
os.environ["IIP_DECADES"] = "2010s"
```

To reduce chunk size:

```python
import os
os.environ["IIP_CHUNKSIZE"] = "200000"
```

Do not set `IIP_FORCE_REBUILD=1` unless you want to recompute everything from raw text files.

## Main outputs

```text
output/tables/yearly_ai_patent_trends.csv
output/tables/main_regression_summary.csv
output/tables/robustness/robustness_summary.csv
output/figures/ai_share_trend.png
output/figures/claims_trend.png
output/models/*.txt
```

## License

MIT License. See `LICENSE`.
