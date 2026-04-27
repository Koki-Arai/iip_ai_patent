# Colab Quickstart

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
