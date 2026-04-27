from src.build_dataset import build_all_memory_safe
from src.descriptive import create_descriptive_outputs
from src.regressions import run_main_regressions
from src.robustness import run_robustness
from src.extract_results import extract_main_results, extract_robustness_results

if __name__ == '__main__':
    build_all_memory_safe()
    create_descriptive_outputs()
    run_main_regressions()
    run_robustness(max_n=500000)
    extract_main_results()
    extract_robustness_results()
