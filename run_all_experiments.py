"""
Run every experiment. Each writes its summary to ./results/<name>.txt
next to the script.

Run:
    python run_all_experiments.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    ("Mediation structures comparison",
     "pw-causal-comparison",   "mediation_structures_comparison.py",
     "results/mediation_structures.txt"),
    ("German Credit attribution",
     "german_credit",          "german_credit_attribution.py",
     "results/german_credit.txt"),
    ("Methods on synthetic benchmarks (S1, S2, S3)",
     "path_wise_experiments",  "compare_methods_on_benchmarks.py",
     "results/methods_on_benchmarks.txt"),
    ("Local attributions on S2 (per-sample)",
     "path_wise_experiments",  "local_attributions_s2.py",
     "results/local_attributions_s2.txt"),
]


def main():
    print(f"Running {len(EXPERIMENTS)} experiments ...")
    t_start = time.time()
    results = []
    for name, sub, script, out in EXPERIMENTS:
        cwd = os.path.join(HERE, sub)
        print('\n' + '#' * 80)
        print(f"# {name}")
        print(f"# cwd: {cwd}   cmd: python {script}")
        print('#' * 80)
        t0 = time.time()
        rc = subprocess.run([sys.executable, script], cwd=cwd).returncode
        out_path = os.path.join(cwd, out)
        success = rc == 0 and os.path.exists(out_path)
        results.append((name, out_path, time.time() - t0, success, rc))

    print('\n' + '=' * 80)
    print("Summary")
    print('=' * 80)
    for name, out_path, elapsed, success, rc in results:
        status = "OK" if success else f"FAILED (rc={rc})"
        print(f"  [{status}]  {name}")
        print(f"            -> {out_path}   ({elapsed:.1f}s)")
    print(f"\nTotal wall time: {time.time() - t_start:.1f}s")

    sys.exit(0 if all(r[3] for r in results) else 1)


if __name__ == "__main__":
    main()
