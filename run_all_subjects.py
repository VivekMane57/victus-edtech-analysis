# run_all_subjects.py
import os, sys, glob, pandas as pd

# ensure the repo root is importable
sys.path.append(os.path.dirname(__file__))

from pipeline import run_subject

root = r"D:\IITB\STData"  # folders: 1, 2, 3, ...

subjects = [d for d in sorted(glob.glob(os.path.join(root, "*"))) if os.path.isdir(d)]

rows = []
for i, subdir in enumerate(subjects, start=1):
    sid = os.path.basename(subdir)
    print(f"▶ [{i}/{len(subjects)}] Processing subject {sid} ...", end=" ")
    try:
        res = run_subject(subdir)
        rows.append(res)
        print("✅ done")
    except Exception as e:
        print(f"❌ error: {e}")

out = os.path.join(root, "summary_results.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("📄 Saved summary table:", out)
