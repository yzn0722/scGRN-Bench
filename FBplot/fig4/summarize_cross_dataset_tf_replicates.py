#!/usr/bin/env python3
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


DATASETS = ["hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]
EPS = 1e-12


def load_rows(path):
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    return report["tf_finite_difference_probe"]["rows"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="outputs/tf_cross_dataset_replicates",
    )
    args = parser.parse_args()
    root = Path(args.root)
    records = []
    dataset_summary = []

    for dataset in DATASETS:
        paths = sorted(
            glob.glob(str(root / dataset / "seed*" / dataset / "dynamic_grn_validation.json"))
        )
        if len(paths) != 7:
            raise RuntimeError(f"{dataset}: expected 7 runs, found {len(paths)}")
        runs = []
        for path in paths:
            rows = load_rows(path)
            runs.append({row["tf"]: row for row in rows})
        tfs = sorted(set.intersection(*(set(run) for run in runs)))

        for tf in tfs:
            ratios = np.asarray(
                [run[tf]["target_all_nontarget_ratio"] for run in runs],
                dtype=float,
            )
            empirical_pvalues = np.asarray(
                [run[tf]["random_label_empirical_p"] for run in runs],
                dtype=float,
            )
            empirical_significant_runs = int(
                np.sum((ratios > 1) & (empirical_pvalues < 0.05))
            )
            degrees = np.asarray([run[tf]["n_targets"] for run in runs], dtype=float)
            log_ratios = np.log(np.maximum(ratios, EPS))
            if np.allclose(log_ratios, 0):
                pvalue = 1.0
            else:
                pvalue = float(
                    wilcoxon(
                        log_ratios,
                        alternative="greater",
                        method="exact",
                    ).pvalue
                )
            median_ratio = float(np.median(ratios))
            median_targets = int(round(float(np.median(degrees))))
            nominal = pvalue < 0.05
            records.append(
                {
                    "dataset": dataset,
                    "tf": tf,
                    "runs": len(ratios),
                    "median_targets": median_targets,
                    "mean_ratio": float(np.mean(ratios)),
                    "median_ratio": median_ratio,
                    "min_ratio": float(np.min(ratios)),
                    "max_ratio": float(np.max(ratios)),
                    "positive_runs": int(np.sum(ratios > 1)),
                    "empirical_significant_runs": empirical_significant_runs,
                    "any_empirical_significant_run": empirical_significant_runs >= 1,
                    "wilcoxon_p_greater": pvalue,
                    "nominal_p_lt_0_05": nominal,
                    "targets_ge_20": median_targets >= 20,
                    "median_ratio_ge_1_10": median_ratio >= 1.10,
                    "stable_strong_response": bool(
                        nominal and median_targets >= 20 and median_ratio >= 1.10
                    ),
                }
            )

        current = [record for record in records if record["dataset"] == dataset]
        eligible = [record for record in current if record["targets_ge_20"]]
        significant = [record for record in current if record["nominal_p_lt_0_05"]]
        significant_eligible = [
            record
            for record in eligible
            if record["nominal_p_lt_0_05"]
        ]
        stable_strong = [record for record in current if record["stable_strong_response"]]
        dataset_summary.append(
            {
                "dataset": dataset,
                "runs": len(paths),
                "tested_tfs": len(current),
                "median_ratio_across_tfs": float(
                    np.median([record["median_ratio"] for record in current])
                ),
                "tfs_median_ratio_gt_1": int(
                    sum(record["median_ratio"] > 1 for record in current)
                ),
                "nominal_significant_tfs": len(significant),
                "eligible_tfs_targets_ge_20": len(eligible),
                "eligible_nominal_significant_tfs": len(significant_eligible),
                "stable_strong_tfs": len(stable_strong),
                "stable_strong_names": [record["tf"] for record in stable_strong],
            }
        )

    csv_path = root / "tf_replicate_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    json_path = root / "tf_dataset_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset_summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False))
    print("\nTOP TFs by dataset")
    for dataset in DATASETS:
        current = [record for record in records if record["dataset"] == dataset]
        current.sort(key=lambda record: record["median_ratio"], reverse=True)
        print(f"\n{dataset}")
        for record in current[:8]:
            print(
                f"{record['tf']}: n={record['median_targets']}, "
                f"median={record['median_ratio']:.3f}, "
                f"range={record['min_ratio']:.3f}-{record['max_ratio']:.3f}, "
                f"positive={record['positive_runs']}/7, "
                f"p={record['wilcoxon_p_greater']:.6f}"
            )


if __name__ == "__main__":
    main()
