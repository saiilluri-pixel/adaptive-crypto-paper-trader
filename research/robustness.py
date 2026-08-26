"""
Phase 2, step 11: parameter-neighborhood robustness.

Reads research/reports/walkforward.json (each fold's full candidate grid,
scored on train+val only -- test was never used in selection). For each
fold's chosen combo, checks how many "neighbor" combos (identical except
one grid dimension changed by one step) also scored reasonably, vs. the
winner being an isolated spike. Flags isolated winners explicitly.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DIMS = ["SWING_LEN", "NW_MULT", "ST_ATR_PERIOD", "ST_ATR_MULT", "ST_INIT_STOP"]


def neighbors(params, all_candidates):
    out = []
    for cand in all_candidates:
        p = cand["params"]
        diffs = [k for k in DIMS if p[k] != params[k]]
        if len(diffs) == 1:
            out.append(cand)
    return out


def analyze():
    path = os.path.join(HERE, "reports", "walkforward.json")
    if not os.path.exists(path):
        print("walkforward.json not found -- run walkforward.py first")
        return None
    data = json.load(open(path))
    report = {}
    for sym, folds in data["folds"].items():
        fold_reports = []
        for i, fold in enumerate(folds):
            chosen = fold["chosen_params"]
            cands = fold["all_candidates_summary"]
            nbrs = neighbors(chosen, cands)
            nbr_scores = [n["score"] for n in nbrs if n["score"] is not None]
            chosen_score = fold["chosen_score"]
            positive_neighbors = sum(1 for s in nbr_scores if s > 0)
            fold_reports.append({
                "fold": i,
                "chosen_score": chosen_score,
                "n_neighbors_in_grid": len(nbrs),
                "n_neighbors_scored": len(nbr_scores),
                "n_neighbors_positive": positive_neighbors,
                "neighbor_score_mean": (sum(nbr_scores) / len(nbr_scores)) if nbr_scores else None,
                "neighbor_score_min": min(nbr_scores) if nbr_scores else None,
                "neighbor_score_max": max(nbr_scores) if nbr_scores else None,
                "isolated_spike": bool(nbr_scores and chosen_score is not None
                                       and chosen_score > 0
                                       and (sum(nbr_scores) / len(nbr_scores)) < 0),
                "oos_test_ret_pct": fold["oos_test_ret_pct"],
                "n_candidates_this_fold": len(cands),
            })
        report[sym] = fold_reports
    with open(os.path.join(HERE, "reports", "robustness.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    analyze()
