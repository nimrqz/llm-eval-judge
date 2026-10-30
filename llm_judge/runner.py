import itertools
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .judge import Rubric, compare_pair, score_response

def load_cases(path: Path) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for case in cases:
        if len(case.get("outputs", {})) < 1:
            raise ValueError(f"Caso {case.get('id')} sem 'outputs'")
    return cases

def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Intervalo de confiança de 95% da média. Com poucos casos, ele é largo — e isso é informativo."""
    if len(values) < 2:
        return (values[0], values[0]) if values else (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]

def evaluate(client, rubric: Rubric, cases: list[dict], workers: int = 4, pairwise: bool = True) -> dict:
    variants = sorted({v for c in cases for v in c["outputs"]})
    jobs = [(c["id"], v, c["prompt"], c["outputs"][v]) for c in cases for v in c["outputs"]]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        scored = list(pool.map(lambda j: score_response(client, rubric, j[2], j[3]), jobs))
    rows = [{"case": j[0], "variant": j[1], **s} for j, s in zip(jobs, scored)]

    summary = {}
    for v in variants:
        mine = [r for r in rows if r["variant"] == v]
        overall = [r["overall"] for r in mine]
        low, high = bootstrap_ci(overall)
        summary[v] = {
            "n": len(mine),
            "overall": round(statistics.fmean(overall), 3),
            "ci95": [round(low, 3), round(high, 3)],
            "criteria": {c.name: round(statistics.fmean(r["criteria"][c.name] for r in mine), 3)
                         for c in rubric.criteria},
        }

    pairs = []
    if pairwise and len(variants) >= 2:
        for a, b in itertools.combinations(variants, 2):
            shared = [c for c in cases if a in c["outputs"] and b in c["outputs"]]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(
                    lambda c: compare_pair(client, rubric, c["prompt"], c["outputs"][a], c["outputs"][b]), shared))
            n = len(results) or 1
            pairs.append({
                "a": a, "b": b, "n": len(results),
                "a_wins": sum(r["winner"] == "A" for r in results),
                "b_wins": sum(r["winner"] == "B" for r in results),
                "ties": sum(r["winner"] == "tie" for r in results),
                "position_bias_rate": round(sum(not r["consistent"] for r in results) / n, 3),
            })
    return {"rubric": rubric.name, "summary": summary, "pairwise": pairs, "rows": rows}
