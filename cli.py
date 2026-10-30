import argparse
import json
from pathlib import Path

import anthropic

from llm_judge.judge import Rubric
from llm_judge.report import render_html
from llm_judge.runner import evaluate, load_cases

def main() -> None:
    parser = argparse.ArgumentParser(description="Compara variantes de prompt com LLM-as-judge")
    parser.add_argument("cases", type=Path)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-pairwise", action="store_true")
    args = parser.parse_args()

    rubric = Rubric.from_dict(json.loads(args.rubric.read_text(encoding="utf-8")))
    result = evaluate(anthropic.Anthropic(), rubric, load_cases(args.cases),
                      workers=args.workers, pairwise=not args.no_pairwise)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "report.html").write_text(render_html(result), encoding="utf-8")
    for variant, s in result["summary"].items():
        print(f"{variant}: {s['overall']:.2f}  IC95% {s['ci95']}")
    print(f"Relatório: {args.out / 'report.html'}")

if __name__ == "__main__":
    main()
