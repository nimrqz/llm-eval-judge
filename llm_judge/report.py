from html import escape

def render_html(result: dict) -> str:
    summary = result["summary"]
    criteria = list(next(iter(summary.values()))["criteria"]) if summary else []
    head = "".join(f"<th>{escape(c)}</th>" for c in criteria)
    rows = ""
    for variant, s in summary.items():
        cells = "".join(f"<td>{s['criteria'][c]:.2f}</td>" for c in criteria)
        low, high = s["ci95"]
        pct = s["overall"] / 5 * 100
        rows += (f"<tr><td><b>{escape(variant)}</b></td><td>{s['n']}</td>"
                 f"<td><div class='bar'><span style='width:{pct:.0f}%'></span></div>"
                 f"{s['overall']:.2f} <small>IC95% [{low:.2f}, {high:.2f}]</small></td>{cells}</tr>")
    pairs = "".join(
        f"<tr><td>{escape(p['a'])} vs {escape(p['b'])}</td><td>{p['a_wins']}</td><td>{p['b_wins']}</td>"
        f"<td>{p['ties']}</td><td>{p['position_bias_rate']:.0%}</td></tr>" for p in result["pairwise"])
    details = "".join(
        f"<tr><td>{escape(r['case'])}</td><td>{escape(r['variant'])}</td><td>{r['overall']:.2f}</td>"
        f"<td>{escape(' · '.join(f'{k}: {v}' for k, v in r['reasons'].items()))}</td></tr>"
        for r in result["rows"])
    return f"""<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<title>Relatório de avaliação</title>
<style>body{{font:15px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}td,th{{border:1px solid #ccc;padding:6px 8px;text-align:left;vertical-align:top}}
.bar{{background:#eee;height:8px;border-radius:4px;width:160px}}.bar span{{display:block;height:8px;background:#2a7;border-radius:4px}}</style>
<h1>Avaliação: {escape(result['rubric'])}</h1>
<h2>Resumo por variante</h2>
<table><tr><th>Variante</th><th>n</th><th>Nota geral (1-5)</th>{head}</tr>{rows}</table>
<h2>Comparação pareada (posições trocadas)</h2>
<table><tr><th>Par</th><th>Vitórias A</th><th>Vitórias B</th><th>Empates</th><th>Viés de posição</th></tr>{pairs}</table>
<h2>Detalhes por caso</h2>
<table><tr><th>Caso</th><th>Variante</th><th>Nota</th><th>Justificativas</th></tr>{details}</table>
</html>"""
