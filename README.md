# Avaliador de prompts com LLM

Compara respostas geradas por versões diferentes de um prompt. Um LLM dá nota de 1 a 5 em cada critério (correção, completude, clareza, tom) e o programa calcula a média ponderada e gera um relatório HTML.

## Como rodar

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
python cli.py examples/cases.jsonl --rubric examples/rubric.json --out reports
python -m pytest tests
```

Cada linha do `cases.jsonl` tem a pergunta e as respostas de cada versão: `{"id", "prompt", "outputs": {"v1": "...", "v2": "..."}}`. Os critérios e pesos ficam em `examples/rubric.json`.

## O que ele calcula

- Nota por critério e nota geral de cada versão, com intervalo de confiança por bootstrap (com poucos casos o intervalo fica grande).
- Comparação direta entre duas versões, rodando duas vezes com a ordem trocada. Se o juiz muda de opinião quando a ordem muda, conta como empate e entra na taxa de "viés de posição".

## Limitações

- Os 5 casos de exemplo são só para demonstrar. Para uma conclusão confiável precisaria de bem mais casos e de comparar as notas do LLM com notas humanas.

## Próximos passos

- Montar um conjunto maior de casos e registrar o relatório gerado
- Comparar as notas do juiz com notas dadas por pessoas
