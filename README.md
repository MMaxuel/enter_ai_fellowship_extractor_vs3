# ENTER-AI Fellowship Extractor VS3

Solução de extração de dados estruturados a partir de PDFs e “telas” (capturas/relatórios) brasileiros, com foco em:

1. heurísticas locais rápidas (regex, BR-validators, KV),
2. uso opcional e controlado de LLM apenas como fallback,
3. desempenho consistente (< 10 s), custo reduzido e ≥ 80% de acurácia média em layouts variáveis.

O sistema expõe:

* API FastAPI com endpoints para extração unitária e em lote.
* Modo CLI para processamento de diretórios ou datasets.
* Métricas Prometheus.
* Cache leve de conhecimento (SqliteDict) com “aprendizado” de exemplos e overrides por template.

---

## Sumário

1. Visão geral e objetivos
2. Principais decisões e desafios endereçados
3. Requisitos e instalação
4. Variáveis de ambiente
5. Executando a API (uvicorn)
6. Endpoints e exemplos de uso
7. Esquema (schema) e labels
8. Como o extrator funciona (arquitetura)
9. Precisão, confiança e Strict Mode
10. LLM: quando e como é usada; custo e tempo
11. Métricas Prometheus
12. Modo CLI e formatos de saída
13. Boas práticas e troubleshooting
14. Segurança
15. Extensões e customização
16. Anatomia do código (explicação dos imports e blocos)

---

## 1) Visão geral e objetivos

* Extrair campos estruturados de documentos PDF heterogêneos e telas de sistemas.
* Priorizar heurísticas locais baratas (regex, padrões BR e grids KV) para reduzir custo e latência.
* Acionar LLM apenas para campos faltantes e somente se houver folga dentro do SLA.
* Aprender ao longo do uso: gravar exemplos e padrões (“overrides”) específicos por template para melhorar recall em execuções futuras.

---

## 2) Principais decisões e desafios endereçados

* Variabilidade de layout: combinação de estratégias (Docling → Markdown e tabelas; PyPDF2/pdfminer; KV; regex canônicas por label; BR-validators).
* Custo vs. precisão: LLM é fallback, controlada por tempo restante e flags; heurísticas cobrem a maior parte dos casos.
* SLA e estabilidade: DEADLINE_S e LLM_SAFETY_MARGIN_S controlam quando a LLM pode rodar.
* Auto-detecção de label: quando `label=auto`, o sistema estima o label a partir do schema e do texto (assinaturas por label + pistas textuais).
* Robustez a OCR/typos: normalização de texto e correções pontuais comuns.
* Aprendizado leve: exemplos e overrides por label/template em SqliteDict (ancoragem por “palavra vizinha”).
* Observabilidade: métricas de latência, taxa de sucesso, uso de LLM e confiança agregada.

---

## 3) Requisitos e instalação

Python 3.10+ recomendado.

Dependências principais:

* fastapi, uvicorn
* PyPDF2 e pdfminer.six (fallback de extração de texto)
* docling e docling-core (opcional; melhora extração baseada em layout)
* openai e httpx (LLM opcional)
* sqlitedict
* prometheus-client
* rapidfuzz (fuzzy matching)
* tqdm (barra de progresso no CLI)
* pydantic

Instalação típica:

```
pip install -r requirements.txt
```

Docling é opcional. Se não for instalar Docling, o código funciona com PyPDF2/pdfminer. Se for instalar Docling, siga as instruções oficiais do pacote para o seu sistema.

---

## 4) Variáveis de ambiente

Controle fino do comportamento:

* Diretórios e arquivos padrão

  * `PDF_DIR` (default: pasta files do dataset)
  * `SCHEMA_PATH` (default: schema.json no diretório do script)
  * `DATASET_PATH` (default: dataset.json no diretório do script)
  * `OUTDIR` (default: outputs)
  * `LABEL` (default: carteira_oab)

* SLA e qualidade

  * `DEADLINE_S` (padrão 9.0). Tempo máximo por extração.
  * `LLM_SAFETY_MARGIN_S` (padrão 2.0). Janela de segurança para decidir se LLM pode rodar.
  * `MIN_CONF` (padrão 0.72). Confiança mínima por campo.
  * `STRICT_MODE` (“1” liga, “0” desliga). Quando ligado, derruba valores com confiança abaixo do limiar para `null`.

* LLM (opcional)

  * `USE_LLM` (“1” ativa, “0” desativa; padrão “0”).
  * `FORCE_LLM_ALL` (“1” tenta LLM em todos os campos, geralmente para testes; desaconselhável em produção).
  * `OPENAI_API_KEY` (chave; sem isso o LLM não roda).
  * `LLM_MODEL` (padrão “gpt-5-mini”).
  * `LLM_PRICE_IN_1K`, `LLM_PRICE_OUT_1K` (custo por 1K tokens; usados só para estimativa).

---

## 5) Executando a API (uvicorn)

Exemplo no PowerShell (Windows):

```
cd C:\Users\SEU_USUARIO\PycharmProjects\treino\Enter

$env:STRICT_MODE="1"
$env:MIN_CONF="0.85"
$env:USE_LLM="1"
$env:OPENAI_API_KEY="sk-XXXXXXXX"
$env:FORCE_LLM_ALL="0"
$env:DEADLINE_S="20"
$env:LLM_MODEL="gpt-5-mini"

python -m uvicorn enter_ai_fellowship_extractor_vs3:app --host 0.0.0.0 --port 8765 --reload --app-dir "C:\Users\SEU_USUARIO\PycharmProjects\treino\Enter"
```

No Linux/macOS (bash):

```
export STRICT_MODE=1
export MIN_CONF=0.85
export USE_LLM=1
export OPENAI_API_KEY="sk-XXXXXXXX"
export FORCE_LLM_ALL=0
export DEADLINE_S=20
export LLM_MODEL="gpt-5-mini"

uvicorn enter_ai_fellowship_extractor_vs3:app --host 0.0.0.0 --port 8765 --reload
```

Verifique saúde:

```
curl http://127.0.0.1:8765/health
```

---

## 6) Endpoints e exemplos de uso

### GET /health

Retorna status do serviço, flags de LLM, versão, limiares.

### GET /metrics

Exibe métricas Prometheus.

### POST /warmup

Aquece regex e hints. Útil após start.

### POST /extract

Campos do formulário:

* `label`: string (ou “auto” para detecção automática)
* `extraction_schema`: JSON em string. Pode ser:

  * `{"campo1": "descrição", "campo2": ""}`
  * `["campo1", "campo2"]`
* `pdf`: arquivo PDF

Exemplo `curl`:

```
curl -X POST http://127.0.0.1:8765/extract \
  -F "label=carteira_oab" \
  -F "extraction_schema={\"nome\":\"\",\"inscricao\":\"\",\"seccional\":\"\",\"subsecao\":\"\",\"categoria\":\"\",\"telefone_profissional\":\"\",\"endereco_profissional\":\"\",\"situacao\":\"\"}" \
  -F "pdf=@oab_1.pdf"
```

Resposta:

* `output`: dict de campos extraídos (ou null)
* `meta`: diagnóstico (confianças, validação por campo, latência, uso de LLM, etc.)

### POST /extract_bulk

Múltiplos PDFs num mesmo request.

Form-data:

* `label`: string (pode ser “auto”)
* `extraction_schema`: string JSON
* `pdfs`: múltiplos arquivos `application/pdf`

Dica para evitar 422:

* Em PowerShell, use `MultipartFormDataContent` e garanta `ContentType` em cada PDF.
* No `curl`, use várias flags `-F "pdfs=@arquivo.pdf"`.

### GET /label_info

Inspeciona campos já vistos para um `label` e mostra overrides e exemplos aprendidos.

### GET /label_schema

Lista o “schema completo” aprendido por label, frequência por campo e overrides.

### POST /suggest_schema

Sugere campos a partir de um PDF, combinando hints heurísticos e, se habilitado, LLM.

---

## 7) Esquema (schema) e labels

* Schema pode ser dict com descrições por campo (melhor para hints) ou lista simples de nomes.
* `label` pode ser fixo (ex. “carteira_oab”, “tela_sistema”) ou “auto”.
* Com `label=auto`, o sistema estima o label com base em:

  * Interseção de campos característicos do schema com assinaturas por label.
  * Pistas de texto (“OAB”, “Inscrição Seccional Subseção”, “Data Base”, etc.).

---

## 8) Como o extrator funciona (arquitetura)

1. Leitura de PDF

* Primeiro tenta Docling (se instalado) para obter Markdown e HTML com tabelas.
* Caso Docling não esteja disponível ou falhe, tenta PyPDF2; se falhar, pdfminer.

2. Normalização e correções de OCR

* Remoção de espaços estranhos, unificação de quebras, correções comuns (“verncimento” → “vencimento”).

3. Extração base (ensemble por campo)

* `KV`: pares “Label: Valor” no texto/Markdown e tabelas 2-colunas.
* `Default regex`: padrões específicos por label/campo (âncoras fortes).
* `BR`: validadores e formatos nacionais (CPF, CNPJ, CEP, telefone, data).
* `Hints`: rótulos e sinônimos com captura na linha e na linha seguinte.
* Escolhe o melhor candidato por confiança; aplica “boost” se dois métodos concordarem.

4. Ajustes por label e pós-processamento

* Campos de telas padronizados (título detectado; normalização de strings).
* Campos OAB tratados pelo “bloco triplo” (Inscrição UF Subseção), nome e endereço profissional.
* `postprocess_value` aplica normalizações, filtros e validações finais (ex.: datas ISO, CEP, CPF/CNPJ formatados, números limpos).

5. Strict Mode e validação

* Cada campo recebe uma confiança.
* Se `STRICT_MODE=1` e `conf < MIN_CONF`, o valor é zerado (null).
* `meta.validation` mostra `value`, `conf` e `meets_min_conf` por campo.

6. Fallback LLM (opcional)

* Só roda se `USE_LLM=1`, houver tempo antes do DEADLINE e o campo precisar de fallback.
* Constrói janelas curtas de contexto por campo e solicita JSON estrito.
* Após LLM, revalida e pós-processa.

7. Aprendizado leve

* Exemplo do campo é salvo.
* Override por âncora (global e por template) é atualizado.
* Frequência de campos por label é incrementada.

---

## 9) Precisão, confiança e Strict Mode

* Confianças calibradas por origem:

  * BR (formatos nacionais validados) e overrides por template recebem pesos mais altos.
  * KV e hints têm pesos menores.
* Acordo entre métodos aumenta a confiança.
* `MIN_CONF` e `STRICT_MODE` permitem forçar qualidade mínima de saída.

---

## 10) LLM: quando e como é usada; custo e tempo

* LLM é fallback. Se o tempo restante for menor que `LLM_SAFETY_MARGIN_S`, não roda.
* `FORCE_LLM_ALL=1` obriga tentar LLM para todos os campos (útil só em diagnóstico).
* Custos estimados a partir de `LLM_PRICE_IN_1K` e `LLM_PRICE_OUT_1K`.
* `meta.llm_usage` e `meta.cost_est_usd` mostram uso estimado e custo por requisição.

---

## 11) Métricas Prometheus

* `extract_latency_seconds` e `extract_bulk_latency_seconds`
* `extract_success_total`, `extract_fail_total`
* `extract_used_llm_total`
* `extract_conf_agg` (histograma)
* `extract_strict_mode` (gauge 1/0)

Acesse em `/metrics`.

---

## 12) Modo CLI e formatos de saída

Execução unitária:

```
python enter_ai_fellowship_extractor_vs3.py --label carteira_oab --schema schema.json --pdf caminho\doc.pdf
```

Execução em lote por diretório:

```
python enter_ai_fellowship_extractor_vs3.py --pdf_dir "C:\path\to\files" --label carteira_oab --schema schema.json --outdir outputs
```

Execução por dataset:

```
python enter_ai_fellowship_extractor_vs3.py --dataset dataset.json --outdir outputs
```

Saídas geradas:

* JSONL consolidado
* CSV com colunas `out.<campo>`
* JSON individual por PDF

---

## 13) Boas práticas e troubleshooting

* Erro 422 no `/extract_bulk`: geralmente é `multipart/form-data` malformado. Garanta que:

  * `extraction_schema` é string JSON válida.
  * Cada PDF vem com `Content-Type: application/pdf`.
  * O campo do arquivo se chama exatamente `pdfs`.

* Baixa cobertura de campos:

  * Ative Docling para melhorar KV de layout.
  * Forneça descrições no schema (dict) para ajudar hints.
  * Ajuste `MIN_CONF` e `STRICT_MODE` conforme a necessidade.
  * Habilite LLM somente quando necessário.

* Tempo alto:

  * Reduza `DEADLINE_S` e mantenha `USE_LLM=0` por padrão.
  * Utilize o modo CLI para processar lotes sem overhead HTTP.

---

## 14) Segurança

* Nunca comprometa sua chave. Use `OPENAI_API_KEY` via ambiente.
* Os metadados mascaram dígitos de CPF/CNPJ em `meta`.
* Se armazenar PDFs sensíveis, proteja diretórios e SQLite.

---

## 15) Extensões e customização

* Novos labels:

  * Adicione assinaturas em `LABEL_SIGNATURES` e regex em `LABEL_DEFAULT_REGEX`.
* Novas validações:

  * Amplie `BR_PATTERNS` e validadores no pós-processamento.
* Estratégias de aprendizado:

  * Ajuste como overrides são gerados e priorizados.
* UI Web:

  * A API está pronta para uma interface simples em cima de `/extract` e `/extract_bulk`.

---

## 16) Anatomia do código (explicação dos imports e blocos)

* `from __future__ import annotations`: usa anotações de tipo mais leves e adia avaliação de tipos.
* Bloco de imports padrão: `io, os, re, json, time, csv, hashlib, argparse, logging, sys`.

  * Manipulação de I/O, caminhos, regex, JSON/CSV, hashing, CLI e logs.
* Tipagem: `typing` com `Dict, Any, Optional, Tuple, List, Iterable, Union`.
* `Pathlib` e `datetime`: caminhos portáveis e datas.
* FastAPI: `FastAPI, UploadFile, File, Form, Query` e `JSONResponse, PlainTextResponse` para API REST.
* Pydantic `BaseModel`: define modelos de entrada/saída (`ExtractResponse`).
* `PyPDF2.PdfReader`: extração rápida de texto, com fallback para `pdfminer.six`.
* Docling (opcional): `DocumentConverter` e serializers (Markdown/HTML) para extrair KV e tabelas respeitando layout.
* `SqliteDict`: armazenamento simples chave-valor para “memória” da aplicação (exemplos, overrides e estatísticas).
* `httpx` e `certifi`: cliente HTTP robusto e cadeia de certificados; usado para cliente OpenAI.
* `openai`: chamadas LLM (chat e responses).
* Prometheus: `Counter, Histogram, Gauge, generate_latest` para métricas.
* `tqdm`: barra de progresso no modo CLI.
* `rapidfuzz`: fuzzy matching (token/partial) para mapear rótulos a campos.
* Constantes de caminhos padrão: `DEFAULT_*` controladas por ambiente.
* SLA/qualidade: `DEADLINE_S`, `LLM_SAFETY_MARGIN_S`, `MIN_CONF`, `STRICT_MODE`.
* Controle LLM: `USE_LLM`, `FORCE_LLM_ALL`, `OPENAI_API_KEY`, `LLM_MODEL`, preços por 1K tokens.
* Cliente OpenAI com `httpx.Client` e captura robusta de uso de tokens para estimar custo.
* Funções LLM padronizadas: `llm_json_complete` e `llm_fallback_extract` com respostas estritamente em JSON.
* Logger JSON: formata logs como objetos estruturados (nível, timestamp, mensagem).
* Métricas: histograms, counters e gauges nomeados de forma explícita.
* Padrões nacionais: regex e validadores de CPF, CNPJ, CEP, telefone, data.
* Normalização de texto: limpeza agressiva anti-OCR.
* Heurísticas e ensemble: `_fx_kv`, `_fx_default_regex`, `_fx_br`, `_fx_hints` e `run_field_ensemble`.
* Pós-processamento por campo: `postprocess_value` (datas ISO, números, nomes, cidade/UF, etc.).
* Regras por label: “carteira_oab” (bloco triplo; nome; endereço) e “tela_sistema” (título; campos de produto/sistema/operacao).
* Aprendizado leve: `learn_from_result`, armazenamento de exemplos e regex com âncora global e por template.
* Cache de sessão: evita retrabalho dentro da mesma execução (chave por versão, label, hash do PDF e hash do schema).
* Endpoints: `/health`, `/metrics`, `/warmup`, `/extract`, `/extract_bulk`, `/label_info`, `/label_schema`, `/suggest_schema`.
* CLI: processamento serial de dataset ou diretório, gerando JSONL, CSV e JSONs individuais.

---

## Exemplo mínimo de uso da API

Unitário:

```
curl -X POST http://127.0.0.1:8765/extract \
  -F "label=auto" \
  -F "extraction_schema={\"nome\":\"\",\"inscricao\":\"\",\"seccional\":\"\",\"subsecao\":\"\",\"categoria\":\"\",\"telefone_profissional\":\"\",\"endereco_profissional\":\"\",\"situacao\":\"\"}" \
  -F "pdf=@C:\caminho\oab_3.pdf"
```

Lote:

```
curl -X POST http://127.0.0.1:8765/extract_bulk \
  -F "label=auto" \
  -F "extraction_schema={\"data_base\":\"\",\"data_vencimento\":\"\",\"quantidade_parcelas\":\"\",\"produto\":\"\",\"sistema\":\"\",\"tipo_de_operacao\":\"\",\"tipo_de_sistema\":\"\",\"valor_parcela\":\"\",\"pesquisa_por\":\"\",\"pesquisa_tipo\":\"\",\"selecao_de_parcelas\":\"\",\"total_de_parcelas\":\"\",\"cidade\":\"\"}" \
  -F "pdfs=@tela_sistema_1.pdf" \
  -F "pdfs=@tela_sistema_2.pdf"
```

---

## Exemplo mínimo de uso via CLI

```
python enter_ai_fellowship_extractor_vs3.py --dataset dataset.json --outdir outputs
```

Gera:

* `outputs/results_YYYYMMDD_HHMMSS.jsonl`
* `outputs/results_YYYYMMDD_HHMMSS.csv`
* `outputs/<nome_do_pdf>.json`

---
