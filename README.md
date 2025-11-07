
# ENTER AI Fellowship — PDF Extractor (VS3)

Extrator de campos estruturados a partir de PDFs brasileiros. Combina:
- heurísticas layout‑aware (regex + validadores BR),
- aprendizado leve por **overrides** e exemplos (SqliteDict),
- fallback opcional por **LLM** com orçamento de tempo (SLA).

> **Entrega**: pronto para uso com FastAPI. Você pode rodar localmente (uvicorn), via Docker, ou usar o script CLI.
> Endpoints: `/extract`, `/extract_bulk`, `/health`, `/metrics`, `/llm_selftest`.

---

## 1) Desafios mapeados → decisões e soluções

1. **OCR ruidoso e variação de layout**  
   _Solução_: normalização de texto + regex canônicas por campo; quando disponível, parsing de quadros/KV; limpeza de ruído.

2. **Campos com múltiplos formatos BR** (CPF, CNPJ, CEP, telefone, datas)  
   _Solução_: validadores/normalizadores específicos (aumentam confiança quando válidos).

3. **Generalização sem overfitting em rótulos**  
   _Solução_: ensemble por campo (KV → regex default → padrões BR → hints + fuzzy) com **boost por acordo**.

4. **Templates variados da mesma entidade** (ex.: OAB)  
   _Solução_: **overrides** aprendidos (global + por `template_id`) com persistência em SqliteDict.

5. **SLA e custo com LLM**  
   _Solução_: orçamento temporal — só aciona LLM se `time_left > safety_margin` e houver campos faltando/baixa confiança. Flags para forçar.

6. **Qualidade/segurança do retorno**  
   _Solução_: `STRICT_MODE` + `MIN_CONF`. Campo abaixo do limiar → `null`. Metadados trazem `conf`, `conf_agg`, `validation` por campo.

---

## 2) Arquitetura (resumo)

- **FastAPI** (`enter_ai_fellowship_extractor_vs3.py`)
  - Núcleo `extract_impl`: extrai texto (PyPDF2/pdfminer), aplica heurísticas, pós‑processa, decide LLM, valida e aprende overrides.
  - Persistência leve: `extract_knowledge.sqlite` (SqliteDict).
  - Métricas Prometheus em `/metrics`.

```text
repo/
├─ enter_ai_fellowship_extractor_vs3.py      # API e CLI (já no seu projeto)
├─ README.md
├─ requirements.txt
├─ Dockerfile
├─ schema_oab.json                           # exemplo de schema (dict)
├─ schema_tela_sistema.json                  # exemplo de schema (dict)
└─ dataset.json                              # exemplo para CLI/batch
```

> **Observação:** Se você já tem o arquivo `enter_ai_fellowship_extractor_vs3.py`, basta colocar estes arquivos na raiz do repositório. A API continuará a funcionar como antes.

---

## 3) Como executar

### 3.1. Requisitos

- Python 3.10+  
- (Opcional) Chave OpenAI para fallback LLM.

### 3.2. Instalação

```bash
python -m venv .venv
source .venv/bin/activate           # Linux/macOS
# .venv\Scripts\Activate.ps1      # PowerShell (Windows)

pip install -r requirements.txt
```

### 3.3. Variáveis de ambiente (opcionais)

```bash
# LLM
export USE_LLM=1                    # 0 desativa, 1 ativa
export OPENAI_API_KEY=sk-...        # obrigatório se USE_LLM=1
export LLM_MODEL=gpt-5-mini         # default gpt-5-mini
export FORCE_LLM_ALL=0              # 1 força LLM em todos os campos

# SLA e qualidade
export DEADLINE_S=20                # prazo por documento (s)
export LLM_SAFETY_MARGIN_S=1.0      # margem mínima para ainda chamar LLM
export MIN_CONF=0.85                # limiar para STRICT_MODE
export STRICT_MODE=1                # 1 aplica MIN_CONF → valores abaixo viram null
```

PowerShell equivalente:
```powershell
$env:USE_LLM="1"
$env:OPENAI_API_KEY="sk-..."
$env:LLM_MODEL="gpt-5-mini"
$env:FORCE_LLM_ALL="0"
$env:DEADLINE_S="20"
$env:LLM_SAFETY_MARGIN_S="1.0"
$env:MIN_CONF="0.85"
$env:STRICT_MODE="1"
```

### 3.4. Subir a API

```bash
uvicorn enter_ai_fellowship_extractor_vs3:app --host 0.0.0.0 --port 8765
```

### 3.5. Healthcheck e auto‑teste LLM

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/llm_selftest
```

Saída esperada (trecho):
```json
{"use_llm_effective": true, "llm_init": "ok", "llm_model":"gpt-5-mini", ...}
{"ok": true, "api_mode": "chat.completions", "raw": "ok"}
```

---

## 4) Como usar

### 4.1. Endpoint `/extract` (API)

- **Método**: `POST multipart/form-data`  
- **Campos**:
  - `label`: string (ex.: `carteira_oab` ou `tela_sistema` ou `auto`)
  - `extraction_schema`: JSON string (dict ou lista)
  - `pdf`: arquivo PDF

#### Exemplo (bash)

```bash
curl -X POST "http://127.0.0.1:8765/extract"   -F 'label=carteira_oab'   -F 'extraction_schema={"nome":"","inscricao":"","seccional":"","subsecao":"","categoria":"","telefone_profissional":"","endereco_profissional":"","situacao":""}'   -F 'pdf=@files/oab_1.pdf' | jq .
```

#### Exemplo (PowerShell)

```powershell
$mp = [System.Net.Http.MultipartFormDataContent]::new()
$mp.Add([System.Net.Http.StringContent]::new('carteira_oab'), 'label')
$schema = '{"nome":"","inscricao":"","seccional":"","subsecao":"","categoria":"","telefone_profissional":"","endereco_profissional":"","situacao":""}'
$mp.Add([System.Net.Http.StringContent]::new($schema), 'extraction_schema')
$mp.Add([System.Net.Http.StreamContent]::new([System.IO.File]::OpenRead('files\oab_1.pdf')), 'pdf', 'oab_1.pdf')
$client = [System.Net.Http.HttpClient]::new()
$r = $client.PostAsync('http://127.0.0.1:8765/extract', $mp).Result
$r.Content.ReadAsStringAsync().Result
```

**Resposta (exemplo OAB):**
```json
{
  "output": {
    "nome": "SON GOKU",
    "inscricao": "101943",
    "seccional": "PR",
    "subsecao": "Conselho Seccional - Paraná",
    "categoria": "Suplementar",
    "telefone_profissional": null,
    "endereco_profissional": "Rua ... 123, Curitiba/PR, CEP 80000-000",
    "situacao": "Situação Regular"
  },
  "meta": {
    "label": "carteira_oab",
    "used_llm": true,
    "conf": { "...": 0.93 },
    "conf_agg": 0.91,
    "strict_mode": true,
    "validation": { "inscricao": {"meets_min_conf": true} },
    "llm_debug": { "need_fallback": ["categoria","subsecao"], "model": "gpt-5-mini" }
  }
}
```

### 4.2. Endpoint `/extract_bulk`

- **Método**: `POST multipart/form-data`  
- Mesmos campos, mas `pdfs` recebe N arquivos.

```bash
curl -X POST "http://127.0.0.1:8765/extract_bulk"   -F 'label=tela_sistema'   -F 'extraction_schema={"data_base":"","data_vencimento":"","quantidade_parcelas":"","produto":"","sistema":"","tipo_de_operacao":"","tipo_de_sistema":"","valor_parcela":"","pesquisa_por":"","pesquisa_tipo":"","selecao_de_parcelas":"","total_de_parcelas":"","cidade":""}'   -F 'pdfs=@files/tela_sistema_1.pdf'   -F 'pdfs=@files/tela_sistema_2.pdf' | jq .
```

### 4.3. Script (CLI)

Processar um `dataset.json`:

```bash
python enter_ai_fellowship_extractor_vs3.py --dataset dataset.json --outdir outputs
```

Ou uma pasta de PDFs com o mesmo schema:

```bash
python enter_ai_fellowship_extractor_vs3.py   --pdf_dir files/   --label carteira_oab   --schema schema_oab.json   --outdir outputs
```

Gera:
- `outputs/results_YYYYmmdd_HHMMSS.jsonl`
- `outputs/results_YYYYmmdd_HHMMSS.csv`
- `outputs/<arquivo>.json` (por PDF)

---

## 5) Esquema (schema) — formato

- **Lista**: `["nome","inscricao","seccional","subsecao", ...]`  
- **Dict** (recomendado):  
  ```json
  {
    "inscricao": "apenas dígitos",
    "seccional": "UF",
    "subsecao": "texto livre (sem UF colada)",
    "telefone_profissional": "telefone BR",
    "data_vencimento": "dd/mm/aaaa"
  }
  ```
  > Dica: descrições como “apenas dígitos” habilitam um pós‑processo que remove tudo que não é número.

---

## 6) Ajuste fino do uso da LLM

- Garanta `USE_LLM=1` e `OPENAI_API_KEY` definido.  
- Se as heurísticas passam do limiar e você **quer** LLM:
  - `FORCE_LLM_ALL=1` (força em todos os campos) **ou**
  - aumente `MIN_CONF` (ex.: `0.90`) para tornar mais campos elegíveis ao fallback.
- Se o SLA está curto:
  - aumente `DEADLINE_S` (ex.: `20`)
  - reduza `LLM_SAFETY_MARGIN_S` (ex.: `1.0`).

Verifique `meta.llm_debug` para auditar a decisão.

---

## 7) Métricas Prometheus

```bash
curl http://127.0.0.1:8765/metrics
```

---

## 8) Segurança e privacidade

- Metadados mascaram PII (últimos dígitos de CPF/CNPJ).  
- Não persistimos o PDF. Apenas overrides/exemplos de valores/regex.  
- Recomenda-se rodar em ambiente isolado quando usar chaves LLM.

---

## 9) Roadmap curto

- Aprimorar learning‑to‑extract por few‑shot memory.  
- UI mínima (grids + upload).  
- Melhorar `template_id` com embeddings de cabeçalho/rodapé.

---

## 10) Licença

MIT (sugestão).
