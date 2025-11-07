from __future__ import annotations
import io, os, re, json, time, csv, hashlib, argparse, logging, sys
from typing import Dict, Any, Optional, Tuple, List, Iterable, Union
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader
try:
    from docling.document_converter import DocumentConverter
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
    DOCILING_AVAILABLE = True
    _docling_converter = DocumentConverter()
except Exception:
    DOCILING_AVAILABLE = False
    _docling_converter = None

from sqlitedict import SqliteDict
import httpx, certifi
from openai import OpenAI
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
try:
    from rapidfuzz import fuzz, process
except Exception:
    fuzz = None
    process = None
ROOT_DIR = Path(__file__).resolve().parent

# Caminhos padrão (usados no CLI e batch)
DEFAULT_PDF_DIR      = Path(os.getenv("PDF_DIR", r"C:\Users\Marlon_Kelly\PycharmProjects\treino\Enter\ai-fellowship-data-main\files"))
DEFAULT_SCHEMA_PATH  = Path(os.getenv("SCHEMA_PATH", str(ROOT_DIR / "schema.json")))
DEFAULT_DATASET_PATH = Path(os.getenv("DATASET_PATH", str(ROOT_DIR / "dataset.json")))
DEFAULT_OUTDIR       = Path(os.getenv("OUTDIR", str(ROOT_DIR / "outputs")))
DEFAULT_LABEL        = os.getenv("LABEL", "carteira_oab")

# Controle de SLA e qualidade (usados no núcleo de extração)
DEADLINE_S           = float(os.getenv("DEADLINE_S", "9.0"))
LLM_SAFETY_MARGIN_S  = float(os.getenv("LLM_SAFETY_MARGIN_S", "2.0"))
MIN_CONF             = float(os.getenv("MIN_CONF", "0.72"))
STRICT_MODE          = os.getenv("STRICT_MODE", "1") == "1"

# LLM (todos usados em init/decisão/custo)
USE_LLM              = os.getenv("USE_LLM", "0") == "1"           # default OFF
FORCE_LLM_ALL        = os.getenv("FORCE_LLM_ALL", "0") == "1"     # não forçar em produção
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL            = os.getenv("LLM_MODEL", "gpt-5-mini")

# Preços por 1K tokens (valores “$0.15/1M” e “$0.60/1M” → por 1K)
LLM_PRICE_IN_1K      = float(os.getenv("LLM_PRICE_IN_1K",  "0.00015"))
LLM_PRICE_OUT_1K     = float(os.getenv("LLM_PRICE_OUT_1K", "0.00060"))

# Acumulador de uso de tokens (consumido e zerado a cada extração)
LLM_TOKENS_LAST = {"prompt": 0, "completion": 0, "total": 0}

oai_client = None
_llm_init_reason = "disabled"
try:
    if USE_LLM and OPENAI_API_KEY:
        http_client = httpx.Client(
            http2=False,
            timeout=30.0,
            verify=certifi.where(),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        oai_client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
        _llm_init_reason = "ok"
    else:
        _llm_init_reason = "USE_LLM=0 or no API key"
except Exception as e:
    oai_client = None
    _llm_init_reason = f"erro: {type(e).__name__}"
    USE_LLM = False
# =================== LLM — CHAMADA PADRONIZADA (SEM TEMPERATURE) ===================
# Observações:
# - NÃO passamos 'temperature' (modelo não aceita).
# - Para Chat Completions: usamos 'response_format={"type":"json_object"}' SOMENTE se require_json=True.
# - Para Responses API: NÃO usar 'response_format'; forçamos JSON pelo prompt.
# - Nunca levantamos exceção; retornamos None em caso de falha/vazio.
# - Contabilizamos tokens em LLM_TOKENS_LAST se a API retornar usage.

def _get_openai_client() -> OpenAI:
    # Reutiliza o cliente já inicializado (oai_client); se não houver, cria um default.
    try:
        return oai_client if oai_client is not None else OpenAI()
    except Exception:
        return OpenAI()

def _update_token_usage_from_chat(resp) -> None:
    try:
        pu = int(getattr(resp, "usage", {}).get("prompt_tokens", 0))
        cu = int(getattr(resp, "usage", {}).get("completion_tokens", 0))
    except Exception:
        pu = cu = 0
    LLM_TOKENS_LAST["prompt"] = pu
    LLM_TOKENS_LAST["completion"] = cu
    LLM_TOKENS_LAST["total"] = pu + cu

def _update_token_usage_from_responses(resp) -> None:
    # Alguns SDKs retornam usage diferente; tentamos obter com segurança.
    try:
        usage = getattr(resp, "usage", None)
        if isinstance(usage, dict):
            pu = int(usage.get("prompt_tokens", 0))
            cu = int(usage.get("completion_tokens", 0))
        else:
            pu = int(getattr(usage, "prompt_tokens", 0))
            cu = int(getattr(usage, "completion_tokens", 0))
    except Exception:
        pu = cu = 0
    LLM_TOKENS_LAST["prompt"] = pu
    LLM_TOKENS_LAST["completion"] = cu
    LLM_TOKENS_LAST["total"] = pu + cu

def llm_json_complete(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 768,
    require_json: bool = True,
    use_responses_api: Optional[bool] = None,  # None → decide por USE_RESPONSES_API env; False → Chat; True → Responses
) -> Optional[Dict[str, Any]]:
    """
    Executa uma chamada LLM e tenta retornar um JSON (dict) quando require_json=True;
    caso contrário retorna {"text": <conteúdo>}. Em falha, retorna None.
    """
    if not USE_LLM or not OPENAI_API_KEY:
        return None

    client = _get_openai_client()
    # Decide rota: Chat Completions (padrão) vs Responses API (opcional)
    if use_responses_api is None:
        use_responses_api = bool(int(os.getenv("USE_RESPONSES_API", "0")))

    try:
        if not use_responses_api:
            # ===== Caminho 1 — Chat Completions =====
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt if not require_json else (
                    user_prompt + "\n\nResponda APENAS com um JSON válido (sem texto fora do JSON)."
                )},
            ]
            kwargs = {
                "model": LLM_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if require_json:
                # 'response_format' é aceito nos SDKs mais novos para forçar JSON no chat
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)
            finish = resp.choices[0].finish_reason
            content = (resp.choices[0].message.content or "").strip()

            _update_token_usage_from_chat(resp)

            if not content:
                logger.info(json.dumps({"event": "llm_empty_content", "method": "chat.completions", "finish_reason": finish}, ensure_ascii=False))
                return None
            if finish == "length":
                logger.warning(json.dumps({"event": "llm_truncated", "method": "chat.completions", "hint": "aumente max_tokens"}, ensure_ascii=False))

            return json.loads(content) if require_json else {"text": content}

        else:
            # ===== Caminho 2 — Responses API =====
            # NUNCA passar 'response_format' aqui; reforçamos JSON pelo prompt.
            input_msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + (
                    "\n\nResponda APENAS com um JSON válido (sem texto fora do JSON)." if require_json else ""
                )},
            ]
            resp = client.responses.create(
                model=LLM_MODEL,
                input=input_msgs,
                max_output_tokens=max_tokens,
            )

            # Responses API: use 'output_text' (campo agregado) quando existir.
            content = (getattr(resp, "output_text", None) or "").strip()
            finish = getattr(resp, "finish_reason", None)

            _update_token_usage_from_responses(resp)

            if not content:
                logger.info(json.dumps({"event": "llm_empty_content", "method": "responses"}, ensure_ascii=False))
                return None
            if finish == "length":
                logger.warning(json.dumps({"event": "llm_truncated", "method": "responses", "hint": "aumente max_output_tokens"}, ensure_ascii=False))

            return json.loads(content) if require_json else {"text": content}

    except json.JSONDecodeError as e:
        logger.warning(json.dumps({"event": "llm_json_decode_error", "err": str(e)}, ensure_ascii=False))
        return None
    except TypeError as e:
        # Padrão do seu log: "unexpected keyword argument 'response_format'".
        logger.warning(json.dumps({"event": "llm_type_error", "err": str(e), "hint": "verifique argumentos do SDK/rota escolhida"}, ensure_ascii=False))
        return None
    except Exception as e:
        logger.warning(json.dumps({"event": "llm_generic_error", "err": str(e)}, ensure_ascii=False))
        return None

# =================== Logging JSON ===================

class JsonFormatter(logging.Formatter):
    def format(self, record):
        d = {
            "level": record.levelname,
            "time": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "msg": record.getMessage(),
            "name": record.name,
        }
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, ensure_ascii=False)

logger = logging.getLogger("extractor")
logger.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(JsonFormatter())
logger.handlers = [h]

# =================== Métricas Prometheus ===================

MET_LATENCY = Histogram("extract_latency_seconds", "Latência por extração (s)")
MET_LATENCY_BULK = Histogram("extract_bulk_latency_seconds", "Latência por lote (s)")
MET_SUCCESS = Counter("extract_success_total", "Total de extrações com sucesso")
MET_FAIL = Counter("extract_fail_total", "Total de extrações com falha")
MET_USED_LLM = Counter("extract_used_llm_total", "Extrações que acionaram LLM")
MET_CONF_AGG = Histogram("extract_conf_agg", "Confiança agregada por documento", buckets=[0, .5, .6, .7, .8, .9, .95, 1.0])
GAUGE_STRICT = Gauge("extract_strict_mode", "Strict mode ativo (1/0)")
GAUGE_STRICT.set(1 if STRICT_MODE else 0)

# =================== Tipos e utilidades ===================

SchemaType = Union[Dict[str, str], List[str]]
EXTRACTOR_VERSION = "VS3-1.1.1"

DB_PATH = os.getenv("EXTRACT_DB", str(ROOT_DIR / "extract_knowledge.sqlite"))
KNOWLEDGE = SqliteDict(DB_PATH, autocommit=True)
SESSION_CACHE: dict[str, Dict[str, Any]] = {}
# ===== Patch A: helpers de hints, template-id e "digits only" =====

def dynamic_hints_for_field(field: str, description: str) -> list[str]:
    """Gera hints a partir do nome do campo + descrição do schema (semântica local)."""
    base = {field}
    desc = (description or "").lower()
    toks = re.findall(r"[a-zà-ÿ]{3,}", desc, flags=re.I)
    stop = {"numero","número","data","nome","valor","tipo","endereco","endereço","documento","id","codigo","código"}
    base |= {t for t in toks if t not in stop and len(t) >= 4}

    # sinônimos comuns por família de campo
    f = field.lower()
    syn = {
        "inscricao": {"inscrição","registro","matrícula","nº","n°"},
        "telefone": {"telefone","celular","contato"},
        "cnpj": {"cnpj","empresa"},
        "cpf": {"cpf","pessoal"},
        "rg": {"rg","identidade"},
        "cep": {"cep","postal"},
        "cidade": {"cidade","município","municipio","localidade"},
    }
    for k, vals in syn.items():
        if k in f:
            base |= vals
    return list({x.strip() for x in base if x.strip()})

def template_id_from_text(text: str) -> str:
    """Hash leve do cabeçalho/rodapé p/ identificar template e permitir overrides por template."""
    lines = [ln.strip().lower() for ln in text.splitlines() if ln.strip()]
    head = "\n".join(lines[:8]); tail = "\n".join(lines[-8:])
    return hashlib.sha1((head+"||"+tail).encode()).hexdigest()[:12]

def k_override_tpl(label: str, tpl: str, field: str) -> str:
    return f"override_regex_tpl::{label}::{tpl}::{field}"

def wants_digits_only(desc: str) -> bool:
    return bool(re.search(r"(somente|apenas)\s*d[ií]gitos|digits\s*only|(apenas|somente)\s*n[úu]meros", (desc or "").lower()))

def k_schema_seen(label: str) -> str:
    return f"schema_seen::{label}"
def k_full_schema(label: str) -> str:
    return f"full_schema::{label}"

def k_field_freq(label: str, field: str) -> str:
    return f"field_freq::{label}::{field}"


def k_override(label: str, field: str) -> str:
    return f"override_regex::{label}::{field}"

def k_examples(label: str, field: str) -> str:
    return f"examples::{label}::{field}"
def get_full_schema(label: str) -> set[str]:
    fs = _compat_get(k_full_schema(label)) or []
    return set(str(x).strip().lower() for x in fs if isinstance(x, str))

def add_fields_to_full_schema(label: str, fields: List[str]) -> None:
    fs = get_full_schema(label)
    for f in fields:
        if isinstance(f, str) and f.strip():
            fs.add(f.strip().lower())
    _compat_set(k_full_schema(label), sorted(fs))
# >>> ADICIONE ESTE HELPER (por exemplo logo após _compat_set) <<<
def _coerce_regex_pat(obj) -> Optional[str]:
    if isinstance(obj, str):
        return obj.strip() or None
    if isinstance(obj, list):
        for el in reversed(obj):
            if isinstance(el, str) and el.strip():
                return el.strip()
    if isinstance(obj, dict):
        for k in ("pattern", "regex", "pat"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def bump_field_freq(label: str, field: str) -> None:
    key = k_field_freq(label, field)
    try:
        _compat_set(key, int((_compat_get(key) or 0)) + 1)
    except Exception:
        _compat_set(key, 1)

def _compat_get(key_str: str):
    try:
        return KNOWLEDGE[key_str]
    except KeyError:
        return None

def _compat_set(key_str: str, value):
    KNOWLEDGE[key_str] = value

# topo do arquivo
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None

def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    """Extrai texto com fallback (PyPDF2 -> pdfminer)."""
    # 1) PyPDF2 (rápido)
    try:
        with io.BytesIO(pdf_bytes) as bio:
            reader = PdfReader(bio, strict=False)
            if getattr(reader, "is_encrypted", False):
                try: reader.decrypt("")
                except Exception: return ""
            if not getattr(reader, "pages", None) or len(reader.pages)==0:
                return ""
            txt = reader.pages[0].extract_text() or ""
            if txt.strip():
                return txt
    except Exception:
        pass
    # 2) pdfminer (recall ↑)
    if pdfminer_extract_text:
        try:
            with io.BytesIO(pdf_bytes) as bio:
                txt2 = pdfminer_extract_text(bio) or ""
                return txt2
        except Exception:
            return ""
    return ""

def docling_pdf_to_markdown(pdf_bytes: bytes) -> Optional[str]:
    """
    Tenta converter o PDF em Markdown estruturado usando Docling.
    Retorna Markdown (str) ou None se falhar/desabilitado.
    """
    if not DOCILING_AVAILABLE or _docling_converter is None:
        return None
    try:
        # Conversão → DoclingDocument
        doc = _docling_converter.convert(source=pdf_bytes).document  # docling aceita bytes também
        # Serialização para Markdown
        md = MarkdownDocSerializer(doc=doc).serialize().text
        return md
    except Exception:
        return None

def extract_kv_grid_from_docling_markdown(md: str) -> dict[str, str]:
    """
    Extrai pares chave:valor de um Markdown Docling (linhas 'Label: Valor' e células simples).
    Deixa o resultado pronto para _fx_kv().
    """
    kv = {}
    for raw in md.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        # Captura "Label: value"
        m = re.match(r"(?i)^([A-Za-zÀ-ÿ0-9\/\.\-\s]{2,40})\s*[:\-–]\s*(.+)$", ln)
        if m:
            lab = re.sub(r"\s{2,}", " ", m.group(1)).strip().lower()
            val = clean_value(m.group(2))
            if lab and val:
                kv[lab] = val
            continue
        # Captura células de tabela Markdown "| Label | Valor |"
        if ln.startswith("|") and ln.endswith("|"):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) == 2 and 2 <= len(cells[0]) <= 40:
                lab, val = cells
                if lab and val:
                    kv[lab.lower()] = clean_value(val)
    return kv
try:
    from docling_core.transforms.serializer.html import HtmlDocSerializer
    DOCILING_HTML = True
except Exception:
    DOCILING_HTML = False

def docling_pdf_to_html_tables(pdf_bytes: bytes) -> List[List[List[str]]]:
    """
    Extrai todas as tabelas como matriz de células usando Docling→HTML.
    Retorna lista de tabelas; cada tabela é uma lista de linhas; cada linha é lista de células.
    """
    if not DOCILING_AVAILABLE or _docling_converter is None or not DOCILING_HTML:
        return []
    try:
        doc = _docling_converter.convert(source=pdf_bytes).document
        html = HtmlDocSerializer(doc=doc).serialize().text
        # Parse leve de markdown-table a partir de HTML para evitar dependências extras
        # Estratégia simples: linhas <tr>, células <td>/<th>
        tables: List[List[List[str]]] = []
        for tbl in re.split(r"</?table[^>]*>", html, flags=re.I):
            if not re.search(r"</tr>", tbl, flags=re.I):
                continue
            rows: List[List[str]] = []
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, flags=re.I|re.S):
                cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.I|re.S)
                if cells:
                    row = [re.sub(r"<[^>]+>", " ", c) for c in cells]
                    row = [clean_value(re.sub(r"\s+", " ", c)) for c in row]
                    rows.append(row)
            if rows:
                tables.append(rows)
        return tables
    except Exception:
        return []

def extract_simple_kv_from_tables(tables: List[List[List[str]]]) -> dict[str, str]:
    """
    De tabelas 2-colunas (Label | Valor) ou 2 células/linha, cria um grid KV adicional.
    """
    kv = {}
    for rows in tables:
        for r in rows:
            if len(r) == 2 and 2 <= len(r[0]) <= 40:
                lab = r[0].strip().lower()
                val = clean_value(r[1])
                if lab and val:
                    kv[lab] = val
    return kv
# --- Correções automáticas de OCR/typos frequentes ---
OCR_CORRECTIONS = {
    r"\bverncimento\b": "vencimento",
    r"\bref[eê]rencia\b": "referência",
    r"\bsubse[çc]\w*\b": "subsecao",  # normaliza para 'subsecao'
    r"\bsitua[çc][aã]o\b": "situação",
}

def fix_ocr_typos(s: str) -> str:
    for k, v in OCR_CORRECTIONS.items():
        s = re.sub(k, v, s, flags=re.I)
    return s

def normalize_text(s: str) -> str:
    # limpeza anti-OCR e normalização básica
    s = s.replace("\u00A0", " ")
    s = fix_ocr_typos(s)
    s = re.sub(r"[ \t\r\f]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r" +", " ", s)
    s = re.sub(r"\s*\|\s*", " | ", s)  # colunas OCR com pipe
    return s.strip()


def normalize_schema(schema: SchemaType) -> Dict[str, str]:
    if isinstance(schema, dict):
        for k in schema.keys():
            if not isinstance(k, str):
                raise ValueError("Chaves do schema (dict) devem ser strings.")
        return schema
    if isinstance(schema, list):
        if not all(isinstance(k, str) for k in schema):
            raise ValueError("Schema (lista) deve conter somente nomes de campos (strings).")
        return {str(k): "" for k in schema}
    raise ValueError("Schema deve ser dict {campo: descrição} ou lista [campo,...].")

def schema_keys(schema: SchemaType) -> List[str]:
    return list(normalize_schema(schema).keys())

def schema_desc(schema: SchemaType, field: str) -> str:
    sd = normalize_schema(schema)
    return str(sd.get(field, ""))

def schema_subset(schema: SchemaType, fields: List[str]) -> Dict[str, str]:
    sd = normalize_schema(schema)
    return {f: sd.get(f, "") for f in fields}

# --------- Padrões BR ----------
BR_PATTERNS = {
    "cpf": r"(?<!\d)(\d{3}[\.\s]?\d{3}[\.\s]?\d{3}[-\s]?\d{2})(?!\d)",
    "cnpj": r"(?<!\d)(\d{2}[\.\s]?\d{3}[\.\s]?\d{3}[\/\s]?\d{4}[-\s]?\d{2})(?!\d)",
    "cep": r"(?<!\d)(\d{5}[-\s]?\d{3})(?!\d)",
    "telefone": r"(?:(?:\+?55\s*)?(?:\(?\d{2}\)?\s*)?(?:9\s*)?\d{4}[-\s]?\d{4}|\b\d{10,11}\b)",
    "data": r"(?<!\d)(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})(?!\d)",
    "inscricao": r"(?<!\d)(\d{4,12})(?!\d)",
}
BR_REGEX = {k: re.compile(v, re.UNICODE) for k, v in BR_PATTERNS.items()}
BR_REGEX["telefone"] = re.compile(BR_PATTERNS["telefone"], re.UNICODE)
# === Regras de cabeçalho e normalização por label (TELAS) ===
HEADER_TITLES = {
    "tela_sistema": re.compile(r"(?i)\bconsulta\s+de\s+cobran[çc]a\b")
}

def detect_screen_title(label: str, text: str) -> Optional[str]:
    if label in HEADER_TITLES:
        head = "\n".join([ln.strip() for ln in text.splitlines()[:8] if ln.strip()])
        if HEADER_TITLES[label].search(head):
            return "Consulta de Cobrança"
    return None

def _std_title_case(s: str) -> str:
    s = re.sub(r"\s{2,}", " ", (s or "").strip())
    return " ".join(w.capitalize() for w in s.split()) if s else s

def standardize_tela_values(out: Dict[str, Optional[str]], screen_title: Optional[str]) -> None:

    if not out:
        return
    # Corrige 'de Cobrança' truncado
    def _fix_cobranca(v: Optional[str]) -> Optional[str]:
        if not v:
            return v
        txt = v.strip()
        if re.search(r"(?i)(^|\b)de\s+cobran[çc]a\b", txt) or re.search(r"(?i)\bconsulta\s+de\s+cobran[çc]a\b", txt):
            return "Consulta de Cobrança"
        return _std_title_case(txt)

    for f in ("produto", "sistema", "tipo_de_operacao", "tipo_de_sistema"):
        if f in out and out.get(f):
            out[f] = _std_title_case(str(out[f]))

    for f in ("pesquisa_por", "pesquisa_tipo"):
        if f in out and out.get(f):
            out[f] = _fix_cobranca(str(out[f]))

    # Se detectamos claramente o título no topo, e 'pesquisa_por' estiver ausente/ruim, definimos
    if screen_title and (not out.get("pesquisa_por") or len(str(out.get("pesquisa_por"))) < 6):
        out["pesquisa_por"] = "Consulta de Cobrança"

FIELD_HINTS = {
    "nome": ["nome", "titular", "advogado", "advogada"],
    "inscricao": ["inscrição", "nº", "numero", "n°", "registro"],
    "seccional": ["seccional", "oab/", "uf", "seção"],
    "subsecao": ["subseção", "subsecao", "conselho seccional", "subsecção"],
    "categoria": ["categoria", "tipo", "classe"],
    "telefone_profissional": ["telefone", "contato", "celular"],
    "endereco_profissional": ["endereço", "endereco", "logradouro", "rua", "avenida"],
    "situacao": ["situação", "situacao", "status"],
    "pesquisa_por": ["pesquisa por"],
    "pesquisa_tipo": ["tipo de pesquisa", "pesquisa - tipo", "tipo"],
    "selecao_de_parcelas": ["seleção de parcelas", "selecao de parcelas"],
    "total_de_parcelas": ["total de parcelas", "qtd total", "quantidade total"],
}
# Vocabulário canônico (evita variação textual derrubar acerto)
CANON = {
    "produto": {
        "refinanciamento": "Refinanciamento",
        "empréstimo": "Empréstimo",
        "emprestimo": "Empréstimo",
        "consignado": "Consignado",
        "crédito pessoal": "Crédito Pessoal",
        "credito pessoal": "Crédito Pessoal",
        "cartao": "Cartão",
        "cartão": "Cartão",
        "veiculo": "Veículo",
        "veículo": "Veículo",
        "habitacao": "Habitação",
        "habitação": "Habitação",
    },
    "tipo_de_operacao": {
        "liquidacao": "Liquidação",
        "liquidação": "Liquidação",
        "parcelamento": "Parcelamento",
        "renegociacao": "Renegociação",
        "renegociação": "Renegociação",
        "consulta": "Consulta",
        "baixa": "Baixa",
    },
    "tipo_de_sistema": {
        "siscom": "Siscom",
        "sap": "SAP",
        "rm": "RM",
        "protheus": "Protheus",
        "totvs": "TOTVS",
    }
}

def _canonize(field: str, value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    f = field.lower()
    v = value.strip().lower()
    if f in CANON:
        # normaliza (contém/começa-com) para maior tolerância
        for k, canon in CANON[f].items():
            if v == k or v.startswith(k) or k in v:
                return canon
    return value

# ===== Auto/Genérico: mapeamento de campos → labels conhecidos =====
LABEL_SIGNATURES = {
    "carteira_oab": {
        "must_any": {"inscricao", "seccional", "subsecao", "categoria", "endereco_profissional", "situacao", "nome"},
        "text_clues": [r"(?i)Inscri[cç][aã]o\s+Seccional\s+Subse[cç][aã]o", r"(?i)\bOAB\b"]
    },
    "tela_sistema": {
        "must_any": {"data_base","data_vencimento","quantidade_parcelas","produto","sistema","tipo_de_operacao","tipo_de_sistema","valor_parcela","cidade"},
        "text_clues": [r"(?i)\bData\s*Base\b", r"(?i)\bSaldo\b\s.*\b(Vencido|Vencer|Geral)\b"]
    }
}

def _norm_keys(schema_obj: SchemaType) -> set[str]:
    try:
        sd = normalize_schema(schema_obj)
        return {str(k).strip().lower() for k in sd.keys()}
    except Exception:
        return set()

def guess_label_from_schema_and_text(schema_obj: SchemaType, text: str) -> str:
    keys = _norm_keys(schema_obj)
    # 1) pelo schema (interseção de campos)
    best = None; best_score = -1
    for lbl, sig in LABEL_SIGNATURES.items():
        overlap = len(keys & set(sig["must_any"]))
        if overlap > best_score:
            best, best_score = lbl, overlap

    # Se bateu pelo menos 1 campo característico, candidate-se
    candidate = best if best_score > 0 else None

    # 2) reforço por pistas no texto (se houver)
    if text and candidate:
        clues = LABEL_SIGNATURES[candidate].get("text_clues", [])
        for pat in clues:
            if re.search(pat, text):
                return candidate

    # 3) checa se outro label tem pistas no texto (mesmo com sobreposição menor)
    if text:
        for lbl, sig in LABEL_SIGNATURES.items():
            for pat in sig.get("text_clues", []):
                if re.search(pat, text):
                    return lbl

    # 4) fallback genérico
    return candidate or "generic_doc"
# Campos digitados/rotulados de forma diferente no dataset/ocr
FIELD_SYNONYMS = {
    "telefone": "telefone_profissional",
    "endereco": "endereco_profissional",
    "subseção": "subsecao", "subsecção": "subsecao",
    "oab_uf": "seccional",
    "data_verncimento": "data_vencimento",   # typo comum
    "data_referencia":  "data_base",         # variação aceita
}
FIELD_SYNONYMS.update({
    "data_verncimento": "data_vencimento",     # reforça o typo
    "pesquisa - tipo": "pesquisa_tipo",        # variação com hífen
    "tipo": "pesquisa_tipo",                   # quando a tela usa só "Tipo"
    "sist": "sistema",                         # abreviação comum
    "valor_da_parcela": "valor_parcela",       # rótulo alternativo
})



LABEL_DEFAULT_REGEX: Dict[str, Dict[str, str]] = {
    "carteira_oab": {
        "nome": r"(?i)\bNome\b\s*[:\-–]?\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ '\-]{2,})",
        "inscricao": r"(?i)\bInscri[cç][aã]o\b\s*[:\-–]\s*(\d{4,12})",
        "seccional": r"(?i)\bSeccional\b\s*[:\-–]\s*([A-Za-z]{2})",
        "subsecao": r"(?i)\b(Subse[cç][aã]o|Conselho Seccional)\b\s*[:\-–]?\s*(.+)",
        "categoria": r"(?i)\bCategoria\b\s*[:\-–]\s*(.+)",
        "endereco_profissional": r"(?i)\bEndere[cç]o\s+Profissional\b\s*[:\-–]?\s*(.+)",
        "situacao": r"(?i)\bSitua[cç][aã]o\b\s*[:\-–]?\s*(.+)",
    }
}
# Padrões de captura ancorados para telas 'tela_sistema'
OVERRIDES_BY_LABEL = {
        "tela_sistema": {
        # Produto costuma aparecer com rótulo
        "produto": r"(?im)^\s*(?:produto|prod\.?)\s*[:\-–]\s*([A-Za-zÀ-ÿ ]{3,40})\b",
        # Tipo de operação (quando aparece no topo/combobox)
        "tipo_de_operacao": r"(?im)^\s*(?:opera[cç][aã]o|tipo\s*de\s*opera[cç][aã]o)\s*[:\-–]\s*([A-Za-zÀ-ÿ ]{3,40})\b",
        # Tipo de sistema
        "tipo_de_sistema": r"(?im)^\s*(?:sistema|tipo\s*de\s*sistema)\s*[:\-–]\s*([A-Za-zÀ-ÿ ]{3,40})\b",
        # Valor parcela (se vier com rótulo)
        "valor_parcela": r"(?im)^\s*(?:valor\s*da\s*parcela|valor\s*parcela|parcela)\s*[:\-–]\s*([^\s].{0,40}?)\s*$",
    },

}

# TELAS DE SISTEMA
LABEL_DEFAULT_REGEX.setdefault("tela_sistema", {}).update({
    # Datas (mantém)
    "data_base": r"(?mi)\bData\s*Base\b[:\s]*([0-9]{2}[/\-\.][0-9]{2}[/\-\.][0-9]{2,4})",
    "data_vencimento": r"(?mi)\bData\s*Venc(?:imento)?\b[:\s]*([0-9]{2}[/\-\.][0-9]{2}[/\-\.][0-9]{2,4})",

    # Qtd parcelas — cobre "Qtd/Qtde/Quantidade" e variações com/sem "de"
    "quantidade_parcelas": r"(?mi)\b(?:Qtd\.?|Qtde\.?|Quantidade)(?:\s*de)?\s*Parcelas?\b[:\s]*([0-9]{1,3})",



    # Sistema/Produto/Operação — somente quando houver rótulo claro
    "produto": r"(?im)^\s*(?:Produto|Prod\.?)\s*[:\-–]\s*([A-Za-zÀ-ÿ0-9 _/\-\.]{3,40})\s*$",
    "sistema": r"(?im)^\s*(?:Sistema|Sist\.?)\s*[:\-–]\s*([A-Za-zÀ-ÿ0-9 _/\-\.]{3,40})\s*$",
    "tipo_de_operacao": r"(?im)^\s*(?:Tipo\s*de\s*Opera[cç][aã]o|Opera[cç][aã]o|Selecionada)\s*[:\-–]?\s*([A-Za-zÀ-ÿ0-9 _/\-\.]{3,40})\s*$",
    "tipo_de_sistema": r"(?im)^\s*(?:Tipo\s*de\s*Sistema|Tipo\s*Sistema)\s*[:\-–]?\s*([A-Za-zÀ-ÿ0-9 _/\-\.]{3,40})\s*$",
    # Valor parcela
    "valor_parcela": r"(?im)^(?:Vlr\.?\s*Parc\.?|Val(?:or)?\s*(?:da\s*)?parcela|Parcela)\s*[:\-–]?\s*(?:R?\$?\s*)?([0-9]{1,3}(?:\.[0-9]{3})*(?:,\d{2})|\d+(?:,\d{2})?)\b",

    # Pesquisa (mantém, mas aceita "Sistema: X" em pesquisa_tipo)
    "pesquisa_por": r"(?mi)\bPesquisa\s*por\b[:\s\-]*([A-Za-zÀ-ÿ0-9 _\-/]+)",
    "pesquisa_tipo": r"(?mi)\b(?:Tipo\s*de\s*Pesquisa|Pesquisa\s*-\s*Tipo|Tipo|Sistema)\b[:\s\-]*([A-Za-zÀ-ÿ0-9 _\-/\.]+)",

    # Seleção/Total de parcelas — cobre "Selecionada(s)" e "Selecionadas: N"
    "selecao_de_parcelas": r"(?mi)\bSele[cç][aã]o\s*de\s*Parcelas?\b[:\s\-]*([0-9]{1,3})|^\s*Selecionad[ao]s?\s*[:\-–]?\s*([0-9]{1,3})\b",
    "total_de_parcelas": r"(?mi)\bTotal\s*de\s*Parcelas?\b[:\s\-]*([0-9]{1,3})",

    # Cidade (mantém)
    "cidade": r"(?mi)\bCidade\b[:\s\-]*([A-Za-zÀ-ÿ ]+)(?:\s+U\.?F[:\s\-]*[A-Z]{2})?(?:\s+CEP[:\s\-]*\d{5}\-?\d{3})?"

})
LABEL_DEFAULT_REGEX.setdefault("tela_sistema", {}).update({
    "data_referencia": r"(?i)\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b"
})
# --- Reforço OAB: categoria/subsecao/telefone/endereco com ancoragem e multiline ---
LABEL_DEFAULT_REGEX.setdefault("carteira_oab", {}).update({
    # Categoria: aceita rótulo explícito e evita capturar "Situação"
    "categoria": r"(?im)^\s*Categoria\s*[:\-–]\s*([A-Za-zÀ-ÿ ]{3,40})\s*$",

    # Subsecao: aceita "Subseção", "Subsecao" e "Conselho Seccional", usando toda a linha até o fim
    "subsecao": r"(?im)^\s*(?:Subse[cç][aã]o|Conselho\s+Seccional)\s*[:\-–]?\s*(.+)$",

    # Telefone profissional: caça labels e formatos brasileiros
    "telefone_profissional": r"(?im)^\s*(?:Telefone|Contato|Telefone\s+Profissional)\s*[:\-–]?\s*([+()0-9 \-]{8,20})\s*$",

    # Endereço profissional: pega a linha após o rótulo e concatena até bater um stopper comum
    "endereco_profissional": r"(?is)\bEndere[cç]o\s+Profissional\b\s*[:\-–]?\s*([^\n]+?)(?:\n\s*(?:CEP|Telefone|Situa[cç][aã]o|Categoria)\b|$)"
})

LABEL_DEFAULT_REGEX.setdefault("generic_doc", {}).update({
    # tenta campos frequentes em documentos brasileiros
    "nome": r"(?i)\bNome\b\s*[:\-–]?\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ '\-]{2,})",
    "cpf": BR_PATTERNS["cpf"],
    "cnpj": BR_PATTERNS["cnpj"],
    "cep": BR_PATTERNS["cep"],
    "data": BR_PATTERNS["data"],
})


def _digits(s: str) -> str:
    return re.sub(r"\D+", "", s or "")

def validate_cpf(s: str) -> bool:
    n = _digits(s)
    if len(n) != 11 or len(set(n)) == 1:
        return False
    def dv(ns, k):
        s_ = sum(int(d)*w for d, w in zip(ns, range(k, 1, -1)))
        r = (s_*10) % 11
        return 0 if r == 10 else r
    d1 = dv(n[:9], 10)
    d2 = dv(n[:10], 11)
    return d1 == int(n[9]) and d2 == int(n[10])

def validate_cnpj(s: str) -> bool:
    n = _digits(s)
    if len(n) != 14 or len(set(n)) == 1:
        return False
    def calc(ns, pes):
        s_ = sum(int(d)*p for d, p in zip(ns, pes))
        r = s_ % 11
        return 0 if r < 2 else 11 - r
    p1 = [5,4,3,2,9,8,7,6,5,4,3,2]
    p2 = [6] + p1
    d1 = calc(n[:12], p1)
    d2 = calc(n[:13], p2)
    return d1 == int(n[12]) and d2 == int(n[13])

def normalize_cpf(s: str) -> str:
    n = _digits(s)
    return f"{n[0:3]}.{n[3:6]}.{n[6:9]}-{n[9:11]}" if len(n)==11 else s

def normalize_cnpj(s: str) -> str:
    n = _digits(s)
    return f"{n[0:2]}.{n[2:5]}.{n[5:8]}/{n[8:12]}-{n[12:14]}" if len(n)==14 else s

def normalize_cep(s: str) -> str:
    n = _digits(s)
    return f"{n[:5]}-{n[5:8]}" if len(n)==8 else s

def parse_date_to_iso(s: str) -> Optional[str]:
    s = s.strip()
    m = re.match(r"^\s*(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})\s*$", s)
    if not m:
        return None
    d, mo, y = map(int, m.groups())
    if y < 100:
        y += 2000 if y < 50 else 1900
    try:
        return datetime(y, mo, d).strftime("%Y-%m-%d")
    except ValueError:
        return None

def compile_hint_regex(hints: List[str]):
    if not hints:
        return None
    pattern = r"(?:" + "|".join(map(re.escape, hints)) + r")\s*[:\-–]?\s*(.+)"
    return re.compile(pattern, re.IGNORECASE)

def clean_value(v: str) -> str:
    v = v.strip()
    v = v.split("\n")[0].strip()
    v = re.sub(r"^(?:nº|n°|no\.|n\.|numero|num\.|inscri[cç][aã]o)\s*[:\-–]?\s*", "", v, flags=re.I)
    v = re.sub(r"\s{2,}", " ", v)
    return v.strip().strip(':').strip('-–').strip()

LABEL_ONLY = {
    "profissional", "telefone profissional", "endereço profissional",
    "endereco profissional", "situação regular", "situacao regular",
    "categoria", "seccional", "subseção", "subsecao", "situação"
}

def postprocess_value(field: str, value: Optional[str], label: Optional[str] = None) -> Optional[str]:
    if value is None:
        return None

    v = value.strip().split("\n")[0].strip()
    v = re.sub(
        r"^(?:nº|n°|no\.|n\.|numero|num\.|inscri[cç][aã]o|seccional|subse[cç][aã]o|categoria|"
        r"situa[cç][aã]o|endere[cç]o\s+profissional|telefone|contato)\s*[:\-–]?\s*",
        "", v, flags=re.I
    )
    v = re.sub(r"\s{2,}", " ", v).strip()

    LABEL_ONLY = {
        "profissional", "telefone profissional", "endereço profissional", "endereco profissional",
        "situação regular", "situacao regular", "categoria", "seccional", "subseção", "subsecao", "situação"
    }
    if v.lower() in LABEL_ONLY:
        return None

    fl = field.lower()

    # Bloqueia “regular” fora de situacao
    if fl != "situacao" and v.lower() in {"regular", "situação regular", "situacao regular"}:
        return None

    # TELEFONE precisa ter dígitos suficientes
    if "telefone" in fl and not re.search(r"\d{8,}", v):
        return None

    # TELAS — filtro de poluição e normalização de strings curtas
    if label == "tela_sistema" and fl in {"sistema", "tipo_de_sistema", "produto", "tipo_de_operacao"}:
        s = re.sub(r"\s+", " ", v).strip()
        # rejeita cabeçalhos, datas, números longos, CPFs, CNPJs
        if re.search(r"\bsaldo\b", s.lower()) and re.search(r"\b(vencido|vencer|geral)\b", s.lower()):
            return None
        if re.search(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b", s):
            return None
        if sum(c.isdigit() for c in s) >= 6:
            return None
        if re.search(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b", s) or re.search(r"\b\d{2}\.?\d{3}\.?\d{3}\/?\d{4}-?\d{2}\b", s):
            return None
        s = re.sub(r"[^0-9A-Za-zÀ-ÿ _\-/\.]", " ", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        return s if len(s) >= 3 else None

    # Documentos BR
    if "cpf" in fl:
        return normalize_cpf(v) if validate_cpf(v) else None
    if "cnpj" in fl:
        return normalize_cnpj(v) if validate_cnpj(v) else None
    if "cep" in fl:
        return normalize_cep(v)

    # RG (evita datas coladas tipo ddmmaaaa)
    if fl == "numero_rg":
        d = re.sub(r"\D+", "", v)
        if len(d) == 8:
            return None
        return d if 5 <= len(d) <= 12 else None

    # Valor parcela — normaliza para "1234.56"
    if fl == "valor_parcela":
        if not re.search(r"\d", v):
            return None
        s = v.replace("R$", "").strip()
        s = re.sub(r"\.(?=\d{3}\b)", "", s)        # remove milhar
        s = s.replace(",", ".")
        s = re.sub(r"[^\d\.]", "", s)
        if s.count(".") > 1:
            parts = s.split(".")
            s = "".join(parts[:-1]) + "." + parts[-1]
        return s if re.match(r"^\d+(\.\d{1,2})?$", s) else s

    # Datas (qualquer campo com 'data' no nome): aceita e retorna ISO
    if any(k in fl for k in ["data", "nascimento", "emissao", "emissão", "referencia", "referência"]):
        iso = parse_date_to_iso(v)
        return iso or v

    # Cidade com padrão "Cidade/UF, CEP 00000-000"
    if fl == "cidade":
        m = re.search(r"(?i)^\s*([A-Za-zÀ-ÿ ]+)\s*/\s*([A-Za-z]{2})(?:\s*[,;]\s*|\s+)(?:CEP|Cep|cep)?\s*[:\s]*([0-9\- ]{8,9})?", v)
        if m:
            city = re.sub(r"\s{2,}", " ", m.group(1)).strip().title()
            uf = m.group(2).upper()
            cep_grp = m.group(3) or ""
            digits = re.sub(r"\D+", "", cep_grp)
            cep_fmt = f"{digits[:5]}-{digits[5:8]}" if len(digits) == 8 else None
            return f"{city}/{uf}" + (f", CEP {cep_fmt}" if cep_fmt else "")
        return " ".join(w.capitalize() for w in v.split())

    # Seccional: deixa só UF
    if fl == "seccional":
        m = re.search(r"([A-Za-z]{2})", v)
        return m.group(1).upper() if m else v

    # Subsecao: elimina UFs/números colados do início
    if fl == "subsecao":
        s = re.sub(r"^\s*(?:\d{4,12}\s+)?(?:[A-Z]{2}\s+)?", "", v).strip()
        if re.match(r"^\d{3,}|^[A-Z]{2}\w+", s):
            return None
        return s.title()

    # Situação
    if fl == "situacao":
        return "Situação Regular" if "regular" in v.lower() else " ".join(w.capitalize() for w in v.split())

    # Nome: evita endereço/número; exige 2 tokens alfa
    if fl == "nome":
        if re.search(r"\d", v) or re.search(r"(?i)\b(avenida|av\.?|rua|rodovia|alameda|travessa|praça|praca|estrada|lote|quadra|bairro|cep|nº|no\.|n\.)\b", v):
            return None
        tokens = re.findall(r"[A-Za-zÀ-ÿ'\-]{2,}", v)
        if len(tokens) < 2 or len(v) < 5:
            return None
        if re.fullmatch(r"[A-Za-zÀ-ÿ '\-]+", v):
            v = " ".join(w.capitalize() for w in v.split())
        return v

    # Inteiros esperados (parcelas) — corta datas coladas
    if fl in {"quantidade_parcelas", "selecao_de_parcelas", "total_de_parcelas"}:
        digits = re.sub(r"\D+", "", v)
        if not digits:
            return None
        d = digits.lstrip("0") or "0"
        if len(digits) == 8 or len(d) > 3:
            return None
        return d

    return v or None

# ========= Confidences calibradas por origem =========
CONF_BASE = {
    "kv": 0.88,
    "default_regex": 0.90,
    "br": 0.93,          # ↑ ligeiro (validações BR são fortes)
    "hints": 0.78,
    "template_override": 0.94,
    "global_override": 0.92,
}

def _agree_boost(c1: float, c2: float) -> float:
    """
    Se dois extratores independentes concordam no MESMO valor (após postprocess),
    aplique um boost de confiança (cap em 0.98).
    """
    base = max(c1, c2)
    inc  = 0.06 if base >= 0.85 else 0.10
    return min(0.98, base + inc)

def _value_eq(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    return re.sub(r"\s+", " ", a.strip().lower()) == re.sub(r"\s+", " ", b.strip().lower())

# --- Ensemble de extratores por campo ---
from typing import Callable  # garanta que Callable esteja importado

FieldExtractor = Callable[[str, str, str], Optional[Tuple[str, float]]]

def _fx_kv(field: str, label: str, text: str) -> Optional[Tuple[str, float]]:
    # Usa extract_kv_grid global se existir; caso contrário, uma versão local leve
    def _local_kv_grid(tx: str) -> dict[str, str]:
        kv = {}
        for raw in tx.splitlines():
            ln = re.sub(r"\s{2,}", "  ", raw.strip())
            if not ln:
                continue
            m = re.match(r"(?i)^([A-Za-zÀ-ÿ0-9\/\.\-\s]{2,30})\s*[:\-–]\s*(.+)$", ln)
            if m:
                lab = re.sub(r"\s{2,}", " ", m.group(1)).strip().lower()
                val = clean_value(m.group(2))
                if lab and val:
                    kv[lab] = val
                continue
            if "|" in ln:
                cols = [c.strip() for c in ln.split("|") if c.strip()]
                if len(cols) == 2 and 2 <= len(cols[0]) <= 30:
                    lab, val = cols
                    kv[lab.lower()] = clean_value(val)
        return kv

    _kv_fn = globals().get("extract_kv_grid", None)
    kv = _kv_fn(text) if callable(_kv_fn) else _local_kv_grid(text)

    # 1) match direto pelo nome do campo e hints
    for nm in [field] + FIELD_HINTS.get(field.lower(), []):
        if nm.lower() in kv:
            v = kv[nm.lower()]
            pv = postprocess_value(field, v, label=label)
            if pv:
                return pv, 0.88

    # 2) se houver grid (Docling), tenta fuzzy entre rótulos do grid e o nome do campo
    if kv and fuzz is not None:
        best = None
        best_score = -1
        fl = field.lower()
        for k_lab, v_val in kv.items():
            sc = max(fuzz.partial_ratio(fl, k_lab), fuzz.token_sort_ratio(fl, k_lab))
            if sc > best_score:
                best_score = sc
                best = (k_lab, v_val)
        if best and best_score >= 80:
            pv = postprocess_value(field, best[1], label=label)
            if pv:
                return pv, 0.82

    return None

def _fx_default_regex(field: str, label: str, text: str) -> Optional[Tuple[str, float]]:
    dm = LABEL_DEFAULT_REGEX.get(label, {})
    if field in dm:
        try:
            rg = re.compile(dm[field], re.I)
            m = rg.search(text)
            if m:
                val = clean_value(m.group(1) if m.groups() else m.group(0))
                pv = postprocess_value(field, val, label=label)
                if pv:
                    return pv, 0.90
        except re.error:
            pass
    return None

def _fx_br(field: str, label: str, text: str) -> Optional[Tuple[str, float]]:
    f = field.lower()
    if "cpf" in f:
        m = BR_REGEX["cpf"].search(text)
        if m and validate_cpf(m.group(1)):
            return normalize_cpf(m.group(1)), 0.97
    if "cnpj" in f:
        m = BR_REGEX["cnpj"].search(text)
        if m and validate_cnpj(m.group(1)):
            return normalize_cnpj(m.group(1)), 0.97
    if "cep" in f:
        m = BR_REGEX["cep"].search(text)
        if m:
            return normalize_cep(m.group(1)), 0.92
    if any(k in f for k in ["telefone","phone","celular"]):
        m = BR_REGEX["telefone"].search(text)
        if m:
            return postprocess_value(field, m.group(0), label=label), 0.86
    if any(k in f for k in ["data","nascimento","emissao","emissão"]):
        m = BR_REGEX["data"].search(text)
        if m:
            iso = parse_date_to_iso(clean_value(m.group(1)))
            if iso:
                return iso, 0.90
    if any(k in f for k in ["inscricao","inscrição","registro","matricula","matrícula"]):
        m = BR_REGEX["inscricao"].search(text)
        if m:
            return postprocess_value(field, clean_value(m.group(1)), label=label), 0.82
    return None

def _fx_hints(field: str, label: str, text: str, desc: str) -> Optional[Tuple[str, float]]:
    hints = list({*FIELD_HINTS.get(field.lower(), []), *dynamic_hints_for_field(field, desc)})
    if not hints:
        return None
    rg = compile_hint_regex(hints)
    if rg:
        for line in text.splitlines():
            m = rg.search(line)
            if m:
                pv = postprocess_value(field, clean_value(m.group(1)), label=label)
                if pv:
                    return pv, 0.78
    # Fallback: próxima linha após o hint
    lines = text.splitlines()
    for i, line in enumerate(lines[:-1]):
        if any(h.lower() in line.lower() for h in hints):
            nxt = lines[i+1].strip()
            if nxt:
                pv = postprocess_value(field, clean_value(nxt), label=label)
                if pv:
                    return pv, 0.73
    return None

def run_field_ensemble(field: str, label: str, text: str, desc: str) -> Tuple[Optional[str], float]:
    """
    Executa os 4 extratores (KV, default_regex, BR, Hints),
    escolhe o melhor e dá boost se houver acordo entre 2+.
    """
    tried: List[Tuple[str, Optional[str], float]] = []

    # 1) KV
    kv = _fx_kv(field, label, text)
    if kv: tried.append(("kv", kv[0], CONF_BASE["kv"]))

    # 2) Default regex
    dr = _fx_default_regex(field, label, text)
    if dr: tried.append(("default_regex", dr[0], CONF_BASE["default_regex"]))

    # 3) BR
    br = _fx_br(field, label, text)
    if br: tried.append(("br", br[0], CONF_BASE["br"]))

    # 4) Hints
    hi = _fx_hints(field, label, text, desc)
    if hi: tried.append(("hints", hi[0], CONF_BASE["hints"]))

    if not tried:
        return None, 0.0

    # Melhor individual
    best_src, best_val, best_c = max(tried, key=lambda t: t[2])

    # Acordo com outro extrator?
    for src, val, c in tried:
        if src == best_src:
            continue
        if _value_eq(best_val, val):
            best_c = _agree_boost(best_c, c)
            break

    return best_val, best_c


# ------------------------------------------------------------------
def extract_field_value(label: str, field: str, text: str, description: str) -> Tuple[Optional[str], float]:
    def _kv_after_label(tx: str, labels: List[str]) -> Optional[str]:
        lines = tx.splitlines()
        for i, ln in enumerate(lines):
            low = ln.lower()
            for lab in labels:
                if lab.lower() in low:
                    m = re.search(rf"{re.escape(lab)}\s*[:\-–]\s*(.+)$", ln, flags=re.I)
                    if m:
                        return clean_value(m.group(1))
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines):
                        nxt = clean_value(lines[j])
                        if nxt and not re.match(r"(?i)^(saldo|valor|total)\b", nxt):
                            return nxt
        return None

    # === Overrides aprendidos (TEMPLATE → GLOBAL) com pesos maiores ===
    tpl_id = template_id_from_text(text)

    # 1) Override por TEMPLATE (mais específico)
    _rg_tpl_obj = _compat_get(k_override_tpl(label, tpl_id, field))
    rg_pat_tpl = _coerce_regex_pat(_rg_tpl_obj)
    if rg_pat_tpl:
        try:
            rg = re.compile(rg_pat_tpl, re.I)
            m = rg.search(text)
            if m:
                val = clean_value(m.group(1) if m.groups() else m.group(0))
                val = postprocess_value(field, val, label=label)
                if val is not None:
                    if _rg_tpl_obj != rg_pat_tpl:
                        _compat_set(k_override_tpl(label, tpl_id, field), rg_pat_tpl)
                    return val, 0.94
        except re.error:
            pass

    # 2) Override GLOBAL (menos específico)
    _rg_obj = _compat_get(k_override(label, field))
    rg_pat = _coerce_regex_pat(_rg_obj)
    if rg_pat:
        try:
            rg = re.compile(rg_pat, re.I)
            m = rg.search(text)
            if m:
                val = clean_value(m.group(1) if m.groups() else m.group(0))
                val = postprocess_value(field, val, label=label)
                if val is not None:
                    if _rg_obj != rg_pat:
                        _compat_set(k_override(label, field), rg_pat)
                    return val, 0.92
        except re.error:
            pass

    # === Ensemble (KV → default regex → BR → hints) ===
    val, conf = run_field_ensemble(field, label, text, description)
    if val:
        return val, conf

    # === Heurística reforçada para telas ===
    fname = field.lower()
    if label == "tela_sistema" and fname in {"produto", "sistema", "tipo_de_operacao", "tipo_de_sistema"}:
        m = re.search(r"(?i)\b(Refinanciamento|Empr[eé]stimo|Consignado|Cr[eé]dito\s*Pessoal|Cart[aã]o|Ve[ií]culo|Habita[cç][aã]o)\b", text)
        if m:
            vclean = postprocess_value(field, m.group(1), label=label)
            if vclean:
                return vclean, 0.88

    # === Fallback fuzzy com descrição ===
    if fuzz is not None and process is not None and description:
        tokens = re.findall(r"[\wÀ-ÿ\-\.\'\,\/]+", text, flags=re.UNICODE)
        candidates: List[str] = []
        for n in (8,7,6,5,4,3,2):
            for i in range(0, max(0, len(tokens)-n)):
                s = " ".join(tokens[i:i+n])
                if len(s) >= 4:
                    candidates.append(s)
            if len(candidates) > 2000:
                break
        if candidates:
            best, score, _ = process.extractOne(description, candidates, scorer=fuzz.token_sort_ratio)
            if score >= 80:
                return postprocess_value(field, clean_value(best), label=label), 0.62

    # === Fallback literal "label: valor" ===
    labels = [field] + FIELD_HINTS.get(fname, [])
    cand = _kv_after_label(text, labels)
    if cand:
        cand = postprocess_value(field, cand, label=label)
        if cand:
            return cand, 0.76

    return None, 0.0



def learn_from_result(label: str, field: str, value: str, text: str):
    if not value:
        return

    # registra campos já vistos
    seen = set(_compat_get(k_schema_seen(label)) or [])
    if field not in seen:
        seen.add(field)
        _compat_set(k_schema_seen(label), list(seen))

    # salva exemplos recentes
    exs = list(_compat_get(k_examples(label, field)) or [])
    if value not in exs:
        exs.append(value)
        _compat_set(k_examples(label, field), exs[-20:])

    # ===== overrides aprendidos =====
    anchor = None  # <- inicializa para evitar UnboundLocalError
    # 1) override "global" por âncora
    for line in text.splitlines():
        if value in line and len(line) <= 200:
            idx = line.find(value)
            prefix = line[:idx].strip()
            tokens = re.findall(r"[A-Za-zÀ-ÿ]{3,25}", prefix)
            if tokens:
                anchor = tokens[-1]
                pat = rf"(?i)\b{re.escape(anchor)}\b\s*[:\-–]?\s*(.+)"
                _compat_set(k_override(label, field), pat)
            break

    # 2) override por template (mais específico) — só se achamos âncora
    if anchor:
        tpl_id = template_id_from_text(text)
        pat_tpl = rf"(?i)\b{re.escape(anchor)}\b\s*[:\-–]?\s*(.+)"
        _compat_set(k_override_tpl(label, tpl_id, field), pat_tpl)
    # Eleva o peso de overrides aprendidos por template (confiança futura)
    try:
        tpl_id = template_id_from_text(text)
        if anchor:
            pat_tpl = rf"(?i)\b{re.escape(anchor)}\b\s*[:\-–]?\s*(.+)"
            _compat_set(k_override_tpl(label, tpl_id, field), pat_tpl)
    except Exception:
        pass


def _windows_for_field(text: str, field: str, max_windows: int = 3, radius: int = 1, label: Optional[str] = None) -> List[str]:
    lines = text.splitlines()
    fl = field.lower()
    desc = ""  # legacy; descrição detalhada entra via llm_fallback_extract
    hints = list({*FIELD_HINTS.get(fl, []), *dynamic_hints_for_field(field, desc)})
    wins: List[str] = []

    # 0) Para telas, inclua topo e, se presente, linhas com "Consulta de Cobrança"
    if label == "tela_sistema":
        head = "\n".join(lines[:8])
        wins.append(head)
        for i, ln in enumerate(lines[:20]):  # header costuma estar no topo
            if re.search(r"(?i)\bconsulta\s+de\s+cobran[çc]a\b", ln):
                start = max(0, i - 1); end = min(len(lines), i + 2)
                wins.append("\n".join(lines[start:end]))
                break

    # 1) linhas que batem hints
    for i, line in enumerate(lines):
        if any(h.lower() in line.lower() for h in hints):
            start = max(0, i - radius)
            end = min(len(lines), i + radius + 1)
            wins.append("\n".join(lines[start:end]))

    # 2) fuzzy no nome do campo
    if not wins and fuzz is not None and process is not None:
        candidates = [(i, ln) for i, ln in enumerate(lines) if len(ln.strip()) >= 3]
        if candidates:
            best = sorted(
                ((i, ln, fuzz.partial_ratio(fl, ln)) for i, ln in candidates),
                key=lambda x: x[2],
                reverse=True
            )[:max_windows]
            for i, ln, _ in best:
                start = max(0, i - radius)
                end = min(len(lines), i + radius + 1)
                wins.append("\n".join(lines[start:end]))

    # 3) fallback topo
    if not wins:
        wins.append("\n".join(lines[:min(8, len(lines))]))

    # dedup e corte
    seen = set(); uniq = []
    for w in wins:
        k = (w or "").strip()
        if k and k not in seen:
            uniq.append(k[:800])
            seen.add(k)
        if len(uniq) >= max_windows:
            break
    return uniq


def llm_fallback_extract(
    text: str,
    label: str,
    extraction_schema: SchemaType,
    timeout_s: float = 6.0
) -> Dict[str, Optional[str]]:
    """Usa a LLM para preencher campos faltantes no PDF, respeitando schema (compatível com gpt-5-mini)."""
    if not USE_LLM or not oai_client:
        return {k: None for k in schema_keys(extraction_schema)}

    sub_schema = normalize_schema(extraction_schema)
    fields = list(sub_schema.keys())
    # ... dentro de llm_fallback_extract, logo antes de montar 'messages':
    screen_title = detect_screen_title(label, text)

    fields_payload = {
        f: {
            "desc": sub_schema.get(f, ""),
            "windows": _windows_for_field(text, f, max_windows=3, radius=2, label=label),
        }
        for f in fields
    }

    messages = [
        {
            "role": "system",
            "content": (
                "Você é um extrator estrito. Responda APENAS com JSON contendo exatamente as chaves solicitadas. "
                "Use null quando não houver valor. Não invente campos. "
                "Baseie-se nas 'windows' fornecidas por campo e nas descrições do schema. "
                "Para telas (label='tela_sistema'), considere o título do topo (ex.: 'Consulta de Cobrança') como referência da tela."
            )
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "label": label,
                    "detected_title": screen_title,
                    "schema_descriptions": {f: sub_schema.get(f, "") for f in fields},
                    "context_by_field": fields_payload
                },
                ensure_ascii=False,
            )
        }
    ]


    def _empty_result():
        return {k: None for k in fields}

    def _coerce_output(obj: Any) -> Dict[str, Optional[str]]:
        out = _empty_result()
        if isinstance(obj, dict):
            for k in fields:
                v = obj.get(k)
                if v is None:
                    out[k] = None
                elif isinstance(v, (str, int, float)):
                    s = str(v).strip()
                    out[k] = s if s else None
                else:
                    try:
                        s = json.dumps(v, ensure_ascii=False)
                        out[k] = s if s else None
                    except Exception:
                        out[k] = None
        return out

    def _find_first_json_object(s: str) -> Optional[str]:
        start = s.find("{")
        while start != -1:
            depth = 0; in_str = False; esc = False
            for i in range(start, len(s)):
                ch = s[i]
                if in_str:
                    if esc: esc = False
                    elif ch == '\\': esc = True
                    elif ch == '"': in_str = False
                else:
                    if ch == '"': in_str = True
                    elif ch == '{': depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            return s[start:i+1]
            start = s.find("{", start + 1)
        return None

    def _try_parse_json(txt: str) -> Dict[str, Optional[str]]:
        if not txt:
            return _empty_result()
        try:
            return _coerce_output(json.loads(txt))
        except Exception:
            pass
        frag = _find_first_json_object(txt)
        if frag:
            try:
                return _coerce_output(json.loads(frag))
            except Exception:
                pass
        return _empty_result()

    def _extract_text_from_responses(resp_obj) -> str:
        txt = (getattr(resp_obj, "output_text", None) or "").strip()
        if txt:
            return txt
        try:
            parts = []
            for b in getattr(resp_obj, "output", None) or []:
                for item in getattr(b, "content", None) or []:
                    t = getattr(item, "text", None)
                    if isinstance(t, str) and t:
                        parts.append(t)
                    elif isinstance(t, list):
                        for span in t:
                            st = getattr(span, "text", "")
                            if st: parts.append(st)
            txt2 = "".join(parts).strip()
            if txt2:
                return txt2
        except Exception:
            pass
        try:
            data = getattr(resp_obj, "data", None)
            if data:
                c0 = getattr(data[0], "content", None) or []
                if c0:
                    t = getattr(c0[0], "text", None)
                    if isinstance(t, str): return t.strip()
                    if isinstance(t, list) and t:
                        st = getattr(t[0], "text", "")
                        return (st or "").strip()
        except Exception:
            pass
        return ""

    # Acúmulo local de tokens para depois somar ao global
    used_prompt = used_completion = used_total = 0

    # ===== Tentativa 1: Chat Completions =====
    try:
        max_comp = int(min(512, max(32, timeout_s * 40)))
        resp = oai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=max_comp,
            timeout=timeout_s
        )
        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            logger.info(json.dumps({"event": "llm_raw_empty", "method": "chat.completions"}, ensure_ascii=False))
        u = getattr(resp, "usage", None)
        if u:
            used_prompt     += getattr(u, "prompt_tokens", 0) or 0
            used_completion += getattr(u, "completion_tokens", 0) or 0
            used_total      += getattr(u, "total_tokens", 0) or (used_prompt + used_completion)

        parsed = _try_parse_json(raw)
        if any(parsed.values()):
            # flush para o global
            LLM_TOKENS_LAST["prompt"]     += used_prompt
            LLM_TOKENS_LAST["completion"] += used_completion
            LLM_TOKENS_LAST["total"]      += used_total
            logger.info(json.dumps({
                "event": "llm_success", "method": "chat.completions",
                "fields_filled": sum(1 for v in parsed.values() if v)
            }, ensure_ascii=False))
            return parsed
        else:
            logger.info(json.dumps({
                "event": "llm_empty_content",
                "method": "chat.completions",
                "finish_reason": getattr(getattr(resp.choices[0], "finish_reason", None), "__str__", lambda: "")(),
            }, ensure_ascii=False))
    except Exception as e:
        logger.warning(json.dumps({
            "event": "llm_chat_failed",
            "error": f"{type(e).__name__}: {str(e)[:180]}"
        }, ensure_ascii=False))

    # ===== Tentativa 2: Responses API =====
    # ===== Tentativa 2: Responses API =====
    try:
        max_out = int(min(512, max(32, timeout_s * 40)))
        r = oai_client.responses.create(
            model=LLM_MODEL,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text":
                    ("Você é um extrator estrito. Responda APENAS com JSON contendo exatamente as "
                     "chaves solicitadas. Use null quando não houver valor. Não invente campos. "
                     "Baseie-se nas 'windows' fornecidas por campo e nas descrições do schema. "
                     "Para telas (label='tela_sistema'), considere o título do topo (ex.: 'Consulta de Cobrança').")}]},
                {"role": "user", "content": [{"type": "input_text", "text": json.dumps({
                    "label": label,
                    "detected_title": screen_title,
                    "schema_descriptions": {f: sub_schema.get(f, "") for f in fields},
                    "context_by_field": fields_payload
                }, ensure_ascii=False)}]}
            ],
            max_output_tokens=max_out,
            timeout=timeout_s
        )
        txt = _extract_text_from_responses(r).strip()
        parsed = _try_parse_json(txt)
        # ... (resto igual)

        # flush para o global
        LLM_TOKENS_LAST["prompt"]     += used_prompt
        LLM_TOKENS_LAST["completion"] += used_completion
        LLM_TOKENS_LAST["total"]      += used_total

        if any(parsed.values()):
            logger.info(json.dumps({
                "event": "llm_success",
                "method": "responses",
                "fields_filled": sum(1 for v in parsed.values() if v)
            }, ensure_ascii=False))
            return parsed
        else:
            logger.info(json.dumps({
                "event": "llm_empty_content",
                "method": "responses"
            }, ensure_ascii=False))
    except Exception as e:
        logger.warning(json.dumps({
            "event": "llm_responses_failed",
            "error": f"{type(e).__name__}: {str(e)[:180]}"
        }, ensure_ascii=False))

    logger.warning(json.dumps({
        "event": "llm_all_methods_failed",
        "label": label,
        "fields": fields
    }, ensure_ascii=False))
    return _empty_result()

class ExtractResponse(BaseModel):
    output: Dict[str, Optional[str]]
    meta: Dict[str, Any]

def _mask_pii(d: Dict[str, Any]) -> Dict[str, Any]:
    def _digits_only(x: str) -> str:
        return re.sub(r"\D+", "", x or "")
    out = {}
    for k, v in d.items():
        if not isinstance(v, str):
            out[k] = v
            continue
        kl = k.lower()
        if "cpf" in kl:
            n = _digits_only(v)
            out[k] = f"***{n[-3:]}" if len(n) >= 3 else "***"
        elif "cnpj" in kl:
            n = _digits_only(v)
            out[k] = f"***{n[-4:]}" if len(n) >= 4 else "***"
        else:
            out[k] = v
    return out

def _oab_special_blocks(label: str, text: str, out: Dict[str, Optional[str]], conf: Dict[str, float], allowed_fields: List[str]):
    if label != "carteira_oab":
        return
    lines = text.splitlines()

    if any(f in allowed_fields for f in ("inscricao", "seccional", "subsecao")):
        # piso de confiança quando veio do bloco triplo
        if "inscricao" in out and out["inscricao"]:
            conf["inscricao"] = max(conf.get("inscricao", 0.0), 0.92)
        if "seccional" in out and out["seccional"]:
            conf["seccional"] = max(conf.get("seccional", 0.0), 0.92)
        if "subsecao" in out and out["subsecao"]:
            conf["subsecao"] = max(conf.get("subsecao", 0.0), 0.90)
        for i, l in enumerate(lines):
            if re.search(r"(?i)Inscri[cç][aã]o\s+Seccional\s+Subse[cç][aã]o", l):
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    triple = re.sub(r"\s+", " ", lines[j]).strip()
                    # dentro de _oab_special_blocks, logo após montar 'triple'
                    mm = re.match(r"(?i)^\s*(\d{4,12})\s+([A-Z]{2})\s+(.+)$", triple)
                    if mm:
                        insc, uf, subse = mm.group(1), mm.group(2), mm.group(3)
                        subse = re.sub(r"^\s*(?:\d{4,12}\s+)?(?:[A-Z]{2}\s+)?", "", subse).strip()
                        # dentro do if mm: (logo após montar insc, uf, subse)
                        # piso extra quando linha tripla está perfeita
                        if re.fullmatch(r"\d{4,12}", insc) and re.fullmatch(r"[A-Z]{2}", uf) and len(subse) >= 3:
                            conf["inscricao"] = max(conf.get("inscricao", 0.0), 0.95)
                            conf["seccional"] = max(conf.get("seccional", 0.0), 0.95)
                            conf["subsecao"] = max(conf.get("subsecao", 0.0), 0.93)

                        # se ainda começar com dígitos ou “UF colada”, descarte
                        if re.match(r"^\d{3,}|^[A-Z]{2}\w+", subse):
                            subse = ""
                        if subse and subse.isupper():
                            subse = subse.title()

                        if "inscricao" in allowed_fields and not out.get("inscricao"):
                            out["inscricao"] = insc
                        if "seccional" in allowed_fields and not out.get("seccional"):
                            out["seccional"] = uf
                        if "subsecao" in allowed_fields and not out.get("subsecao") and subse:
                            out["subsecao"] = subse.title()
                break


    # (2) nome — tenta "Nome:" rotulado; senão, escolhe linha "limpa"
    if "nome" in allowed_fields and not out.get("nome"):
        # 2.a) rótulo explícito
        m = re.search(r"(?i)\bNome\b\s*[:\-–]\s*(.+)", text)
        if m:
            cand = m.group(1).strip()
            cand = re.sub(r"\s{2,}", " ", cand)
            if cand and not re.search(r"\d", cand):
                out["nome"] = " ".join(w.capitalize() for w in cand.split())

        # 2.b) fallback: primeira linha sem dígitos/sem “termos de endereço”
        if not out.get("nome"):
            street_words = r"(?i)\b(avenida|av\.?|rua|rodovia|alameda|travessa|praça|praca|estrada|lote|quadra|bairro|cep|nº|no\.|n\.)\b"
            for l in lines:
                s = l.strip()
                if (not s) or re.search(street_words, s) or re.search(r"\d", s):
                    continue
                if len(re.findall(r"[A-Za-zÀ-ÿ]{2,}", s)) >= 2:
                    out["nome"] = " ".join(w.capitalize() for w in s.split())
                    break

    # (3) endereço profissional
    if "endereco_profissional" in allowed_fields and not out.get("endereco_profissional"):
        for i, l in enumerate(lines):
            if re.match(r"(?i)^Endere[cç]o\s+Profissional\s*$", l.strip()):
                t = []
                for s in (1, 2, 3):
                    idx = i + s
                    if idx < len(lines) and lines[idx].strip():
                        t.append(lines[idx].strip())
                if t:
                    out["endereco_profissional"] = " ".join(t)
                break

# =================== Núcleo de extração ===================

def extract_impl(label: str, extraction_schema: SchemaType, pdf_bytes: bytes) -> ExtractResponse:
    t0 = time.time()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]

    # Extrai e normaliza texto
    raw_text = pdf_bytes_to_text(pdf_bytes)
    text = normalize_text(raw_text)
    # 1) Tenta Docling primeiro (layout-aware) → Markdown
    # 1) Tenta Docling primeiro (layout-aware) → Markdown
    md_text = docling_pdf_to_markdown(pdf_bytes)

    if md_text and len(md_text) > 50:
        # Usa o Markdown como “texto principal” para heurísticas e KV
        md_norm = normalize_text(md_text)
        # Fusão com OCR plano para aumentar recall
        ocr_plain = normalize_text(pdf_bytes_to_text(pdf_bytes))
        text = (md_norm + "\n" + ocr_plain).strip()

        # injeta um extrator KV global baseado no Docling Markdown, reutilizado em _fx_kv()
        def extract_kv_grid(_: str) -> dict[str, str]:
            return extract_kv_grid_from_docling_markdown(md_text)

        globals()["extract_kv_grid"] = extract_kv_grid
    else:
        # fallback para o comportamento antigo (PyPDF2 → pdfminer)
        raw_text = pdf_bytes_to_text(pdf_bytes)
        text = normalize_text(raw_text)
        if "extract_kv_grid" in globals():
            del globals()["extract_kv_grid"]  # evita grid “velho” ficar preso

    # Tabelas em HTML (Docling) → KV merge
    tbls = docling_pdf_to_html_tables(pdf_bytes)
    if tbls:
        kv_extra = extract_simple_kv_from_tables(tbls)
        if kv_extra:
            # se já injetamos extract_kv_grid, faça merge
            _kv_fn = globals().get("extract_kv_grid")
            if callable(_kv_fn):
                base_grid = _kv_fn(text)
                def extract_kv_grid(_tx: str) -> dict[str, str]:
                    merged = dict(base_grid)
                    merged.update(kv_extra)
                    return merged
                globals()["extract_kv_grid"] = extract_kv_grid
            else:
                def extract_kv_grid(_tx: str) -> dict[str, str]:
                    return kv_extra
                globals()["extract_kv_grid"] = extract_kv_grid

    # ===== Auto-label: deduzir o label antes de aprender schema e antes do cache =====
    orig_label = (label or "").strip().lower()
    if not orig_label or orig_label in {"auto", "generic", "generico", "gen"}:
        label = guess_label_from_schema_and_text(extraction_schema, text)

    # (Re)detecta o título depois do label final
    screen_title = detect_screen_title(label, text)

    # ===== Aprendizado do schema completo por label (com o label final) =====
    req_fields = schema_keys(extraction_schema)
    add_fields_to_full_schema(label, req_fields)
    known_full = get_full_schema(label)
    unknown = [f for f in req_fields if f.strip().lower() not in known_full]
    if unknown:
        logger.info(json.dumps(
            {"event": "schema_subset_warn", "label": label, "unknown_fields": unknown},
            ensure_ascii=False
        ))

    # ===== cache_key robusto e estável (usa label final, hash do PDF e SHA-1 do schema) =====
    skey = sorted(normalize_schema(extraction_schema).items())
    skey_json = json.dumps(skey, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    schema_sha = hashlib.sha1(skey_json.encode("utf-8")).hexdigest()
    cache_key = f"{EXTRACTOR_VERSION}|{label}|{pdf_hash}|{schema_sha}"

    if cache_key in SESSION_CACHE:
        cached = SESSION_CACHE[cache_key]
        return ExtractResponse(**cached)

    out: Dict[str, Optional[str]] = {}
    conf: Dict[str, float] = {}
    validation: Dict[str, Dict[str, Any]] = {}

    fields = schema_keys(extraction_schema)

    # Heurísticas/regex primeiro
    for field in fields:
        desc = schema_desc(extraction_schema, field)
        eff = FIELD_SYNONYMS.get(field, field)  # <- usa sinônimo no mecanismo
        val, c = extract_field_value(label, eff, text, desc)
        out[field] = val
        conf[field] = c

    # Ajustes OAB
    _oab_special_blocks(label, text, out, conf, allowed_fields=fields)
    # Normalização específica por label
    if label == "tela_sistema":
        standardize_tela_values(out, screen_title)

    # --- Resolver conflito pesquisa_tipo × sistema ---
    if label == "tela_sistema":
        pt = (out.get("pesquisa_tipo") or "").strip()
        if pt:
            m = re.match(r"(?i)^\s*(?:Sistema)\s*[:\-–]\s*(.+)$", pt)
            if m:
                val = m.group(1).strip()
                if not out.get("sistema"):
                    out["sistema"] = postprocess_value("sistema", val, label=label)
                    conf["sistema"] = max(conf.get("sistema", 0.0), 0.90)
                # zera pesquisa_tipo se virou container
                out["pesquisa_tipo"] = None
                conf["pesquisa_tipo"] = 0.0

    # Se algum campo foi preenchido por bloco especial/hint mas sem confiança, dê um piso seguro
    for f in fields:
        if out.get(f) and conf.get(f, 0.0) == 0.0:
            conf[f] = 0.88  # piso razoável para heurística robusta

    # Pós-processamento pré-LLM
    for f in list(out.keys()):
        eff = FIELD_SYNONYMS.get(f, f)
        out[f] = postprocess_value(eff, out[f], label=label)
        # Enforça "somente dígitos" quando a descrição do schema pedir
        desc_f = schema_desc(extraction_schema, f)
        if out[f] and wants_digits_only(desc_f):
            out[f] = re.sub(r"\D+", "", out[f])

    # ===== Decisão de LLM conforme SLA (bloco único) =====
    elapsed = time.time() - t0
    time_left = DEADLINE_S - elapsed
    need_fallback = [f for f in fields if (out[f] is None or conf[f] < MIN_CONF)]
    used_llm_flag = False

    if not USE_LLM:
        llm_skip_reason = "USE_LLM=0"
    elif not oai_client:
        llm_skip_reason = "oai_client_not_initialized"
    else:
        if FORCE_LLM_ALL:
            target_fields = fields[:]  # força tentar todos
        else:
            target_fields = need_fallback[:]  # só os que precisam

        if not target_fields:
            llm_skip_reason = "no_fields_need_fallback"
        elif time_left <= LLM_SAFETY_MARGIN_S:
            llm_skip_reason = f"no_time_left({time_left:.3f}s <= safety_margin {LLM_SAFETY_MARGIN_S:.3f}s)"
        else:
            # Chama a LLM
            llm_timeout = max(1.0, min(6.0, time_left - LLM_SAFETY_MARGIN_S))
            sub_schema = schema_subset(extraction_schema, target_fields)
            llm_ans = llm_fallback_extract(text, label, sub_schema, timeout_s=llm_timeout)
            filled = 0
            for f in target_fields:
                if llm_ans.get(f):
                    out[f] = postprocess_value(f, llm_ans[f], label=label)
                    if out[f]:
                        conf[f] = max(conf[f], 0.75)
                    filled += 1
            used_llm_flag = (filled > 0)
            llm_skip_reason = None

    # Log de debug da LLM
    logger.info(json.dumps({
        "event": "llm_decision",
        "forced": bool(FORCE_LLM_ALL),
        "need_fallback": need_fallback,
        "target_fields": (fields if FORCE_LLM_ALL else need_fallback),
        "time_left_s": round(time_left, 3),
        "timeout_s": float(llm_timeout) if 'llm_timeout' in locals() else None,
        "skip_reason": llm_skip_reason,
        "used_llm": used_llm_flag,
    }, ensure_ascii=False))

    # Pós-processamento final + aprendizado leve (idempotente e seguro)
    for f in list(out.keys()):
        out[f] = postprocess_value(f, out[f], label=label)
        # repete a salvaguarda digits-only (caso LLM tenha alterado)
        desc_f = schema_desc(extraction_schema, f)
        if out[f] and wants_digits_only(desc_f):
            out[f] = re.sub(r"\D+", "", out[f])
    for f in fields:
        fl = f.lower()
        v = out.get(f)
        if not v:
            continue
        # Se CPF/CNPJ válidos passaram no validador, suba piso
        if ("cpf" in fl and validate_cpf(v)) or ("cnpj" in fl and validate_cnpj(v)):
            conf[f] = max(conf.get(f, 0.0), 0.95)
        # Datas ISO válidas
        if any(k in fl for k in ["data", "nascimento", "emissao", "emissão", "referencia", "referência"]):
            if parse_date_to_iso(v):
                conf[f] = max(conf.get(f, 0.0), 0.90)
    # --- Consistências e saneamentos adicionais ---
    # (a) seccional colada na subseção (ex.: "PR Curitiba")
    if label == "carteira_oab" and out.get("seccional") and out.get("subsecao"):
        uf = str(out["seccional"]).upper()
        sub = str(out["subsecao"])
        out["subsecao"] = re.sub(rf"^\s*{re.escape(uf)}\s*", "", sub, flags=re.I).strip().title() or sub

    # (b) telefone_profissional precisa ter dígitos válidos
    if out.get("telefone_profissional") and not re.search(r"\d{8,}", str(out["telefone_profissional"])):
        out["telefone_profissional"] = None
        conf["telefone_profissional"] = 0.0

    # (c) se 'pesquisa_por' ausente e título detectado na tela, usar título
    if label == "tela_sistema" and not out.get("pesquisa_por"):
        if screen_title:
            out["pesquisa_por"] = screen_title
            conf["pesquisa_por"] = max(conf.get("pesquisa_por", 0.0), 0.88)

    # STRICT_MODE aplica MIN_CONF por campo + registra validação detalhada
    for k in fields:
        k_conf = round(conf.get(k, 0.0), 3)
        k_val_ok = k_conf >= MIN_CONF
        validation[k] = {
            "value": out.get(k),
            "conf": k_conf,
            "meets_min_conf": k_val_ok
        }
        if STRICT_MODE and not k_val_ok:
            # derruba o campo abaixo do limiar para evitar “meias verdades”
            out[k] = None

    # Score agregado (somente campos não-nulos pós-STRICT)
    filled_confs = [conf.get(k, 0.0) for k in fields if out.get(k) is not None]
    conf_agg = round(sum(filled_confs) / len(filled_confs), 3) if filled_confs else 0.0

    # Depuração da decisão LLM
    llm_debug = {
        "need_fallback": need_fallback,
        "elapsed_s": round(elapsed, 3),
        "time_left_s": round(time_left, 3),
        "safety_margin_s": LLM_SAFETY_MARGIN_S,
        "use_llm": bool(USE_LLM),
        "oai_client": bool(oai_client),
        "model": LLM_MODEL,
        "skip_reason": llm_skip_reason,
        "forced": bool(FORCE_LLM_ALL),
    }

    # ===== Uso de LLM e custo (por extração) =====
    llm_usage = {
        "prompt": LLM_TOKENS_LAST.get("prompt", 0),
        "completion": LLM_TOKENS_LAST.get("completion", 0),
        "total": LLM_TOKENS_LAST.get("total", 0),
    }
    cost_est = (
        (llm_usage["prompt"] / 1000.0) * LLM_PRICE_IN_1K +
        (llm_usage["completion"] / 1000.0) * LLM_PRICE_OUT_1K
    )
    cost_est = round(cost_est, 6)

    # Zera acumulador global para a próxima chamada
    LLM_TOKENS_LAST["prompt"] = 0
    LLM_TOKENS_LAST["completion"] = 0
    LLM_TOKENS_LAST["total"] = 0

    meta = {
        "label": label,
        "screen_title_detected": bool(screen_title) if label == "tela_sistema" else None,
        "pdf_hash": pdf_hash,
        "latency_ms": int((time.time() - t0) * 1000),
        "used_llm": used_llm_flag,
        "conf": {k: round(conf.get(k, 0.0), 3) for k in fields},
        "conf_agg": conf_agg,
        "deadline_s": DEADLINE_S,
        "min_conf": MIN_CONF,
        "strict_mode": STRICT_MODE,
        "validation": validation,
        "llm_debug": llm_debug,
        "extractor_version": EXTRACTOR_VERSION,
        "llm_usage": llm_usage,
        "cost_est_usd": cost_est,
    }

    resp = ExtractResponse(
        output={k: (out.get(k) if out.get(k) else None) for k in fields},
        meta=_mask_pii(meta)
    )
    SESSION_CACHE[cache_key] = json.loads(resp.model_dump_json())

    # métricas
    MET_CONF_AGG.observe(conf_agg)
    if any(resp.output.values()):
        MET_SUCCESS.inc()
    else:
        MET_FAIL.inc()
    if used_llm_flag:
        MET_USED_LLM.inc()

    logger.info(json.dumps({
        "event": "extract_done",
        "label": label,
        "used_llm": used_llm_flag,
        "conf_agg": conf_agg,
        "latency_ms": meta["latency_ms"],
        "strict_mode": STRICT_MODE
    }, ensure_ascii=False))

    for f, v in out.items():
        if v:
            learn_from_result(label, f, v, text)
            bump_field_freq(label, f)

    return resp


# =================== FastAPI ===================

app = FastAPI(title="ENTER-AI Fellowship Extractor VS3", version=EXTRACTOR_VERSION)

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {
        "ok": True,
        "use_llm_env": bool(os.getenv("USE_LLM", "0") == "1"),
        "openai_key_present": bool(bool(OPENAI_API_KEY)),
        "use_llm_effective": bool(USE_LLM),
        "llm_init": _llm_init_reason,
        "llm_model": LLM_MODEL,
        "version": EXTRACTOR_VERSION,
        "deadline_s": DEADLINE_S,
        "min_conf": MIN_CONF,
        "strict_mode": STRICT_MODE
    }
# --- no endpoint /llm_selftest ---
@app.get("/llm_selftest")
def llm_selftest():
    def diag(payload=None, **kw):
        d = {
            "ok": False,
            "api_mode": None,
            "model": LLM_MODEL,
            "reason": None,
            "raw": "",
            "exception": None,
            "use_llm_effective": bool(USE_LLM),
            "oai_client_initialized": bool(oai_client),
            "env_has_key": bool(bool(OPENAI_API_KEY)),
        }
        if payload and isinstance(payload, dict):
            d.update(payload)
        d.update(kw)
        return d

    if not USE_LLM:
        return diag(reason="USE_LLM=0 (desativado)")
    if not OPENAI_API_KEY:
        return diag(reason="OPENAI_API_KEY ausente")
    if not oai_client:
        return diag(reason=f"Cliente OpenAI não inicializado: {_llm_init_reason}")

    messages = [
        {"role": "system", "content": "Responda exatamente: ok"},
        {"role": "user",   "content": "ping"}
    ]

    # A) Chat Completions
    try:
        resp = oai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_completion_tokens=16,
            timeout=15  # ↑ mais folga
        )
        content = (resp.choices[0].message.content or "").strip()
        if content:
            return {
                "ok": (content.lower() == "ok"),
                "api_mode": "chat.completions",
                "model": LLM_MODEL,
                "raw": content,
                "reason": None if content else "resposta vazia"
            }
    except Exception as e1:
        chat_err = f"{type(e1).__name__}: {str(e1)[:180]}"
    else:
        chat_err = "n/a"

    # B) Responses API (usar blocos input_text)
    def _extract_text_from_responses(resp_obj) -> str:
        txt = (getattr(resp_obj, "output_text", None) or "").strip()
        if txt:
            return txt
        try:
            blocks = getattr(resp_obj, "output", None) or []
            parts = []
            for b in blocks:
                for it in getattr(b, "content", []) or []:
                    t = getattr(it, "text", "")
                    if isinstance(t, str) and t:
                        parts.append(t)
                    elif isinstance(t, list) and t:
                        for s in t:
                            st = getattr(s, "text", "")
                            if st:
                                parts.append(st)
            txt = "".join(parts).strip()
            if txt:
                return txt
        except Exception:
            pass
        try:
            data = getattr(resp_obj, "data", None) or []
            if data:
                c0 = getattr(data[0], "content", None) or []
                if c0:
                    t = getattr(c0[0], "text", "")
                    if isinstance(t, str):
                        return t.strip()
                    if isinstance(t, list) and t:
                        return (getattr(t[0], "text", "") or "").strip()
        except Exception:
            pass
        return ""

    try:
        r = oai_client.responses.create(
            model=LLM_MODEL,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": "Responda exatamente: ok"}]},
                {"role": "user",   "content": [{"type": "input_text", "text": "ping"}]}
            ],
            max_output_tokens=16,
            timeout=15  # ↑ mais folga
        )
        txt = _extract_text_from_responses(r).strip()
        return {
            "ok": (txt.lower() == "ok"),
            "api_mode": "responses",
            "model": LLM_MODEL,
            "raw": txt,
            "reason": None if txt else f"resposta vazia (chat_err={chat_err})"
        }
    except Exception as e2:
        return diag(
            api_mode="responses",
            reason="exceção na Responses API",
            exception=f"{type(e2).__name__}: {str(e2)[:180]}"
        )



@app.post("/warmup")
def warmup():
    _ = BR_REGEX["cpf"]
    _ = compile_hint_regex(["nome", "inscrição"])
    return {"warmed": True}

@app.post("/extract", response_model=ExtractResponse)
async def extract_endpoint(
    label: str = Form(...),
    extraction_schema: str = Form(...),  # aceita dict ou lista (string JSON)
    pdf: UploadFile = File(...),
):
    t0 = time.time()
    try:
        raw = json.loads(extraction_schema)
        assert isinstance(raw, (dict, list)) and raw
        schema_obj: SchemaType = raw
    except Exception:
        MET_FAIL.inc()
        return JSONResponse(
            status_code=400,
            content={"error": "extraction_schema inválido (dict {campo: descrição} ou lista [campo,...])"}
        )

    pdf_bytes = await pdf.read()
    with MET_LATENCY.time():
        resp = extract_impl(label, schema_obj, pdf_bytes)
    latency = int((time.time() - t0) * 1000)
    logger.info(json.dumps({"event": "http_extract", "latency_ms": latency, "label": label}, ensure_ascii=False))
    return JSONResponse(content=json.loads(resp.model_dump_json()))
@app.get("/label_schema")
def label_schema(label: str = Query(..., description="Label a inspecionar")):
    fs = sorted(get_full_schema(label))
    details = {}
    for f in fs:
        details[f] = {
            "override_regex": _compat_get(k_override(label, f)),
            "examples": _compat_get(k_examples(label, f)) or [],
            "freq": int(_compat_get(k_field_freq(label, f)) or 0),
        }
    return {"label": label, "full_schema": fs, "details": details}

@app.post("/extract_bulk")
async def extract_bulk_endpoint(
    label: str = Form(...),
    extraction_schema: str = Form(...),
    pdfs: List[UploadFile] = File(...),
):
    """
    Faz extração em lote. Retorna lista de {file, output, meta}.
    """
    try:
        raw = json.loads(extraction_schema)
        assert isinstance(raw, (dict, list)) and raw
        schema_obj: SchemaType = raw
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "extraction_schema inválido (dict {campo: descrição} ou lista [campo,...])"}
        )

    results = []
    start = time.time()
    with MET_LATENCY_BULK.time():
        for up in pdfs:
            try:
                b = await up.read()
                resp = extract_impl(label, schema_obj, b)
                results.append({"file": up.filename, "output": resp.output, "meta": resp.meta})
            except Exception as e:
                MET_FAIL.inc()
                results.append({"file": up.filename, "error": str(e)})
    dur = int((time.time() - start) * 1000)
    logger.info(json.dumps({"event": "http_extract_bulk", "files": len(pdfs), "latency_ms": dur}, ensure_ascii=False))
    return {"results": results, "latency_ms": dur}

@app.get("/label_info")
def label_info(label: str = Query(..., description="Nome do label a inspecionar")):
    seen = list(_compat_get(k_schema_seen(label)) or [])
    fields = {}
    for f in seen:
        rg = _compat_get(k_override(label, f))
        ex = _compat_get(k_examples(label, f)) or []
        fields[f] = {"override_regex": rg, "examples": ex}
    return {"label": label, "fields_seen": seen, "details": fields}

@app.post("/suggest_schema")
async def suggest_schema(
    label: str = Form(...),
    pdf: UploadFile = File(...),
    k: int = Form(6),
):
    pdf_bytes = await pdf.read()
    text = normalize_text(pdf_bytes_to_text(pdf_bytes))

    candidates = set()
    for fname, hints in FIELD_HINTS.items():
        pat = compile_hint_regex(hints)
        if not pat:
            continue
        for line in text.splitlines():
            if pat.search(line):
                candidates.add(fname)

    llm_added: List[str] = []
    if USE_LLM and oai_client:
        try:
            instr = {"role": "system",
                     "content": "Retorne apenas JSON no formato {\"fields\":[\"...\"]}. Sem explicações."}
            user = {"role": "user",
                    "content": "Texto do PDF (trecho):\n" + text[:8000]}
            resp = oai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[instr, user],
                max_completion_tokens=16, timeout=5
            )
            js = (resp.choices[0].message.content or "").strip()
            try:
                data = json.loads(js)
                llm_added = [str(x).strip().lower() for x in data.get("fields", []) if isinstance(x, str)]
            except Exception:
                llm_added = []
        except Exception:
            llm_added = []

    merged = list(dict.fromkeys([*candidates, *llm_added]))[:max(1, int(k))]
    return {"label": label, "suggested_fields": merged, "heuristic": list(candidates), "llm": llm_added}

# =================== Helpers batch CLI ===================

def _iter_pdfs_in_dir(pdf_dir: Path) -> Iterable[Path]:
    for p in sorted(pdf_dir.glob("*.pdf")):
        if p.is_file():
            yield p

def _load_schema_from_entry(entry: Dict[str, Any]) -> SchemaType:
    if "schema" in entry and isinstance(entry["schema"], (dict, list)):
        return entry["schema"]
    if "schema_path" in entry:
        p = Path(entry["schema_path"]).expanduser()
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
            if not isinstance(raw, (dict, list)):
                raise ValueError("schema_path deve conter dict ou lista.")
            return raw
    if "extraction_schema" in entry and isinstance(entry["extraction_schema"], (dict, list)):
        return entry["extraction_schema"]
    raise ValueError("Item do dataset precisa de 'schema' (obj/list) ou 'schema_path' (arquivo).")

def _coerce_dataset_item(entry: Dict[str, Any]) -> Dict[str, Any]:
    if "label" not in entry:
        raise ValueError("Entrada do dataset sem 'label'.")
    label = entry["label"]
    if "pdf" in entry and isinstance(entry["pdf"], str):
        pdf = entry["pdf"]
    elif "pdf_path" in entry and isinstance(entry["pdf_path"], str):
        p = entry["pdf_path"]
        pdf = p if (":" in p or p.startswith("/")) else str(DEFAULT_PDF_DIR / p)
    else:
        raise ValueError("Entrada do dataset precisa de 'pdf' (caminho) ou 'pdf_path' (legado).")

    if "schema" in entry:
        schema_obj = entry["schema"]
    elif "extraction_schema" in entry:
        schema_obj = entry["extraction_schema"]
    elif "schema_path" in entry:
        schema_obj = _load_schema_from_entry(entry)
    else:
        raise ValueError("Entrada do dataset precisa de 'schema' (obj/list) ou 'extraction_schema' (legado) ou 'schema_path'.")

    _ = normalize_schema(schema_obj)
    return {"label": label, "pdf": pdf, "schema": schema_obj}

def run_dataset(dataset_path: Path, outdir: Path, sleep_first: float = 0.0):
    with dataset_path.open("r", encoding="utf-8") as f:
        raw_items = json.load(f)
    if not isinstance(raw_items, list):
        raise ValueError("dataset.json deve ser uma lista de itens.")
    items = []
    for it in raw_items:
        try:
            items.append(_coerce_dataset_item(it))
        except Exception as e:
            print(f"[erro] Item inválido no dataset: {it} → {e}")
    return _run_items_serial(items, outdir, sleep_first)

def run_pdf_dir(pdf_dir: Path, label: str, schema_path: Path, outdir: Path, sleep_first: float = 0.0):
    raw = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(raw, (dict, list)):
        raise ValueError("schema.json deve ser dict {campo: descrição} ou lista [campo,...].")
    schema = raw
    items = [{"label": label, "pdf": str(pdfp), "schema": schema} for pdfp in _iter_pdfs_in_dir(pdf_dir)]
    return _run_items_serial(items, outdir, sleep_first)

def _run_items_serial(items: List[Dict[str, Any]], outdir: Path, sleep_first: float = 0.0):
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = outdir / f"results_{ts}.jsonl"
    csv_path   = outdir / f"results_{ts}.csv"

    rows: List[Dict[str, Any]] = []
    all_out_keys: set[str] = set()

    if sleep_first > 0:
        time.sleep(sleep_first)

    iterator = items
    if tqdm is not None:
        iterator = tqdm(items, desc="Processando PDFs (serial)")

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for entry in iterator:
            try:
                norm = _coerce_dataset_item(entry)
            except Exception as e:
                print(f"[erro] Item inválido no dataset: {entry} → {e}")
                continue

            label = norm["label"]
            pdf_path = Path(norm["pdf"]).expanduser()
            if not pdf_path.exists():
                print(f"[aviso] PDF não encontrado: {pdf_path}")
                continue

            schema_obj = norm["schema"]
            pdf_bytes = pdf_path.read_bytes()
            t0 = time.time()
            resp = extract_impl(label, schema_obj, pdf_bytes)
            elapsed = int((time.time() - t0) * 1000)

            rec = {"label": label, "pdf": str(pdf_path), "output": resp.output, "meta": resp.meta}
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            single_json_path = outdir / f"{pdf_path.stem}.json"
            with single_json_path.open("w", encoding="utf-8") as sf:
                json.dump({"label": label, "pdf": str(pdf_path), "output": resp.output}, sf, ensure_ascii=False, indent=2)

            row = {
                "label": label,
                "pdf": str(pdf_path),
                "latency_ms": resp.meta.get("latency_ms", elapsed),
                "used_llm": resp.meta.get("used_llm", False),
            }
            for k, v in resp.output.items():
                row[f"out.{k}"] = v
                all_out_keys.add(f"out.{k}")
            rows.append(row)

    base_cols = ["label", "pdf", "latency_ms", "used_llm"]
    fieldnames = base_cols + sorted(all_out_keys)
    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nOK! Salvo:\n- JSONL: {jsonl_path}\n- CSV:   {csv_path}\n- JSONs individuais em: {outdir}")
    print("Dica: USE_LLM=0 p/ custo mínimo; USE_LLM=1 p/ cobrir campos faltantes via LLM.")
    return {"jsonl": str(jsonl_path), "csv": str(csv_path)}

# =================== CLI ===================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", help="label para extração unitária")
    ap.add_argument("--schema", help="arquivo JSON {campo: descrição} OU lista [campos] para extração unitária")
    ap.add_argument("--pdf", help="arquivo PDF (1 página) para extração unitária")
    ap.add_argument("--dataset", default=None, help="arquivo dataset.json (processamento serial)")
    ap.add_argument("--pdf_dir", default=None, help="pasta contendo .pdf a processar com o mesmo label+schema")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="pasta de saída para batch (JSONL/CSV)")
    ap.add_argument("--sleep_first", type=float, default=0.0, help="delay opcional antes do 1º item do batch")
    args = ap.parse_args()

    if not any([args.label, args.schema, args.pdf, args.dataset, args.pdf_dir]):
        if DEFAULT_DATASET_PATH.exists():
            run_dataset(DEFAULT_DATASET_PATH, Path(args.outdir), sleep_first=args.sleep_first); raise SystemExit(0)
        if DEFAULT_PDF_DIR.exists() and DEFAULT_SCHEMA_PATH.exists():
            run_pdf_dir(DEFAULT_PDF_DIR, DEFAULT_LABEL, DEFAULT_SCHEMA_PATH, Path(args.outdir), sleep_first=args.sleep_first); raise SystemExit(0)
        print("Nenhum argumento e não encontrei dataset.json ou a pasta padrão + schema.json.\n"
              "Use: --dataset dataset.json  OU  --pdf_dir <dir> --label <label> --schema schema.json")
        raise SystemExit(2)

    if args.pdf_dir and args.label and args.schema:
        pdf_dir = Path(args.pdf_dir)
        if not pdf_dir.exists(): print(f"pasta de PDFs não encontrada: {pdf_dir}"); raise SystemExit(1)
        schema_path = Path(args.schema)
        if not schema_path.exists(): print(f"schema não encontrado: {schema_path}"); raise SystemExit(1)
        run_pdf_dir(pdf_dir, args.label, schema_path, Path(args.outdir), sleep_first=args.sleep_first); raise SystemExit(0)

    if args.dataset:
        ds_path = Path(args.dataset)
        if not ds_path.exists(): print(f"dataset não encontrado: {ds_path}"); raise SystemExit(1)
        run_dataset(ds_path, Path(args.outdir), sleep_first=args.sleep_first); raise SystemExit(0)

    if args.label and args.schema and args.pdf:
        raw = json.loads(Path(args.schema).read_text(encoding="utf-8"))
        if not isinstance(raw, (dict, list)):
            print("schema deve ser dict ou lista."); raise SystemExit(1)
        with open(args.pdf, "rb") as f:
            pdf_bytes = f.read()
        r = extract_impl(args.label, raw, pdf_bytes)
        print(json.dumps(json.loads(r.model_dump_json()), ensure_ascii=False, indent=2)); raise SystemExit(0)

    print("Uso (unitário): python enter_ai_fellowship_extractor_vs3.py --label <label> --schema schema.json --pdf caminho\\doc.pdf\n"
          "Batch (dataset): python enter_ai_fellowship_extractor_vs3.py --dataset dataset.json --outdir outputs\n"
          "Batch (pasta):   python enter_ai_fellowship_extractor_vs3.py --pdf_dir \"C:\\path\\to\\files\" --label <label> --schema schema.json --outdir outputs")
    raise SystemExit(2)
