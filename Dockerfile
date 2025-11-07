
# Dockerfile — executor mínimo
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# copie seu código para /app (ajuste o nome do arquivo principal se necessário)
# COPY enter_ai_fellowship_extractor_vs3.py /app/
# COPY schema_oab.json schema_tela_sistema.json dataset.json /app/

EXPOSE 8765
CMD ["uvicorn", "enter_ai_fellowship_extractor_vs3:app", "--host", "0.0.0.0", "--port", "8765"]
