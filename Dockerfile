FROM python:3.9-slim-buster AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OPENAI_API_KEY=${OPENAI_API_KEY}

# Use a non-root user for security.
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser

CMD ["uvicorn", "shap_analysis:app", "--host", "0.0.0.0"]