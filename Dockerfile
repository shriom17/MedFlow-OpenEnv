FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

WORKDIR /workspace

COPY requirements.txt .

# install dependencies (no resolver loop now)
RUN pip install --no-cache-dir --retries 5 --use-deprecated=legacy-resolver -r requirements.txt

COPY app ./app
COPY templates ./templates
COPY openenv.yaml .

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]