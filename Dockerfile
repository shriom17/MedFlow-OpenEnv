# Docker configuration
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY models.py tasks.py client.py __init__.py /app/
COPY server/ /app/server/
EXPOSE 7860
ENV PORT=7860
ENV ENABLE_WEB_INTERFACE=true
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]