# The scoring service: FastAPI at /, /model, /predict and /docs, and the Gradio form at /ui.
#
#   docker build -t home-credit-api .
#   docker run -p 8000:8000 home-credit-api
#
# The image holds the code and the saved model in src/serving/model (checksum checked on start), not the data.
FROM python:3.13-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src/ src/

# run as an unprivileged user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4)"

CMD ["uvicorn", "src.app.app:app", "--host", "0.0.0.0", "--port", "8000"]
