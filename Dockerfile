FROM python:3.12-slim
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src/app.py src/app.py
COPY models/ models/
ENV MODEL_DIR=/app/models
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "--app-dir", "src", "app:app", "--host", "0.0.0.0", "--port", "8000"]
