# Prepared configuration only: Docker build/run was not executed for this delivery.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY antispam.py api.py service.py maildata.py experiment_common.py ./
COPY models/ ./models/
USER 65534:65534
EXPOSE 8000
CMD ["python", "antispam.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
