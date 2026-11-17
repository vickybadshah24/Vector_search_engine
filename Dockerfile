FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir . fastapi "uvicorn[standard]" numpy
ENV VS_DIM=384 VS_MAX_ELEMENTS=100000 VS_SPACE=cosine
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
