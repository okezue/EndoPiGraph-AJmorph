FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .

ENV PIMORPH_DATA_DIR=/data
RUN mkdir -p /data

EXPOSE 7860
CMD ["gunicorn", "labeler:app", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "600"]
