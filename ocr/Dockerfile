FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      libglib2.0-0 \
      libgl1 \
      libsm6 \
      libxrender1 \
      libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ocr/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ocr/app.py ./app.py
COPY ocr/models ./models

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
