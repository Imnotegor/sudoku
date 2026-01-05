FROM python:3.11-slim
WORKDIR /app

RUN apt-get update \
  && apt-get install -y tesseract-ocr libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY ocr/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ocr /app/ocr

EXPOSE 8000
CMD ["uvicorn", "ocr.main:app", "--host", "0.0.0.0", "--port", "8000"]
