FROM rust:1.82-bookworm AS builder

WORKDIR /app

COPY Cargo.toml ./
COPY src ./src

RUN cargo build --release

FROM debian:bookworm-slim

WORKDIR /app

COPY --from=builder /app/target/release/sudoku-server /app/sudoku-server
COPY index.html ./static/index.html
COPY styles.css ./static/styles.css
COPY app.js ./static/app.js
COPY sw.js ./static/sw.js
COPY manifest.webmanifest ./static/manifest.webmanifest
COPY favicon-32.png ./static/favicon-32.png
COPY icon-192.png ./static/icon-192.png
COPY icon-512.png ./static/icon-512.png
COPY apple-touch-icon.png ./static/apple-touch-icon.png
COPY fonts ./static/fonts

ENV STATIC_DIR=/app/static
ENV OCR_URL=http://ocr:8000/scan

EXPOSE 8080

CMD ["/app/sudoku-server"]
