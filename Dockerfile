FROM node:20-alpine AS frontend
WORKDIR /app
COPY package.json package-lock.json build.js tsconfig.json ./
COPY src/client ./src/client
COPY static ./static
RUN npm ci
RUN node build.js

FROM rust:1.83-slim AS backend
WORKDIR /app
RUN apt-get update \
  && apt-get install -y --no-install-recommends pkg-config libssl-dev curl \
  && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml ./
COPY src/server ./src/server
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update \
  && apt-get install -y ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=backend /app/target/release/sudoku /app/sudoku
COPY --from=frontend /app/static /app/static
ENV STATIC_DIR=/app/static
ENV BIND_ADDR=0.0.0.0:8080
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/api/health || exit 1
CMD ["/app/sudoku"]
