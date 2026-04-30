FROM rust:1.88-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-dev python3-pip make ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYO3_PYTHON=python3

# Repository root is /app; crate lives in /app/backtester (matches monorepo layout).
WORKDIR /app/backtester

COPY backtester/Cargo.toml backtester/Cargo.lock backtester/pyproject.toml ./
COPY backtester/Makefile backtester/README.md ./
COPY backtester/LICENSE-APACHE backtester/LICENSE-MIT ./
COPY backtester/src ./src
COPY backtester/scripts ./scripts
COPY datasets /app/datasets
COPY traders /app/traders

RUN cargo build --release

RUN /app/backtester/target/release/rust_backtester --products off > /tmp/smoke_output.txt

CMD ["/app/backtester/target/release/rust_backtester"]
