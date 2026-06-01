default: check

# Run the same gates as CI.
check:
    cargo fmt --all -- --check
    cargo clippy --all-targets -- -D warnings
    cargo clippy --no-default-features --all-targets --features "tap" -- -D warnings
    cargo clippy --no-default-features --all-targets --features "model-catalog" -- -D warnings
    cargo clippy --no-default-features --all-targets --features "cache-foyer" -- -D warnings
    cargo clippy --no-default-features --all-targets --features "cache-memvid" -- -D warnings
    cargo clippy --no-default-features --all-targets --features "eval" -- -D warnings
    cargo clippy --all-targets --all-features -- -D warnings
    cargo test
    cargo test --no-default-features --features "cache-foyer"
    cargo test --no-default-features --features "eval"
    cargo test --all-features

fmt:
    cargo fmt --all

clippy:
    cargo clippy --all-targets -- -D warnings
    cargo clippy --all-targets --all-features -- -D warnings

test:
    cargo test
    cargo test --all-features

# Phase 6.F: real-user chat-savings benchmark.
measure provider="ollama":
    cargo run --release --example measure_savings --features "tap,model-catalog" -- --provider {{provider}}
