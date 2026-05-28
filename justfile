default: check

# Run the same gates as CI.
check:
    cargo fmt --all -- --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo test --all-features

fmt:
    cargo fmt --all

clippy:
    cargo clippy --all-targets --all-features -- -D warnings

test:
    cargo test --all-features

# Phase 6.F: real-user chat-savings benchmark.
measure provider="ollama":
    cargo run --release --example measure_savings --features "tap,model-catalog" -- --provider {{provider}}
