set -e
cargo build --release 2>/dev/null

RUST_OUTPUT=$(./target/release/rust-inference2)
PYTHON_OUTPUT=$(python $1)

if [ "$RUST_OUTPUT" = "$PYTHON_OUTPUT" ]; then
    echo "PASS"
else
    echo "FAIL"
    diff <(echo "$RUST_OUTPUT") <(echo "$PYTHON_OUTPUT")
fi
