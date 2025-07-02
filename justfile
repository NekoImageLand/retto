set windows-shell := ["pwsh.exe", "-NoLogo", "-Command"]

clean:
    cargo clean; rm -r retto-core/models; rm -r retto-wasm/fe/retto_wasm.*; rm -r retto-wasm/fe/dist; rm -r retto-wasm/fe/node_modules

setup:
    rustup toolchain install nightly --profile complete
    rustup target add wasm32-unknown-emscripten --toolchain nightly

fmt-check: setup
    cargo +nightly fmt --all -- --check

clippy-check: setup
    cargo +nightly clippy --locked --all-targets -- --deny warnings

build-check: setup
    cargo +nightly check --locked --all-targets --all-features

build-cli: setup
    cargo build -p retto-cli --features "hf-hub" --release

build-wasm-lib: setup
    rustup toolchain install nightly && rustup target add wasm32-unknown-emscripten --toolchain nightly
    cd retto-wasm && cargo +nightly build --target wasm32-unknown-emscripten --all-features --release

build-wasm-fe: setup
    cp target/wasm32-unknown-emscripten/release/retto_wasm.* retto-wasm/fe && cd retto-wasm/fe && pnpm i && pnpm build

build-wasm: build-wasm-lib build-wasm-fe

publish-wasm-fe:
    cd retto-wasm/fe && pnpm publish --access public --no-git-checks