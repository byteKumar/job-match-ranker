#!/bin/sh
# macOS only. faiss-cpu and torch wheels each bundle their own libomp.dylib. Loading both in
# one process aborts ("OMP: Error #15") or deadlocks when torch runs on CPU. Point faiss at
# torch's copy so a single OpenMP runtime is loaded. Run once after `pip install -r requirements.txt`.
set -eu
SP="$(dirname "$0")/../.venv/lib/python3.12/site-packages"
FAISS_OMP="$SP/faiss/.dylibs/libomp.dylib"
TORCH_OMP="$SP/torch/lib/libomp.dylib"
[ -f "$TORCH_OMP" ] || { echo "torch libomp not found"; exit 1; }
if [ -L "$FAISS_OMP" ]; then echo "already linked"; exit 0; fi
mv "$FAISS_OMP" "$FAISS_OMP.bundled"
ln -s "$(cd "$(dirname "$TORCH_OMP")" && pwd)/libomp.dylib" "$FAISS_OMP"
echo "faiss now uses torch's libomp: $(readlink "$FAISS_OMP")"
