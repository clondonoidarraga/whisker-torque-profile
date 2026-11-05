#!/usr/bin/env bash
# Clona WHISKiT Physics (SeNSE lab), aplica nuestros parches y lo compila. Ejecutar desde la raíz del repo.
set -e
sudo apt-get update && sudo apt-get install -y build-essential cmake libboost-all-dev
git clone https://github.com/SeNSE-lab/whiskitphysics.git whiskitphysics
cd whiskitphysics && git apply ../patches/whiskit_patches.diff
cd code && mkdir -p build && cd build && cmake .. && make -j"$(nproc)"
echo "Listo: whiskitphysics/code/build/whiskit"
