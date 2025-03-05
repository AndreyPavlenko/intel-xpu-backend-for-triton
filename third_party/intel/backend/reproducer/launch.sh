#!/bin/sh

cd "$(dirname "$0")"
mkdir -p build
cmake -B build -S .
cmake --build build
exec ./build/KernelLauncher
