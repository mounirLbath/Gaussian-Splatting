# Gaussian Splatting

Real-time viewer for 3D Gaussian Splatting `.ply` scenes. Built with OpenGL 4.3.

## Setup

1. Choose a `.ply` file at `assets/nike/scene.ply` (or change the path in `src/scene.cpp`).
2. Build and run from the repo root. (e.g. CMake relwithdebinfo)

## How it works

**PLY loader** : On startup, loads a 3DGS `.ply` (position, SH color, scale, rotation, opacity), converts SH DC to RGB, and precomputes 3×3 covariances on the CPU. Data is uploaded once to SSBOs.

Then each frame the renderer runs three GPU stages:

1. **Prepass** : A vertex-shader pass projects each splat, writes a depth key, and records visible splat indices via an atomic counter (rasterizer discard, no fragments).
2. **Radix sort** : A compute shader sorts visible splats back-to-front using float-flipped depth keys (16–32 bit precision, configurable in the GUI).
3. **Draw** : Instanced indirect draw renders one billboard quad per visible splat. The fragment shader evaluates a 2D Gaussian falloff and alpha-blends splats in sorted order.
