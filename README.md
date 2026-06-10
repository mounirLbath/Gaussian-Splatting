# Gaussian Splatting

Real-time viewer for 3D Gaussian Splatting `.ply` scenes, with a playable physics sandbox powered by [Bullet Physics](https://github.com/bulletphysics/bullet3). Built with OpenGL 4.3.

## Setup

1. Choose assets in `assets/nike_retrained/`:
   - `scene.ply` — splat template (position, SH color, scale, rotation, opacity)
   - `scene_mesh.ply` — mesh used for collision hulls and shadows  
   (Or change the folder in `src/scene.cpp`.)

2. Build and run from the repo root. (e.g. CMake relwithdebinfo)

## How it works

**PLY loader** : On startup, loads the splat template from `scene.ply`, converts SH DC to RGB, centers splats at the origin, and precomputes 3×3 covariances on the CPU. The mesh from `scene_mesh.ply` is used to build a convex-hull collision shape for each physics object.

**Physics** : Bullet simulates multiple rigid bodies on a ground plane. Each frame, body positions and rotations are applied to the splat template (CPU-side transform), then re-uploaded to GPU SSBOs. A first-person camera walks on the plane with sphere-sweep collision; left-click raycasts grab and carry objects.

Then each frame the renderer runs three GPU stages:

1. **Prepass** : A vertex-shader pass projects each splat, writes a depth key, and records visible splat indices via an atomic counter (rasterizer discard, no fragments).
2. **Radix sort** : A compute shader sorts visible splats back-to-front using float-flipped depth keys (16–32 bit precision, configurable in the GUI).
3. **Draw** : Instanced indirect draw renders one billboard quad per visible splat. The fragment shader evaluates a 2D Gaussian falloff and alpha-blends splats in sorted order.

## Controls

- **WASD / ZQSD + mouse** — walk and look around
- **Left click (hold)** — grab and carry a physics object
- **Escape** — release / recapture mouse (for GUI)
- **GUI** — object count, spacing, alpha cutoff, sort settings
