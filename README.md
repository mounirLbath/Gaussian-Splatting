# Gaussian Splatting

A real-time **3D Gaussian Splatting** renderer written in C++/OpenGL on top of the
[CGP](https://github.com/drohmer/cgp) library. It loads a trained `.ply` splat
scene and renders it through a GPU-driven pipeline: per-frame projection,
visibility culling, depth sorting, and indirect instanced drawing — the CPU never
reads back the visible count.

## Pipeline

Each frame runs three GPU stages (`scene_structure::display_frame`):

1. **Prepass** — the instancing vertex shader (`instancing.vert.glsl`) runs over
   every splat with `GL_RASTERIZER_DISCARD` enabled. It projects each splat,
   culls invisible ones, computes a sortable depth key, and atomically appends the
   visible `(key, index)` pairs into an SSBO while a counter tracks how many
   passed.
2. **Sort** — a GPU **radix sort** (`shaders/compute/radix_sort.comp.glsl`)
   orders the visible splats back-to-front for correct alpha blending. It runs one
   8-bit pass per byte of the depth key (clear → histogram → prefix → scatter
   phases), ping-ponging between two buffers. The number of passes follows the
   selected depth precision (16/24/32 bits → 2/3/4 passes).
3. **Draw** — the visible count is copied GPU→GPU into a
   `DrawElementsIndirectCommand`, and a single `glDrawElementsIndirect` instances a
   textured quad per visible sorted splat (`instancing.vert/frag.glsl`), evaluating
   the Gaussian with an alpha cutoff.

## Build

Requires a C++17 compiler, **GLFW3**, and a GPU supporting **OpenGL 4.3**
(compute shaders are mandatory).

### CMake (Windows / Linux / macOS)
```bash
cmake -B build
cmake --build build --config Release
```

### Make (Linux / macOS)
```bash
make -j
```

> Note: the Makefile defaults to `CGP_OPENGL_3_3`. Since the renderer relies on
> compute shaders, build with the CMake preset (`CGP_OPENGL_4_3`) or bump that
> define. GLFW3 must be discoverable via `pkg-config`
> (`apt install libglfw3-dev`, `brew install glfw`).

## Adding a splat scene (`.ply`)

The renderer does **not** ship with a splat scene — the `assets/` folder is
git-ignored, so you must supply your own `.ply`. By default the code loads:

```
assets/nike/scene.ply
```

Create that folder and drop in a trained 3D Gaussian Splatting `.ply`, or edit the
path in the `read_points_from_ply_file(...)` call in `src/scene.cpp`.

The loader expects a **binary** PLY exported by a standard 3DGS trainer, with these
per-vertex float properties: `x y z` (position), `f_dc_0/1/2` (SH DC color),
`scale_0/1/2`, `rot_0/1/2/3` (quaternion), and `opacity`. The last argument of
`read_points_from_ply_file` (default `0.7`) randomly keeps that fraction of splats
— set it to `1.0` to load the full cloud.

## Run

Run the resulting `project` executable from the **project root** (asset and shader
paths are relative):
```bash
./project
```

## Controls

- **Mouse** — orbit / pan / zoom the camera (CGP orbit controller; the usage is
  printed to the console on startup).
- **GUI**:
  - *Frame* — toggle the world-axis frame.
  - *Alpha cutoff* — Gaussian transparency threshold.
  - *Depth bits* — sort precision (16 / 24 / 32), trading passes for accuracy.
  - *Switch to looping / manual camera* — toggle the animated orbit camera.

## Layout

```
src/        application, scene, ply loading, GPU pipeline
shaders/    instancing (prepass + draw) and radix_sort compute shaders
cgp/        CGP graphics library
assets/     splat scenes and textures
```
