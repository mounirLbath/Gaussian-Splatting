#pragma once

#include "cgp/cgp.hpp"

#include <string>
#include <vector>


struct Property {
    size_t offset;
    std::string type;
};


// read point coordinates of cloud from ply file
void read_points_from_ply_file(
    const std::string& filepath,
    cgp::numarray<cgp::vec3>& points,
    cgp::numarray<cgp::vec3>& colors,
    cgp::numarray<cgp::vec3>& scales,
    cgp::numarray<cgp::vec4>& rotations,
    cgp::numarray<float>& opacities,
    float percentage = 1.0f
);


// sort points by depth
void sortPoints(cgp::numarray<int>& indices, const cgp::numarray<cgp::vec3>& points, const cgp::vec3& view);


// Precompute the symmetric 3D covariance matrices Sigma = R * diag(scale^2) * R^T for each splat
void compute_covariances_from_scales_and_rotations(
    cgp::numarray<cgp::vec3> const& scales,
    cgp::numarray<cgp::vec4> const& rotations,
    cgp::numarray<cgp::vec4>& out_covariances
);


// Load and link an OpenGL compute shader from a single .comp.glsl file.
// Returns the linked program id, or 0 on failure (and prints the log to stderr).
GLuint load_compute_program(std::string const& path);

// Load and link a vert+frag program directly (no cgp drawable wiring).
// Returns the program id, or 0 on failure.
GLuint load_graphics_program(std::string const& vert_path, std::string const& frag_path);
