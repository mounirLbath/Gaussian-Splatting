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


// Read triangle mesh vertices from a PLY file (x,y,z per vertex).
void read_mesh_vertices_from_ply_file(
    const std::string& filepath,
    cgp::numarray<cgp::vec3>& vertices
);

// Read a triangle mesh (vertices + faces) from a PLY file.
void read_mesh_from_ply_file(
    const std::string& filepath,
    cgp::mesh& mesh
);
