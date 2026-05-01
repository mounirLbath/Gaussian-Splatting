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