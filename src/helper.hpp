#pragma once

#include "cgp/cgp.hpp"
using cgp::vec3;

#include <string>
#include <vector>


struct Property {
    size_t offset;
    std::string type;
};


// read point coordinates of cloud from ply file
void read_points_from_ply_file(const std::string &filepath, cgp::numarray<vec3> &points, const float percentage = 1.0f);
