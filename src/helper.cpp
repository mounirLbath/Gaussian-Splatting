#include "helper.hpp" 
#include "cgp/cgp.hpp"

#include <iostream>
#include <sstream>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <cmath>
#include <algorithm>

float sigmoid(float x)
{
	return 1.0f / (1.0f + std::exp(-x));
}


// Spherical Harmonics to RGB conversion 
constexpr float C0 = 0.28209479177387814f; // Y^0_0 = 1 / sqrt(4pi)
cgp::vec3 sh_dc_to_rgb(float f0, float f1, float f2)
{
	float r = 0.5f + C0 * f0;
	float g = 0.5f + C0 * f1;
	float b = 0.5f + C0 * f2;
	r = std::max(0.0f, std::min(1.0f, r));
	g = std::max(0.0f, std::min(1.0f, g));
	b = std::max(0.0f, std::min(1.0f, b));
	return { r, g, b };
}

// PLY stores quaternion as (w,x,y,z) in rot_0..3; GLSL uses vec4(x,y,z,w).
cgp::vec4 ply_rot_to_shader_quat(float w, float x, float y, float z)
{
	float len = std::sqrt(w * w + x * x + y * y + z * z);
	if (len > 1e-6f) {
		w /= len;
		x /= len;
		y /= len;
		z /= len;
	}
	return { x, y, z, w };
}

void read_points_from_ply_file(
    const std::string& filepath,
    cgp::numarray<cgp::vec3>& points,
    cgp::numarray<cgp::vec3>& colors,
    cgp::numarray<cgp::vec3>& scales,
    cgp::numarray<cgp::vec4>& rotations,
    cgp::numarray<float>& opacities,
    float percentage
)
{

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filepath << std::endl;
        return;
    }

	std::string line;


	std::string format;
	int nbVertex = 0;
	std::unordered_map<std::string,Property> properties;
	size_t vertexSize = 0;



	while(std::getline(file, line))
	{
		if(line.rfind("format", 0) == 0)
		{
			std::istringstream iss(line);
			std::string tmp;
			iss >> tmp >> format >> tmp;
		}

		if(line.rfind("element vertex", 0) == 0)
		{
			std::istringstream iss(line);
			std::string tmp;
			iss >> tmp >> tmp >> nbVertex;
		}

		if(line.rfind("property", 0) == 0)
		{
			std::istringstream iss(line);
			std::string tmp, type, name;
			iss >> tmp >> type >> name;

			properties[name] = {vertexSize, type};

			size_t size = 0;
			if(type == "float") size = 4;
			else if(type == "int") size = 4;

			vertexSize += size;
		}

		if (line == "end_header") break;
	}

	std::streampos dataStart = file.tellg();
	file.seekg(0, std::ios::end);
	std::streampos fileEnd = file.tellg();

	size_t dataSize = static_cast<size_t>(fileEnd - dataStart);

	file.seekg(dataStart);

	std::vector<uint8_t> buffer(dataSize);
	file.read(reinterpret_cast<char*>(buffer.data()), dataSize);
    file.close();

    std::cout << "Loaded " << dataSize << " bytes from " << filepath << std::endl;

	uint8_t* ptr = buffer.data(); 

	for(int i = 0; i < nbVertex; i++)
	{
		if(properties["x"].type != "float" || properties["y"].type != "float" || properties["z"].type != "float")
		{
			std::cerr << "x,y and z should be of type float in " << filepath << std::endl;
        	return;
		}
		float x, y, z;
		float f0, f1, f2;
		float scale0, scale1, scale2;
		float rot0, rot1, rot2, rot3;
		float opacity;

		std::memcpy(&x, ptr + properties["x"].offset, 4);
		std::memcpy(&y, ptr + properties["y"].offset, 4);
		std::memcpy(&z, ptr + properties["z"].offset, 4);

		std::memcpy(&f0, ptr + properties["f_dc_0"].offset, 4);
		std::memcpy(&f1, ptr + properties["f_dc_1"].offset, 4);
		std::memcpy(&f2, ptr + properties["f_dc_2"].offset, 4);

		std::memcpy(&scale0, ptr + properties["scale_0"].offset, 4);
		std::memcpy(&scale1, ptr + properties["scale_1"].offset, 4);
		std::memcpy(&scale2, ptr + properties["scale_2"].offset, 4);

		std::memcpy(&rot0, ptr + properties["rot_0"].offset, 4);
		std::memcpy(&rot1, ptr + properties["rot_1"].offset, 4);
		std::memcpy(&rot2, ptr + properties["rot_2"].offset, 4);
		std::memcpy(&rot3, ptr + properties["rot_3"].offset, 4);

		std::memcpy(&opacity, ptr + properties["opacity"].offset, 4);

		if(percentage > 0.99f || static_cast<float>(rand()) / RAND_MAX <= percentage)
		{
			points.push_back({x, y, z});
			colors.push_back(sh_dc_to_rgb(f0, f1, f2));
			scales.push_back({std::exp(scale0), std::exp(scale1), std::exp(scale2)});
			rotations.push_back(ply_rot_to_shader_quat(rot0, rot1, rot2, rot3));
			opacities.push_back(sigmoid(opacity));
		}

		ptr += vertexSize;
	}
}


void sortPoints(cgp::numarray<int>& indices, const cgp::numarray<cgp::vec3>& points, const cgp::vec3& view)
{
	auto depth_sq = [&](int idx) {
		return dot(points[idx] - view, points[idx] - view);
	};

	std::sort(indices.begin(), indices.end(), [&](int a, int b) {
		float depth_a = depth_sq(a);
		float depth_b = depth_sq(b);
		if(depth_a != depth_b) return depth_a > depth_b;
		return a < b;
	});
}