#include "helper.hpp" 
#include "cgp/cgp.hpp"

#include <iostream>
#include <sstream>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <fstream>

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
			colors.push_back({f0, f1, f2});
			scales.push_back({scale0, scale1, scale2});
			rotations.push_back({rot0, rot1, rot2, rot3});
			opacities.push_back(opacity);
		}

		ptr += vertexSize;
	}
}
