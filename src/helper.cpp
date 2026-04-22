#include "helper.hpp" 

#include <iostream>
#include <sstream>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <fstream>

std::vector<vec3> read_points_from_ply_file(const std::string &filepath)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filepath << std::endl;
        return {};
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

	std::vector<vec3> points;

	for(int i = 0; i < nbVertex; i++)
	{
		if(properties["x"].type != "float" || properties["y"].type != "float" || properties["z"].type != "float")
		{
			std::cerr << "x,y and z should be of type float in " << filepath << std::endl;
        	return {};
		}
		float x, y, z;

		std::memcpy(&x, ptr + properties["x"].offset, 4);
		std::memcpy(&y, ptr + properties["y"].offset, 4);
		std::memcpy(&z, ptr + properties["z"].offset, 4);

		points.emplace_back(x, y, z);

		ptr += vertexSize;
	}

	for(int i = 0; i < 10; i++)
		std::cout << points[i] << std::endl;

	return points;
}
