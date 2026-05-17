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


// ---------------------------------------------------------------------------
// Compute / graphics shader loading helpers (Phase 1).
// ---------------------------------------------------------------------------
namespace {
std::string read_file_text(std::string const& path)
{
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[shader] Cannot open " << path << std::endl;
        return {};
    }
    std::string out;
    f.seekg(0, std::ios::end);
    out.resize(static_cast<size_t>(f.tellg()));
    f.seekg(0, std::ios::beg);
    f.read(out.data(), out.size());
    return out;
}

GLuint compile_one(GLenum stage, std::string const& src, char const* tag)
{
    GLuint id = glCreateShader(stage);
    char const* p = src.c_str();
    glShaderSource(id, 1, &p, nullptr);
    glCompileShader(id);
    GLint ok = 0;
    glGetShaderiv(id, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(static_cast<size_t>(len) + 1);
        glGetShaderInfoLog(id, len, &len, log.data());
        std::cerr << "[shader] Compile error in " << tag << ":\n" << log.data() << std::endl;
        glDeleteShader(id);
        return 0;
    }
    return id;
}

GLuint link_program(std::vector<GLuint> const& shaders)
{
    GLuint prog = glCreateProgram();
    for (GLuint s : shaders) glAttachShader(prog, s);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(static_cast<size_t>(len) + 1);
        glGetProgramInfoLog(prog, len, &len, log.data());
        std::cerr << "[shader] Link error:\n" << log.data() << std::endl;
        glDeleteProgram(prog);
        for (GLuint s : shaders) glDeleteShader(s);
        return 0;
    }
    for (GLuint s : shaders) {
        glDetachShader(prog, s);
        glDeleteShader(s);
    }
    return prog;
}
} // namespace

GLuint load_compute_program(std::string const& path)
{
    std::string const src = read_file_text(path);
    if (src.empty()) return 0;
    GLuint cs = compile_one(GL_COMPUTE_SHADER, src, path.c_str());
    if (!cs) return 0;
    GLuint const prog = link_program({cs});
    if (prog) std::cout << "  [info] Compute shader compiled successfully [ID=" << prog << "] (" << path << ")" << std::endl;
    return prog;
}

GLuint load_graphics_program(std::string const& vert_path, std::string const& frag_path)
{
    std::string const vsrc = read_file_text(vert_path);
    std::string const fsrc = read_file_text(frag_path);
    if (vsrc.empty() || fsrc.empty()) return 0;
    GLuint vs = compile_one(GL_VERTEX_SHADER,   vsrc, vert_path.c_str());
    GLuint fs = compile_one(GL_FRAGMENT_SHADER, fsrc, frag_path.c_str());
    if (!vs || !fs) {
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        return 0;
    }
    GLuint const prog = link_program({vs, fs});
    if (prog) std::cout << "  [info] Graphics shader compiled successfully [ID=" << prog << "] (" << vert_path << ", " << frag_path << ")" << std::endl;
    return prog;
}


void compute_covariances_from_scales_and_rotations(
    cgp::numarray<cgp::vec3> const& scales,
    cgp::numarray<cgp::vec4> const& rotations,
    cgp::numarray<cgp::vec4>& out_covariances)
{
	int const n = scales.size();
	out_covariances.resize(n * 2);

	for (int k = 0; k < n; ++k) {
		cgp::vec3 const& s = scales[k];
		cgp::vec4 const& q = rotations[k]; // (x, y, z, w)

		// Rotation R from the unit quaternion 
		cgp::mat3 const R = cgp::mat3::build_rotation_from_quaternion(cgp::quaternion{q.x, q.y, q.z, q.w});
		cgp::mat3 const S2 = cgp::mat3::build_diagonal(s.x * s.x, s.y * s.y, s.z * s.z);
		cgp::mat3 const Sigma = R * S2 * transpose(R);

		out_covariances[2 * k + 0] = { Sigma(0,0), Sigma(1,1), Sigma(2,2), Sigma(0,1) };
		out_covariances[2 * k + 1] = { Sigma(0,2), Sigma(1,2), 0.0f, 0.0f};
	}
}