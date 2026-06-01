#include "scene.hpp"
#include "helper.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

using namespace cgp;

namespace {

constexpr size_t radix_block_size = 128u * 4u;

	numarray<vec4> pad_vec3_to_vec4(numarray<vec3> const& src)
{
	int const n = src.size();
	numarray<vec4> out;
	out.resize(n);
	for (int i = 0; i < n; ++i) {
		vec3 const& p = src[i];
		out[i] = { p.x, p.y, p.z, 0.0f };
	}
	return out;
}

// Create a SSBO and upload the given array to it (static data).
template <typename T>
void create_static_ssbo(GLuint& ssbo, numarray<T> const& data)
{
	if (data.size() <= 0)
		return;
	glGenBuffers(1, &ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
	glBufferData(
		GL_SHADER_STORAGE_BUFFER,
		GLsizeiptr(data.size()) * GLsizeiptr(sizeof(T)),
		ptr(data),
		GL_STATIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

// Bind an SSBO to a fixed binding point matching the shader's
void bind_ssbo(GLuint ssbo, GLuint binding_point)
{
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point, ssbo);
}

void create_dynamic_ssbo(GLuint& ssbo, size_t count, size_t elem_size)
{
	if (count == 0)
		return;
	glGenBuffers(1, &ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
	glBufferData(GL_SHADER_STORAGE_BUFFER, GLsizeiptr(count) * GLsizeiptr(elem_size), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void setup_instance_index_vao(GLuint vao, GLuint index_vbo, int location)
{
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, index_vbo);
	glVertexAttribIPointer(location, 1, GL_UNSIGNED_INT, 0, nullptr);
	glVertexAttribDivisor(location, 1); // advance this attribute once per instance instead of once per vertex
	glEnableVertexAttribArray(location);
	glBindVertexArray(0);
}

struct DrawElementsIndirectCommand
{
	GLuint count;
	GLuint instanceCount;
	GLuint firstIndex;
	GLint baseVertex;
	GLuint baseInstance;
};

// Minimal equivalent of cgp::draw() for an indirect instanced draw.
// The instance count lives in `indirect_buffer` and is filled on the GPU from
// the visible-splat counter, so the CPU does not need to read visible_count.
void draw_mesh_indirect(mesh_drawable const& drawable, environment_generic_structure const& environment, GLuint indirect_buffer, GLenum draw_mode = GL_TRIANGLES)
{
	bool expected_uniforms = true;

	// Send the same standard uniforms/textures as the normal cgp draw path.
	glUseProgram(drawable.shader.id);

	drawable.send_opengl_uniform(expected_uniforms); // per-drawable uniforms
	environment.send_opengl_uniform(drawable.shader, expected_uniforms && environment.default_expected_uniform); // per-frame uniforms

	
	glActiveTexture(GL_TEXTURE0); // base texture unit
	drawable.texture.bind();
	opengl_uniform(drawable.shader, "image_texture", 0, expected_uniforms);

	int texture_count = 1; // extra textures start at unit 1
	for (auto const& element : drawable.supplementary_texture) {
		std::string const& additional_texture_name = element.first;
		opengl_texture_image_structure const& additional_texture = element.second;
		glActiveTexture(GL_TEXTURE0 + texture_count);
		additional_texture.bind();
		opengl_uniform(drawable.shader, additional_texture_name, texture_count, expected_uniforms);
		texture_count++;
	}

	// Draw once per visible sorted instance using the indirect command.
	glBindVertexArray(drawable.vao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, drawable.ebo_connectivity.id); // index buffer
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_buffer); // count + instanceCount
	glDrawElementsIndirect(draw_mode, GL_UNSIGNED_INT, nullptr);
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
	glBindVertexArray(0);
	glUseProgram(0);
}

GLuint load_compute_program(std::string const& path)
{
	// Open and read the shader source code from file
	std::ifstream file(path);
	std::ostringstream buffer;
	buffer << file.rdbuf();
	std::string const source = buffer.str();

	// Compile the single compute shader object.
	GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
	char const* src = source.c_str();
	glShaderSource(shader, 1, &src, nullptr);
	glCompileShader(shader);

	// Create the program, attach the shader, and link.
	GLuint program = glCreateProgram();
	glAttachShader(program, shader);
	glLinkProgram(program);
	glDeleteShader(shader);

	return program;
}

} // namespace


// Main initialization function called once at program startup
// Sets up the camera, 3D scene elements, and the image animation system
void scene_structure::initialize()
{
	
	std::cout << "Start function scene_structure::initialize()" << std::endl;

	// Set the behavior of the camera and its initial position
	// ********************************************** //
	camera_control.initialize(inputs, window); 
	camera_control.set_rotation_axis_z(); // camera rotates around z-axis
	//   look_at(camera_position, targeted_point, up_direction)
	camera_control.look_at(
		{ 5.0f, -4.0f, 3.5f } /* position of the camera in the 3D scene */,
		{0,0,0} /* targeted point in 3D scene */,
		{0,0,1} /* direction of the "up" vector */);

	camera_projection = camera_projection_perspective{
		50.0f * Pi/180, // Field of view
		1.0f,           // Aspect ratio
		0.01f,          // Depth min
		1000            // Depth max
	};


	// General information
	display_info();

	// Create 3D coordinate frame (x, y, z axes) for visual reference
	global_frame.initialize_data_on_gpu(mesh_primitive_frame());

	// Change the background color to black
	environment.background_color = {0, 0, 0};

	// Initialize the shapes of the scene
	// ***************************************** //

	gui.display_frame = true;

	mesh quad_mesh = mesh_primitive_quadrangle(
		{ -1,-1,0 },
		{  1,-1,0 },
		{  1, 1,0 },
		{ -1, 1,0 }
	);

	numarray<vec3> splat_scales;
	numarray<vec4> splat_rotations;
	read_points_from_ply_file("./assets/nike/scene.ply", splat_points, splat_colors, splat_scales, splat_rotations, splat_opacities, 0.7);

	std::cout << splat_points.size() << " points" << std::endl;

	quad1.initialize_data_on_gpu(quad_mesh);
	quad1.shader.load(project::path + "shaders/instancing/instancing.vert.glsl", project::path + "shaders/instancing/instancing.frag.glsl");
	// quad1.model.rotation = rotation_transform::from_axis_angle({1,0,0}, -3.14/2.0);
	// quad1.initialize_supplementary_data_on_gpu(splat_points, /*location*/ 4, /*divisor: 1=per instance, 0=per vertex*/ 1);

	int const n = splat_points.size();
	splat_indices.resize(n);
	for (int k = 0; k < n; ++k)
		splat_indices[k] = k;

	// Precompute the 3D covariance Sigma = R * diag(s^2) * R^T on the CPU
	numarray<vec4> splat_covariances;
	compute_covariances_from_scales_and_rotations(splat_scales, splat_rotations, splat_covariances);

	// Upload per-splat data to SSBOs. Points and colors are padded to vec4
	numarray<vec4> const pos4 = pad_vec3_to_vec4(splat_points);
	numarray<vec4> const col4 = pad_vec3_to_vec4(splat_colors);
	create_static_ssbo(ssbo_points, pos4);
	create_static_ssbo(ssbo_colors, col4);
	create_static_ssbo(ssbo_covariances, splat_covariances);
	create_static_ssbo(ssbo_opacities, splat_opacities);
	create_dynamic_ssbo(ssbo_depth_keys, n, sizeof(unsigned int));
	create_dynamic_ssbo(ssbo_sort_ping, n, sizeof(unsigned int) * 2u);
	create_dynamic_ssbo(ssbo_sort_pong, n, sizeof(unsigned int) * 2u);
	size_t const radix_block_count_max = (size_t(n) + radix_block_size - 1u) / radix_block_size;
	create_dynamic_ssbo(ssbo_radix_hist, radix_block_count_max * 256u, sizeof(unsigned int));
	create_dynamic_ssbo(ssbo_radix_prefix, 256, sizeof(unsigned int));
	create_dynamic_ssbo(ssbo_indirect_draw, 5, sizeof(unsigned int));

	// Set up the indirect draw parameters.
	DrawElementsIndirectCommand indirect_cmd = {};
	indirect_cmd.count = GLuint(quad1.ebo_connectivity.size * 3);
	indirect_cmd.instanceCount = GLuint(splat_points.size());
	indirect_cmd.firstIndex = 0u;
	indirect_cmd.baseVertex = 0;
	indirect_cmd.baseInstance = 0u;
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, ssbo_indirect_draw);
	glBufferSubData(GL_DRAW_INDIRECT_BUFFER, 0, sizeof(DrawElementsIndirectCommand), &indirect_cmd);
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

	// Load the radix sort shader.
	compute_radix_program = load_compute_program(project::path + "shaders/compute/radix_sort.comp.glsl");


	glGenBuffers(1, &ssbo_visible_counter);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, ssbo_visible_counter);
	glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

	glGenBuffers(1, &vbo_indices);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_indices);
	glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(n) * GLsizeiptr(sizeof(int)), ptr(splat_indices), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	setup_instance_index_vao(quad1.vao, vbo_indices, /* location = */ 4);

	// Bind each SSBO once to its binding point
	bind_ssbo(ssbo_points, 0);
	bind_ssbo(ssbo_colors, 1);
	bind_ssbo(ssbo_covariances, 2);
	bind_ssbo(ssbo_opacities, 3);
	bind_ssbo(ssbo_depth_keys, 4);
	glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 5, ssbo_visible_counter);
	bind_ssbo(ssbo_sort_ping, 6);
	bind_ssbo(ssbo_sort_pong, 7);
	bind_ssbo(ssbo_radix_hist, 8);
	bind_ssbo(ssbo_radix_prefix, 9);
	bind_ssbo(ssbo_indirect_draw, 10);


	std::cout << "End function scene_structure::initialize()" << std::endl;

}




// This function is called permanently at every new frame
// Note that you should avoid having costly computation and large allocation defined there. This function is mostly used to call the draw() functions on pre-existing data.
void scene_structure::display_frame()
{
	// Set the light to the current position of the camera
	camera_projection.aspect_ratio = window.aspect_ratio();
	environment.camera_projection = camera_projection.matrix();
	environment.camera_view = camera_control.camera_model.matrix_view();
	environment.light = camera_control.camera_model.position();

	// Draw the 3D reference frame axes if enabled
	if (gui.display_frame)
		draw(global_frame, environment);

	// Update time
	timer.update();

	// send alpha_cutoff uniform 
	environment.uniform_generic.uniform_float["alpha_cutoff"] = gui.alpha_cutoff;
	environment.uniform_generic.uniform_int["depth_bits"] = gui.depth_bits;
	environment.uniform_generic.uniform_int["prepass_only"] = 0;

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDepthMask(false);

	if (ssbo_visible_counter != 0) {
		unsigned int zero = 0;
		glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, ssbo_visible_counter);
		glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(unsigned int), &zero);
		glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
	}

	// If we have splats, run prepass -> sort -> draw.
	if (splat_points.size() > 0){
		unsigned int total_count = static_cast<unsigned int>(splat_points.size());

		// Prepass starts from the full instance count.
		glBindBuffer(GL_DRAW_INDIRECT_BUFFER, ssbo_indirect_draw);
		glBufferSubData(GL_DRAW_INDIRECT_BUFFER, sizeof(unsigned int), sizeof(unsigned int), &total_count);
		glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

		// Prepass: write depth keys + visible indices into SSBOs.
		environment.uniform_generic.uniform_int["prepass_only"] = 1;
		glUseProgram(quad1.shader.id);

		// Prepass writes to ping (binding 6).
		bind_ssbo(ssbo_sort_ping, 6);
		quad1.send_opengl_uniform(false);
		environment.send_opengl_uniform(quad1.shader, false);

		// Prepass must scan all splats, so bind the original index stream.
		glBindVertexArray(quad1.vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_indices);
		glVertexAttribIPointer(4, 1, GL_UNSIGNED_INT, 0, nullptr);
		glVertexAttribDivisor(4, 1);
		glEnableVertexAttribArray(4);

		// Run vertex shader only (no fragments) for prepass.
		glEnable(GL_RASTERIZER_DISCARD);
		glDisable(GL_BLEND);
		glDrawArraysInstanced(GL_POINTS, 0, 1, total_count);
		glDisable(GL_RASTERIZER_DISCARD);
		glEnable(GL_BLEND);
		glBindVertexArray(0);
		glUseProgram(0);
		environment.uniform_generic.uniform_int["prepass_only"] = 0;

		// Make prepass writes visible to compute.
		glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

		// Copy visible count into the indirect draw command (GPU->GPU).
		glBindBuffer(GL_COPY_READ_BUFFER, ssbo_visible_counter);
		glBindBuffer(GL_COPY_WRITE_BUFFER, ssbo_indirect_draw);
		glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, sizeof(unsigned int), sizeof(unsigned int));
		glBindBuffer(GL_COPY_READ_BUFFER, 0);
		glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
		glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

		glUseProgram(compute_radix_program);

		// Bind compute uniforms.
		GLint loc_phase = glGetUniformLocation(compute_radix_program, "radix_phase");
		GLint loc_shift = glGetUniformLocation(compute_radix_program, "shift");
		GLint loc_block_count = glGetUniformLocation(compute_radix_program, "block_count");

		// One 8-bit radix pass per byte of the depth key.
		int passes = (gui.depth_bits + 7) / 8;
		GLuint const radix_block_count = GLuint((size_t(total_count) + radix_block_size - 1u) / radix_block_size);
		bool ping_input = true;
		for (int pass = 0; pass < passes; ++pass) {
			// Select the current radix digit.
			int shift = pass * 8;
			glUniform1i(loc_shift, shift);
			glUniform1ui(loc_block_count, radix_block_count);

			// Bind ping-pong buffers correctly.
			if (ping_input) {
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, ssbo_sort_ping);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, ssbo_sort_pong);
			} else {
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, ssbo_sort_pong);
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, ssbo_sort_ping);
			}

			// Step 1: clear histogram/prefix.
			glUniform1i(loc_phase, 0);
			glDispatchCompute(radix_block_count, 1, 1);
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

			// Step 2: build histogram.
			glUniform1i(loc_phase, 1);
			glDispatchCompute(radix_block_count, 1, 1);
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

			// Step 3: prefix sums.
			glUniform1i(loc_phase, 2);
			glDispatchCompute(1, 1, 1);
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

			// Step 4: scatter into output.
			glUniform1i(loc_phase, 3);
			glDispatchCompute(radix_block_count, 1, 1);
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

			ping_input = !ping_input;
		}

		// Unbind compute resources.
		glUseProgram(0);

		// Bind sorted indices: SortPair is {key,index} -> attribute 4 reads index.
		bool sorted_in_ping = ping_input;
		GLuint sorted_buffer = sorted_in_ping ? ssbo_sort_ping : ssbo_sort_pong;
		glBindVertexArray(quad1.vao);
		glBindBuffer(GL_ARRAY_BUFFER, sorted_buffer);
		glVertexAttribIPointer(4, 1, GL_UNSIGNED_INT, sizeof(unsigned int) * 2, reinterpret_cast<void*>(sizeof(unsigned int)));
		glVertexAttribDivisor(4, 1);
		glEnableVertexAttribArray(4);
		glBindVertexArray(0);

		// Final draw: uses visible count + sorted indices.
		draw_mesh_indirect(quad1, environment, ssbo_indirect_draw);
	}

	glDepthMask(true);
	glDisable(GL_BLEND);
}


void scene_structure::display_gui()
{
	ImGui::Checkbox("Frame", &gui.display_frame);
	ImGui::SliderFloat("Alpha cutoff", &gui.alpha_cutoff, 1e-6f, 1.0f, "%.5f", ImGuiSliderFlags_Logarithmic);
	const char* depth_labels[] = { "16", "24", "32" };
	int depth_index = (gui.depth_bits == 16) ? 0 : (gui.depth_bits == 24) ? 1 : 2;
	if (ImGui::Combo("Depth bits", &depth_index, depth_labels, 3)) {
		gui.depth_bits = (depth_index == 0) ? 16 : (depth_index == 1) ? 24 : 32;
	}
}




void scene_structure::mouse_move_event()
{
	if (!inputs.keyboard.shift)
		camera_control.action_mouse_move();
	
}
void scene_structure::mouse_click_event()
{
	camera_control.action_mouse_click();
}
void scene_structure::keyboard_event()
{
	camera_control.action_keyboard();
}
void scene_structure::idle_frame()
{
	camera_control.idle_frame();
	
}

void scene_structure::display_info()
{
	std::cout << "\nCAMERA CONTROL:" << std::endl;
	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << camera_control.doc_usage() << std::endl;
	std::cout << "-----------------------------------------------\n" << std::endl;


	std::cout << "\nSCENE INFO:" << std::endl;
	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << "Gaussian splat renderer" << std::endl;
	std::cout << "-----------------------------------------------\n" << std::endl;
}