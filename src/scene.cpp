#include "scene.hpp"
#include "helper.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

using namespace cgp;

namespace {

constexpr size_t radix_block_size = 128u * 4u;

vec3 compute_bounds_center(numarray<vec3> const& points)
{
	if (points.size() <= 0)
		return {0, 0, 0};
	vec3 min_p = points[0];
	vec3 max_p = points[0];
	for (int i = 1; i < points.size(); ++i) {
		min_p = { std::min(min_p.x, points[i].x), std::min(min_p.y, points[i].y), std::min(min_p.z, points[i].z) };
		max_p = { std::max(max_p.x, points[i].x), std::max(max_p.y, points[i].y), std::max(max_p.z, points[i].z) };
	}
	return 0.5f * (min_p + max_p);
}

float compute_mesh_to_splat_scale(numarray<vec3> const& mesh_vertices, numarray<vec3> const& splat_points)
{
	if (mesh_vertices.size() == 0 || splat_points.size() == 0)
		return 1.0f;

	vec3 mesh_min = mesh_vertices[0];
	vec3 mesh_max = mesh_vertices[0];
	vec3 splat_min = splat_points[0];
	vec3 splat_max = splat_points[0];
	for (int i = 1; i < mesh_vertices.size(); ++i) {
		mesh_min = { std::min(mesh_min.x, mesh_vertices[i].x), std::min(mesh_min.y, mesh_vertices[i].y), std::min(mesh_min.z, mesh_vertices[i].z) };
		mesh_max = { std::max(mesh_max.x, mesh_vertices[i].x), std::max(mesh_max.y, mesh_vertices[i].y), std::max(mesh_max.z, mesh_vertices[i].z) };
	}
	for (int i = 1; i < splat_points.size(); ++i) {
		splat_min = { std::min(splat_min.x, splat_points[i].x), std::min(splat_min.y, splat_points[i].y), std::min(splat_min.z, splat_points[i].z) };
		splat_max = { std::max(splat_max.x, splat_points[i].x), std::max(splat_max.y, splat_points[i].y), std::max(splat_max.z, splat_points[i].z) };
	}

	vec3 const mesh_extent = mesh_max - mesh_min;
	vec3 const splat_extent = splat_max - splat_min;
	float const mesh_span = std::max({ mesh_extent.x, mesh_extent.y, mesh_extent.z, 1e-4f });
	float const splat_span = std::max({ splat_extent.x, splat_extent.y, splat_extent.z, 1e-4f });
	return splat_span / mesh_span;
}

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
	// Reset ssbo if needed
	if (ssbo != 0) {
		glDeleteBuffers(1, &ssbo);
		ssbo = 0;
	}
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
	// Reset ssbo if needed
	if (ssbo != 0) {
		glDeleteBuffers(1, &ssbo);
		ssbo = 0;
	}
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

mat4 build_planar_shadow_matrix(vec3 const& light_direction)
{
	vec3 const dir = normalize(light_direction);
	vec4 const plane = {0.0f, 0.0f, 1.0f, 0.0f};
	vec4 const light = {dir.x, dir.y, dir.z, 0.0f};
	float const dot = plane.x * light.x + plane.y * light.y + plane.z * light.z + plane.w * light.w;
	return mat4(
		dot - light.x * plane.x, -light.x * plane.y, -light.x * plane.z, -light.x * plane.w,
		-light.y * plane.x, dot - light.y * plane.y, -light.y * plane.z, -light.y * plane.w,
		-light.z * plane.x, -light.z * plane.y, dot - light.z * plane.z, -light.z * plane.w,
		-light.w * plane.x, -light.w * plane.y, -light.w * plane.z, dot - light.w * plane.w);
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
	camera_control.set_rotation_axis_z();
	camera_control.look_at({0.0f, -4.0f, 1.26f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f});
	camera_control.enable_play_mode();

	environment.light = normalize(vec3{0.45f, 0.25f, 0.85f});

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

	gui.display_frame = false;

	mesh quad_mesh = mesh_primitive_quadrangle(
		{ -1,-1,0 },
		{  1,-1,0 },
		{  1, 1,0 },
		{ -1, 1,0 }
	);

	numarray<vec3> splat_scales;
	numarray<vec4> splat_rotations;

	// Open the asset folder (containing scene.ply splats and scene_mesh.ply)
	std::string const asset_folder = "./assets/nike_retrained/";
	read_points_from_ply_file(asset_folder + "scene.ply", template_splat_points, template_splat_colors, splat_scales, splat_rotations, template_splat_opacities, 0.7);

	vec3 const splat_center = compute_bounds_center(template_splat_points);
	for (int i = 0; i < template_splat_points.size(); ++i)
		template_splat_points[i] = template_splat_points[i] - splat_center;

	// Get the mesh (vertices for physics, full mesh for shadows)
	numarray<vec3> mesh_vertices;
	read_mesh_vertices_from_ply_file(asset_folder + "scene_mesh.ply", mesh_vertices);
	mesh shadow_template;
	read_mesh_from_ply_file(asset_folder + "scene_mesh.ply", shadow_template);
	float const mesh_scale = compute_mesh_to_splat_scale(mesh_vertices, template_splat_points);

	template_splat_count = template_splat_points.size();
	// Precompute covariances
	compute_covariances_from_scales_and_rotations(splat_scales, splat_rotations, template_splat_covariances);
	std::cout << template_splat_count << " template splats" << std::endl;

	// Initialize the physics system 
	physics.initialize(mesh_vertices, mesh_scale);
	physics.set_object_count(gui.num_objects);

	// Grey collision table (50x larger play area)
	mesh table_mesh = mesh_primitive_grid(
		{-200.0f, -200.0f, 0.0f},
		{ 200.0f, -200.0f, 0.0f},
		{ 200.0f,  200.0f, 0.0f},
		{-200.0f,  200.0f, 0.0f},
		100, 100);
	table_plane.initialize_data_on_gpu(table_mesh);
	table_plane.material.color = {1.0f, 1.0f, 1.0f};
	table_plane.material.phong.ambient = 0.18f;
	table_plane.material.phong.diffuse = 0.82f;
	table_plane.material.phong.specular = 0.05f;
	table_plane.material.texture_settings.two_sided = true;

	vec3 const mesh_center = compute_bounds_center(shadow_template.position);
	for (int i = 0; i < shadow_template.position.size(); ++i)
		shadow_template.position[i] = (shadow_template.position[i] - mesh_center) * mesh_scale;
	shadow_shader.load(project::path + "shaders/shadow/shadow.vert.glsl", project::path + "shaders/shadow/shadow.frag.glsl");
	shadow_mesh.initialize_data_on_gpu(shadow_template, shadow_shader);

	mesh shadow_overlay_mesh = mesh_primitive_grid(
		{-200.0f, -200.0f, 0.015f},
		{ 200.0f, -200.0f, 0.015f},
		{ 200.0f,  200.0f, 0.015f},
		{-200.0f,  200.0f, 0.015f},
		100, 100);
	shadow_overlay_plane.initialize_data_on_gpu(shadow_overlay_mesh, shadow_shader);

	quad1.initialize_data_on_gpu(quad_mesh);
	quad1.shader.load(project::path + "shaders/instancing/instancing.vert.glsl", project::path + "shaders/instancing/instancing.frag.glsl");

	rebuild_splat_gpu_buffers();
	reset_objects();


	std::cout << "End function scene_structure::initialize()" << std::endl;

}


void scene_structure::reset_objects()
{
	int const old_total = physics.object_count() * template_splat_count;
	physics.set_object_count(gui.num_objects);
	physics.reset_objects(gui.vertical_spacing, gui.horizontal_variance);
	int const new_total = physics.object_count() * template_splat_count;
	shadow_object_positions.resize(0);
	shadow_object_rotations.resize(0);
	sync_shadow_transforms();
	if (new_total != old_total)
		rebuild_splat_gpu_buffers();
	else
		update_splats_from_physics();
}

void scene_structure::sync_shadow_transforms()
{
	int const num_objects = physics.object_count();
	if (num_objects <= 0) {
		shadow_object_positions.resize(0);
		shadow_object_rotations.resize(0);
		return;
	}

	if (shadow_object_positions.size() != num_objects) {
		shadow_object_positions.resize(num_objects);
		shadow_object_rotations.resize(num_objects);
		for (int i = 0; i < num_objects; ++i) {
			shadow_object_positions[i] = physics.get_position(i);
			shadow_object_rotations[i] = physics.get_rotation(i);
		}
		return;
	}

	float const blend = shadow_transform_blend;
	for (int i = 0; i < num_objects; ++i) {
		vec3 const target_position = physics.get_position(i);
		shadow_object_positions[i] = (1.0f - blend) * shadow_object_positions[i] + blend * target_position;

		mat3 const target_rotation = physics.get_rotation(i);
		mat3& smoothed_rotation = shadow_object_rotations[i];
		for (int r = 0; r < 3; ++r) {
			for (int c = 0; c < 3; ++c)
				smoothed_rotation(r, c) = (1.0f - blend) * smoothed_rotation(r, c) + blend * target_rotation(r, c);
		}
	}
}

void scene_structure::fill_splats_from_physics()
{
	int const num_objects = physics.object_count();
	int const n_template = template_splat_count;
	int const total = num_objects * n_template;
	if (total <= 0)
		return;

	splat_points.resize(total);
	splat_colors.resize(total);
	splat_opacities.resize(total);
	splat_covariances.resize(total * 2);

	for (int object_index = 0; object_index < num_objects; ++object_index) {
		mat3 const R = physics.get_rotation(object_index);
		vec3 const T = physics.get_position(object_index);
		for (int i = 0; i < n_template; ++i) {
			int const dst = object_index * n_template + i;
			splat_points[dst] = R * template_splat_points[i] + T;
			splat_colors[dst] = template_splat_colors[i];
			splat_opacities[dst] = template_splat_opacities[i];

			vec4 const cov_a = template_splat_covariances[2 * i + 0];
			vec4 const cov_b = template_splat_covariances[2 * i + 1];
			mat3 sigma_local = mat3(
				cov_a.x, cov_a.w, cov_b.x,
				cov_a.w, cov_a.y, cov_b.y,
				cov_b.x, cov_b.y, cov_a.z);
			mat3 const sigma_world = R * sigma_local * transpose(R);
			splat_covariances[2 * dst + 0] = { sigma_world(0,0), sigma_world(1,1), sigma_world(2,2), sigma_world(0,1) };
			splat_covariances[2 * dst + 1] = { sigma_world(0,2), sigma_world(1,2), 0.0f, 0.0f };
		}
	}
}

void scene_structure::update_splats_from_physics()
{
	fill_splats_from_physics();

	numarray<vec4> const pos4 = pad_vec3_to_vec4(splat_points);
	numarray<vec4> const col4 = pad_vec3_to_vec4(splat_colors);
	if (ssbo_points == 0)
		return;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_points);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(pos4.size()) * GLsizeiptr(sizeof(vec4)), ptr(pos4));
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_colors);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(col4.size()) * GLsizeiptr(sizeof(vec4)), ptr(col4));
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_covariances);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(splat_covariances.size()) * GLsizeiptr(sizeof(vec4)), ptr(splat_covariances));
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_opacities);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, GLsizeiptr(splat_opacities.size()) * GLsizeiptr(sizeof(float)), ptr(splat_opacities));
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void scene_structure::rebuild_splat_gpu_buffers()
{
	int const n = physics.object_count() * template_splat_count;
	splat_indices.resize(n);
	for (int k = 0; k < n; ++k)
		splat_indices[k] = k;

	fill_splats_from_physics();

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

	DrawElementsIndirectCommand indirect_cmd = {};
	indirect_cmd.count = GLuint(quad1.ebo_connectivity.size * 3);
	indirect_cmd.instanceCount = GLuint(n);
	indirect_cmd.firstIndex = 0u;
	indirect_cmd.baseVertex = 0;
	indirect_cmd.baseInstance = 0u;
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, ssbo_indirect_draw);
	glBufferSubData(GL_DRAW_INDIRECT_BUFFER, 0, sizeof(DrawElementsIndirectCommand), &indirect_cmd);
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

	if (compute_radix_program == 0)
		compute_radix_program = load_compute_program(project::path + "shaders/compute/radix_sort.comp.glsl");

	if (ssbo_visible_counter == 0) {
		glGenBuffers(1, &ssbo_visible_counter);
		glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, ssbo_visible_counter);
		glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
	}

	if (vbo_indices == 0)
		glGenBuffers(1, &vbo_indices);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_indices);
	glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(n) * GLsizeiptr(sizeof(int)), ptr(splat_indices), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	setup_instance_index_vao(quad1.vao, vbo_indices, 4);

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
}




// This function is called permanently at every new frame
// Note that you should avoid having costly computation and large allocation defined there. This function is mostly used to call the draw() functions on pre-existing data.
void scene_structure::display_frame()
{
	camera_projection.aspect_ratio = window.aspect_ratio();
	environment.camera_projection = camera_projection.matrix();
	environment.camera_view = camera_control.camera_model.matrix_view();

	// Draw the 3D reference frame axes if enabled
	if (gui.display_frame)
		draw(global_frame, environment);

	draw(table_plane, environment);
	display_shadows();

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


void scene_structure::display_shadows()
{
	mat4 const shadow_matrix = build_planar_shadow_matrix(environment.light);

	int const num_objects = physics.object_count();
	if (shadow_object_positions.size() != num_objects)
		return;

	GLboolean const cull_was_enabled = glIsEnabled(GL_CULL_FACE);
	GLboolean const stencil_was_enabled = glIsEnabled(GL_STENCIL_TEST);
	if (cull_was_enabled)
		glDisable(GL_CULL_FACE);

	uniform_generic_structure shadow_uniforms;
	shadow_uniforms.uniform_vec3["color"] = {0.04f, 0.04f, 0.04f};
	shadow_uniforms.uniform_float["alpha"] = 0.40f;

	// Pass 1: mark shadow coverage in stencil (at most one bit per pixel, regardless of triangle overlap).
	glEnable(GL_STENCIL_TEST);
	glStencilMask(0xFF);
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glDepthMask(GL_FALSE);
	glStencilFunc(GL_ALWAYS, 1, 0xFF);
	glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

	for (int object_index = 0; object_index < num_objects; ++object_index) {
		mat3 const R = shadow_object_rotations[object_index];
		vec3 const T = shadow_object_positions[object_index];
		mat4 const object_matrix = affine_rt(rotation_transform::from_matrix(R), T).matrix();
		shadow_mesh.supplementary_model_matrix = shadow_matrix * object_matrix;
		shadow_mesh.model = affine();
		draw(shadow_mesh, environment, 1, false, shadow_uniforms);
	}

	// Pass 2: paint uniform shadow color once on all marked table pixels.
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glStencilFunc(GL_EQUAL, 1, 0xFF);
	glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	shadow_overlay_plane.supplementary_model_matrix = mat4::build_identity();
	shadow_overlay_plane.model = affine();
	draw(shadow_overlay_plane, environment, 1, true, shadow_uniforms);

	glDepthMask(GL_TRUE);
	glDepthFunc(GL_LESS);
	glDisable(GL_STENCIL_TEST);
	if (stencil_was_enabled)
		glEnable(GL_STENCIL_TEST);
	if (cull_was_enabled)
		glEnable(GL_CULL_FACE);
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
	ImGui::Spacing();
	ImGui::Text("Mouse: %s (Esc to toggle)", camera_control.cursor_trapped() ? "captured" : "free");
	if (grab.active)
		ImGui::Text("Grabbing object %d", grab.body_index);
	if (ImGui::CollapsingHeader("Physics")) {
		ImGui::SliderInt("Number of objects", &gui.num_objects, 1, 20);
		ImGui::SliderFloat("Vertical spacing", &gui.vertical_spacing, 0.05f, 1.0f);
		ImGui::SliderFloat("Horizontal variance", &gui.horizontal_variance, 0.0f, 1.0f);
		if (ImGui::Button("Reset objects"))
			reset_objects();
	}
}




void scene_structure::mouse_move_event()
{
	camera_control.action_mouse_move();
}

void scene_structure::mouse_click_event()
{
	if (inputs.mouse.click.last_action == last_mouse_cursor_action::click_left)
		try_start_grab();
	if (inputs.mouse.click.last_action == last_mouse_cursor_action::release_left)
		release_grab();
}

void scene_structure::keyboard_event()
{
	camera_control.action_keyboard_playable();
}

bool scene_structure::validate_camera_move(vec3 const& from, vec3 const& to) const
{
	int const exclude = grab.active ? grab.body_index : -1;
	if (!physics.can_move_camera(from, to, exclude))
		return false;

	return true;
}

vec3 scene_structure::aim_direction() const
{
	return camera_ray_direction(
		camera_control.camera_model.matrix_frame(),
		inverse(camera_projection.matrix()),
		{0.0f, 0.0f});
}

void scene_structure::draw_crosshair() const
{
	if (!camera_control.cursor_trapped())
		return;

	ImDrawList* draw = ImGui::GetForegroundDrawList();
	ImVec2 const center = ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.5f);
	float const size = 10.0f;
	ImU32 const color = IM_COL32(255, 255, 255, 220);
	draw->AddLine(ImVec2(center.x - size, center.y), ImVec2(center.x + size, center.y), color, 1.5f);
	draw->AddLine(ImVec2(center.x, center.y - size), ImVec2(center.x, center.y + size), color, 1.5f);
}

void scene_structure::try_start_grab()
{
	if (!camera_control.cursor_trapped())
		return;

	vec3 const origin = camera_control.camera_model.position();
	vec3 const ray_dir = aim_direction();

	physics_ray_hit const hit = physics.raycast_dynamic(origin, ray_dir, 200.0f);
	if (!hit.hit)
		return;

	grab.active = true;
	grab.body_index = hit.body_index;
	grab.distance = std::max(hit.distance, 0.5f);
	grab.locked_rotation = physics.get_rotation(hit.body_index);
	physics.activate_all();
}

void scene_structure::release_grab()
{
	grab.active = false;
	grab.body_index = -1;
}

void scene_structure::update_grabbed_object()
{
	if (!grab.active || grab.body_index < 0)
		return;

	physics.activate_all();

	vec3 const camera_pos = camera_control.camera_model.position();
	vec3 const front = camera_control.camera_model.front();
	vec3 const target = camera_pos + grab.distance * front;
	vec3 const current = physics.get_position(grab.body_index);

	vec3 vel = (target - current) * grab_pull_strength;
	float const speed = norm(vel);
	if (speed > grab_max_speed)
		vel = vel * (grab_max_speed / speed);
	physics.set_linear_velocity(grab.body_index, vel);
	physics.set_angular_velocity(grab.body_index, {0, 0, 0});
}

void scene_structure::idle_frame()
{
	vec3 const camera_pos_before = camera_control.camera_model.position_camera;

	camera_control.idle_frame();

	vec3 const camera_pos_after = camera_control.camera_model.position_camera;
	if (!validate_camera_move(camera_pos_before, camera_pos_after))
		camera_control.camera_model.position_camera = camera_pos_before;

	update_grabbed_object();

	physics.step(inputs.time_interval);

	if (grab.active)
		physics.set_rotation(grab.body_index, grab.locked_rotation);

	sync_shadow_transforms();
	update_splats_from_physics();
}

void scene_structure::display_info()
{
	std::cout << "\nPLAY CONTROLS:" << std::endl;
	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << "ZQSD / WASD + mouse: walk on the plane and look around" << std::endl;
	std::cout << "Left click: grab physics object (hold to carry)" << std::endl;
	std::cout << "Escape: release / recapture mouse (for GUI)" << std::endl;
	std::cout << "-----------------------------------------------\n" << std::endl;


	std::cout << "\nSCENE INFO:" << std::endl;
	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << "Gaussian splat renderer" << std::endl;
	std::cout << "-----------------------------------------------\n" << std::endl;
}