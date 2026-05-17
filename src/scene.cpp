#include "scene.hpp"
#include "helper.hpp"

#include <chrono>
#include <utility>

using namespace cgp;

namespace {
// Explicit GL error check used in the Phase 1 GPU pipeline so we can localize
// which call leaks an error (rather than being caught later by cgp::draw).
inline void check_gl(char const* tag)
{
	GLenum const e = glGetError();
	if (e != GL_NO_ERROR) {
		std::cerr << "[gl-error] " << tag << " -> 0x" << std::hex << e << std::dec << std::endl;
	}
}
} // namespace


// ---------------------------------------------------------------------------
// Phase 0: profiling helpers
// ---------------------------------------------------------------------------

void gpu_timer_scope::initialize()
{
	glGenQueries(2, queries);
}
void gpu_timer_scope::destroy()
{
	if (queries[0]) glDeleteQueries(2, queries);
	queries[0] = queries[1] = 0;
	has_pending = false;
}
void gpu_timer_scope::begin()
{
	glBeginQuery(GL_TIME_ELAPSED, queries[write_index]);
}
void gpu_timer_scope::end()
{
	glEndQuery(GL_TIME_ELAPSED);
	has_pending = true;
}
void gpu_timer_scope::resolve()
{
	// Read the OTHER buffer (last frame's result) so we never stall.
	int const read_index = 1 - write_index;
	GLint available = 0;
	glGetQueryObjectiv(queries[read_index], GL_QUERY_RESULT_AVAILABLE, &available);
	if (available) {
		GLuint64 ns = 0;
		glGetQueryObjectui64v(queries[read_index], GL_QUERY_RESULT, &ns);
		last_ms = double(ns) * 1.0e-6;
	}
	write_index = read_index;
}

void frame_profiler::initialize()
{
	gpu_project.initialize();
	gpu_sort.initialize();
	gpu_draw.initialize();
}
void frame_profiler::destroy()
{
	gpu_project.destroy();
	gpu_sort.destroy();
	gpu_draw.destroy();
}
void frame_profiler::resolve_all()
{
	gpu_project.resolve();
	gpu_sort.resolve();
	gpu_draw.resolve();
}

namespace {

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

// Create a SSBO and upload the given array to it.
template <typename T>
void create_ssbo(GLuint& ssbo, numarray<T> const& data)
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

void setup_instance_index_vao(GLuint vao, GLuint index_vbo, int location)
{
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, index_vbo);
	glVertexAttribIPointer(location, 1, GL_UNSIGNED_INT, 0, nullptr);
	glVertexAttribDivisor(location, 1); // advance this attribute once per instance instead of once per vertex
	glEnableVertexAttribArray(location);
	glBindVertexArray(0);
}


// Phase 1: set up a minimal unit-quad VAO/VBO/EBO used by the GPU pipeline.
//   - location 0: vec2 in [-1, 1]^2
//   - 6 indices for two triangles
void create_unit_quad(GLuint& vao, GLuint& vbo, GLuint& ebo)
{
	float const verts[8] = {
		-1.f, -1.f,
		 1.f, -1.f,
		 1.f,  1.f,
		-1.f,  1.f,
	};
	unsigned int const idx[6] = { 0u, 1u, 2u, 0u, 2u, 3u };

	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glGenBuffers(1, &ebo);

	glBindVertexArray(vao);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

	glBindVertexArray(0);
}

struct DrawElementsIndirectCommand_t {
	GLuint count;
	GLuint instanceCount;
	GLuint firstIndex;
	GLuint baseVertex;
	GLuint baseInstance;
};

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
	create_ssbo(ssbo_points, pos4);
	create_ssbo(ssbo_colors, col4);
	create_ssbo(ssbo_covariances, splat_covariances);
	create_ssbo(ssbo_opacities, splat_opacities);

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


	// =======================================================================
	// Phase 1: GPU projection + visibility + indirect draw
	// =======================================================================
	{
		// Per-splat record: 3 vec4 + 1 uvec4 = 64 bytes. Match the layout in project.comp.glsl.
		size_t const record_bytes = 4u * sizeof(float) * 4u; // 4 vec4 (last is uvec4)
		glGenBuffers(1, &ssbo_view_data);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_view_data);
		glBufferData(GL_SHADER_STORAGE_BUFFER,
			GLsizeiptr(n) * GLsizeiptr(record_bytes),
			nullptr, GL_DYNAMIC_DRAW);

		glGenBuffers(1, &ssbo_visible_indices);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_visible_indices);
		glBufferData(GL_SHADER_STORAGE_BUFFER,
			GLsizeiptr(n) * GLsizeiptr(sizeof(GLuint)),
			nullptr, GL_DYNAMIC_DRAW);

		glGenBuffers(1, &ssbo_indirect_draw);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_indirect_draw);
		DrawElementsIndirectCommand_t const cmd0 = { 6u, 0u, 0u, 0u, 0u };
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(cmd0), &cmd0, GL_DYNAMIC_DRAW);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		// Bind to compute-shader binding points.
		bind_ssbo(ssbo_view_data,       5);
		bind_ssbo(ssbo_visible_indices, 6);
		bind_ssbo(ssbo_indirect_draw,   8);

		// Programs.
		program_project = load_compute_program(project::path + "shaders/instancing/project.comp.glsl");
		program_splat   = load_graphics_program(
			project::path + "shaders/instancing/splat.vert.glsl",
			project::path + "shaders/instancing/splat.frag.glsl");
		assert_cgp(program_project != 0, "Failed to load project.comp.glsl");
		assert_cgp(program_splat   != 0, "Failed to load splat.vert.glsl/splat.frag.glsl");

		// Phase 2 radix sort programs.
		program_radix_histogram = load_compute_program(project::path + "shaders/instancing/radix_histogram.comp.glsl");
		program_radix_block_prefix = load_compute_program(project::path + "shaders/instancing/radix_block_prefix.comp.glsl");
		program_radix_bin_prefix = load_compute_program(project::path + "shaders/instancing/radix_bin_prefix.comp.glsl");
		program_radix_scatter = load_compute_program(project::path + "shaders/instancing/radix_scatter.comp.glsl");
		assert_cgp(program_radix_histogram != 0, "Failed to load radix_histogram.comp.glsl");
		assert_cgp(program_radix_block_prefix != 0, "Failed to load radix_block_prefix.comp.glsl");
		assert_cgp(program_radix_bin_prefix != 0, "Failed to load radix_bin_prefix.comp.glsl");
		assert_cgp(program_radix_scatter != 0, "Failed to load radix_scatter.comp.glsl");

		// Phase 2 radix sort buffers.
		radix_block_count_max = (GLuint(n) + radix_block_size - 1u) / radix_block_size;
		glGenBuffers(1, &ssbo_sort_a);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_sort_a);
		glBufferData(GL_SHADER_STORAGE_BUFFER, GLsizeiptr(n) * GLsizeiptr(sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
		glGenBuffers(1, &ssbo_sort_b);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_sort_b);
		glBufferData(GL_SHADER_STORAGE_BUFFER, GLsizeiptr(n) * GLsizeiptr(sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
		glGenBuffers(1, &ssbo_radix_hist);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_radix_hist);
		glBufferData(GL_SHADER_STORAGE_BUFFER, GLsizeiptr(radix_block_count_max) * GLsizeiptr(256u) * GLsizeiptr(sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
		glGenBuffers(1, &ssbo_radix_bins);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_radix_bins);
		glBufferData(GL_SHADER_STORAGE_BUFFER, GLsizeiptr(256u) * GLsizeiptr(sizeof(GLuint)), nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		// Quad geometry for the GPU draw path.
		create_unit_quad(quad_vao, quad_vbo, quad_ebo);

		// Profiler.
		profiler.initialize();
		profiler.splat_count = n;

		// Drain any leftover GL error from setup so the per-frame opengl_check macros
		// inside cgp::draw don't fire on a stale error from init.
		while (glGetError() != GL_NO_ERROR) {}
	}


	std::cout << "End function scene_structure::initialize()" << std::endl;

}




// This function is called permanently at every new frame
// Note that you should avoid having costly computation and large allocation defined there. This function is mostly used to call the draw() functions on pre-existing data.
void scene_structure::display_frame()
{
	// Drain stale GL errors at frame start so localized check_gl() calls and CGP's
	// opengl_check macro inside cgp::draw report only this frame's errors.
	while (glGetError() != GL_NO_ERROR) {}

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

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDepthMask(false);

	int const n = splat_points.size();
	if (n > 0) {
		if (use_gpu_pipeline) {
			// ---------------- Phase 1: GPU pipeline ----------------
			// Drain any prior pending error so localized checks below are meaningful.
			(void)glGetError();

			// 1) Reset the indirect-draw command (instanceCount==0 acts as the atomic counter).
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_indirect_draw);
			DrawElementsIndirectCommand_t const cmd0 = { 6u, 0u, 0u, 0u, 0u };
			glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(cmd0), &cmd0);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			check_gl("reset indirect cmd");

			// 2) Run the projection compute shader.
			profiler.gpu_project.begin();
			glUseProgram(program_project);                           check_gl("useProgram project");
			bind_ssbo(ssbo_points,          0);
			bind_ssbo(ssbo_colors,          1);
			bind_ssbo(ssbo_covariances,     2);
			bind_ssbo(ssbo_opacities,       3);
			bind_ssbo(ssbo_view_data,       5);
			bind_ssbo(ssbo_visible_indices, 6);
			bind_ssbo(ssbo_indirect_draw,   8);
			check_gl("bind ssbos");
			// CGP stores matrices row-major in memory, so transpose=GL_TRUE.
			glUniformMatrix4fv(glGetUniformLocation(program_project, "view"),       1, GL_TRUE, ptr(environment.camera_view));
			glUniformMatrix4fv(glGetUniformLocation(program_project, "projection"), 1, GL_TRUE, ptr(environment.camera_projection));
			glUniform1f (glGetUniformLocation(program_project, "alpha_cutoff"), gui.alpha_cutoff);
			glUniform1ui(glGetUniformLocation(program_project, "splat_count"), GLuint(n));
			check_gl("compute uniforms");

			GLuint const groups = (GLuint(n) + 255u) / 256u;
			glDispatchCompute(groups, 1u, 1u);                       check_gl("dispatchCompute");
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
			check_gl("memoryBarrier");
			profiler.gpu_project.end();

			// 2.5) Read back the visible count (instanceCount slot of the indirect cmd).
			DrawElementsIndirectCommand_t cmd_back{};
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_indirect_draw);
			glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(cmd_back), &cmd_back);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			profiler.visible_count = int(cmd_back.instanceCount);
			profiler.splat_count   = n;

			GLuint visible_count = cmd_back.instanceCount;
			GLuint sorted_buffer = ssbo_visible_indices;
			if (visible_count > 1u) {
				profiler.gpu_sort.begin();
				GLuint const block_count = (visible_count + radix_block_size - 1u) / radix_block_size;

				// Copy the visible list into the ping-pong input buffer.
				glBindBuffer(GL_COPY_READ_BUFFER, ssbo_visible_indices);
				glBindBuffer(GL_COPY_WRITE_BUFFER, ssbo_sort_a);
				glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
					GLsizeiptr(visible_count) * GLsizeiptr(sizeof(GLuint)));
				glBindBuffer(GL_COPY_READ_BUFFER, 0);
				glBindBuffer(GL_COPY_WRITE_BUFFER, 0);

				GLuint sort_in = ssbo_sort_a;
				GLuint sort_out = ssbo_sort_b;
				for (GLuint pass = 0; pass < 4u; ++pass) {
					GLuint const shift = pass * 8u;

					// Clear block histograms.
					GLuint const zero = 0u;
					glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_radix_hist);
					glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
					glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

					// Histogram per block.
					glUseProgram(program_radix_histogram);
					bind_ssbo(sort_in, 9);
					bind_ssbo(ssbo_view_data, 5);
					bind_ssbo(ssbo_radix_hist, 11);
					glUniform1ui(glGetUniformLocation(program_radix_histogram, "element_count"), visible_count);
					glUniform1ui(glGetUniformLocation(program_radix_histogram, "pass_shift"), shift);
					glUniform1ui(glGetUniformLocation(program_radix_histogram, "block_count"), block_count);
					glDispatchCompute(block_count, 1u, 1u);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

					// Prefix over blocks (writes block prefixes + per-bin totals).
					glUseProgram(program_radix_block_prefix);
					bind_ssbo(ssbo_radix_hist, 11);
					bind_ssbo(ssbo_radix_bins, 12);
					glUniform1ui(glGetUniformLocation(program_radix_block_prefix, "block_count"), block_count);
					glDispatchCompute(1u, 1u, 1u);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

					// Prefix over bins to get global base offsets.
					glUseProgram(program_radix_bin_prefix);
					bind_ssbo(ssbo_radix_bins, 12);
					glDispatchCompute(1u, 1u, 1u);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

					// Scatter into the output buffer (stable within each block).
					glUseProgram(program_radix_scatter);
					bind_ssbo(sort_in, 9);
					bind_ssbo(sort_out, 10);
					bind_ssbo(ssbo_view_data, 5);
					bind_ssbo(ssbo_radix_hist, 11);
					bind_ssbo(ssbo_radix_bins, 12);
					glUniform1ui(glGetUniformLocation(program_radix_scatter, "element_count"), visible_count);
					glUniform1ui(glGetUniformLocation(program_radix_scatter, "pass_shift"), shift);
					glUniform1ui(glGetUniformLocation(program_radix_scatter, "block_count"), block_count);
					glDispatchCompute(block_count, 1u, 1u);
					glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

					std::swap(sort_in, sort_out);
				}

				profiler.gpu_sort.end();
				sorted_buffer = sort_in;
			}

			// 3) Indirect draw of the quad, instanceCount = visible count.
			profiler.gpu_draw.begin();
			glUseProgram(program_splat);                              check_gl("useProgram splat");
			glUniform1f(glGetUniformLocation(program_splat, "alpha_cutoff"), gui.alpha_cutoff);
			bind_ssbo(ssbo_view_data,       5);
			bind_ssbo(sorted_buffer, 6);
			check_gl("splat uniforms+ssbos");

			glBindVertexArray(quad_vao);                              check_gl("bindVertexArray");
			glBindBuffer(GL_DRAW_INDIRECT_BUFFER, ssbo_indirect_draw); check_gl("bind DRAW_INDIRECT");
			glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, nullptr); check_gl("drawElementsIndirect");
			glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
			glBindVertexArray(0);
			glUseProgram(0);
			profiler.gpu_draw.end();
			check_gl("end gpu draw");

			check_gl("end gpu draw");
		}
		else {
			// ---------------- Legacy CPU path ----------------
			auto const t0 = std::chrono::high_resolution_clock::now();
			sortPoints(splat_indices, splat_points, camera_control.camera_model.position());
			auto const t1 = std::chrono::high_resolution_clock::now();
			glBindBuffer(GL_ARRAY_BUFFER, vbo_indices);
			glBufferSubData(GL_ARRAY_BUFFER, 0, GLsizeiptr(splat_indices.size()) * GLsizeiptr(sizeof(int)), ptr(splat_indices));
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			auto const t2 = std::chrono::high_resolution_clock::now();

			profiler.cpu_sort_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
			profiler.cpu_upload_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

			profiler.gpu_draw.begin();
			draw(quad1, environment, splat_indices.size());
			profiler.gpu_draw.end();

			profiler.visible_count = n;
			profiler.splat_count   = n;
		}
	}

	// Resolve double-buffered GPU timer queries (fetch the previous frame's results).
	profiler.resolve_all();

	glDepthMask(true);
	glDisable(GL_BLEND);
}


void scene_structure::display_gui()
{
	ImGui::Checkbox("Frame", &gui.display_frame);
	ImGui::SliderFloat("Alpha cutoff", &gui.alpha_cutoff, 1e-6f, 1.0f, "%.5f", ImGuiSliderFlags_Logarithmic);

	ImGui::Separator();
	ImGui::Checkbox("GPU pipeline (Phase 1)", &use_gpu_pipeline);
	ImGui::Checkbox("Show profiler", &gui.show_profiler);

	if (gui.show_profiler) {
		ImGui::Text("Splats:    %d", profiler.splat_count);
		ImGui::Text("Visible:   %d", profiler.visible_count);
		ImGui::Separator();
		if (use_gpu_pipeline) {
			ImGui::Text("GPU project:  %.3f ms", profiler.gpu_project.last_ms);
			ImGui::Text("GPU sort:     %.3f ms", profiler.gpu_sort.last_ms);
			ImGui::Text("GPU draw:     %.3f ms", profiler.gpu_draw.last_ms);
		} else {
			ImGui::Text("CPU sort:     %.3f ms", profiler.cpu_sort_ms);
			ImGui::Text("CPU upload:   %.3f ms", profiler.cpu_upload_ms);
			ImGui::Text("GPU draw:     %.3f ms", profiler.gpu_draw.last_ms);
		}
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