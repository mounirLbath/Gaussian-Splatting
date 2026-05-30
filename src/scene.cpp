#include "scene.hpp"
#include "helper.hpp"

using namespace cgp;

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
	create_dynamic_ssbo(ssbo_visible_indices, n, sizeof(unsigned int));

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
	bind_ssbo(ssbo_visible_indices, 5);
	glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 6, ssbo_visible_counter);


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

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDepthMask(false);

	if (ssbo_visible_counter != 0) {
		unsigned int zero = 0;
		glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, ssbo_visible_counter);
		glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(unsigned int), &zero);
		glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
	}

	if (splat_points.size() > 0){
		
		// sort points for alpha blending then send the list of indices to the vbo
		sortPoints(splat_indices, splat_points, camera_control.camera_model.position());
		glBindBuffer(GL_ARRAY_BUFFER, vbo_indices);
		glBufferSubData(GL_ARRAY_BUFFER, 0, GLsizeiptr(splat_indices.size()) * GLsizeiptr(sizeof(int)), ptr(splat_indices));
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// display all the instances of quad
		draw(quad1, environment, splat_indices.size());
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