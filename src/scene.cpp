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

template <typename T>
void create_tbo(GLuint& buffer, GLuint& texture,
                numarray<T> const& data,
                GLenum internal_format)
{
	if (data.size() <= 0)
		return;
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_TEXTURE_BUFFER, buffer);
	glBufferData(
		GL_TEXTURE_BUFFER,
		GLsizeiptr(data.size()) * GLsizeiptr(sizeof(T)),
		ptr(data),
		GL_STATIC_DRAW);

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_BUFFER, texture);
	glTexBuffer(GL_TEXTURE_BUFFER, internal_format, buffer);

	glBindTexture(GL_TEXTURE_BUFFER, 0);
	glBindBuffer(GL_TEXTURE_BUFFER, 0);
}

void bind_tbo_to_shader(
    GLuint shader,
    GLuint texture,
    char const* uniform_name,
    int texture_unit)
{
    glUseProgram(shader);
    glActiveTexture(GL_TEXTURE0 + texture_unit);
    glBindTexture(GL_TEXTURE_BUFFER, texture);
    glUniform1i(glGetUniformLocation(shader, uniform_name), texture_unit);
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

	numarray<vec4> const pos4 = pad_vec3_to_vec4(splat_points);
	numarray<vec4> const col4 = pad_vec3_to_vec4(splat_colors);
	numarray<vec4> const scl4 = pad_vec3_to_vec4(splat_scales);
	create_tbo(tbo_points, tex_points, pos4, GL_RGBA32F);
	create_tbo(tbo_colors, tex_colors, col4, GL_RGBA32F);
	create_tbo(tbo_scales, tex_scales, scl4, GL_RGBA32F);
	create_tbo(tbo_rotations, tex_rotations, splat_rotations, GL_RGBA32F);
	create_tbo(tbo_opacities, tex_opacities, splat_opacities, GL_R32F);

	glGenBuffers(1, &vbo_indices);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_indices);
	glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(n) * GLsizeiptr(sizeof(int)), ptr(splat_indices), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	setup_instance_index_vao(quad1.vao, vbo_indices, /* location = */ 4);

	GLuint const shader = quad1.shader.id;
	bind_tbo_to_shader(shader, tex_points, "splat_points_tbo", 1);
	bind_tbo_to_shader(shader, tex_colors, "splat_colors_tbo", 2);
	bind_tbo_to_shader(shader, tex_scales, "splat_scales_tbo", 3);
	bind_tbo_to_shader(shader, tex_rotations, "splat_rotations_tbo", 4);
	bind_tbo_to_shader(shader, tex_opacities, "splat_opacities_tbo", 5);


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

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDepthMask(false);

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