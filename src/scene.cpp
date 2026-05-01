#include "scene.hpp"
#include "helper.hpp"


using namespace cgp;

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

	// Initialize the shapes of the scene
	// ***************************************** //

	gui.display_frame = true;

	float delta = 0.01;
	mesh quad_mesh = mesh_primitive_quadrangle(
		{ -delta,-delta,0 },
		{  delta,-delta,0 },
		{  delta, delta,0 },
		{ -delta, delta,0 }
	);

	read_points_from_ply_file("./assets/nike/scene.ply", splat_points, splat_colors, splat_scales, splat_rotations, splat_opacities, 0.1);

	quad1.initialize_data_on_gpu(quad_mesh);
	quad1.shader.load(project::path + "shaders/instancing/instancing.vert.glsl", project::path + "shaders/instancing/instancing.frag.glsl");
	
	quad1.initialize_supplementary_data_on_gpu(splat_points, /*location*/ 4, /*divisor: 1=per instance, 0=per vertex*/ 1);
	quad1.initialize_supplementary_data_on_gpu(splat_colors, 5, 1);
	quad1.initialize_supplementary_data_on_gpu(splat_scales, 6, 1);
	quad1.initialize_supplementary_data_on_gpu(splat_rotations, 7, 1);
	// quad1.initialize_supplementary_data_on_gpu(splat_opacities, 8, 1);

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

	// quad1.model.rotation = rotation_transform::from_axis_angle({0,0,1}, timer.t * 0.5f);

	draw(quad1, environment, splat_points.size());

	if (gui.display_wireframe) {
		//draw_wireframe(terrain, environment);
	}


// 	glEnable(GL_BLEND);
// 	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
// 	glDepthMask(false);


// draw(quad1, environment, splat_points.size());

// 	glDepthMask(true);
// 	glDisable(GL_BLEND);


}


void scene_structure::display_gui()
{
	ImGui::Checkbox("Frame", &gui.display_frame);
	ImGui::Checkbox("Wireframe", &gui.display_wireframe);
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
	std::cout << "Example of scene to start a project." << std::endl;
	std::cout << "-----------------------------------------------\n" << std::endl;
}