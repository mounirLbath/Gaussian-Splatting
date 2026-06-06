#pragma once


#include "cgp/cgp.hpp"

#include "environment.hpp"
#include "physics.hpp"

using cgp::mesh_drawable;



struct gui_parameters {
	bool display_frame = false;
	float alpha_cutoff = 0.004f;
	int depth_bits = 32;
	int num_objects = 5;
	float vertical_spacing = 0.5f;
	float horizontal_variance = 0.15f;
};



// The structure of the custom scene
struct scene_structure : cgp::scene_inputs_generic {
	
	// ****************************** //
	// Standard Functions
	// ****************************** //

	void initialize();  // Standard initialization to be called before the animation loop
	void display_frame();     // The frame display to be called within the animation loop
	void display_gui(); // The display of the GUI, also called within the animation loop

	// ****************************** //
	// Context
	// ****************************** //

	// Environment controller (background color, )
	environment_structure environment; 
	// Window where the scene is displayed
	window_structure window; 
	// Storage for inputs status (mouse, keyboard, window dimension)
	input_devices inputs; 
	// Standard GUI element storage
	gui_parameters gui; 

	// Display information at the start of the program
	void display_info();

	// ****************************** //
	// Camera controller
	// ****************************** //

	// Controller of the camera (extrinsic parameters: position/orientation) -- to be adapted to the desired model and behavior
	camera_controller_orbit_euler camera_control; 

	// The model of camera projection (intrinsic parameters)
	camera_projection_perspective camera_projection;

	// Camera animation state
	bool animation_mode = false;
	float animation_time = 0.0f;
	float animation_period = 20.0f;
	float animation_pitch_amplitude = 0.25f;
	float animation_zoom_amplitude = 0.2f;
	float animation_base_distance = 7.5f;
	vec3 animation_center = {0.0f, -0.15f, 0.0f};

	
	
	// ****************************** //
	// Elements and shapes of the scene
	// ****************************** //

	mesh_drawable global_frame;          // The standard global frame
	mesh_drawable table_plane;

	timer_basic timer;

	mesh_drawable quad1;

	physics_world physics;

	numarray<vec3> template_splat_points;
	numarray<vec3> template_splat_colors;
	numarray<vec4> template_splat_covariances;
	numarray<float> template_splat_opacities;
	int template_splat_count = 0;

	numarray<vec3> splat_points;
	numarray<vec3> splat_colors;
	numarray<float> splat_opacities;
	numarray<vec4> splat_covariances;


	// Per-splat data stored as SSBOs
	GLuint ssbo_points = 0;
	GLuint ssbo_colors = 0;
	GLuint ssbo_covariances = 0;
	GLuint ssbo_opacities = 0;
	GLuint ssbo_depth_keys = 0;
	GLuint ssbo_visible_counter = 0;
	GLuint ssbo_sort_ping = 0;
	GLuint ssbo_sort_pong = 0;
	GLuint ssbo_radix_hist = 0;
	GLuint ssbo_radix_prefix = 0;
	GLuint ssbo_indirect_draw = 0;
	GLuint compute_radix_program = 0;

	GLuint vbo_indices = 0;
	cgp::numarray<int> splat_indices;


	// ****************************** //
	// Callback functions
	// ****************************** //
	void mouse_move_event();
	void mouse_click_event();
	void keyboard_event();
	void idle_frame();
	void update_camera_animation(float dt);
	void reset_objects();
	void fill_splats_from_physics();
	void update_splats_from_physics();
	void rebuild_splat_gpu_buffers();

};





