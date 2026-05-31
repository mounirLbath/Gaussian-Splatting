#pragma once


#include "cgp/cgp.hpp"

#include "environment.hpp"

using cgp::mesh_drawable;



struct gui_parameters {
	bool display_frame = false;
	float alpha_cutoff = 0.004f;
	int depth_bits = 32;
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

	
	
	// ****************************** //
	// Elements and shapes of the scene
	// ****************************** //

	mesh_drawable global_frame;          // The standard global frame

	timer_basic timer;

	mesh_drawable quad1;

	numarray<vec3> splat_points;
	numarray<vec3> splat_colors;
	numarray<float> splat_opacities;


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

	GLuint vbo_indices = 0;
	cgp::numarray<int> splat_indices;


	// ****************************** //
	// Callback functions
	// ****************************** //
	void mouse_move_event();
	void mouse_click_event();
	void keyboard_event();
	void idle_frame();

};





