#pragma once


#include "cgp/cgp.hpp"

#include "environment.hpp"
#include "physics.hpp"
#include "playable_camera_controller.hpp"

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

	// First-person camera (ZQSD/WASD + mouse look)
	playable_camera_controller camera_control;

	// The model of camera projection (intrinsic parameters)
	camera_projection_perspective camera_projection;

	struct object_grab_state {
		bool active = false;
		int body_index = -1;
		float distance = 0.0f;
		mat3 locked_rotation = mat3::build_identity();
	};
	object_grab_state grab;
	float grab_pull_strength = 12.0f;
	float grab_max_speed = 8.0f;

	
	
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
	void reset_objects();
	void update_grabbed_object();
	bool validate_camera_move(vec3 const& from, vec3 const& to) const;
	void try_start_grab();
	void release_grab();
	void draw_crosshair() const;
	vec3 aim_direction() const;
	void fill_splats_from_physics();
	void update_splats_from_physics();
	void rebuild_splat_gpu_buffers();

};





