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
	GLuint compute_radix_program = 0;

	GLuint vbo_indices = 0;
	cgp::numarray<int> splat_indices;


	// Phase 1: GPU projection + visibility + indirect draw
	// ------------------------------------------------------------------ //
	// Per-splat record produced each frame by the projection compute shader.
	// Layout (must match shaders/instancing/project.comp.glsl):
	//   vec4 center_axis1;   // xy = NDC center, zw = ellipse axis 1 in NDC
	//   vec4 axis2_aabb;     // xy = ellipse axis 2 in NDC, zw = pixel-space AABB radius (rx, ry)
	//   vec4 color_opacity;  // rgb = color, a = opacity
	//   uint depth_key;      // float-flipped 32-bit sortable depth (back-to-front: larger key = farther)
	//   float view_depth;    // raw view-space -z (>0 in front of camera)
	//   uint visible;        // 0 or 1
	//   uint pad;
	GLuint ssbo_view_data = 0;        // 64 bytes per splat (3 vec4 + 1 uvec4)
	GLuint ssbo_visible_indices = 0;  // uint[] of visible splat indices, populated each frame
	GLuint ssbo_indirect_draw = 0;    // single DrawElementsIndirectCommand; instanceCount also acts as the atomic visible counter

	// GPU programs
	GLuint program_project = 0;       // compute shader
	GLuint program_splat = 0;         // graphics program (vert + frag) for the thin instancing path
	GLuint program_radix_histogram = 0;
	GLuint program_radix_block_prefix = 0;
	GLuint program_radix_bin_prefix = 0;
	GLuint program_radix_scatter = 0;

	// Phase 2: GPU radix sort buffers
	GLuint ssbo_sort_a = 0;           // ping-pong buffer A for visible indices
	GLuint ssbo_sort_b = 0;           // ping-pong buffer B for visible indices
	GLuint ssbo_radix_hist = 0;       // block histograms / block prefixes (block_count * 256)
	GLuint ssbo_radix_bins = 0;       // per-bin totals / base offsets (256)
	GLuint radix_block_count_max = 0; // max block count for allocation
	static constexpr GLuint radix_block_size = 1024u; // 256 threads * 4 items

	// Quad geometry for the splat draw (replaces the cgp mesh_drawable path)
	GLuint quad_vao = 0;
	GLuint quad_vbo = 0;
	GLuint quad_ebo = 0;

	// Toggles. Default OFF until Phase 2 lands the GPU sort -- the GPU path currently
	// draws splats in atomic-append order so blending is wrong on heavy alpha overlap.
	bool use_gpu_pipeline = true;    // true = GPU sort + indirect draw path

	// Profiler
	frame_profiler profiler;


	// ****************************** //
	// Callback functions
	// ****************************** //
	void mouse_move_event();
	void mouse_click_event();
	void keyboard_event();
	void idle_frame();
	void update_camera_animation(float dt);

};





