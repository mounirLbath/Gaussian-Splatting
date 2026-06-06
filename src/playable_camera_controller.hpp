#pragma once

#include "cgp/cgp.hpp"

// First-person camera with Escape toggling mouse capture (play mode default).
struct playable_camera_controller : cgp::camera_controller_first_person_euler {
	void action_mouse_move();
	void idle_frame();
	void action_keyboard_playable();
	void set_cursor_trapped(bool trapped);
	bool cursor_trapped() const { return is_cursor_trapped; }
	void enable_play_mode();

	static constexpr float mouse_sensitivity = 0.56f;

private:
	using cgp::camera_controller_first_person_euler::is_cursor_trapped;
};
