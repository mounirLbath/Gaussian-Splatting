#include "playable_camera_controller.hpp"

using namespace cgp;

void playable_camera_controller::set_cursor_trapped(bool trapped)
{
	is_cursor_trapped = trapped;
	if (window != nullptr) {
		glfwSetInputMode(window->glfw_window, GLFW_CURSOR,
			trapped ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
	}
}

void playable_camera_controller::enable_play_mode()
{
	set_cursor_trapped(true);
}

void playable_camera_controller::action_keyboard_playable()
{
	if (inputs == nullptr)
		return;

	if (inputs->keyboard.last_action.is_pressed(GLFW_KEY_C) && inputs->keyboard.shift)
		set_cursor_trapped(!is_cursor_trapped);

	if (inputs->keyboard.last_action.is_pressed(GLFW_KEY_ESCAPE))
		set_cursor_trapped(!is_cursor_trapped);
}

void playable_camera_controller::action_mouse_move()
{
	assert_cgp_no_msg(inputs != nullptr);
	assert_cgp_no_msg(window != nullptr);
	if (!is_active)
		return;

	vec2 const& p1 = inputs->mouse.position.current;
	vec2 const& p0 = inputs->mouse.position.previous;
	vec2 const dp = mouse_sensitivity * (p1 - p0);

	bool const event_valid = !inputs->mouse.on_gui;
	bool const click_left = inputs->mouse.click.left;
	bool const click_right = inputs->mouse.click.right;
	bool const ctrl = inputs->keyboard.ctrl;

	if (event_valid) {
		if (click_left || (is_cursor_trapped && !click_right)) {
			if (!ctrl)
				camera_model.manipulator_rotate_roll_pitch_yaw(0, dp.y, -dp.x);
			else
				camera_model.manipulator_twist_rotation_axis(dp.x);
		}
		else if (click_right)
			camera_model.manipulator_translate_front((p1 - p0).y);
	}
}

void playable_camera_controller::idle_frame()
{
	assert_cgp_no_msg(inputs != nullptr);
	if (!is_active)
		return;

	float const magnitude = 2.0f * inputs->time_interval;
	float move_forward = 0.0f;
	float move_right = 0.0f;

	if (inputs->keyboard.is_pressed(GLFW_KEY_W))
		move_forward += magnitude;
	if (inputs->keyboard.is_pressed(GLFW_KEY_S))
		move_forward -= magnitude;
	if (inputs->keyboard.is_pressed(GLFW_KEY_A))
		move_right += magnitude;
	if (inputs->keyboard.is_pressed(GLFW_KEY_D))
		move_right -= magnitude;

	if (!inputs->keyboard.ctrl) {
		if (inputs->keyboard.up)
			move_forward += magnitude;
		if (inputs->keyboard.down)
			move_forward -= magnitude;
	}
	if (inputs->keyboard.left)
		move_right += magnitude;
	if (inputs->keyboard.right)
		move_right -= magnitude;

	if (move_forward == 0.0f && move_right == 0.0f)
		return;

	vec3 const front_3d = camera_model.front();
	vec3 front_plane = { front_3d.x, front_3d.y, 0.0f };
	if (norm(front_plane) < 1e-6f)
		front_plane = {0.0f, 1.0f, 0.0f};
	else
		front_plane = normalize(front_plane);

	vec3 const right_plane = normalize(cross(vec3{0.0f, 0.0f, 1.0f}, front_plane));
	camera_model.position_camera += move_forward * front_plane + move_right * right_plane;
}
