#pragma once

#include "cgp/cgp.hpp"

#include <vector>

class btDiscreteDynamicsWorld;
class btCollisionShape;
class btRigidBody;

struct physics_ray_hit {
	bool hit = false;
	int body_index = -1;
	float distance = 0.0f;
	cgp::vec3 point = {0, 0, 0};
};

struct physics_world {
	~physics_world();
	void initialize(cgp::numarray<cgp::vec3> const& mesh_vertices, float mesh_scale = 1.0f);
	void set_object_count(int count);
	void reset_objects(float vertical_spacing, float horizontal_variance);
	void step(float dt);
	cgp::mat3 get_rotation(int index) const;
	cgp::vec3 get_position(int index) const;
	int object_count() const { return static_cast<int>(bodies.size()); }

	physics_ray_hit raycast_dynamic(cgp::vec3 origin, cgp::vec3 direction, float max_distance) const;
	bool can_move_camera(cgp::vec3 from, cgp::vec3 to, int exclude_body_index = -1) const;
	bool can_move_object(int body_index, cgp::vec3 from, cgp::vec3 to) const;
	void set_linear_velocity(int index, cgp::vec3 velocity);
	void set_angular_velocity(int index, cgp::vec3 velocity);
	void set_rotation(int index, cgp::mat3 rotation);

private:
	bool convex_sweep_blocked(btCollisionShape* shape, cgp::mat3 const& rotation,
		cgp::vec3 from, cgp::vec3 to, int exclude_body_index) const;

	btDiscreteDynamicsWorld* world = nullptr;
	btCollisionShape* ground_shape = nullptr;
	btCollisionShape* object_shape = nullptr;
	btCollisionShape* camera_shape = nullptr;
	std::vector<btRigidBody*> bodies;
	cgp::vec3 mesh_center = {0, 0, 0};
	float mesh_half_height = 0.5f;
};
