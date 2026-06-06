#pragma once

#include "cgp/cgp.hpp"

#include <vector>

class btDiscreteDynamicsWorld;
class btCollisionShape;
class btRigidBody;

struct physics_world {
	~physics_world();
	void initialize(cgp::numarray<cgp::vec3> const& mesh_vertices, float mesh_scale = 1.0f);
	void set_object_count(int count);
	void reset_objects(float vertical_spacing, float horizontal_variance);
	void step(float dt);
	cgp::mat3 get_rotation(int index) const;
	cgp::vec3 get_position(int index) const;
	int object_count() const { return static_cast<int>(bodies.size()); }

private:
	btDiscreteDynamicsWorld* world = nullptr;
	btCollisionShape* ground_shape = nullptr;
	btCollisionShape* object_shape = nullptr;
	std::vector<btRigidBody*> bodies;
	cgp::vec3 mesh_center = {0, 0, 0};
	float mesh_half_height = 0.5f;
};
