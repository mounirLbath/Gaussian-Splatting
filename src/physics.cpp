#include "physics.hpp"

#include <btBulletDynamicsCommon.h>
#include <LinearMath/btVector3.h>

#include <algorithm>
#include <cmath>
#include <random>

namespace {

cgp::vec3 compute_center(cgp::numarray<cgp::vec3> const& vertices)
{
	cgp::vec3 sum = {0, 0, 0};
	if (vertices.size() == 0)
		return sum;
	for (auto const& v : vertices)
		sum = sum + v;
	return sum / float(vertices.size());
}

float compute_half_height(cgp::numarray<cgp::vec3> const& vertices, cgp::vec3 const& center)
{
	float max_z = 0.0f;
	for (auto const& v : vertices) {
		float dz = std::abs(v.z - center.z);
		max_z = std::max(max_z, dz);
	}
	return std::max(max_z, 0.05f);
}

btTransform to_bt_transform(cgp::vec3 const& position, cgp::mat3 const& rotation = cgp::mat3::build_identity())
{
	btTransform transform;
	transform.setIdentity();
	transform.setOrigin(btVector3(position.x, position.y, position.z));
	btMatrix3x3 basis(
		rotation(0, 0), rotation(0, 1), rotation(0, 2),
		rotation(1, 0), rotation(1, 1), rotation(1, 2),
		rotation(2, 0), rotation(2, 1), rotation(2, 2));
	transform.setBasis(basis);
	return transform;
}

cgp::mat3 from_bt_rotation(btMatrix3x3 const& basis)
{
	return {
		basis[0][0], basis[0][1], basis[0][2],
		basis[1][0], basis[1][1], basis[1][2],
		basis[2][0], basis[2][1], basis[2][2]
	};
}

} // namespace

physics_world::~physics_world()
{
	for (auto* body : bodies) {
		if (world && body) {
			world->removeRigidBody(body);
			delete body->getMotionState();
			delete body;
		}
	}
	bodies.clear();
	delete object_shape;
	delete ground_shape;
	delete world;
}

void physics_world::initialize(cgp::numarray<cgp::vec3> const& mesh_vertices, float mesh_scale)
{
	mesh_center = compute_center(mesh_vertices);
	mesh_half_height = compute_half_height(mesh_vertices, mesh_center) * mesh_scale;

	auto* collision_config = new btDefaultCollisionConfiguration();
	auto* dispatcher = new btCollisionDispatcher(collision_config);
	auto* broadphase = new btDbvtBroadphase();
	auto* solver = new btSequentialImpulseConstraintSolver();
	world = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, collision_config);
	world->setGravity(btVector3(0, 0, -9.81f));

	ground_shape = new btStaticPlaneShape(btVector3(0, 0, 1), 0.0f);
	btTransform ground_transform = to_bt_transform({0, 0, 0});
	auto* ground_body = new btRigidBody(0.0f, nullptr, ground_shape, btVector3(0, 0, 0));
	ground_body->setWorldTransform(ground_transform);
	ground_body->setFriction(0.8f);
	world->addRigidBody(ground_body);

	btConvexHullShape* hull = new btConvexHullShape();
	for (int i = 0; i < mesh_vertices.size(); ++i) {
		cgp::vec3 const local = (mesh_vertices[i] - mesh_center) * mesh_scale;
		hull->addPoint(btVector3(local.x, local.y, local.z), false);
	}
	hull->optimizeConvexHull();
	hull->initializePolyhedralFeatures();
	object_shape = hull;
}

void physics_world::set_object_count(int count)
{
	for (auto* body : bodies) {
		world->removeRigidBody(body);
		delete body->getMotionState();
		delete body;
	}
	bodies.clear();
	bodies.resize(std::max(count, 0), nullptr);

	btVector3 local_inertia(0, 0, 0);
	object_shape->calculateLocalInertia(1.0f, local_inertia);

	for (int i = 0; i < count; ++i) {
		btTransform start_transform = to_bt_transform({0, 0, 1.0f + float(i)});
		auto* motion_state = new btDefaultMotionState(start_transform);
		auto* body = new btRigidBody(1.0f, motion_state, object_shape, local_inertia);
		body->setFriction(0.7f);
		body->setRestitution(0.05f);
		world->addRigidBody(body);
		bodies[i] = body;
	}
}

void physics_world::reset_objects(float vertical_spacing, float horizontal_variance)
{
	std::mt19937 rng(42);
	std::normal_distribution<float> horizontal(0.0f, std::sqrt(std::max(horizontal_variance, 0.0f)));

	float const base_z = mesh_half_height + 0.05f;
	for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
		cgp::vec3 position = {
			horizontal(rng),
			horizontal(rng),
			base_z + float(i) * vertical_spacing
		};
		btTransform transform = to_bt_transform(position);
		bodies[i]->setWorldTransform(transform);
		bodies[i]->getMotionState()->setWorldTransform(transform);
		bodies[i]->setLinearVelocity(btVector3(0, 0, 0));
		bodies[i]->setAngularVelocity(btVector3(0, 0, 0));
		bodies[i]->activate(true);
	}
}

void physics_world::step(float dt)
{
	if (!world)
		return;
	int const max_substeps = 4;
	world->stepSimulation(dt, max_substeps);
}

cgp::mat3 physics_world::get_rotation(int index) const
{
	if (index < 0 || index >= static_cast<int>(bodies.size()))
		return cgp::mat3::build_identity();
	btTransform transform;
	bodies[index]->getMotionState()->getWorldTransform(transform);
	return from_bt_rotation(transform.getBasis());
}

cgp::vec3 physics_world::get_position(int index) const
{
	if (index < 0 || index >= static_cast<int>(bodies.size()))
		return {0, 0, 0};
	btTransform transform;
	bodies[index]->getMotionState()->getWorldTransform(transform);
	btVector3 const origin = transform.getOrigin();
	return {origin.x(), origin.y(), origin.z()};
}
