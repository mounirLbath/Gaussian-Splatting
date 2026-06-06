#include "physics.hpp"

#include "cgp/12_shape/intersection/intersection.hpp"

#include <btBulletCollisionCommon.h>
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
	delete camera_shape;
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

	camera_shape = new btSphereShape(0.2f);
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

namespace {

struct closest_dynamic_ray_callback : btCollisionWorld::RayResultCallback {
	int found_index = -1;
	std::vector<btRigidBody*> const* bodies = nullptr;

	explicit closest_dynamic_ray_callback(std::vector<btRigidBody*> const& body_list)
		: bodies(&body_list)
	{
		m_closestHitFraction = 1.0f;
	}

	btScalar addSingleResult(btCollisionWorld::LocalRayResult& rayResult, bool) override
	{
		auto* body = btRigidBody::upcast(rayResult.m_collisionObject);
		if (!body)
			return m_closestHitFraction;

		bool is_dynamic_object = false;
		for (int i = 0; i < static_cast<int>(bodies->size()); ++i) {
			if ((*bodies)[i] == body) {
				is_dynamic_object = true;
				if (rayResult.m_hitFraction < m_closestHitFraction) {
					m_closestHitFraction = rayResult.m_hitFraction;
					m_collisionObject = body;
					found_index = i;
				}
				break;
			}
		}
		if (!is_dynamic_object)
			return 1.0f;
		return m_closestHitFraction;
	}
};

struct sweep_callback : btCollisionWorld::ClosestConvexResultCallback {
	int exclude_index = -1;
	std::vector<btRigidBody*> const* bodies = nullptr;

	sweep_callback(btVector3 const& from, btVector3 const& to, int exclude, std::vector<btRigidBody*> const& body_list)
		: ClosestConvexResultCallback(from, to), exclude_index(exclude), bodies(&body_list) {}

	btScalar addSingleResult(btCollisionWorld::LocalConvexResult& convexResult, bool normalInWorldSpace) override
	{
		auto* body = btRigidBody::upcast(convexResult.m_hitCollisionObject);
		if (body && exclude_index >= 0) {
			for (int i = 0; i < static_cast<int>(bodies->size()); ++i) {
				if (i == exclude_index && (*bodies)[i] == body)
					return 1.0f;
			}
		}
		return ClosestConvexResultCallback::addSingleResult(convexResult, normalInWorldSpace);
	}
};

} // namespace

physics_ray_hit physics_world::raycast_dynamic(cgp::vec3 origin, cgp::vec3 direction, float max_distance) const
{
	physics_ray_hit result;
	if (!world || max_distance <= 0.0f || cgp::norm(direction) < 1e-6f)
		return result;

	cgp::vec3 const dir = cgp::normalize(direction);
	btVector3 const from(origin.x, origin.y, origin.z);
	btVector3 const to = from + btVector3(dir.x, dir.y, dir.z) * max_distance;

	closest_dynamic_ray_callback callback(bodies);
	world->rayTest(from, to, callback);

	if (callback.found_index >= 0) {
		result.hit = true;
		result.body_index = callback.found_index;
		result.distance = callback.m_closestHitFraction * max_distance;
		btVector3 p;
		p.setInterpolate3(from, to, callback.m_closestHitFraction);
		result.point = {p.x(), p.y(), p.z()};
		return result;
	}

	float const pick_radius = mesh_half_height * 2.5f;
	int best_index = -1;
	float best_distance = max_distance;
	for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
		cgp::vec3 const center = get_position(i);
		cgp::intersection_structure const inter = cgp::intersection_ray_sphere(origin, dir, center, pick_radius);
		if (!inter.valid)
			continue;
		float const distance = cgp::dot(inter.position - origin, dir);
		if (distance > 0.0f && distance < best_distance) {
			best_distance = distance;
			best_index = i;
			result.point = inter.position;
		}
	}

	if (best_index >= 0) {
		result.hit = true;
		result.body_index = best_index;
		result.distance = best_distance;
	}
	return result;
}

bool physics_world::convex_sweep_blocked(btCollisionShape* shape, cgp::mat3 const& rotation,
	cgp::vec3 from, cgp::vec3 to, int exclude_body_index) const
{
	if (!world || !shape)
		return false;

	btTransform const from_t = to_bt_transform(from, rotation);
	btTransform const to_t = to_bt_transform(to, rotation);
	btVector3 const from_bt(from.x, from.y, from.z);
	btVector3 const to_bt(to.x, to.y, to.z);

	auto const* convex_shape = dynamic_cast<btConvexShape const*>(shape);
	if (!convex_shape)
		return false;

	sweep_callback callback(from_bt, to_bt, exclude_body_index, bodies);
	world->convexSweepTest(convex_shape, from_t, to_t, callback);
	return callback.hasHit();
}

bool physics_world::can_move_camera(cgp::vec3 from, cgp::vec3 to, int exclude_body_index) const
{
	if (!camera_shape)
		return true;
	cgp::mat3 const identity = cgp::mat3::build_identity();
	return !convex_sweep_blocked(camera_shape, identity, from, to, exclude_body_index);
}

bool physics_world::can_move_object(int body_index, cgp::vec3 from, cgp::vec3 to) const
{
	if (!object_shape || body_index < 0 || body_index >= static_cast<int>(bodies.size()))
		return true;
	cgp::mat3 const rotation = get_rotation(body_index);
	return !convex_sweep_blocked(object_shape, rotation, from, to, body_index);
}

void physics_world::set_linear_velocity(int index, cgp::vec3 velocity)
{
	if (index < 0 || index >= static_cast<int>(bodies.size()))
		return;
	bodies[index]->setLinearVelocity(btVector3(velocity.x, velocity.y, velocity.z));
	bodies[index]->activate(true);
}

void physics_world::set_angular_velocity(int index, cgp::vec3 velocity)
{
	if (index < 0 || index >= static_cast<int>(bodies.size()))
		return;
	bodies[index]->setAngularVelocity(btVector3(velocity.x, velocity.y, velocity.z));
}

void physics_world::set_rotation(int index, cgp::mat3 rotation)
{
	if (index < 0 || index >= static_cast<int>(bodies.size()))
		return;
	cgp::vec3 const position = get_position(index);
	btTransform transform = to_bt_transform(position, rotation);
	bodies[index]->setWorldTransform(transform);
	bodies[index]->getMotionState()->setWorldTransform(transform);
	bodies[index]->setAngularVelocity(btVector3(0, 0, 0));
}
