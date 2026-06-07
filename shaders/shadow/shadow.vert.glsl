#version 330 core

layout (location = 0) in vec3 vertex_position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	vec4 position = model * vec4(vertex_position, 1.0);
	// Snap to a single plane height to avoid z-fighting with the table.
	position.z = 0.015;
	gl_Position = projection * view * position;
}
