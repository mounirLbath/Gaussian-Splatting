#version 330 core


layout(location=0) out vec4 FragColor;
uniform vec3 color; // Uniform color of the object


in vec3 instanceColor;

void main()
{
	FragColor = vec4(color + instanceColor, 1.0);
}
