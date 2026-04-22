
// CGP library
#include "cgp/cgp.hpp" 

// Running application
#include "application.hpp"

// Custom scene of this code
#include "scene.hpp"

#include "helper.hpp"


int main(int, char* argv[])
{

	read_points_from_ply_file("./assets/nike/scene.ply");
	// scene_structure scene;
	// application_structure app;
	// app.initialize(argv[0], &scene);
	
	// app.start_loop();

	return 0;
}

