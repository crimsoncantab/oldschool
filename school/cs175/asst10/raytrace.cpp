#include <iostream>
#include <vector>
#include <cmath>
#include "raycast.h"
#include "parse.h"
#include "ppm.h"


static std::vector <Surface *> scene;								// the geometry of the scene
static std::vector <Light> light;									// the lights in the scene


static Vector3 cameraPosition;											// camera position
static int resolutionX;
static int resolutionY;
static double fov;
static int multiresSamples;


static double pixelSize;											// this is computed from FOV




static Ray computeScreenRay(const double x, const double y)			// (x,y) are between (0,0) and (resolutionX, resolutionY)
{
    const Vector3 pixelPosition((x - resolutionX/2) * pixelSize, (y - resolutionY/2) * pixelSize, -1);
    return Ray(cameraPosition, pixelPosition);
}



template <int base> static float haltonNumber(const int i)			// this function is basically a random-number generator (that generates samples that are well-distributed for sampling (Halton numbers))
{
	static const float coef = 1.0 / static_cast <float> (base);
	float inc = coef;
	float re = 0;
	unsigned int c = i;
	while (c)
	{
		const int d = c % base;
		c -= d;
		re += d * inc;
		inc *= coef;
		c /= base;
	}
	return re;
}





int main()
{
    // parse the input
    Parser parser("scene.txt");
    parser.parse(fov, resolutionX, resolutionY, multiresSamples, light, scene);

    cameraPosition = Vector3(0,0,0);								// we asume that the camera is at (0,0,0)
    pixelSize = 2. * std::sin(0.5 * fov * CS175_PI / 180.) / static_cast <double> (resolutionY);
    
    // we precompute the multiresolution sampling pattern for a pixel
    Vector3 * sampleLocation = new Vector3[multiresSamples];
    for (int i = 0; i < multiresSamples; ++i)
    {
        sampleLocation[i][0] = haltonNumber <17> (19836 + i);		// some pseudo-random number that is well-distributed for sampling
        sampleLocation[i][1] = haltonNumber <7> (1836 + i);			// some pseudo-random number that is well-distributed for sampling
    }

    std::cerr << "Rendering... ";
    packed_pixel_t * frameBuffer = new packed_pixel_t[resolutionX * resolutionY];
    for (int y = 0; y < resolutionY; ++y)
    {
        for (int x = 0; x < resolutionX; ++x)
        {
			if (x == 326 && y == 214) {
				std::cout<<"";
			}
            Vector3 color(0,0,0);
            for (int i = 0; i < multiresSamples; ++i)
            {
                color += rayTrace(light, scene, computeScreenRay(x + sampleLocation[i][0], resolutionY - 1 - y + sampleLocation[i][1]));
            }
            color *= (1. / static_cast <double> (multiresSamples));
            frameBuffer[x + y*resolutionX].r = static_cast <unsigned char> (std::min(255.9, std::max(0., 256.*color[0])));
            frameBuffer[x + y*resolutionX].g = static_cast <unsigned char> (std::min(255.9, std::max(0., 256.*color[1])));
            frameBuffer[x + y*resolutionX].b = static_cast <unsigned char> (std::min(255.9, std::max(0., 256.*color[2])));
        }
        std::cerr << ".";
    }
    std::cerr << "done.\n";
    
    ppmwrite("output_image.ppm", frameBuffer, resolutionX, resolutionY);
    
    for (std::size_t i = 0; i < scene.size(); ++i) delete scene[i];	// destroy all data
    delete [] frameBuffer;
    delete [] sampleLocation;
    return 0;
}

