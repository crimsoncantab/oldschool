#ifndef PARSE_H
#define PARSE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "vec.h"
#include "raylight.h"


class Parser
{
	std::ifstream fileStream_;

private:
	Parser() {}

public:
	Parser(const char *filename)		{ fileStream_.open(filename); }
	~Parser()							{ close(); }

	void close(void)
	{
		if (fileStream_.is_open())
			fileStream_.close();
	}

	void parse(double& fovy, int& width, int& height, int& samples, 
			   std::vector<Light>& light, std::vector<Surface *>& scene)
	{
		while(!fileStream_.eof())
		{
			std::string tag;
			fileStream_ >> std::ws >> tag >> std::ws;
			if (tag == std::string("TRIANGLE"))
			{
				Vector3 data[6];
				std::string subtag;
				int exponent;
				double mirror;

				for (int i=0; i<6; i++)
					fileStream_ >> std::ws >> subtag >> std::ws >> 
						data[i][0] >> std::ws >> data[i][1] >> std::ws >> data[i][2] >> std::ws;
				fileStream_ >> std::ws >> subtag >> std::ws >> exponent >> std::ws;
				fileStream_ >> std::ws >> subtag >> std::ws >> mirror >> std::ws;
				scene.push_back(new Triangle(data[0], data[1], data[2], data[3], data[4], data[5], exponent, mirror));
			}
			else if (tag == std::string("PLANE"))
			{
				Vector3 data[5];
				std::string subtag;
				int exponent;
				double mirror;

				for (int i=0; i<5; i++) 
					fileStream_ >> std::ws >> subtag >> std::ws >> 
						data[i][0] >> std::ws >> data[i][1] >> std::ws >> data[i][2] >> std::ws;
				fileStream_ >> std::ws >> subtag >> std::ws >> exponent >> std::ws;
				fileStream_ >> std::ws >> subtag >> std::ws >> mirror >> std::ws;
				scene.push_back(new Plane(data[0], data[1], data[2], data[3], data[4], exponent, mirror));
			}
			else if (tag == std::string("SPHERE"))
			{
				Vector3 center;
				Vector3 data[3];
				std::string subtag;
				float radius;
				int exponent;
				double mirror;

				fileStream_ >> std::ws >> subtag >> std::ws >> 
						center[0] >> std::ws >> center[1] >> std::ws >> center[2] >> std::ws;
				fileStream_ >> std::ws >> subtag >> std::ws >> radius >> std::ws;
				for (int i=0; i<3; i++)
					fileStream_ >> std::ws >> subtag >> std::ws >> 
						data[i][0] >> std::ws >> data[i][1] >> std::ws >> data[i][2] >> std::ws;
				fileStream_ >> std::ws >> subtag >> std::ws >> exponent >> std::ws;
				fileStream_ >> std::ws >> subtag >> std::ws >> mirror >> std::ws;
				scene.push_back(new Sphere(center, radius, data[0], data[1], data[2], exponent, mirror));
			}
			else if (tag == std::string("LIGHT"))
			{
				Vector3 data[2];
				std::string subtag;

				for (int i=0; i<2; i++) 
					fileStream_ >> std::ws >> subtag >> std::ws >> 
						data[i][0] >> std::ws >> data[i][1] >> std::ws >> data[i][2] >> std::ws;
				light.push_back(Light(data[0], data[1]));
			}
			else if (tag == std::string("CAMERA"))
			{
				std::string subtag;
				fileStream_ >> std::ws >> subtag >> std::ws >> fovy >> std::ws;
				fileStream_ >> std::ws >> subtag >> std::ws >> width >> std::ws;
				fileStream_ >> std::ws >> subtag >> std::ws >> height >> std::ws;
				fileStream_ >> std::ws >> subtag >> std::ws >> samples >> std::ws;
			}
			else
			{
				// do nothing
			}
		}
	}
};

#endif