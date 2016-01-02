#ifndef ARCBALL_H
#define ARCBALL_H



static bool getScreenSpaceCircle(const Vector3& center, double radius,						// camera/eye coordinate info for sphere
                                 const Matrix4& projection,									// projection matrix
								 const double frust_near, const double frust_fovy,
                                 const int screen_width, const int screen_height, 			// viewport size
                                 Vector2 *out_center, double *out_radius)					// output data in screen coordinates
{																							// returns false if the arcball is behind the viewer
	// get post projection canonical coordinates
	Vector3 postproj = projection * center;
	double w = projection[3][0] * center[0] + projection[3][1] 
	  * center[1] + projection[3][2] * center[2] + projection[3][3] * 1.0;
	double winv = 0.0;
	if (w != 0.0) {
	  winv = 1.0 / w;
	}
	postproj *= winv;

	// convert to screen pixel space
	(*out_center)[0] = postproj[0] * (double)screen_width/2.0 + ((double)screen_width-1.0)/2.0;
	(*out_center)[1] = postproj[1] * (double)screen_height/2.0 + ((double)screen_height-1.0)/2.0;

	// determine some overall radius
	double dist = center[2];
	if (dist < frust_near) {
		*out_radius = -radius/(dist * tan(frust_fovy * CS175_PI/360.0));
	}
	else {
		*out_radius = 1.0;
		return false;
	}

	*out_radius *= screen_height * 0.5;
	return true;
}




#endif

