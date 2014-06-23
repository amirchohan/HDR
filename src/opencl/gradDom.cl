// gradDom.cl (HDR)
// Copyright (c) 2014, Amir Chohan,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.

float GL_to_CL(uint val);
float3 RGBtoXYZ(float3 rgb);

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

//this kernel computes logLum
kernel void computeLogLum( 	__read_only image2d_t image,
							__global float* logLum) {

	int2 pos;
	uint4 pixel;
	float lum;
	for (pos.y = get_global_id(1); pos.y < HEIGHT; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < WIDTH; pos.x += get_global_size(0)) {
			pixel = read_imageui(image, sampler, pos);
			lum = GL_to_CL(pixel.x)*0.2126
				+ GL_to_CL(pixel.y)*0.7152
				+ GL_to_CL(pixel.z)*0.0722;
			logLum[pos.x + pos.y*WIDTH] = log(lum + 0.000001);
		}
	}
}

//computes the next level mipmap
kernel void channel_mipmap(	__global float* mipmap,	//array containing all the mipmap levels
							const int prev_width,	//width of the previous mipmap
							const int prev_offset, 	//start point of the previous mipmap 
							const int m_width,		//width of the mipmap being generated
							const int m_height,		//height of the mipmap being generated
							const int m_offset) { 	//start point to store the current mipmap
	int2 pos;
	for (pos.y = get_global_id(1); pos.y < m_height; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < m_width; pos.x += get_global_size(0)) {
			int _x = 2*pos.x;
			int _y = 2*pos.y;
			mipmap[pos.x + pos.y*m_width + m_offset] = 	(mipmap[_x + _y*prev_width + prev_offset]
														+ mipmap[_x+1 + _y*prev_width + prev_offset]
														+ mipmap[_x + (_y+1)*prev_width + prev_offset]
														+ mipmap[(_x+1) + (_y+1)*prev_width + prev_offset])/4.f;
		}
	}
}

//computing gradient magnitude using central differences at level k
kernel void gradient_mag(	__global float* lum,		//array containing all the luminance mipmap levels
							__global float* gradient,	//array to store all the gradients at different levels
							const int g_width,			//width of the gradient being generated
							const int g_height,			//height of the gradient being generated
							const int offset,			//start point to store the current gradient
							const float divider) { 	
	//k_av_grad = 0.f;
	int x_west;
	int x_east;
	int y_north;
	int y_south;
	float x_grad;
	float y_grad;
	int2 pos;
	for (pos.y = get_global_id(1); pos.y < g_height; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < g_width; pos.x += get_global_size(0)) {
			x_west  = clamp(pos.x-1, 0, g_width-1);
			x_east  = clamp(pos.x+1, 0, g_width-1);
			y_north = clamp(pos.y-1, 0, g_height-1);
			y_south = clamp(pos.y+1, 0, g_height-1);

			x_grad = (lum[x_west + pos.y*g_width + offset]  - lum[x_east + pos.y*g_width + offset])/divider;
			y_grad = (lum[pos.x + y_south*g_width + offset] - lum[pos.x + y_north*g_width + offset])/divider;

			gradient[pos.x + pos.y*g_width + offset] = sqrt(pow(x_grad, 2.f) + pow(y_grad, 2.f));
		}
	}
}

//used to compute the average gradient for the specified mipmap level
kernel void partialReduc(	__global float* gradient,	//array containing all the luminance gradient mipmap levels
							__global float* gradient_partial_sum,
							__local float* gradient_loc,
							const int height,	//height of the given mipmap
							const int width,	//width of the given mipmap
							const int g_offset) {	///index denoting the start point of the mipmap gradient array

	float gradient_acc = 0.f;

	for (int gid = get_global_id(0); gid < height*width; gid += get_global_size(0)) {
		gradient_acc += gradient[g_offset + gid];
	}

	const int lid = get_local_id(0);	//local id in one dimension
	gradient_loc[lid] = gradient_acc;

	// Perform parallel reduction
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int offset = get_local_size(0)/2; offset > 0; offset = offset/2) {
		if (lid < offset) {
			gradient_loc[lid] += gradient_loc[lid + offset];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	const int group_id = get_group_id(0);
	if (lid == 0) {
		gradient_partial_sum[group_id] = gradient_loc[0];
	}
}

//used to compute the average gradient for the specified mipmap level
kernel void finalReduc(	__global float* gradient_partial_sum,
						__global float* alphas,	//array containg alpha for each mipmap level
						const int mipmap_level,
						const int width,	//width of the given mipmap
						const int height,	//height of the given mipmap
						const unsigned int num_reduc_bins) {
	if (get_global_id(0)==0) {

		float sum_grads = 0.f;
	
		for (int i=0; i<num_reduc_bins; i++) {
			sum_grads += gradient_partial_sum[i];
		}
		alphas[mipmap_level] = ADJUST_ALPHA*exp(sum_grads/((float)width*height));
	}
	else return;
}

//computes attenuation function of the coarsest level mipmap
kernel void coarsest_level_attenfunc(	__global float* gradient,	//array containing all the luminance gradient mipmap levels
										__global float* atten_func,	//arrray to store attenuation function for each mipmap
										__global float* k_alpha,	//array containing alpha for each mipmap
										const int width,	//width of the coarsest level mipmap
										const int height,	//height of the coarsest level mipmap
										const int offset) {	//index where the data about the coarsest level mipmap starts in gradient array and atten_func array

	for (int gid = get_global_id(0); gid < width*height; gid+= get_global_size(0) ) {
		atten_func[gid+offset] = (k_alpha[0]/gradient[gid+offset])*pow(gradient[gid+offset]/k_alpha[0], (float)BETA);
	}
}

//computes attenuation function of a given mipmap
kernel void atten_func(	__global float* gradient,	//array containing all the luminance gradient mipmap levels
						__global float* atten_func,	//arrray to store attenuation function for each mipmap
						__global float* k_alpha,	//array containing alpha for each mipmap
						const int width,	//width of the given mipmap
						const int height,	//height of the given mipmap
						const int offset,	//index where the data about the given mipmap level starts in gradient array and atten_func array
						const int c_width,	//width of the coarser mipmap
						const int c_height,	//height of the coarser mipmap
						const int c_offset,	//index where the data about the coarser mipmap level starts in gradient array and atten_func array
						const int level) {	//current mipmap level
	int2 pos;
	int2 c_pos;
	int2 neighbour;
	float k_xy_atten_func;
	float k_xy_scale_factor;
	for (pos.y = get_global_id(1); pos.y < height; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < width; pos.x += get_global_size(0)) {
			if (gradient[pos.x + pos.y*width + offset] != 0) {

				c_pos = pos/2;	//position in the coarser grid

				//neighbours need to be left or right dependent on where we are
				neighbour.x = (pos.x & 1) ? 1 : -1;
				neighbour.y = (pos.y & 1) ? 1 : -1;


				//this stops us from going out of bounds
				if ((c_pos.x + neighbour.x) < 0) neighbour.x = 0;
				if ((c_pos.y + neighbour.y) < 0) neighbour.y = 0;
				if ((c_pos.x + neighbour.x) >= c_width)  neighbour.x = 0;
				if ((c_pos.y + neighbour.y) >= c_height) neighbour.y = 0;
				if (c_pos.x == c_width)  c_pos.x -= 1;
				if (c_pos.y == c_height) c_pos.y -= 1;

				k_xy_atten_func = 9.0*atten_func[c_pos.x 				+ c_pos.y					*c_width	+ c_offset]
								+ 3.0*atten_func[c_pos.x+neighbour.x 	+ c_pos.y					*c_width	+ c_offset]
								+ 3.0*atten_func[c_pos.x 				+ (c_pos.y+neighbour.y)		*c_width	+ c_offset]
								+ 1.0*atten_func[c_pos.x+neighbour.x 	+ (c_pos.y+neighbour.y)		*c_width	+ c_offset];

				k_xy_scale_factor = (k_alpha[level]/gradient[pos.x + pos.y*width + offset])*pow(gradient[pos.x + pos.y*width + offset]/k_alpha[level], (float)BETA);
				atten_func[pos.x + pos.y*width + offset] = (1.f/16.f)*(k_xy_atten_func)*k_xy_scale_factor;
			}
			else atten_func[pos.x + pos.y*width + offset] = 0.f;
		}
	}
}

//finds gradients in x and y direction and attenuates them using the previously computed attenuation function
kernel void grad_atten(	__global float* atten_grad_x,	//array to store the attenuated gradient in x dimension
						__global float* atten_grad_y,	//array to store the attenuated gradeint in y dimension
						__global float* lum,			//original luminance of the image
						__global float* atten_func) {	//attenuation function
	int2 pos;
	float2 grad;
	for (pos.y = get_global_id(1); pos.y < HEIGHT; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < WIDTH; pos.x += get_global_size(0)) {	
			grad.x = (pos.x < WIDTH-1 ) ? (lum[pos.x+1 +  	 pos.y*WIDTH] - lum[pos.x + pos.y*WIDTH]) : 0;
			grad.y = (pos.y < HEIGHT-1) ? (lum[pos.x   + (pos.y+1)*WIDTH] - lum[pos.x + pos.y*WIDTH]) : 0;
			atten_grad_x[pos.x + pos.y*WIDTH] = grad.x*atten_func[pos.x + pos.y*WIDTH];
			atten_grad_y[pos.x + pos.y*WIDTH] = grad.y*atten_func[pos.x + pos.y*WIDTH];
		}
	}
}

//computes the divergence field of the attenuated gradients
kernel void divG(	__global float* atten_grad_x,	//attenuated gradient in x direction
					__global float* atten_grad_y,	//attenuated gradient in y direction
					__global float* div_grad) {		//array to store the divergence field of the gradients
	div_grad[0] = 0;
	int2 pos;
	for (pos.x = get_global_id(0) + 1; pos.x < WIDTH; pos.x += get_global_size(0)) {
		div_grad[pos.x] = atten_grad_x[pos.x] - atten_grad_x[pos.x-1];
	}
	for (pos.y = get_global_id(1) + 1; pos.y < HEIGHT; pos.y += get_global_size(1)) {
		div_grad[pos.y*WIDTH] = atten_grad_y[pos.y*WIDTH] - atten_grad_y[(pos.y-1)*WIDTH];
		for (pos.x = get_global_id(0)+1; pos.x < WIDTH; pos.x += get_global_size(0)) {
			div_grad[pos.x + pos.y*WIDTH] 	= (atten_grad_x[pos.x + pos.y*WIDTH] - atten_grad_x[(pos.x-1) + pos.y*WIDTH])
											+ (atten_grad_y[pos.x + pos.y*WIDTH] - atten_grad_y[pos.x + (pos.y-1)*WIDTH]);
		}
	}	
}


//convert an RGB pixel to XYZ format
float3 RGBtoXYZ(float3 rgb) {
	float3 xyz;
	xyz.x = rgb.x*0.4124 + rgb.y*0.3576 + rgb.z*0.1805;
	xyz.y = rgb.x*0.2126 + rgb.y*0.7152 + rgb.z*0.0722;
	xyz.z = rgb.x*0.0193 + rgb.y*0.1192 + rgb.z*0.9505;
	return xyz;
}

//a function to read an OpenGL texture pixel when using Snapdragon's Android OpenCL implementation
float GL_to_CL(uint val) {
	if (BUGGY_CL_GL) {
		if (val >= 14340) return round(0.1245790*val - 1658.44);	//>=128
		if (val >= 13316) return round(0.0622869*val - 765.408);	//>=64
		if (val >= 12292) return round(0.0311424*val - 350.800);	//>=32
		if (val >= 11268) return round(0.0155702*val - 159.443);	//>=16
	
		float v = (float) val;
		return round(0.0000000000000125922*pow(v,4.f) - 0.00000000026729*pow(v,3.f) + 0.00000198135*pow(v,2.f) - 0.00496681*v - 0.0000808829);
	}
	else return (float)val;
}