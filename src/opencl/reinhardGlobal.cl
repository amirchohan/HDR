// reinhardGlobal.cl (HDR)
// Copyright (c) 2014, Amir Chohan,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.

float GL_to_CL(uint val);
float3 RGBtoXYZ(float3 rgb);

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

//this kernel computes logAvgLum and Lwhite by performing reduction
//the results are stored in an array of size num_work_groups
kernel void computeLogAvgLum( 	__read_only image2d_t image,
								__global float* logAvgLum,
								__global float* Lwhite,
								__local float* Lwhite_loc,
								__local float* logAvgLum_loc) {

	float lum;
	float Lwhite_acc = 0.f;		//maximum luminance in the image
	float logAvgLum_acc = 0.f;

	int2 pos;
	uint4 pixel;
	for (pos.y = get_global_id(1); pos.y < HEIGHT; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < WIDTH; pos.x += get_global_size(0)) {
			pixel = read_imageui(image, sampler, pos);
			lum = GL_to_CL(pixel.x)*0.2126
				+ GL_to_CL(pixel.y)*0.7152
				+ GL_to_CL(pixel.z)*0.0722;

			Lwhite_acc = (lum > Lwhite_acc) ? lum : Lwhite_acc;
			logAvgLum_acc += log(lum + 0.000001);
		}
	}

	pos.x = get_local_id(0);
	pos.y = get_local_id(1);
	const int lid = pos.x + pos.y*get_local_size(0);	//local id in one dimension
	Lwhite_loc[lid] = Lwhite_acc;
	logAvgLum_loc[lid] = logAvgLum_acc;

	// Perform parallel reduction
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int offset = (get_local_size(0)*get_local_size(1))/2; offset > 0; offset = offset/2) {
		if (lid < offset) {
			Lwhite_loc[lid] = (Lwhite_loc[lid+offset] > Lwhite_loc[lid]) ? Lwhite_loc[lid+offset] : Lwhite_loc[lid];
			logAvgLum_loc[lid] += logAvgLum_loc[lid + offset];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	const int num_work_groups = get_global_size(0)/get_local_size(0);	//number of workgroups in x dim
	const int group_id = get_group_id(0) + get_group_id(1)*num_work_groups;
	if (lid == 0) {
		Lwhite[group_id] = Lwhite_loc[0];
		logAvgLum[group_id] = logAvgLum_loc[0];
	}
}

//combines the results of computeLogAvgLum kernel
kernel void finalReduc(	__global float* logAvgLum_acc,
						__global float* Lwhite_acc,
						const unsigned int num_reduc_bins) {
	if (get_global_id(0)==0) {

		float Lwhite = 0.f;
		float logAvgLum = 0.f;
	
		for (int i=0; i<num_reduc_bins; i++) {
			if (Lwhite < Lwhite_acc[i]) Lwhite = Lwhite_acc[i];
			logAvgLum += logAvgLum_acc[i];
		}
		Lwhite_acc[0] = Lwhite;
		logAvgLum_acc[0] = exp(logAvgLum/((float)WIDTH*HEIGHT));
	}
	else return;
}

//Reinhard's Global Tone-Mapping Operator
kernel void reinhardGlobal(	__read_only image2d_t input_image,
							__write_only image2d_t output_image,
							__global float* logAvgLum_acc,
							__global float* Lwhite_acc) {
	float Lwhite = Lwhite_acc[0];
	float logAvgLum = logAvgLum_acc[0];

	int2 pos;
	uint4 pixel;
	float3 rgb, xyz;
	for (pos.y = get_global_id(1); pos.y < HEIGHT; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < WIDTH; pos.x += get_global_size(0)) {
			pixel = read_imageui(input_image, sampler, pos);

			rgb.x = GL_to_CL(pixel.x);
			rgb.y = GL_to_CL(pixel.y);
			rgb.z = GL_to_CL(pixel.z);

			xyz = RGBtoXYZ(rgb);

			float L  = ((KEY*1.f)/logAvgLum) * xyz.y;
			float Ld = (L * (1.f + L/(Lwhite * Lwhite)) )/(1.f + L);

			pixel.x = clamp((pow(rgb.x/xyz.y, (float)SAT)*Ld)*255.f, 0.f, 255.f);
			pixel.y = clamp((pow(rgb.y/xyz.y, (float)SAT)*Ld)*255.f, 0.f, 255.f);
			pixel.z = clamp((pow(rgb.z/xyz.y, (float)SAT)*Ld)*255.f, 0.f, 255.f);
			write_imageui(output_image, pos, pixel);
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