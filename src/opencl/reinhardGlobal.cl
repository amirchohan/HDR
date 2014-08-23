// reinhardGlobal.cl (HDR)
// Copyright (c) 2014, Amir Chohan,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.

float3 GLtoCL(uint3 val);
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
			// lum = pixel.x * 0.2126f + pixel.y * 0.7152f + pixel.z * 0.0722f;
			lum = dot(GLtoCL(pixel.xyz), (float3)(0.2126f, 0.7152f, 0.0722f));

			Lwhite_acc = max(lum, Lwhite_acc);
			logAvgLum_acc += log(lum + 0.000001f);
		}
	}

	size_t lid = get_local_id(0) + get_local_id(1)*get_local_size(0);	//linearised local id
	Lwhite_loc[lid] = Lwhite_acc;
	logAvgLum_loc[lid] = logAvgLum_acc;
	barrier(CLK_LOCAL_MEM_FENCE);

	// Perform parallel reduction
	for(int offset = (get_local_size(0)*get_local_size(1))/2; offset > 0; offset = offset/2) {
		if (lid < offset) {
			Lwhite_loc[lid] = max(Lwhite_loc[lid+offset], Lwhite_loc[lid]);
			logAvgLum_loc[lid] += logAvgLum_loc[lid + offset];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (lid == 0) {
		size_t group_id = get_group_id(0) + get_group_id(1)*get_num_groups(0);
		Lwhite[group_id] = Lwhite_loc[0];
		logAvgLum[group_id] = logAvgLum_loc[0];
	}
}

//combines the results of computeLogAvgLum kernel
kernel void finalReduc(	__global float* logAvgLum_acc,
						__global float* Lwhite_acc,
						const uint num_reduc_bins) {
	if (get_global_id(0) == 0) {

		float Lwhite = 0.f;
		float logAvgLum = 0.f;
	
		for (uint i = 0; i < num_reduc_bins; i++) {
			Lwhite = max(Lwhite, Lwhite_acc[i]);
			logAvgLum += logAvgLum_acc[i];
		}
		Lwhite_acc[0] = Lwhite;
		logAvgLum_acc[0] = exp(logAvgLum/(convert_float(WIDTH*HEIGHT)));
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

			rgb = GLtoCL(pixel.xyz);
			xyz = RGBtoXYZ(rgb);

			float L  = ((KEY*1.f)/logAvgLum) * xyz.y;
			float Ld = (L * (1.f + L/(Lwhite * Lwhite))) / (1.f + L);

			pixel.xyz = convert_uint3((float3)255.f * \
				clamp((pow(rgb.xyz/xyz.y, (float3)SAT)*(float3)Ld), 0.f, 1.f));

			write_imageui(output_image, pos, pixel);
		}
	}
}


//convert an RGB pixel to XYZ format:
//	xyz.x = rgb.x*0.4124f + rgb.y*0.3576f + rgb.z*0.1805f;
//	xyz.y = rgb.x*0.2126f + rgb.y*0.7152f + rgb.z*0.0722f;
//	xyz.z = rgb.x*0.0193f + rgb.y*0.1192f + rgb.z*0.9505f;
float3 RGBtoXYZ(float3 rgb) {
	float3 xyz;
	xyz.x = dot(rgb, (float3)(0.4124f, 0.3576f, 0.1805f)); // will be optimised away
	xyz.y = dot(rgb, (float3)(0.2126f, 0.7152f, 0.0722f));
	xyz.z = dot(rgb, (float3)(0.0193f, 0.1192f, 0.9505f)); // will be optimised away
	return xyz;
}


//convert a single OpenGL texture pixel component to an OpenCL texture pixel component
float GLtoCL1(uint val) {
	float valf = convert_float(val);
#if 1 == BUGGY_CL_GL
	// a workaround for Snapdragon's Android OpenCL implementation
	if (val >= 14340) return round(0.1245790f*valf - 1658.44f);	//>=128
	if (val >= 13316) return round(0.0622869f*valf - 765.408f);	//>=64
	if (val >= 12292) return round(0.0311424f*valf - 350.800f);	//>=32
	if (val >= 11268) return round(0.0155702f*valf - 159.443f);	//>=16
	
	return round(
		pow(valf, 4.f) * 0.0000000000000125922f -
		pow(valf, 3.f) * 0.00000000026729f +
		pow(valf, 2.f) * 0.00000198135f -
		valf * 0.00496681f -
		0.0000808829f);
#else
	return valf;
#endif
}


//convert an OpenGL texture pixel to an OpenCL texture pixel
float3 GLtoCL(uint3 val) {
	return (float3)(GLtoCL1(val.x), GLtoCL1(val.y), GLtoCL1(val.z));
}
