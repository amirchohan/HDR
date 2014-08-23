// reinhardLocal.cl (HDR)
// Copyright (c) 2014, Amir Chohan,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.


float3 GLtoCL(uint3 val);
float3 RGBtoXYZ(float3 rgb);

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

//this kernel computes logAvgLum by performing reduction
//the results are stored in an array of size num_work_groups
kernel void computeLogAvgLum( 	__read_only image2d_t image,
								__global float* lum,
								__global float* logAvgLum,
								__local float* logAvgLum_loc) {

	float lum0;
	float logAvgLum_acc = 0.f;

	int2 pos;
	uint4 pixel;
	for (pos.y = get_global_id(1); pos.y < HEIGHT; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < WIDTH; pos.x += get_global_size(0)) {
			pixel = read_imageui(image, sampler, pos);
			// lum0 = pixel.x * 0.2126f + pixel.y * 0.7152f + pixel.z * 0.0722f;
			lum0 = dot(GLtoCL(pixel.xyz), (float3)(0.2126f, 0.7152f, 0.0722f));

			logAvgLum_acc += log(lum0 + 0.000001f);
			lum[pos.x + pos.y*WIDTH] = lum0;
		}
	}

	pos.x = get_local_id(0);
	pos.y = get_local_id(1);
	const int lid = pos.x + pos.y*get_local_size(0);	//local id in one dimension
	logAvgLum_loc[lid] = logAvgLum_acc;

	// Perform parallel reduction
	barrier(CLK_LOCAL_MEM_FENCE);


	for(int offset = (get_local_size(0)*get_local_size(1))/2; offset > 0; offset = offset/2) {
		if (lid < offset) {
			logAvgLum_loc[lid] += logAvgLum_loc[lid + offset];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	const int num_work_groups = get_global_size(0)/get_local_size(0);	//number of workgroups in x dim
	const int group_id = get_group_id(0) + get_group_id(1)*num_work_groups;
	if (lid == 0) {
		logAvgLum[group_id] = logAvgLum_loc[0];
	}
}

//combines the results of computeLogAvgLum kernel
kernel void finalReduc(	__global float* logAvgLum_acc,
						const unsigned int num_reduc_bins) {
	if (get_global_id(0)==0) {

		float logAvgLum = 0.f;
		for (uint i=0; i < num_reduc_bins; i++) {
			logAvgLum += logAvgLum_acc[i];
		}
		logAvgLum_acc[0] = KEY / exp(logAvgLum/(convert_float(WIDTH*HEIGHT)));
	}
	else return;
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


//computes the mapping for each pixel as per Reinhard's Local TMO
kernel void reinhardLocal(	__global float* Ld_array,	//array to hold the mappings for each pixel. output of this kernel
							__global float* lumMips,	//contains the entire mipmap pyramid for the luminance of the image
							__global int* m_width,	//width of each of the mipmaps
							__global int* m_offset,	///set of indices denotaing the start point of each mipmap in lumMips array
							__global float* logAvgLum_acc) {

	float factor = logAvgLum_acc[0]; // originally (KEY / logAvgLum_acc[0])

#define PHI_IS_EIGHT_POINT_NOUGHT
#ifndef PHI_IS_EIGHT_POINT_NOUGHT
	float k[7];
    const float scale_sq[7] = \
		{ 1.f*1.f, 2.f*2.f, 4.f*4.f, 8.f*8.f, 16.f*16.f, 32.f*32.f, 64.f*64.f };
	for (int i=0; i<NUM_MIPMAPS-1; i++) {
		k[i] = pow(2.f, PHI) * KEY / scale_sq[i];
	}
#else
	constant float k[7] = {
		256.f * KEY / ( 1.f*1.f ),
		256.f * KEY / ( 2.f*2.f ),
		256.f * KEY / ( 4.f*4.f ),
		256.f * KEY / ( 8.f*8.f ),
		256.f * KEY / (16.f*16.f),
		256.f * KEY / (32.f*32.f),
		256.f * KEY / (64.f*64.f)
	};
#endif

	int2 pos, centre_pos, surround_pos;
	for (pos.y = get_global_id(1); pos.y < HEIGHT; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < WIDTH; pos.x += get_global_size(0)) {
			surround_pos = pos;
			float local_logAvgLum = 0.f;
			for (uint i = 0; i < NUM_MIPMAPS-1; i++) {
				centre_pos = surround_pos;
				surround_pos = centre_pos/2;

				int2 m_width_01, m_offset_01;
				m_width_01  = vload2(0, &m_width[i]);
				m_offset_01 = vload2(0, &m_offset[i]);

				int2 index_01 = m_offset_01 + (int2)(centre_pos.x, surround_pos.x);
				index_01 += m_width_01 * (int2)(centre_pos.y, surround_pos.y);

				float2 lumMips_01 = factor;
				lumMips_01 *= (float2)(lumMips[index_01.s0], lumMips[index_01.s1]);

				float centre_logAvgLum, surround_logAvgLum;
				centre_logAvgLum   = lumMips_01.s0;
				surround_logAvgLum = lumMips_01.s1;

				float cs_diff = fabs(centre_logAvgLum - surround_logAvgLum);
				if (cs_diff > (k[i] + centre_logAvgLum) * EPSILON) {
					local_logAvgLum = centre_logAvgLum;
					break;
				} else {
					local_logAvgLum = surround_logAvgLum;
				}
			}
			Ld_array[pos.x + pos.y*WIDTH] = factor / (1.f + local_logAvgLum);
		}
	}
}

//applies the previously computed mappings to image pixels
kernel void tonemap(__read_only image2d_t input_image,
					__write_only image2d_t output_image,
					__global float* Ld_array) {
	int2 pos;
	uint4 pixel;
	float3 rgb, xyz;
	for (pos.y = get_global_id(1); pos.y < HEIGHT; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < WIDTH; pos.x += get_global_size(0)) {
			pixel = read_imageui(input_image, sampler, pos);

			rgb = GLtoCL(pixel.xyz);
			xyz = RGBtoXYZ(rgb);

			float Ld  = Ld_array[pos.x + pos.y*WIDTH] * xyz.y;

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
