// ReinhardLocal.cpp (HDR)
// Copyright (c) 2014, Amir Chohan,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.

#include <string.h>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <omp.h>
#include <vector>

#include "ReinhardLocal.h"
#include "opencl/reinhardLocal.h"

using namespace hdr;

ReinhardLocal::ReinhardLocal(float _key, float _sat, float _epsilon, float _phi) : Filter() {
	m_name = "ReinhardLocal";
	key = _key;
	sat = _sat;
	epsilon = _epsilon;
	phi = _phi;
	num_mipmaps = 8;
}

bool ReinhardLocal::setupOpenCL(cl_context_properties context_prop[], const Params& params) {

	char flags[1024];
	sprintf(flags, "-cl-fast-relaxed-math -D NUM_CHANNELS=%d -D WIDTH=%d -D HEIGHT=%d -D NUM_MIPMAPS=%d -D KEY=%ff -D SAT=%ff -D EPSILON=%ff -D PHI=%ff -D BUGGY_CL_GL=%d",
				NUM_CHANNELS, img_size.x, img_size.y, num_mipmaps, key, sat, epsilon, phi, BUGGY_CL_GL);

	if (!initCL(context_prop, params, reinhardLocal_kernel, flags)) {
		return false;
	}

	cl_int err;

	/////////////////////////////////////////////////////////////////kernels

	//computes the log average luminance of the image
	kernels["computeLogAvgLum"] = clCreateKernel(m_program, "computeLogAvgLum", &err);
	CHECK_ERROR_OCL(err, "creating computeLogAvgLum kernel", return false);

	//computes the next mipmap level of the provided data
	kernels["channel_mipmap"] = clCreateKernel(m_program, "channel_mipmap", &err);
	CHECK_ERROR_OCL(err, "creating channel_mipmap kernel", return false);

	//this kernel computes log average luminance of the image
	kernels["finalReduc"] = clCreateKernel(m_program, "finalReduc", &err);
	CHECK_ERROR_OCL(err, "creating finalReduc kernel", return false);

	//computes the operation to be applied to each pixel and performs the actual tonemapping
	kernels["reinhardLocal"] = clCreateKernel(m_program, "reinhardLocal", &err);
	CHECK_ERROR_OCL(err, "creating reinhardLocal kernel", return false);

	/////////////////////////////////////////////////////////////////kernel sizes
	reportStatus("\nKernels:");

	kernel2DSizes("computeLogAvgLum");
	kernel2DSizes("channel_mipmap");
	kernel2DSizes("reinhardLocal");


	reportStatus("---------------------------------Kernel finalReduc:");

		int num_wg = (global_sizes["computeLogAvgLum"][0]*global_sizes["computeLogAvgLum"][1])
						/(local_sizes["computeLogAvgLum"][0]*local_sizes["computeLogAvgLum"][1]);
		reportStatus("Number of work groups in computeLogAvgLum: %lu", num_wg);
	
		size_t* local = (size_t*) calloc(2, sizeof(size_t));
		size_t* global = (size_t*) calloc(2, sizeof(size_t));
		local[0] = num_wg;	//workgroup size for normal kernels
		global[0] = num_wg;
	
		local_sizes["finalReduc"] = local;
		global_sizes["finalReduc"] = global;
		reportStatus("Kernel sizes: Local=%lu Global=%lu", local[0], global[0]);


	/////////////////////////////////////////////////////////////////allocating memory

	//initialising information regarding all mipmap levels
	m_width = (int*) calloc(num_mipmaps, sizeof(int));
	m_height = (int*) calloc(num_mipmaps, sizeof(int));
	m_offset = (int*) calloc(num_mipmaps, sizeof(int));

	m_offset[0] = 0;
	m_width[0]  = img_size.x;
	m_height[0] = img_size.y;

	for (int level=1; level<num_mipmaps; level++) {
		m_width[level]  = m_width[level-1]/2;
		m_height[level] = m_height[level-1]/2;
		m_offset[level] = m_offset[level-1] + m_width[level-1]*m_height[level-1];
	}

	mems["lumMips"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(float)*img_size.x*img_size.y*2, NULL, &err);
	CHECK_ERROR_OCL(err, "creating logLum_Mip1 memory", return false);

	mems["m_width"] = clCreateBuffer(m_clContext, CL_MEM_COPY_HOST_PTR, sizeof(int)*num_mipmaps, m_width, &err);
	CHECK_ERROR_OCL(err, "creating m_width memory", return false);

	mems["m_height"] = clCreateBuffer(m_clContext, CL_MEM_COPY_HOST_PTR, sizeof(int)*num_mipmaps, m_height, &err);
	CHECK_ERROR_OCL(err, "creating m_height memory", return false);

	mems["m_offset"] = clCreateBuffer(m_clContext, CL_MEM_COPY_HOST_PTR, sizeof(int)*num_mipmaps, m_offset, &err);
	CHECK_ERROR_OCL(err, "creating m_offset memory", return false);

	mems["logAvgLum"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(float)*num_wg, NULL, &err);
	CHECK_ERROR_OCL(err, "creating logAvgLum memory", return false);

	if (params.opengl) {
		mem_images[0] = clCreateFromGLTexture2D(m_clContext, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, in_tex, &err);
		CHECK_ERROR_OCL(err, "creating gl input texture", return false);
		
		mem_images[1] = clCreateFromGLTexture2D(m_clContext, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, out_tex, &err);
		CHECK_ERROR_OCL(err, "creating gl output texture", return false);
	}
	else {
		cl_image_format format;
		format.image_channel_order = CL_RGBA;
		format.image_channel_data_type = CL_UNSIGNED_INT8;
		mem_images[0] = clCreateImage2D(m_clContext, CL_MEM_READ_ONLY, &format, img_size.x, img_size.y, 0, NULL, &err);
		CHECK_ERROR_OCL(err, "creating input image memory", return false);
	
		mem_images[1] = clCreateImage2D(m_clContext, CL_MEM_WRITE_ONLY, &format, img_size.x, img_size.y, 0, NULL, &err);
		CHECK_ERROR_OCL(err, "creating output image memory", return false);
	}

	/////////////////////////////////////////////////////////////////setting kernel arguements

	err = clSetKernelArg(kernels["computeLogAvgLum"], 0, sizeof(cl_mem), &mem_images[0]);
	err = clSetKernelArg(kernels["computeLogAvgLum"], 1, sizeof(cl_mem), &mems["lumMips"]);
	err = clSetKernelArg(kernels["computeLogAvgLum"], 2, sizeof(cl_mem), &mems["logAvgLum"]);
	err = clSetKernelArg(kernels["computeLogAvgLum"], 3,
		sizeof(cl_float)*local_sizes["computeLogAvgLum"][0]*local_sizes["computeLogAvgLum"][1], NULL);
	CHECK_ERROR_OCL(err, "setting computeLogAvgLum arguments", return false);

	err = clSetKernelArg(kernels["channel_mipmap"], 0, sizeof(cl_mem), &mems["lumMips"]);
	CHECK_ERROR_OCL(err, "setting channel_mipmap arguments", return false);

	err = clSetKernelArg(kernels["finalReduc"], 0, sizeof(cl_mem), &mems["logAvgLum"]);
	err = clSetKernelArg(kernels["finalReduc"], 1, sizeof(cl_uint), &num_wg);
	CHECK_ERROR_OCL(err, "setting finalReduc arguments", return false);

	err = clSetKernelArg(kernels["reinhardLocal"], 0, sizeof(cl_mem), &mem_images[0]);
	err = clSetKernelArg(kernels["reinhardLocal"], 1, sizeof(cl_mem), &mem_images[1]);
	err = clSetKernelArg(kernels["reinhardLocal"], 2, sizeof(cl_mem), &mems["lumMips"]);
	err = clSetKernelArg(kernels["reinhardLocal"], 3, sizeof(cl_mem), &mems["m_width"]);
	err = clSetKernelArg(kernels["reinhardLocal"], 4, sizeof(cl_mem), &mems["m_offset"]);
	err = clSetKernelArg(kernels["reinhardLocal"], 5, sizeof(cl_mem), &mems["logAvgLum"]);
	CHECK_ERROR_OCL(err, "setting reinhardLocal arguments", return false);

	reportStatus("\n\n");

	return true;
}

double ReinhardLocal::runCLKernels(std::vector<bool> whichKernelsToRun, bool recomputeMapping) {
	double start, end;

	if (!whichKernelsToRun[0]) {
		// if whichKernelsToRun[0] is false, -kernels was provided;
		// hence, should warn about invalid indices
		for (unsigned kernelIdx = 5; kernelIdx < whichKernelsToRun.size(); ++kernelIdx) {
			if (whichKernelsToRun[kernelIdx]) {
				reportStatus("Warning: no kernel with index %d", kernelIdx);
			}
		}
	}

	start = omp_get_wtime();

	cl_int err = CL_SUCCESS;
	cl_event event = 0;

	if (recomputeMapping) {
		if (whichKernelsToRun[1]) {
			err = clEnqueueNDRangeKernel(m_queue, kernels["computeLogAvgLum"], 2, NULL,
				global_sizes["computeLogAvgLum"], local_sizes["computeLogAvgLum"], 0, NULL, &event);
			CHECK_ERROR_OCL(err, "enqueuing computeLogAvgLum kernel", return false);
			CHECK_PROFILING_OCL(event, "profiling kernel 1: computeLogAvgLum");
		}

		if (whichKernelsToRun[2]) {
			err = clEnqueueNDRangeKernel(m_queue, kernels["finalReduc"], 1, NULL,
				&global_sizes["finalReduc"][0], &local_sizes["finalReduc"][0], 0, NULL, &event);
			CHECK_ERROR_OCL(err, "enqueuing finalReduc kernel", return false);
			CHECK_PROFILING_OCL(event, "profiling kernel 2: finalReduc");
		}
	
		if (whichKernelsToRun[3]) {
			//creating mipmaps
			for (int level=1; level<num_mipmaps; level++) {
				err |= clSetKernelArg(kernels["channel_mipmap"], 1, sizeof(int), &m_width[level-1]);
				err |= clSetKernelArg(kernels["channel_mipmap"], 2, sizeof(int), &m_offset[level-1]);
				err |= clSetKernelArg(kernels["channel_mipmap"], 3, sizeof(int), &m_width[level]);
				err |= clSetKernelArg(kernels["channel_mipmap"], 4, sizeof(int), &m_height[level]);
				err |= clSetKernelArg(kernels["channel_mipmap"], 5, sizeof(int), &m_offset[level]);
				CHECK_ERROR_OCL(err, "setting channel_mipmap arguments", return false);
	
				err = clEnqueueNDRangeKernel(m_queue, kernels["channel_mipmap"], 2, NULL,
					global_sizes["channel_mipmap"], local_sizes["channel_mipmap"], 0, NULL, &event);
				CHECK_ERROR_OCL(err, "enqueuing channel_mipmap kernel", return false);
				CHECK_PROFILING_OCL(event, "profiling kernel 3: channel_mipmap");
			}
		}

		if (whichKernelsToRun[4]) {
			err = clEnqueueNDRangeKernel(m_queue, kernels["reinhardLocal"], 2, NULL,
				global_sizes["reinhardLocal"], local_sizes["reinhardLocal"], 0, NULL, &event);
			CHECK_ERROR_OCL(err, "enqueuing reinhardLocal kernel", return false);
			CHECK_PROFILING_OCL(event, "profiling kernel 4: reinhardLocal");
		}
	}

	err = clFinish(m_queue);
	CHECK_ERROR_OCL(err, "running kernels", return false);

	end = omp_get_wtime();

	return (end - start);
}

bool ReinhardLocal::cleanupOpenCL() {
	clReleaseMemObject(mem_images[0]);
	clReleaseMemObject(mem_images[1]);
	clReleaseMemObject(mems["lumMips"]);
	clReleaseMemObject(mems["m_width"]);
	clReleaseMemObject(mems["m_height"]);
	clReleaseMemObject(mems["m_offset"]);
	clReleaseMemObject(mems["logAvgLum"]);
	clReleaseKernel(kernels["computeLogAvgLum"]);
	clReleaseKernel(kernels["channel_mipmap"]);
	clReleaseKernel(kernels["finalReduc"]);
	clReleaseKernel(kernels["reinhardLocal"]);
	clReleaseKernel(kernels["tonemap"]);
	releaseCL();
	return true;
}


bool ReinhardLocal::runReference(uchar* input, uchar* output) {

	// Check for cached result
	if (m_reference.data) {
		memcpy(output, m_reference.data, img_size.x*img_size.y*NUM_CHANNELS);
		reportStatus("Finished reference (cached)");
		return true;
	}

	reportStatus("Running reference");


	float** mipmap_pyramid = (float**) calloc(num_mipmaps, sizeof(float*));	//the complete mipmap pyramid
	int2* mipmap_sizes = (int2*) calloc(num_mipmaps, sizeof(int2));	//width and height of each of the mipmap
	mipmap_sizes[0] = img_size;


	float logAvgLum = 0.f;
	float lum = 0.f;
	int2 pos;
	mipmap_pyramid[0] = (float*) calloc(img_size.x*img_size.y, sizeof(float));
	for (pos.y = 0; pos.y < img_size.y; pos.y++) {
		for (pos.x = 0; pos.x < img_size.x; pos.x++) {
			lum = getPixelLuminance(input, img_size, pos);
			mipmap_pyramid[0][pos.x + pos.y*img_size.x] = lum; 
			logAvgLum += log(lum + 0.000001);
		}
	}
	logAvgLum = exp(logAvgLum/((float)(img_size.x*img_size.y)));

	float factor = key/logAvgLum;

	float scale_sq[num_mipmaps-1];
	float k[num_mipmaps-1];	//product of multiple constants
	for (int i=1; i<num_mipmaps; i++) {
		mipmap_pyramid[i] = mipmap(mipmap_pyramid[i-1], mipmap_sizes[i-1]);
		mipmap_sizes[i] = (int2){mipmap_sizes[i-1].x/2, mipmap_sizes[i-1].y/2};
		k[i] = pow(2.f, phi)*key/pow(pow(2, i-1), 2);
	}

	int2 centre;
	int2 surround;
	for (pos.y = 0; pos.y < img_size.y; pos.y++) {
		for (pos.x = 0; pos.x < img_size.x; pos.x++) {

			float local_logAvgLum = 0.f;
			surround.x = pos.x;
			surround.y = pos.y;

			float v, centre_logAvgLum, surround_logAvgLum, cs_diff;
			for (int i=0; i<num_mipmaps-1; i++) {
				centre.x = surround.x;
				centre.y = surround.y;
				surround.x = centre.x/2;
				surround.y = centre.y/2;

				centre_logAvgLum = getValue(mipmap_pyramid[i], mipmap_sizes[i], centre)*factor;
				surround_logAvgLum = getValue(mipmap_pyramid[i+1], mipmap_sizes[i+1], surround)*factor;

				cs_diff = centre_logAvgLum - surround_logAvgLum;
				cs_diff = cs_diff >= 0 ? cs_diff : -cs_diff;

				v = cs_diff/(k[i] + centre_logAvgLum);

				if (v > epsilon) {
					local_logAvgLum = centre_logAvgLum;
					break;
				}
				else local_logAvgLum = surround_logAvgLum;
			}

			float3 rgb, xyz;
			rgb.x = getPixel(input, img_size, pos, 0);
			rgb.y = getPixel(input, img_size, pos, 1);
			rgb.z = getPixel(input, img_size, pos, 2);

			xyz = RGBtoXYZ(rgb);

			float Ld = xyz.y*factor/(1.f + local_logAvgLum);

			rgb.x = (pow(rgb.x/xyz.y, sat) * Ld)*PIXEL_RANGE;
			rgb.y = (pow(rgb.y/xyz.y, sat) * Ld)*PIXEL_RANGE;
			rgb.z = (pow(rgb.z/xyz.y, sat) * Ld)*PIXEL_RANGE;

			setPixel(output, img_size, pos, 0, rgb.x);
			setPixel(output, img_size, pos, 1, rgb.y);
			setPixel(output, img_size, pos, 2, rgb.z);
		}
	}



	reportStatus("Finished reference");

	// Cache result
	m_reference.width = img_size.x;
	m_reference.height = img_size.y;
	m_reference.data = new uchar[img_size.x*img_size.y*NUM_CHANNELS];
	memcpy(m_reference.data, output, img_size.x*img_size.y*NUM_CHANNELS);

	return true;
}
