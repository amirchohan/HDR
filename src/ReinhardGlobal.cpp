// ReinhardGlobal.cpp (HDR)
// Copyright (c) 2014, Amir Chohan,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.

#include "ReinhardGlobal.h"
#include "opencl/reinhardGlobal.h"

using namespace hdr;

ReinhardGlobal::ReinhardGlobal(float _key, float _sat) : Filter() {
	m_name = "ReinhardGlobal";
	key = _key;
	sat = _sat;
}

bool ReinhardGlobal::setupOpenCL(cl_context_properties context_prop[], const Params& params) {

	char flags[1024];
	sprintf(flags, "-cl-fast-relaxed-math -D NUM_CHANNELS=%d -D WIDTH=%d -D HEIGHT=%d -D KEY=%f -D SAT=%f -D BUGGY_CL_GL=%d",
				NUM_CHANNELS, img_size.x, img_size.y, key, sat, BUGGY_CL_GL);

	if (!initCL(context_prop, params, reinhardGlobal_kernel, flags)) return false;

	cl_int err;

	/////////////////////////////////////////////////////////////////kernels

	//this kernel computes log average luminance of the image
	kernels["computeLogAvgLum"] = clCreateKernel(m_program, "computeLogAvgLum", &err);
	CHECK_ERROR_OCL(err, "creating computeLogAvgLum kernel", return false);

	//this kernel computes log average luminance of the image
	kernels["finalReduc"] = clCreateKernel(m_program, "finalReduc", &err);
	CHECK_ERROR_OCL(err, "creating finalReduc kernel", return false);

	//performs the reinhard global tone mapping operator
	kernels["reinhardGlobal"] = clCreateKernel(m_program, "reinhardGlobal", &err);
	CHECK_ERROR_OCL(err, "creating reinhardGlobal kernel", return false);


	/////////////////////////////////////////////////////////////////kernel sizes
	reportStatus("\nKernels:");

	kernel2DSizes("computeLogAvgLum");
	kernel2DSizes("reinhardGlobal");

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

	mems["logAvgLum"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(float)*num_wg, NULL, &err);
	CHECK_ERROR_OCL(err, "creating logAvgLum memory", return false);

	mems["Lwhite"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(float)*num_wg, NULL, &err);
	CHECK_ERROR_OCL(err, "creating Lwhite memory", return false);

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

	err  = clSetKernelArg(kernels["computeLogAvgLum"], 0, sizeof(cl_mem), &mem_images[0]);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 1, sizeof(cl_mem), &mems["logAvgLum"]);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 2, sizeof(cl_mem), &mems["Lwhite"]);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 3, sizeof(float*)*local_sizes["computeLogAvgLum"][0]*local_sizes["computeLogAvgLum"][1], NULL);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 4, sizeof(float*)*local_sizes["computeLogAvgLum"][0]*local_sizes["computeLogAvgLum"][1], NULL);
	CHECK_ERROR_OCL(err, "setting computeLogAvgLum arguments", return false);

	err  = clSetKernelArg(kernels["finalReduc"], 0, sizeof(cl_mem), &mems["logAvgLum"]);
	err  = clSetKernelArg(kernels["finalReduc"], 1, sizeof(cl_mem), &mems["Lwhite"]);
	err  = clSetKernelArg(kernels["finalReduc"], 2, sizeof(unsigned int), &num_wg);
	CHECK_ERROR_OCL(err, "setting finalReduc arguments", return false);

	err  = clSetKernelArg(kernels["reinhardGlobal"], 0, sizeof(cl_mem), &mem_images[0]);
	err  = clSetKernelArg(kernels["reinhardGlobal"], 1, sizeof(cl_mem), &mem_images[1]);
	err  = clSetKernelArg(kernels["reinhardGlobal"], 2, sizeof(cl_mem), &mems["logAvgLum"]);
	err  = clSetKernelArg(kernels["reinhardGlobal"], 3, sizeof(cl_mem), &mems["Lwhite"]);
	CHECK_ERROR_OCL(err, "setting globalTMO arguments", return false);

	reportStatus("\n");

	return true;
}

double ReinhardGlobal::runCLKernels(bool recomputeMapping) {
	double start = omp_get_wtime();

	cl_int err;
	err = clEnqueueNDRangeKernel(m_queue, kernels["computeLogAvgLum"], 2, NULL, global_sizes["computeLogAvgLum"], local_sizes["computeLogAvgLum"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing computeLogAvgLum kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["finalReduc"], 1, NULL, &global_sizes["finalReduc"][0], &local_sizes["finalReduc"][0], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing finalReduc kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["reinhardGlobal"], 2, NULL, global_sizes["reinhardGlobal"], local_sizes["reinhardGlobal"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing transfer_data kernel", return false);

	err = clFinish(m_queue);
	CHECK_ERROR_OCL(err, "running kernels", return false);
	return omp_get_wtime() - start;	
}


bool ReinhardGlobal::cleanupOpenCL() {
	clReleaseMemObject(mem_images[0]);
	clReleaseMemObject(mem_images[1]);
	clReleaseMemObject(mems["Lwhite"]);
	clReleaseMemObject(mems["logAvgLum"]);
	clReleaseKernel(kernels["computeLogAvgLum"]);
	clReleaseKernel(kernels["finalReduc"]);
	clReleaseKernel(kernels["reinhardGlobal"]);
	releaseCL();
	return true;
}


bool ReinhardGlobal::runReference(uchar* input, uchar* output) {

	// Check for cached result
	if (m_reference.data) {
		memcpy(output, m_reference.data, img_size.x*img_size.y*NUM_CHANNELS);
		reportStatus("Finished reference (cached)");
		return true;
	}

	reportStatus("Running reference");

	float logAvgLum = 0.f;
	float Lwhite = 0.f;	//smallest luminance that'll be mapped to pure white

	int2 pos;
	for (pos.y = 0; pos.y < img_size.y; pos.y++) {
		for (pos.x = 0; pos.x < img_size.x; pos.x++) {
			float lum = getPixelLuminance(input, img_size, pos);
			logAvgLum += log(lum + 0.000001);

			if (lum > Lwhite) Lwhite = lum;
		}
	}
	logAvgLum = exp(logAvgLum/(img_size.x*img_size.y));

	//Global Tone-mapping operator
	for (pos.y = 0; pos.y < img_size.y; pos.y++) {
		for (pos.x = 0; pos.x < img_size.x; pos.x++) {
			float3 rgb, xyz;
			rgb.x = getPixel(input, img_size, pos, 0);
			rgb.y = getPixel(input, img_size, pos, 1);
			rgb.z = getPixel(input, img_size, pos, 2);

			xyz = RGBtoXYZ(rgb);

			float L  = (key/logAvgLum) * xyz.y;
			float Ld = (L * (1.f + L/(Lwhite * Lwhite)) )/(1.f + L);

			rgb.x = pow(rgb.x/xyz.y, sat) * Ld;
			rgb.y = pow(rgb.y/xyz.y, sat) * Ld;
			rgb.z = pow(rgb.z/xyz.y, sat) * Ld;

			setPixel(output, img_size, pos, 0, rgb.x*PIXEL_RANGE);
			setPixel(output, img_size, pos, 1, rgb.y*PIXEL_RANGE);
			setPixel(output, img_size, pos, 2, rgb.z*PIXEL_RANGE);
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