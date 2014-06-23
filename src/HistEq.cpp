// HistEq.cpp (HDR)
// Copyright (c) 2014, Amir Chohan,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.

#include <string.h>
#include <cstdio>
#include <algorithm>
#include <omp.h>

#include "HistEq.h"
#include "opencl/histEq.h"

using namespace hdr;

HistEq::HistEq() : Filter() {
	m_name = "HistEq";
}

bool HistEq::setupOpenCL(cl_context_properties context_prop[], const Params& params) {
	char flags[1024];
	int hist_size = PIXEL_RANGE+1;

	sprintf(flags, "-cl-fast-relaxed-math -D PIXEL_RANGE=%d -D HIST_SIZE=%d -D NUM_CHANNELS=%d -D WIDTH=%d -D HEIGHT=%d -D BUGGY_CL_GL=%d",
			PIXEL_RANGE, hist_size, NUM_CHANNELS, img_size.x, img_size.y, BUGGY_CL_GL);

	if (!initCL(context_prop, params, histEq_kernel, flags)) return false;


	/////////////////////////////////////////////////////////////////kernels
	cl_int err;

	kernels["transfer_data"] = clCreateKernel(m_program, "transfer_data", &err);
	CHECK_ERROR_OCL(err, "creating transfer_data kernel", return false);

	//compute partial histogram
	kernels["partial_hist"] = clCreateKernel(m_program, "partial_hist", &err);
	CHECK_ERROR_OCL(err, "creating partial_hist kernel", return false);

	//merge partial histograms
	kernels["merge_hist"] = clCreateKernel(m_program, "merge_hist", &err);
	CHECK_ERROR_OCL(err, "creating merge_hist kernel", return false);

	//compute cdf of brightness histogram
	kernels["hist_cdf"] = clCreateKernel(m_program, "hist_cdf", &err);
	CHECK_ERROR_OCL(err, "creating hist_cdf kernel", return false);

	//perfrom histogram equalisation to the original image
	kernels["hist_eq"] = clCreateKernel(m_program, "histogram_equalisation", &err);
	CHECK_ERROR_OCL(err, "creating histogram_equalisation kernel", return false);


	/////////////////////////////////////////////////////////////////kernel sizes

	kernel2DSizes("transfer_data");
	kernel1DSizes("partial_hist");

	//number of workgroups in transfer_data kernel
	int num_wg = (global_sizes["partial_hist"][0])/(local_sizes["partial_hist"][0]);
	reportStatus("Number of work groups in partial_hist: %lu", num_wg);

	//merge_hist kernel size	
	reportStatus("---------------------------------Kernel merge_hist:");
	local_sizes["merge_hist"] = (size_t*) calloc(2, sizeof(size_t));
	global_sizes["merge_hist"] = (size_t*) calloc(2, sizeof(size_t));
	local_sizes["merge_hist"][0] = hist_size;
	global_sizes["merge_hist"][0] = hist_size;
	reportStatus("Kernel sizes: Local=%lu Global=%lu", global_sizes["merge_hist"][0], local_sizes["merge_hist"][0]);

	kernel1DSizes("hist_cdf");
	kernel2DSizes("hist_eq");

	/////////////////////////////////////////////////////////////////allocating memory

	mems["partial_hist"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(unsigned int)*hist_size*num_wg, NULL, &err);
	CHECK_ERROR_OCL(err, "creating histogram memory", return false);

	mems["merge_hist"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(unsigned int)*hist_size, NULL, &err);
	CHECK_ERROR_OCL(err, "creating merge_hist memory", return false);

	mems["image"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(float)*img_size.x*img_size.y*NUM_CHANNELS, NULL, &err);
	CHECK_ERROR_OCL(err, "creating image memory", return false);

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

	err  = clSetKernelArg(kernels["transfer_data"], 0, sizeof(cl_mem), &mem_images[0]);
	err |= clSetKernelArg(kernels["transfer_data"], 1, sizeof(cl_mem), &mems["image"]);
	CHECK_ERROR_OCL(err, "setting transfer_data arguments", return false);

	err  = clSetKernelArg(kernels["partial_hist"], 0, sizeof(cl_mem), &mems["image"]);
	err |= clSetKernelArg(kernels["partial_hist"], 1, sizeof(cl_mem), &mems["partial_hist"]);
	CHECK_ERROR_OCL(err, "setting partial_hist arguments", return false);

	err  = clSetKernelArg(kernels["merge_hist"], 0, sizeof(cl_mem), &mems["partial_hist"]);
	err |= clSetKernelArg(kernels["merge_hist"], 1, sizeof(cl_mem), &mems["merge_hist"]);
	err |= clSetKernelArg(kernels["merge_hist"], 2, sizeof(unsigned int), &num_wg);
	CHECK_ERROR_OCL(err, "setting merge_hist arguments", return false);

	err = clSetKernelArg(kernels["hist_cdf"], 0, sizeof(cl_mem), &mems["merge_hist"]);
	CHECK_ERROR_OCL(err, "setting hist_cdf arguments", return false);

	err  = clSetKernelArg(kernels["hist_eq"], 0, sizeof(cl_mem), &mems["image"]);
	err |= clSetKernelArg(kernels["hist_eq"], 1, sizeof(cl_mem), &mem_images[1]);
	err |= clSetKernelArg(kernels["hist_eq"], 2, sizeof(cl_mem), &mems["merge_hist"]);
	CHECK_ERROR_OCL(err, "setting histogram_equalisation arguments", return false);

	return true;
}

double HistEq::runCLKernels(bool recomputeMapping) {
	cl_int err;
	double start = omp_get_wtime();

	err = clEnqueueNDRangeKernel(m_queue, kernels["transfer_data"], 2, NULL, global_sizes["transfer_data"], local_sizes["transfer_data"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing transfer_data kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["partial_hist"], 1, NULL, &global_sizes["partial_hist"][0], &local_sizes["partial_hist"][0], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing partial_hist kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["merge_hist"], 1, NULL, &global_sizes["merge_hist"][0], &local_sizes["merge_hist"][0], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing merge_hist kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["hist_cdf"], 1, NULL, &global_sizes["hist_cdf"][0], &local_sizes["hist_cdf"][0], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing hist_cdf kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["hist_eq"], 2, NULL, global_sizes["hist_eq"], local_sizes["hist_eq"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing histogram_equalisation kernel", return false);

	err = clFinish(m_queue);
	CHECK_ERROR_OCL(err, "running kernels", return false);

	return omp_get_wtime() - start;
}

bool HistEq::cleanupOpenCL() {
	clReleaseMemObject(mem_images[0]);
	clReleaseMemObject(mem_images[1]);
	clReleaseMemObject(mems["image"]);
	clReleaseMemObject(mems["merge_hist"]);
	clReleaseMemObject(mems["partial_hist"]);
	clReleaseKernel(kernels["transfer_data"]);
	clReleaseKernel(kernels["partial_hist"]);
	clReleaseKernel(kernels["merge_hist"]);
	clReleaseKernel(kernels["hist_cdf"]);
	clReleaseKernel(kernels["hist_eq"]);
	releaseCL();
	return true;
}


bool HistEq::runReference(uchar* input, uchar* output) {
	// Check for cached result
	if (m_reference.data) {
		memcpy(output, m_reference.data, img_size.x*img_size.y*NUM_CHANNELS);
		reportStatus("Finished reference (cached)");
		return true;
	}

	const int hist_size = PIXEL_RANGE+1;
	unsigned int brightness_hist[hist_size] = {0};
	int brightness;
	float red, green, blue;

	reportStatus("Running reference");
	int2 pos;
	for (pos.y = 0; pos.y < img_size.y; pos.y++) {
		for (pos.x = 0; pos.x < img_size.x; pos.x++) {
			red   = getPixel(input, img_size, pos, 0);
			green = getPixel(input, img_size, pos, 1);
			blue  = getPixel(input, img_size, pos, 2);
			brightness = std::max(std::max(red, green), blue);
			brightness_hist[brightness] ++;
		}
	}

	for (int i = 1; i < hist_size; i++) {
		brightness_hist[i] += brightness_hist[i-1];
	}

	float3 rgb;
	float3 hsv;
	for (pos.y = 0; pos.y < img_size.y; pos.y++) {
		for (pos.x = 0; pos.x < img_size.x; pos.x++) {
			rgb.x = getPixel(input, img_size, pos, 0);
			rgb.y = getPixel(input, img_size, pos, 1);
			rgb.z = getPixel(input, img_size, pos, 2);
			hsv = RGBtoHSV(rgb);		//Convert to HSV to get Hue and Saturation

			hsv.z = ((hist_size-1)*(brightness_hist[(int)hsv.z] - brightness_hist[0]))
						/(img_size.x*img_size.y - brightness_hist[0]);

			rgb = HSVtoRGB(hsv);	//Convert back to RGB with the modified brightness for V
			setPixel(output, img_size, pos, 0, rgb.x);
			setPixel(output, img_size, pos, 1, rgb.y);
			setPixel(output, img_size, pos, 2, rgb.z);
		}
	}


	reportStatus("Finished reference");


	// Cache result
	m_reference.width = img_size.y;
	m_reference.height = img_size.x;
	m_reference.data = new uchar[img_size.x*img_size.y*NUM_CHANNELS];
	memcpy(m_reference.data, output, img_size.x*img_size.y*NUM_CHANNELS);

	return true;
}
