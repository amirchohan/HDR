// GradDom.cpp (HDR)
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

#include "GradDom.h"
#include "opencl/gradDom.h"

using namespace hdr;

GradDom::GradDom(float _adjust_alpha, float _beta, float _sat) : Filter() {
	m_name = "GradDom";
	adjust_alpha = _adjust_alpha;
	beta = _beta;
	sat = _sat;
}

bool GradDom::setupOpenCL(cl_context_properties context_prop[], const Params& params) {

	//get the number of mipmaps needed for the image of this size
	num_mipmaps = 0;
	for (int x=img_size.x, y=img_size.y ; x >= 32 && y >= 32; y/=2, x/=2) num_mipmaps++;

	char flags[1024];
	sprintf(flags, "-cl-fast-relaxed-math -D WIDTH=%d -D HEIGHT=%d -D NUM_MIPMAPS=%d -D ADJUST_ALPHA=%f -D BETA=%f -D SAT=%f -D BUGGY_CL_GL=%d",
				img_size.x, img_size.y, num_mipmaps, adjust_alpha, beta, sat, BUGGY_CL_GL);

	if (!initCL(context_prop, params, gradDom_kernel, flags)) return false;

	cl_int err;

	/////////////////////////////////////////////////////////////////kernels

	//this kernel computes log average luminance of the image
	kernels["computeLogLum"] = clCreateKernel(m_program, "computeLogLum", &err);
	CHECK_ERROR_OCL(err, "creating computeLogLum kernel", return false);

	//this kernel computes all the mipmap levels of the luminance
	kernels["channel_mipmap"] = clCreateKernel(m_program, "channel_mipmap", &err);
	CHECK_ERROR_OCL(err, "creating channel_mipmap kernel", return false);

	//this kernel generates gradient magnitude at each of the mipmap levels
	kernels["gradient_mag"] = clCreateKernel(m_program, "gradient_mag", &err);
	CHECK_ERROR_OCL(err, "creating gradient_mag kernel", return false);

	//computes the partial reduction of a given array
	kernels["partialReduc"] = clCreateKernel(m_program, "partialReduc", &err);
	CHECK_ERROR_OCL(err, "creating partialReduc kernel", return false);

	//computes the final reduction of a given array
	kernels["finalReduc"] = clCreateKernel(m_program, "finalReduc", &err);
	CHECK_ERROR_OCL(err, "creating finalReduc kernel", return false);

	//computes attenuation function of the coarsest level mipmap
	kernels["coarsest_level_attenfunc"] = clCreateKernel(m_program, "coarsest_level_attenfunc", &err);
	CHECK_ERROR_OCL(err, "creating coarsest_level_attenfunc kernel", return false);

	//computes attenuation function of a given mipmap
	kernels["atten_func"] = clCreateKernel(m_program, "atten_func", &err);
	CHECK_ERROR_OCL(err, "creating atten_func kernel", return false);

	//finds gradients in x and y direction and attenuates them using the previously computed attenuation function
	kernels["grad_atten"] = clCreateKernel(m_program, "grad_atten", &err);
	CHECK_ERROR_OCL(err, "creating grad_atten kernel", return false);

	//computes the divergence field of the attenuated gradients
	kernels["divG"] = clCreateKernel(m_program, "divG", &err);
	CHECK_ERROR_OCL(err, "creating divG kernel", return false);

	/////////////////////////////////////////////////////////////////kernel sizes

	kernel2DSizes("computeLogLum");
	kernel2DSizes("channel_mipmap");
	kernel2DSizes("gradient_mag");
	kernel1DSizes("partialReduc");
	kernel1DSizes("coarsest_level_attenfunc");
	kernel2DSizes("atten_func");
	kernel2DSizes("grad_atten");
	kernel2DSizes("divG");

	reportStatus("---------------------------------Kernel finalReduc:");

		int num_wg = (global_sizes["partialReduc"][0])
						/(local_sizes["partialReduc"][0]);
		reportStatus("Number of work groups in partialReduc: %lu", num_wg);
	
		size_t* local = (size_t*) calloc(2, sizeof(size_t));
		size_t* global = (size_t*) calloc(2, sizeof(size_t));
		local[0] = num_wg;	//workgroup size for normal kernels
		global[0] = num_wg;
	
		local_sizes["finalReduc"] = local;
		global_sizes["finalReduc"] = global;
		reportStatus("Kernel sizes: Local=%lu Global=%lu", local[0], global[0]);

	/////////////////////////////////////////////////////////////////allocating memory

	//initialising information regarding all mipmap levels
	m_width  = (int*) calloc(num_mipmaps, sizeof(int));
	m_height = (int*) calloc(num_mipmaps, sizeof(int));
	m_offset = (int*) calloc(num_mipmaps, sizeof(int));
	m_divider = (float*) calloc(num_mipmaps, sizeof(float));

	m_offset[0] = 0;
	m_width[0]  = img_size.x;
	m_height[0] = img_size.y;
	m_divider[0] = 2;

	for (int level=1; level<num_mipmaps; level++) {
		m_width[level]  = m_width[level-1]/2;
		m_height[level] = m_height[level-1]/2;
		m_offset[level] = m_offset[level-1] + m_width[level-1]*m_height[level-1];
		m_divider[level] = pow(2, level+1);
	}

	//initialise memory objects
	//TODO: this can be further optmised by reusing the same memory again
	mems["logLum_Mips"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(cl_float)*img_size.x*img_size.y*2, NULL, &err);
	CHECK_ERROR_OCL(err, "creating logLum_Mips memory", return false);

	mems["gradient_Mips"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(cl_float)*img_size.x*img_size.y*2, NULL, &err);
	CHECK_ERROR_OCL(err, "creating gradient_Mips memory", return false);

	mems["attenfunc_Mips"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(cl_float)*img_size.x*img_size.y*2, NULL, &err);
	CHECK_ERROR_OCL(err, "creating attenfunc_Mips memory", return false);

	mems["gradient_PartialSum"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(cl_float)*num_wg, NULL, &err);
	CHECK_ERROR_OCL(err, "creating gradient_PartialSum memory", return false);

	mems["k_alphas"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(cl_float)*num_mipmaps, NULL, &err);
	CHECK_ERROR_OCL(err, "creating k_alphas memory", return false);

	mems["atten_grad_x"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(cl_float)*img_size.x*img_size.y, NULL, &err);
	CHECK_ERROR_OCL(err, "creating atten_grad_x memory", return false);

	mems["atten_grad_y"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(cl_float)*img_size.x*img_size.y, NULL, &err);
	CHECK_ERROR_OCL(err, "creating atten_grad_y memory", return false);

	mems["div_grad"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(cl_float)*img_size.x*img_size.y, NULL, &err);
	CHECK_ERROR_OCL(err, "creating div_grad memory", return false);

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

	err = clSetKernelArg(kernels["computeLogLum"], 0, sizeof(cl_mem), &mem_images[0]);
	err = clSetKernelArg(kernels["computeLogLum"], 1, sizeof(cl_mem), &mems["logLum_Mips"]);
	CHECK_ERROR_OCL(err, "setting computeLogLum arguments", return false);

	err = clSetKernelArg(kernels["channel_mipmap"], 0, sizeof(cl_mem), &mems["logLum_Mips"]);
	CHECK_ERROR_OCL(err, "setting channel_mipmap arguments", return false);

	err = clSetKernelArg(kernels["gradient_mag"], 0, sizeof(cl_mem), &mems["logLum_Mips"]);
	err = clSetKernelArg(kernels["gradient_mag"], 1, sizeof(cl_mem), &mems["gradient_Mips"]);
	CHECK_ERROR_OCL(err, "setting gradient_mag arguments", return false);

	err = clSetKernelArg(kernels["partialReduc"], 0, sizeof(cl_mem), &mems["gradient_Mips"]);
	err = clSetKernelArg(kernels["partialReduc"], 1, sizeof(cl_mem), &mems["gradient_PartialSum"]);
	err = clSetKernelArg(kernels["partialReduc"], 2, sizeof(cl_float)*local_sizes["partialReduc"][0], NULL);
	CHECK_ERROR_OCL(err, "setting partialReduc arguments", return false);

	err = clSetKernelArg(kernels["finalReduc"], 0, sizeof(cl_mem), &mems["gradient_PartialSum"]);
	err = clSetKernelArg(kernels["finalReduc"], 1, sizeof(cl_mem), &mems["k_alphas"]);
	err = clSetKernelArg(kernels["finalReduc"], 5, sizeof(cl_uint), &num_wg);
	CHECK_ERROR_OCL(err, "setting finalReduc arguments", return false);

	err = clSetKernelArg(kernels["coarsest_level_attenfunc"], 0, sizeof(cl_mem), &mems["gradient_Mips"]);
	err = clSetKernelArg(kernels["coarsest_level_attenfunc"], 1, sizeof(cl_mem), &mems["attenfunc_Mips"]);
	err = clSetKernelArg(kernels["coarsest_level_attenfunc"], 2, sizeof(cl_mem), &mems["k_alphas"]);
	err = clSetKernelArg(kernels["coarsest_level_attenfunc"], 3, sizeof(cl_int), &m_width[num_mipmaps-1]);
	err = clSetKernelArg(kernels["coarsest_level_attenfunc"], 4, sizeof(cl_int), &m_height[num_mipmaps-1]);
	err = clSetKernelArg(kernels["coarsest_level_attenfunc"], 5, sizeof(cl_int), &m_offset[num_mipmaps-1]);
	CHECK_ERROR_OCL(err, "setting coarsest_level_attenfunc arguments", return false);

	err = clSetKernelArg(kernels["atten_func"], 0, sizeof(cl_mem), &mems["gradient_Mips"]);
	err = clSetKernelArg(kernels["atten_func"], 1, sizeof(cl_mem), &mems["attenfunc_Mips"]);
	err = clSetKernelArg(kernels["atten_func"], 2, sizeof(cl_mem), &mems["k_alphas"]);
	CHECK_ERROR_OCL(err, "setting atten_func arguments", return false);

	err = clSetKernelArg(kernels["grad_atten"], 0, sizeof(cl_mem), &mems["atten_grad_x"]);
	err = clSetKernelArg(kernels["grad_atten"], 1, sizeof(cl_mem), &mems["atten_grad_y"]);
	err = clSetKernelArg(kernels["grad_atten"], 2, sizeof(cl_mem), &mems["logLum_Mips"]);
	err = clSetKernelArg(kernels["grad_atten"], 3, sizeof(cl_mem), &mems["attenfunc_Mips"]);
	CHECK_ERROR_OCL(err, "setting grad_atten arguments", return false);

	err = clSetKernelArg(kernels["divG"], 0, sizeof(cl_mem), &mems["atten_grad_x"]);
	err = clSetKernelArg(kernels["divG"], 1, sizeof(cl_mem), &mems["atten_grad_y"]);
	err = clSetKernelArg(kernels["divG"], 2, sizeof(cl_mem), &mems["div_grad"]);
	CHECK_ERROR_OCL(err, "setting divG arguments", return false);

	reportStatus("\n\n");

	return true;
}

double GradDom::runCLKernels(std::vector<bool> whichKernelsToRun, bool recomputeMapping) {
	double start, end;

	start = omp_get_wtime();

	cl_int err;
	if (recomputeMapping) {
		err = clEnqueueNDRangeKernel(m_queue, kernels["computeLogLum"], 2, NULL, global_sizes["computeLogLum"], local_sizes["computeLogLum"], 0, NULL, NULL);
		CHECK_ERROR_OCL(err, "enqueuing computeLogLum kernel", return false);

		//compute the gradient magniute of mipmap level 0
		err = clSetKernelArg(kernels["gradient_mag"], 2, sizeof(cl_int), &m_width[0]);
		err = clSetKernelArg(kernels["gradient_mag"], 3, sizeof(cl_int), &m_height[0]);
		err = clSetKernelArg(kernels["gradient_mag"], 4, sizeof(cl_int), &m_offset[0]);
		err = clSetKernelArg(kernels["gradient_mag"], 5, sizeof(cl_float), &m_divider[0]);
		err = clEnqueueNDRangeKernel(m_queue, kernels["gradient_mag"], 2, NULL, global_sizes["gradient_mag"], local_sizes["gradient_mag"], 0, NULL, NULL);
		CHECK_ERROR_OCL(err, "enqueuing gradient_mag kernel", return false);

		err = clSetKernelArg(kernels["partialReduc"], 3, sizeof(cl_int), &m_width[0]);
		err = clSetKernelArg(kernels["partialReduc"], 4, sizeof(cl_int), &m_height[0]);
		err = clSetKernelArg(kernels["partialReduc"], 5, sizeof(cl_int), &m_offset[0]);
		err = clEnqueueNDRangeKernel(m_queue, kernels["partialReduc"], 1, NULL, &global_sizes["partialReduc"][0], &local_sizes["partialReduc"][0], 0, NULL, NULL);
		CHECK_ERROR_OCL(err, "setting partialReduc arguments", return false);

		int level = 0;
		err = clSetKernelArg(kernels["finalReduc"], 2, sizeof(cl_int), &level);
		err = clSetKernelArg(kernels["finalReduc"], 3, sizeof(cl_int), &m_width[level]);
		err = clSetKernelArg(kernels["finalReduc"], 4, sizeof(cl_int), &m_height[level]);
		err = clEnqueueNDRangeKernel(m_queue, kernels["finalReduc"], 1, NULL, &global_sizes["finalReduc"][0], &local_sizes["finalReduc"][0], 0, NULL, NULL);
		CHECK_ERROR_OCL(err, "enqueuing finalReduc kernel", return false);
	
		//creating mipmaps and their gradient magnitudes
		for (int level=1; level<num_mipmaps; level++) {
			err = clSetKernelArg(kernels["channel_mipmap"], 1, sizeof(cl_int), &m_width[level-1]);
			err = clSetKernelArg(kernels["channel_mipmap"], 2, sizeof(cl_int), &m_offset[level-1]);
			err = clSetKernelArg(kernels["channel_mipmap"], 3, sizeof(cl_int), &m_width[level]);
			err = clSetKernelArg(kernels["channel_mipmap"], 4, sizeof(cl_int), &m_height[level]);
			err = clSetKernelArg(kernels["channel_mipmap"], 5, sizeof(cl_int), &m_offset[level]);
			err = clEnqueueNDRangeKernel(m_queue, kernels["channel_mipmap"], 2, NULL, global_sizes["channel_mipmap"], local_sizes["channel_mipmap"], 0, NULL, NULL);
			CHECK_ERROR_OCL(err, "enqueuing channel_mipmap kernel", return false);

			err = clSetKernelArg(kernels["gradient_mag"], 2, sizeof(cl_int), &m_width[level]);
			err = clSetKernelArg(kernels["gradient_mag"], 3, sizeof(cl_int), &m_height[level]);
			err = clSetKernelArg(kernels["gradient_mag"], 4, sizeof(cl_int), &m_offset[level]);
			err = clSetKernelArg(kernels["gradient_mag"], 5, sizeof(cl_float), &m_divider[level]);
			err = clEnqueueNDRangeKernel(m_queue, kernels["gradient_mag"], 2, NULL, global_sizes["gradient_mag"], local_sizes["gradient_mag"], 0, NULL, NULL);
			CHECK_ERROR_OCL(err, "enqueuing gradient_mag kernel", return false);			

			err = clSetKernelArg(kernels["partialReduc"], 3, sizeof(cl_int), &m_width[level]);
			err = clSetKernelArg(kernels["partialReduc"], 4, sizeof(cl_int), &m_height[level]);
			err = clSetKernelArg(kernels["partialReduc"], 5, sizeof(cl_int), &m_offset[level]);
			err = clEnqueueNDRangeKernel(m_queue, kernels["partialReduc"], 1, NULL, &global_sizes["partialReduc"][0], &local_sizes["partialReduc"][0], 0, NULL, NULL);
			CHECK_ERROR_OCL(err, "setting partialReduc arguments", return false);
	
			err = clSetKernelArg(kernels["finalReduc"], 2, sizeof(cl_int), &level);
			err = clSetKernelArg(kernels["finalReduc"], 3, sizeof(cl_int), &m_width[level]);
			err = clSetKernelArg(kernels["finalReduc"], 4, sizeof(cl_int), &m_height[level]);
			err = clEnqueueNDRangeKernel(m_queue, kernels["finalReduc"], 1, NULL, &global_sizes["finalReduc"][0], &local_sizes["finalReduc"][0], 0, NULL, NULL);
			CHECK_ERROR_OCL(err, "enqueuing finalReduc kernel", return false);
		}

		//attenuation function of mipmap at level num_mipmaps-1
		err = clEnqueueNDRangeKernel(m_queue, kernels["coarsest_level_attenfunc"], 1, NULL,
				&global_sizes["coarsest_level_attenfunc"][0], &local_sizes["coarsest_level_attenfunc"][0], 0, NULL, NULL);
		CHECK_ERROR_OCL(err, "enqueuing coarsest_level_attenfunc kernel", return false);

		for (int level=num_mipmaps-2; level>-1; level--) {
			err = clSetKernelArg(kernels["atten_func"], 3, sizeof(cl_int), &m_width[level]);
			err = clSetKernelArg(kernels["atten_func"], 4, sizeof(cl_int), &m_height[level]);
			err = clSetKernelArg(kernels["atten_func"], 5, sizeof(cl_int), &m_offset[level]);
			err = clSetKernelArg(kernels["atten_func"], 6, sizeof(cl_int), &m_width[level+1]);
			err = clSetKernelArg(kernels["atten_func"], 7, sizeof(cl_int), &m_height[level+1]);
			err = clSetKernelArg(kernels["atten_func"], 8, sizeof(cl_int), &m_offset[level+1]);
			err = clSetKernelArg(kernels["atten_func"], 9, sizeof(cl_int), &level);
			err = clEnqueueNDRangeKernel(m_queue, kernels["atten_func"], 2, NULL, global_sizes["atten_func"], local_sizes["atten_func"], 0, NULL, NULL);
			CHECK_ERROR_OCL(err, "enqueuing atten_func kernel", return false);
		}
	
		err = clEnqueueNDRangeKernel(m_queue, kernels["grad_atten"], 2, NULL, global_sizes["grad_atten"], local_sizes["grad_atten"], 0, NULL, NULL);
		CHECK_ERROR_OCL(err, "enqueuing grad_atten kernel", return false);

		err = clEnqueueNDRangeKernel(m_queue, kernels["divG"], 2, NULL, global_sizes["divG"], local_sizes["divG"], 0, NULL, NULL);
		CHECK_ERROR_OCL(err, "enqueuing divG kernel", return false);

	}

	err = clFinish(m_queue);
	CHECK_ERROR_OCL(err, "running kernels", return false);

	end = omp_get_wtime();

	return (end - start);
}

bool GradDom::cleanupOpenCL() {
	clReleaseMemObject(mem_images[0]);
	clReleaseMemObject(mem_images[1]);
	clReleaseMemObject(mems["logLum_Mips"]);
	clReleaseMemObject(mems["gradient_Mips"]);
	clReleaseMemObject(mems["attenfunc_Mips"]);
	clReleaseMemObject(mems["gradient_PartialSum"]);
	clReleaseMemObject(mems["k_alphas"]);
	clReleaseMemObject(mems["atten_grad_x"]);
	clReleaseMemObject(mems["atten_grad_y"]);
	clReleaseMemObject(mems["div_grad"]);
	clReleaseKernel(kernels["computeLogLum"]);
	clReleaseKernel(kernels["channel_mipmap"]);
	clReleaseKernel(kernels["gradient_mag"]);
	clReleaseKernel(kernels["partialReduc"]);
	clReleaseKernel(kernels["finalReduc"]);
	clReleaseKernel(kernels["coarsest_level_attenfunc"]);
	clReleaseKernel(kernels["atten_func"]);
	clReleaseKernel(kernels["grad_atten"]);
	clReleaseKernel(kernels["divG"]);
	
	releaseCL();
	return true;
}


float* GradDom::attenuate_func(float* lum) {
	int k = 0;
	int2 k_dim = img_size;	//width and height of the level k in the pyramid
	float* k_lum = lum;
	float k_av_grad = 0.f;
	float* k_gradient;

	std::vector<float*> pyramid;
	std::vector<float> av_grads;
	std::vector<std::pair<unsigned int, unsigned int> > pyramid_sizes;

	for ( ; k_dim.x >= 32 && k_dim.y >= 32; k_dim.y/=2, k_dim.x/=2, k++) {

		//computing gradient magnitude using central differences at level k
		k_av_grad = 0.f;
		k_gradient = (float*) calloc(k_dim.x*k_dim.y, sizeof(float));
		for (int y = 0; y < k_dim.y; y++) {
			for (int x = 0; x < k_dim.x; x++) {
				int x_west  = clamp(x-1, 0, k_dim.x-1);
				int x_east  = clamp(x+1, 0, k_dim.x-1);
				int y_north = clamp(y-1, 0, k_dim.y-1);
				int y_south = clamp(y+1, 0, k_dim.y-1);

				float x_grad = (k_lum[x_west + y*k_dim.x] - k_lum[x_east + y*k_dim.x])/pow(2.f, k+1);
				float y_grad = (k_lum[x + y_south*k_dim.x] - k_lum[x + y_north*k_dim.x])/pow(2.f, k+1);
				k_gradient[x + y*k_dim.x] = sqrt(pow(x_grad, 2) + pow(y_grad, 2));
				k_av_grad += k_gradient[x + y*k_dim.x];
			}
		}
		pyramid.push_back(k_gradient);
		pyramid_sizes.push_back(std::pair< unsigned int, unsigned int >(k_dim.x, k_dim.y));
		av_grads.push_back(adjust_alpha*exp(k_av_grad/((float)k_dim.x*k_dim.y)));

		k_lum = mipmap(k_lum, k_dim);
	}


	//computing attenuation functions
	k_gradient = pyramid.back();
	k_dim.x = pyramid_sizes.back().first;
	k_dim.y = pyramid_sizes.back().second;
	float k_alpha = av_grads.back();
	float* k_atten_func;
	k--;
	
	//attenuation function for the coarsest level
	k_atten_func = (float*) calloc(k_dim.x*k_dim.y, sizeof(float));
	for (int y = 0; y < k_dim.y; y++) {
		for (int x = 0; x < k_dim.x; x++) {
			k_atten_func[x + y*k_dim.x] = (k_alpha/k_gradient[x + y*k_dim.x])*pow(k_gradient[x + y*k_dim.x]/k_alpha, beta);
		}
	}

	pyramid.pop_back();
	pyramid_sizes.pop_back();
	av_grads.pop_back();

	while (! pyramid.empty()) {
		
		float* k1_atten_func = k_atten_func;
		int k1_width = k_dim.x;
		int k1_height = k_dim.y;

		k_gradient = pyramid.back();
		k_dim.x = pyramid_sizes.back().first;
		k_dim.y = pyramid_sizes.back().second;
		float k_alpha = av_grads.back();
		float k_xy_scale_factor;
		float k_xy_atten_func;
		k--;

		//attenuation function for this level
		k_atten_func = (float*) calloc(k_dim.x*k_dim.y, sizeof(float));
		for (int y = 0; y < k_dim.y; y++) {
			for (int x = 0; x < k_dim.x; x++) {

				if (k_gradient[x + y*k_dim.x] != 0) {

					int k1_x = x/2, k1_y = y/2;		//x and y value of the coarser grid

					//neighbours need to be left or right dependent on where we are
					int n_x = (x & 1) ? 1 : -1;
					int n_y = (y & 1) ? 1 : -1;


					//this stops us from going out of bounds
					if ((k1_x + n_x) < 0) n_x = 0;
					if ((k1_y + n_y) < 0) n_y = 0;
					if ((k1_x + n_x) >= k1_width) n_x = 0;
					if ((k1_y + n_y) >= k1_height) n_y = 0;
					if (k1_x == k1_width) k1_x -= 1;
					if (k1_y == k1_height) k1_y -= 1;

					k_xy_atten_func = 9.0*k1_atten_func[k1_x 		+ k1_y			*k1_width]
									+ 3.0*k1_atten_func[k1_x+n_x 	+ k1_y			*k1_width]
									+ 3.0*k1_atten_func[k1_x 		+ (k1_y+n_y)	*k1_width]
									+ 1.0*k1_atten_func[k1_x+n_x 	+ (k1_y+n_y)	*k1_width];

					k_xy_scale_factor = (k_alpha/k_gradient[x + y*k_dim.x])*pow(k_gradient[x + y*k_dim.x]/k_alpha, beta);
				}
				else k_xy_scale_factor = 0.f;
				k_atten_func[x + y*k_dim.x] = (1.f/16.f)*(k_xy_atten_func)*k_xy_scale_factor;
			}
		}

		pyramid.pop_back();
		pyramid_sizes.pop_back();
		av_grads.pop_back();
	}
	return k_atten_func;
}


float* GradDom::poissonSolver(float* lum, float* div_grad, float convergenceCriteria) {

	float* prev_dr = (float*) calloc(img_size.y*img_size.x, sizeof(float));
	int* converged = (int*) calloc(img_size.y*img_size.x, sizeof(int));
	for (int y = 0; y < img_size.y; y++) {
		for (int x = 0; x < img_size.x; x++) {
			prev_dr[x + y*img_size.x] = lum[x+y*img_size.x];
			converged[x + y*img_size.x] = 0;
		}
	}

	float* new_dr = (float*) calloc(img_size.y*img_size.x, sizeof(float));

	float diff;
	int converged_pixels = 0;
	while (converged_pixels < 0.9*img_size.x*img_size.y) {
		diff = 0;
		for (int y = 0; y < img_size.y; y++) {
			for (int x = 0; x < img_size.x; x++) {

				if (converged[x + y*img_size.x] != 1) {
					float prev  = ((x-1 >=        0) ? prev_dr[x-1 +   y*img_size.x] : 0)
								+ ((x+1 <=  img_size.x-1) ? prev_dr[x+1 +   y*img_size.x] : 0)
								+ ((y-1 >=        0) ? prev_dr[x + (y-1)*img_size.x] : 0)
								+ ((y+1 <= img_size.y-1) ? prev_dr[x + (y+1)*img_size.x] : 0);

					new_dr[x + y*img_size.x] = 0.25f*(prev - div_grad[x + y*img_size.x]);
					diff = new_dr[x + y*img_size.x] - prev_dr[x + y*img_size.x];
					diff = (diff >= 0) ? diff : -diff;

					if (diff < convergenceCriteria) {
						converged_pixels++;
						converged[x + y*img_size.x] = 1;
					}
				}
			}
		}
		//printf("%d, %d\n", converged_pixels, img_size.x*img_size.y);
		float* swap = prev_dr;
		prev_dr = new_dr;
		new_dr = swap;
	}

	return new_dr;
}


bool GradDom::runReference(uchar* input, uchar* output) {

	// Check for cached result
	if (m_reference.data) {
		memcpy(output, m_reference.data, img_size.x*img_size.y*NUM_CHANNELS);
		reportStatus("Finished reference (cached)");
		return true;
	}

	reportStatus("Running reference");

	//computing logarithmic luminace of the image
	float* lum = (float*) calloc(img_size.x * img_size.y, sizeof(float));	//logarithm luminance
	int2 pos;
	for (pos.y = 0; pos.y < img_size.y; pos.y++) {
		for (pos.x = 0; pos.x < img_size.x; pos.x++) {
			lum[pos.x + pos.y*img_size.x] = log(getPixelLuminance(input, img_size, pos) + 0.000001);
		}
	}

	//computing the attenuation function which will then be multiplied by luminance gradient to acheive attenuated gradient
	float* att_func = attenuate_func(lum);	//o(x,y)

	//luminance gradient in forward direction for x and y
	float* grad_x = (float*) calloc(img_size.x * img_size.y, sizeof(float));	//H(x,y)
	float* grad_y = (float*) calloc(img_size.x * img_size.y, sizeof(float));	//H(x,y)
	for (int y = 0; y < img_size.y; y++) {
		for (int x = 0; x < img_size.x; x++) {
			grad_x[x + y*img_size.x] = (x < img_size.x-1) ? (lum[x+1 +     y*img_size.x] - lum[x + y*img_size.x]) : 0;
			grad_y[x + y*img_size.x] = (y < img_size.y-1) ? (lum[x   + (y+1)*img_size.x] - lum[x + y*img_size.x]) : 0;
		}
	}


	//attenuated gradient achieved by using the previously computed attenuation function
	float* att_grad_x = (float*) calloc(img_size.y * img_size.x, sizeof(float));	//G(x,y)
	float* att_grad_y = (float*) calloc(img_size.y * img_size.x, sizeof(float));	//G(x,y)
	for (int y = 0; y < img_size.y; y++) {
		for (int x = 0; x < img_size.x; x++) {
			att_grad_x[x + y*img_size.x] = grad_x[x + y*img_size.x] * att_func[x + y*img_size.x];
			att_grad_y[x + y*img_size.x] = grad_y[x + y*img_size.x] * att_func[x + y*img_size.x];
		}
	}

	//divG(x,y)
	float* div_grad = (float*) calloc(img_size.y * img_size.x, sizeof(float));
	div_grad[0] = 0;
	for (int x = 1; x < img_size.x; x++) {
		div_grad[x] = att_grad_x[x] - att_grad_x[x-1];
	}
	for (int y = 1; y < img_size.y; y++) {
		div_grad[y*img_size.x] = att_grad_y[y*img_size.x] - att_grad_y[(y-1)*img_size.x];
		for (int x = 1; x < img_size.x; x++) {
			div_grad[x + y*img_size.x] = (att_grad_x[x + y*img_size.x] - att_grad_x[(x-1) + y*img_size.x])
										+ (att_grad_y[x + y*img_size.x] - att_grad_y[x + (y-1)*img_size.x]);
		}
	}

	float* new_dr = poissonSolver(lum, div_grad);

	float3 rgb;
	for (pos.y = 0; pos.y < img_size.y; pos.y++) {
		for (pos.x = 0; pos.x < img_size.x; pos.x++) {
			//printf("%f, %f\n", lum[x + y*img_size.x], new_dr[x+y*img_size.x]);
			rgb.x = pow(getPixel(input, img_size, pos, 0)/exp(lum[pos.x + pos.y*img_size.x]), sat)*exp(new_dr[pos.x + pos.y*img_size.x]);
			rgb.y = pow(getPixel(input, img_size, pos, 1)/exp(lum[pos.x + pos.y*img_size.x]), sat)*exp(new_dr[pos.x + pos.y*img_size.x]);
			rgb.z = pow(getPixel(input, img_size, pos, 2)/exp(lum[pos.x + pos.y*img_size.x]), sat)*exp(new_dr[pos.x + pos.y*img_size.x]);

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
