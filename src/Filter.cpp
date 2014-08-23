// Filter.cpp (HDR)
// Copyright (c) 2014, Amir Chohan,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.

#include <stddef.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <exception>
#include <stdexcept>

#include "Filter.h"

namespace hdr
{
Filter::Filter() {
	m_statusCallback = NULL;
	m_clContext = 0;
	m_queue = 0;
	m_program = 0;
	m_reference.data = NULL;
}

Filter::~Filter() {
	clearReferenceCache();
}

void Filter::clearReferenceCache() {
	if (m_reference.data) {
		delete[] m_reference.data;
		m_reference.data = NULL;
	}
}

const char* Filter::getName() const {
	return m_name;
}


bool Filter::initCL(cl_context_properties context_prop[], const Params& params, const char *source, const char *options) {
	// Ensure no existing context
	releaseCL();

	cl_int err;
	cl_uint numPlatforms, numDevices;

	cl_platform_id platform, platforms[params.platformIndex+1];
	err = clGetPlatformIDs(params.platformIndex+1, platforms, &numPlatforms);
	CHECK_ERROR_OCL(err, "getting platforms", return false);
	if (params.platformIndex >= numPlatforms) {
		reportStatus("Platform index %d out of range (%d platforms found)",
			params.platformIndex, numPlatforms);
		return false;
	}
	platform = platforms[params.platformIndex];

	cl_device_id devices[params.deviceIndex+1];
	err = clGetDeviceIDs(platform, params.type, params.deviceIndex+1, devices, &numDevices);
	CHECK_ERROR_OCL(err, "getting devices", return false);
	if (params.deviceIndex >= numDevices) {
		reportStatus("Device index %d out of range (%d devices found)",
			params.deviceIndex, numDevices);
		return false;
	}
	m_device = devices[params.deviceIndex];

	char name[64];
	clGetDeviceInfo(m_device, CL_DEVICE_NAME, 64, name, NULL);
	reportStatus("Using device: %s", name);

	cl_ulong device_size;
	clGetDeviceInfo(m_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_size), &device_size, NULL);
	reportStatus("CL_DEVICE_GLOBAL_MEM_SIZE: %lu bytes", device_size);

	clGetDeviceInfo(m_device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(device_size), &device_size, NULL);
	reportStatus("CL_DEVICE_LOCAL_MEM_SIZE: %lu bytes", device_size);

	clGetDeviceInfo(m_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &max_cu, NULL);
	reportStatus("CL_DEVICE_MAX_COMPUTE_UNITS: %lu", max_cu);

	if (params.opengl) context_prop[5] = (cl_context_properties) platform;

	m_clContext = clCreateContext(context_prop, 1, &m_device, NULL, NULL, &err);
	CHECK_ERROR_OCL(err, "creating context", return false);

	m_queue = clCreateCommandQueue(m_clContext, m_device, CL_QUEUE_PROFILING_ENABLE, &err);
	CHECK_ERROR_OCL(err, "creating command queue", return false);

	m_program = clCreateProgramWithSource(m_clContext, 1, &source, NULL, &err);
	CHECK_ERROR_OCL(err, "creating program", return false);

	err = clBuildProgram(m_program, 1, &m_device, options, NULL, NULL);
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t sz;
		clGetProgramBuildInfo(
			m_program, m_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
		char *log = (char*)malloc(++sz);
		clGetProgramBuildInfo(
			m_program, m_device, CL_PROGRAM_BUILD_LOG, sz, log, NULL);
		reportStatus(log);
		free(log);
	}
	CHECK_ERROR_OCL(err, "building program", return false);

	reportStatus("OpenCL context initialised.");
	return true;
}

cl_int Filter::releaseCL() {
	cl_int err = CL_SUCCESS;

	if (m_program) {
		err = clReleaseProgram(m_program);
		CHECK_ERROR_OCL(err, "releasing program", return false);
		m_program = 0;
	}
	if (m_queue) {
		err = clReleaseCommandQueue(m_queue);
		CHECK_ERROR_OCL(err, "releasing queue", return false);
		m_queue = 0;
	}
	if (m_clContext) {
		err = clReleaseContext(m_clContext);
		CHECK_ERROR_OCL(err, "releasing context", return false);
		m_clContext = 0;
	}

	return err;
}

bool Filter::runOpenCL(std::vector<bool> whichKernelsToRun, bool recomputeMapping) {
	cl_int err;

	err = clEnqueueAcquireGLObjects(m_queue, 2, &mem_images[0], 0, 0, 0);
	CHECK_ERROR_OCL(err, "acquiring GL objects", return false);

	double runTime = runCLKernels(whichKernelsToRun, recomputeMapping);

	err = clEnqueueReleaseGLObjects(m_queue, 2, &mem_images[0], 0, 0, 0);
	CHECK_ERROR_OCL(err, "releasing GL objects", return false);

	reportStatus("Finished running OpenCL kernels in %.3f ms", runTime * 1e3);

	return true;
}

bool Filter::runOpenCL(uchar* input, uchar* output,
		std::vector<bool> whichKernelsToRun, bool recomputeMapping, bool verifyOutput) {
	cl_int err;

 	const size_t origin[] = {0, 0, 0};
 	const size_t region[] = {img_size.x, img_size.y, 1};
	err = clEnqueueWriteImage(m_queue, mem_images[0], CL_TRUE, origin, region, sizeof(uchar)*img_size.x*NUM_CHANNELS, 0, input, 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "writing image memory", return false);

	double runTime = runCLKernels(whichKernelsToRun, recomputeMapping);

	err = clEnqueueReadImage(m_queue, mem_images[1], CL_TRUE, origin, region, sizeof(uchar)*img_size.x*NUM_CHANNELS, 0, output, 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "reading image memory", return false);

	reportStatus("Finished running OpenCL kernels in %.3f ms", runTime * 1e3);

	// Verification
	bool passed = true;
	if (verifyOutput) {
		passed = verify(input, output);
		reportStatus("Verification: %s", passed ? "PASSED" : "FAILED");
	} else {
		reportStatus("Verification: %s", "NOT RUN");
	}

	return passed;
}


void Filter::reportStatus(const char *format, ...) const {
	if (m_statusCallback) {
		va_list args;
		va_start(args, format);
		m_statusCallback(format, args);
		va_end(args);
	}
}


void Filter::setStatusCallback(int (*callback)(const char*, va_list args)) {
	m_statusCallback = callback;
}


bool Filter::verify(uchar* input, uchar* output, float tolerance, float maxErrorPercent) {
	// compute reference image
	uchar* ref = (uchar*) calloc(img_size.x*img_size.y*NUM_CHANNELS, sizeof(uchar));
	runReference(input, ref);

	// compare pixels
	int errors = 0;
	const int maxErrors = img_size.x*img_size.y*maxErrorPercent;
	int2 pos;
	for (pos.y = 0; pos.y < img_size.y; pos.y++) {
		for (pos.x = 0; pos.x < img_size.x; pos.x++) {
			for (int c = 0; c < NUM_CHANNELS; c++) {
				float r = getPixel(ref, img_size, pos, c);
				float o = getPixel(output, img_size, pos, c);
				float diff = r - o;
				diff = diff >= 0 ? diff : -diff;

				if (diff > tolerance) {
					// Only report first few errors
					if (errors < maxErrors) {
						reportStatus("Mismatch at (%d,%d,%d): %f vs %f", pos.x, pos.y, c, r, o);
					}
					if (++errors == maxErrors) {
						reportStatus("Supressing further errors");
					}
				}
			}
		}
	}

	free(ref);
	return errors == 0;
}


bool Filter::kernel1DSizes(const char* kernel_name) {
	reportStatus("---------------------------------Kernel %s:", kernel_name);

	cl_int err;

	size_t max_wg_size;	//max workgroup size for the kernel
	err = clGetKernelWorkGroupInfo (kernels[kernel_name], m_device,
		CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL);
	CHECK_ERROR_OCL(err, "getting CL_KERNEL_WORK_GROUP_SIZE", return false);
	reportStatus("CL_KERNEL_WORK_GROUP_SIZE: %lu", max_wg_size);

	size_t preferred_wg_size;	//workgroup size should be a multiple of this
	err = clGetKernelWorkGroupInfo (kernels[kernel_name], m_device,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferred_wg_size, NULL);
	CHECK_ERROR_OCL(err, "getting CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE", return false);
	reportStatus("CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: %lu", preferred_wg_size);

	size_t* local = (size_t*) calloc(2, sizeof(size_t));
	size_t* global = (size_t*) calloc(2, sizeof(size_t));
	local[0] = preferred_wg_size;	//workgroup size for normal kernels
	global[0] = preferred_wg_size*max_cu;

	local_sizes[kernel_name] = local;
	global_sizes[kernel_name] = global;

	reportStatus("Kernel sizes: Local=%lu Global=%lu", local[0], global[0]);
}


bool Filter::kernel2DSizes(const char* kernel_name) {
	reportStatus("---------------------------------Kernel %s:", kernel_name);

	cl_int err;

	size_t max_wg_size;	//max workgroup size for the kernel
	err = clGetKernelWorkGroupInfo (kernels[kernel_name], m_device,
		CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL);
	CHECK_ERROR_OCL(err, "getting CL_KERNEL_WORK_GROUP_SIZE", return false);
	reportStatus("CL_KERNEL_WORK_GROUP_SIZE: %lu", max_wg_size);

	size_t preferred_wg_size;	//workgroup size should be a multiple of this
	err = clGetKernelWorkGroupInfo (kernels[kernel_name], m_device,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferred_wg_size, NULL);
	CHECK_ERROR_OCL(err, "getting CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE", return false);
	reportStatus("CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: %lu", preferred_wg_size);

	size_t* local = (size_t*) calloc(2, sizeof(size_t));
	size_t* global = (size_t*) calloc(2, sizeof(size_t));

	int i=0;
	local[0] = preferred_wg_size;
	local[1] = 1;
	while (local[0]*local[1] <= max_wg_size) {
		local[i%2] *= 2;
		i++;
	}
	while (local[0]*local[1] > max_wg_size) {
		local[i%2] /= 2;
		i--;
	}

	global[0] = local[0]*max_cu;
	global[1] = local[1]*max_cu;

	global[0] = ceil((float)global[0]/(float)local[0])*local[0];
	global[1] = ceil((float)global[1]/(float)local[1])*local[1];

	local_sizes[kernel_name] = local;
	global_sizes[kernel_name] = global;
	
	reportStatus("Kernel sizes: Local=(%lu, %lu) Global=(%lu, %lu)", local[0], local[1], global[0], global[1]);
}


/////////////////
// Image utils //
/////////////////

void Filter::setImageSize(int width, int height) {
	img_size.x = width;
	img_size.y = height;
}

void Filter::setImageTextures(GLuint input_texture, GLuint output_texture) {
	in_tex = input_texture;
	out_tex = output_texture;
}

float* mipmap(float* input, int2 size, int level) {
	int scale_factor = pow(2, level);
	int m_width = size.x/scale_factor;
	int m_height = size.y/scale_factor;

	float* result = (float*) calloc(m_width*m_height, sizeof(float));

	for (int y = 0; y < m_height; y++) {
		for (int x = 0; x < m_width; x++) {
			int _x = scale_factor*x;
			int _y = scale_factor*y;
			result[x + y*m_width] = (input[_x + _y*size.x] + input[_x+1 + _y*size.x] + input[_x + (_y+1)*size.x] + input[(_x+1) + (_y+1)*size.x])/4.f;
		}
	}
	return result;
}

float clamp(float x, float min, float max) {
	return x < min ? min : x > max ? max : x;
}


float getValue(float* data, int2 size, int2 pos) {
	int _x = clamp(pos.x, 0, size.x-1);
	int _y = clamp(pos.y, 0, size.y-1);
	return data[_x + _y*size.x];
}

float getPixel(uchar* data, int2 image_size, int2 pixel_pos, int c) {
	int _x = clamp(pixel_pos.x, 0, image_size.x-1);
	int _y = clamp(pixel_pos.y, 0, image_size.y-1);
	return ((float) data[(_x + _y*image_size.x)*NUM_CHANNELS + c]);
}

void setPixel(uchar* data, int2 image_size, int2 pixel_pos, int c, float value) {
	int _x = clamp(pixel_pos.x, 0, image_size.x-1);
	int _y = clamp(pixel_pos.y, 0, image_size.y-1);
	data[(_x + _y*image_size.x)*NUM_CHANNELS + c] = clamp(value, 0.f, PIXEL_RANGE*1.f);
}

float getPixelLuminance(uchar* image, int2 image_size, int2 pixel_pos) {
	float3 pixel_val = {getPixel(image, image_size, pixel_pos, 0),
						getPixel(image, image_size, pixel_pos, 1),
						getPixel(image, image_size, pixel_pos, 2)};

	return    pixel_val.x*0.2126
			+ pixel_val.y*0.7152
			+ pixel_val.z*0.0722;
}


float3 RGBtoHSV(float3 rgb) {
	float r = rgb.x;
	float g = rgb.y;
	float b = rgb.z;
	float min, max, delta;
	min = std::min(std::min(r, g), b);
	max = std::max(std::max(r, g), b);

	float3 hsv;

	hsv.z = max;	//Brightness
	delta = max - min;
	if(max != 0) hsv.y = delta/max;//Saturation
	else {	// r = g = b = 0	//Saturation = 0, Value is undefined
		hsv.y = 0;
		hsv.x = -1;
		return hsv;
	}

	//Hue
	if(r == max) 		hsv.x = (g-b)/delta;
	else if(g == max) 	hsv.x = (b-r)/delta + 2;
	else 				hsv.x = (r-g)/delta + 4;
	hsv.x *= 60;				
	if( hsv.x < 0 ) hsv.x += 360;

	return hsv;
}

float3 HSVtoRGB(float3 hsv) {
	int i;
	float h = hsv.x;
	float s = hsv.y;
	float v = hsv.z;
	float f, p, q, t;
	float3 rgb;
	if( s == 0 ) { // achromatic (grey)
		rgb.x = rgb.y = rgb.z = v;
		return rgb;
	}
	h /= 60;			// sector 0 to 5
	i = floor( h );
	f = h - i;			// factorial part of h
	p = v * ( 1 - s );
	q = v * ( 1 - s * f );
	t = v * ( 1 - s * ( 1 - f ) );
	switch( i ) {
		case 0:
			rgb.x = v;
			rgb.y = t;
			rgb.z = p;
			break;
		case 1:
			rgb.x = q;
			rgb.y = v;
			rgb.z = p;
			break;
		case 2:
			rgb.x = p;
			rgb.y = v;
			rgb.z = t;
			break;
		case 3:
			rgb.x = p;
			rgb.y = q;
			rgb.z = v;
			break;
		case 4:
			rgb.x = t;
			rgb.y = p;
			rgb.z = v;
			break;
		default:		// case 5:
			rgb.x = v;
			rgb.y = p;
			rgb.z = q;
			break;
	}
	return rgb;
}


float3 RGBtoXYZ(float3 rgb) {
	float3 xyz;
	xyz.x = rgb.x*0.4124 + rgb.y*0.3576 + rgb.z*0.1805;
	xyz.y = rgb.x*0.2126 + rgb.y*0.7152 + rgb.z*0.0722;
	xyz.z = rgb.x*0.0193 + rgb.y*0.1192 + rgb.z*0.9505;
	return xyz;
}

float3 XYZtoRGB(float3 xyz) {
	float3 rgb;
	rgb.x =   xyz.x*3.240479 - xyz.y*1.53715  - xyz.z*0.498535;
	rgb.y = - xyz.x*0.969256 + xyz.y*1.875991 + xyz.z*0.041556;
	rgb.z =   xyz.x*0.055648 - xyz.y*0.204043 + xyz.z*1.057311;
	return rgb;
}


float weight(float luminance) {
	if (luminance < 0.5) return luminance*2.0;
	else return (1.0 - luminance)*2.0;
}


//////////////////
// Timing utils //
//////////////////

double getCurrentTime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_usec + tv.tv_sec*1e6;
}
}
