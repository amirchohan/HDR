// Filter.h (HDR)
// Copyright (c) 2014, Amir Chohan,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.

#pragma once

#include <map>
#include <vector>
#include <math.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <omp.h>

#ifdef __ANDROID_API__
	#include <GLES/gl.h>
	#define BUGGY_CL_GL 1	//see NOTES
#else
	#include <GL/gl.h>
	#define BUGGY_CL_GL 0
#endif

#define PIXEL_RANGE	255	//8-bit
#define NUM_CHANNELS 4	//RGBA

#define MAX_NUM_KERNELS 10 //maximum number of kernels in any filter

#define CHECK_ERROR_OCL(err, op, action)						\
	if (err != CL_SUCCESS) {							\
		reportStatus("Error during operation '%s' (%d)", op, err);		\
		releaseCL();								\
		action;									\
	}

#define CHECK_PROFILING_OCL(event, op)							\
	do {										\
		cl_int num_events = 1;							\
		clWaitForEvents(num_events, &event);					\
											\
		cl_int err = CL_SUCCESS;						\
		cl_ulong tq, tu, ts, te;						\
		err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED,	\
			sizeof(tq), &tq, NULL);						\
		err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT,	\
			sizeof(tu), &tu, NULL);						\
		err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,	\
			sizeof(ts), &ts, NULL);						\
		err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,		\
			sizeof(te), &te, NULL);						\
		err |= clReleaseEvent(event);						\
		CHECK_ERROR_OCL(err, op, return false);					\
											\
		reportStatus("%s:", op);						\
		reportStatus("\t[queueing, submitting, executing]");			\
		reportStatus("\t[%.3f ms, %.3f ms, %.3f ms]", 				\
			(tu-tq)*1e-6, (ts-tq)*1e-6, (te-ts)*1e-6);			\
	} while(0)


namespace hdr
{
typedef unsigned char uchar;

typedef struct {
	int x, y;
} int2;

typedef struct {
	uchar* data;
	size_t width, height;
} Image;

typedef struct {
	float x;
	float y;
	float z;
} float3;

typedef enum {
	METHOD_NONE = 0,
	METHOD_REFERENCE,
	METHOD_OPENCL,
	METHOD_LAST
} method_t;

class Filter {
public:
	typedef struct _Params_ {
		cl_device_type type;
		cl_uint platformIndex, deviceIndex;
		bool opengl, verify;
		_Params_() {
			type = CL_DEVICE_TYPE_ALL;
			opengl = false;
			deviceIndex = 0;
			platformIndex = 0;
			verify = false;
		}
	} Params;

public:
	Filter();
	virtual ~Filter();

	virtual void clearReferenceCache();
	virtual const char* getName() const;

	//initialise OpenCL kernels and memory objects and anything else which remains the same for each frame in the stream
	virtual bool setupOpenCL(cl_context_properties context_prop[], const Params& params) = 0;
	//acquire OpenGL objects, execute kernels and release the objects again
	virtual bool runOpenCL(std::vector<bool> whichKernelsToRun, bool recomputeMapping=true);
	//transfer data from the input to the GPU, execute requested kernels and read the output from the GPU
	virtual bool runOpenCL(uchar* input, uchar* output,
		std::vector<bool> whichKernelsToRun, bool recomputeMapping=true, bool verifyOutput=false);
	//execute the OpenCL kernels for which the flag is set (all by default)
	virtual double runCLKernels(std::vector<bool> whichKernelsToRun, bool recomputeMapping) = 0;
	//release all the kernels and memory objects
	virtual bool cleanupOpenCL() = 0;

	virtual bool runReference(uchar* input, uchar* output) = 0;

	//compute kernel sizes depending on the hardware being used
	virtual bool kernel1DSizes(const char* kernel_name);
	virtual bool kernel2DSizes(const char* kernel_name);

	//set image properties
	virtual void setImageSize(int width, int height);
	virtual void setImageTextures(GLuint input_texture, GLuint output_texture);

	virtual void setStatusCallback(int (*callback)(const char*, va_list args));

protected:
	const char *m_name;
	Image m_reference;
	int (*m_statusCallback)(const char*, va_list args);
	void reportStatus(const char *format, ...) const;
	virtual bool verify(uchar* input, uchar* output, float tolerance=1.f, float maxErrorPercent=0.05);

	cl_device_id m_device;
	cl_context m_clContext;
	cl_command_queue m_queue;
	cl_program m_program;
	cl_mem mem_images[2];

	size_t max_cu;	//max compute units

	std::map<std::string, cl_mem> mems;
	std::map<std::string, cl_kernel> kernels;
	std::map<std::string, size_t*> local_sizes;
	std::map<std::string, size_t*> global_sizes;

	int2 img_size;
	GLuint in_tex;
	GLuint out_tex;

	bool initCL(cl_context_properties context_prop[], const Params& params, const char *source, const char *options);
	cl_int releaseCL();
};

//timing utils
double getCurrentTime();

//image utils
float* mipmap(float* input, int2 input_size, int level=1);
float clamp(float x, float min, float max);
float getPixelLuminance(uchar* image, int2 image_size, int2 pixel_pos);
float getValue(float* data, int2 size, int2 pos);
float getPixel(uchar* data, int2 img_size, int2 pixel_pos, int c);
void setPixel(uchar* data, int2 img_size, int2 pixel_pos, int c, float value);

//pixel conversion
float3 RGBtoHSV(float3 rgb);
float3 HSVtoRGB(float3 hsv);
float3 RGBtoXYZ(float3 rgb);
float3 XYZtoRGB(float3 xyz);

}
