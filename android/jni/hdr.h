// hdr.h (HDR)
// Copyright (c) 2014, Amir Chohan,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.


#ifndef JNIAPI_H
#define JNIAPI_H

#include <pthread.h>
#include <EGL/egl.h> // requires ndk r5 or newer
#include <GLES/gl.h>

#include <CL/cl.h>
#include <CL/cl_gl.h>

#include "HistEq.h"
#include "ReinhardLocal.h"
#include "ReinhardGlobal.h"
#include "GradDom.h"

using namespace hdr;

extern "C" {
	// Variadic argument wrapper for updateStatus
	void status(const char *fmt, ...);
	int updateStatus(const char *format, va_list args);

	Filter* filter;
	Filter::Params params;

    cl_context_properties cl_prop[7];

	JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_MyGLRenderer_initCL(JNIEnv* jenv, jobject obj, jint width, jint height, jint in_tex, jint out_tex);
	JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_MyGLRenderer_processFrame(JNIEnv* jenv, jobject obj, jboolean recomputeMapping);
	JNIEXPORT void JNICALL Java_com_uob_achohan_hdr_MyGLRenderer_killCL(JNIEnv* jenv, jobject obj);

};

#endif // JNIAPI_H