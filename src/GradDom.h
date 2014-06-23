// GradDom.h (HDR)
// Copyright (c) 2014, Amir Chohan,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.

#include "Filter.h"

namespace hdr
{
class GradDom : public Filter {
public:
	GradDom(float _adjust_alpha=0.1f, float _beta=0.85f, float _sat=0.5f);

	virtual bool setupOpenCL(cl_context_properties context_prop[], const Params& params);
	virtual double runCLKernels(bool recomputeMapping);
	virtual bool cleanupOpenCL();
	virtual bool runReference(uchar* input, uchar* output);

	//computes the attenuation function for the gradients
	float* attenuate_func(float* lum);
	//solves the poisson equation to derive a new compressed dynamic range
	float* poissonSolver(float* lum, float* div_grad, float terminationCriterea=0.0005);

protected:
	float adjust_alpha;	//to adjust the gradients at each mipmap level. gradients smaller than alpha are slightly magnified
	float beta;	//used to attenuate larger gradients
	float sat;	//increase this for more colourful pictures

	//information regarding all mipmap levels
	int num_mipmaps;
	int* m_width;		//at index i this contains the width of the mipmap at index i
	int* m_height;		//at index i this contains the height of the mipmap at index i
	int* m_offset;		//at index i this contains the start point to store the mipmap at level i
	float* m_divider;		//at index i this contains the value the pixels of gradient magnitude are going to be divided by	

};
}