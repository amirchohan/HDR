// ReinhardLocal.h (HDR)
// Copyright (c) 2014, Amir Chohan,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.

#include "Filter.h"

namespace hdr
{
class ReinhardLocal : public Filter {
public:
	ReinhardLocal(float _key=0.18f, float _sat=1.6f, float _epsilon=0.05, float _phi=8.0);

	virtual bool setupOpenCL(cl_context_properties context_prop[], const Params& params);
	virtual double runCLKernels(bool recomputeMapping);
	virtual bool cleanupOpenCL();
	virtual bool runReference(uchar* input, uchar* output);

protected:
	float key;	//increase this to allow for more contrast in the darker regions
	float sat;	//increase this for more colourful pictures
	float epsilon;	//serves as an edge enhancing parameter
	float phi;	//similar to epsilon, however the effects are only noticeable on smaller scales

	//information regarding all mipmap levels
	int num_mipmaps;
	int* m_width;		//at index i this contains the width of the mipmap at index i
	int* m_height;		//at index i this contains the height of the mipmap at index i
	int* m_offset;		//at index i this contains the start point to store the mipmap at level i

};
}