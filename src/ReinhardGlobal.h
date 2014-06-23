// ReinhardGlobal.h (HDR)
// Copyright (c) 2014, Amir Chohan,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.

#include "Filter.h"

namespace hdr
{
class ReinhardGlobal : public Filter {
public:
	ReinhardGlobal(float _key=0.18f, float _sat=1.6f);

	virtual bool setupOpenCL(cl_context_properties context_prop[], const Params& params);
	virtual double runCLKernels(bool recomputeMapping);
	virtual bool cleanupOpenCL();
	virtual bool runReference(uchar* input, uchar* output);

protected:
	float key;	//increase this to allow for more contrast in the darker regions
	float sat;	//increase this for more colourful pictures
};
}