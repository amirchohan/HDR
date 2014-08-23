const char *histEq_kernel =
"// histEq.cl (HDR)\n"
"// Copyright (c) 2014, Amir Chohan,\n"
"// University of Bristol. All rights reserved.\n"
"//\n"
"// This program is provided under a three-clause BSD license. For full\n"
"// license terms please see the LICENSE file distributed with this\n"
"// source code.\n"
"\n"
"float GL_to_CL(uint val);\n"
"float3 RGBtoHSV(uint4 rgb);\n"
"uint4 HSVtoRGB(float3 hsv);\n"
"\n"
"const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
"\n"
"//use this one for android because android's opencl specification is buggy\n"
"kernel void transfer_data(__read_only image2d_t input_image, __global float* image) {\n"
"	int2 pos;\n"
"	uint4 pixel;\n"
"	for (pos.y = get_global_id(1); pos.y < HEIGHT; pos.y += get_global_size(1)) {\n"
"		for (pos.x = get_global_id(0); pos.x < WIDTH; pos.x += get_global_size(0)) {\n"
"			pixel = read_imageui(input_image, sampler, pos);\n"
"			image[(pos.x + pos.y*WIDTH)*NUM_CHANNELS + 0] = GL_to_CL(pixel.x);\n"
"			image[(pos.x + pos.y*WIDTH)*NUM_CHANNELS + 1] = GL_to_CL(pixel.y);\n"
"			image[(pos.x + pos.y*WIDTH)*NUM_CHANNELS + 2] = GL_to_CL(pixel.z);		\n"
"			image[(pos.x + pos.y*WIDTH)*NUM_CHANNELS + 3] = GL_to_CL(pixel.w);\n"
"		}\n"
"	}\n"
"}\n"
"\n"
"//computes the histogram for brightness\n"
"kernel void partial_hist(__global float* image, __global uint* partial_histogram) {\n"
"	const int global_size = get_global_size(0);\n"
"	const int group_size = get_local_size(0);\n"
"	const int group_id = get_group_id(0);\n"
"	const int lid = get_local_id(0);\n"
"\n"
"	__local uint l_hist[HIST_SIZE];\n"
"	for (int i = lid; i < HIST_SIZE; i+=group_size) {\n"
"		l_hist[i] = 0;\n"
"	}\n"
"\n"
"	int brightness;\n"
"	for (int i = get_global_id(0); i < WIDTH*HEIGHT; i += global_size) {\n"
"		brightness = max(max(image[i*NUM_CHANNELS + 0], image[i*NUM_CHANNELS + 1]), image[i*NUM_CHANNELS + 2]);\n"
"		barrier(CLK_LOCAL_MEM_FENCE);\n"
"		atomic_inc(&l_hist[brightness]);\n"
"	}\n"
"\n"
"	barrier(CLK_LOCAL_MEM_FENCE);\n"
"	for (int i = lid; i < HIST_SIZE; i+=group_size) {\n"
"		partial_histogram[i + group_id * HIST_SIZE] = l_hist[i];\n"
"	}\n"
"}\n"
"\n"
"\n"
"//requires global work group size to be equal to HIST_SIZE\n"
"kernel void merge_hist(	__global uint* partial_histogram,\n"
"						__global uint* histogram,\n"
"						const int num_hists) {	//number of histograms in partial histogram, i.e number of workgroups in previous kernel\n"
"	const int gid = get_global_id(0);\n"
"\n"
"	uint sum = 0;\n"
"	for(uint i = 0; i < num_hists; i++) {\n"
"		sum += partial_histogram[gid + i*HIST_SIZE];\n"
"	}\n"
"\n"
"	histogram[gid] = sum;\n"
"}\n"
"\n"
"//TODO: even though this takes barely anytime at all, could look into parrallel scan in future\n"
"//computes the cdf of the brightness histogram\n"
"kernel void hist_cdf( __global uint* hist) {\n"
"	const int gid = get_global_id(0);\n"
"\n"
"	if (gid==0)\n"
"		for (int i=1; i<HIST_SIZE; i++) {\n"
"			hist[i] += hist[i-1];\n"
"		}\n"
"}\n"
"\n"
"//kernel to perform histogram equalisation using the modified brightness cdf\n"
"kernel void histogram_equalisation(__global float* image, write_only image2d_t output_image, __global uint* brightness_cdf) {\n"
"	int2 pos;\n"
"	uint4 pixel;\n"
"	float3 hsv;\n"
"	for (pos.y = get_global_id(1); pos.y < HEIGHT; pos.y += get_global_size(1)) {\n"
"		for (pos.x = get_global_id(0); pos.x < WIDTH; pos.x += get_global_size(0)) {\n"
"			pixel.x = image[(pos.x + pos.y*WIDTH)*NUM_CHANNELS + 0];\n"
"			pixel.y = image[(pos.x + pos.y*WIDTH)*NUM_CHANNELS + 1];\n"
"			pixel.z = image[(pos.x + pos.y*WIDTH)*NUM_CHANNELS + 2];\n"
"			pixel.w = image[(pos.x + pos.y*WIDTH)*NUM_CHANNELS + 3];\n"
"\n"
"			hsv = RGBtoHSV(pixel);		//Convert to HSV to get Hue and Saturation\n"
"\n"
"			hsv.z = ((HIST_SIZE-1)*(brightness_cdf[(int)hsv.z] - brightness_cdf[0]))\n"
"						/(HEIGHT*WIDTH - brightness_cdf[0]);\n"
"\n"
"			pixel = HSVtoRGB(hsv);	//Convert back to RGB with the modified brightness for V\n"
"\n"
"			write_imageui(output_image, pos, pixel);\n"
"		}\n"
"	}\n"
"}\n"
"\n"
"float3 RGBtoHSV(uint4 rgb) {\n"
"	float r = rgb.x;\n"
"	float g = rgb.y;\n"
"	float b = rgb.z;\n"
"	float rgb_min, rgb_max, delta;\n"
"	rgb_min = min(min(r, g), b);\n"
"	rgb_max = max(max(r, g), b);\n"
"\n"
"	float3 hsv;\n"
"\n"
"	hsv.z = rgb_max;	//Brightness\n"
"	delta = rgb_max - rgb_min;\n"
"	if(rgb_max != 0) hsv.y = delta/rgb_max;//Saturation\n"
"	else {	// r = g = b = 0	//Saturation = 0, Value is undefined\n"
"		hsv.y = 0;\n"
"		hsv.x = -1;\n"
"		return hsv;\n"
"	}\n"
"\n"
"	//Hue\n"
"	if(r == rgb_max) 		hsv.x = (g-b)/delta;\n"
"	else if(g == rgb_max) 	hsv.x = (b-r)/delta + 2;\n"
"	else 			 		hsv.x = (r-g)/delta + 4;\n"
"	hsv.x *= 60;				\n"
"	if( hsv.x < 0 ) hsv.x += 360;\n"
"\n"
"	return hsv;\n"
"}\n"
"\n"
"uint4 HSVtoRGB(float3 hsv) {\n"
"	int i;\n"
"	float h = hsv.x;\n"
"	float s = hsv.y;\n"
"	float v = hsv.z;\n"
"	float f, p, q, t;\n"
"	uint4 rgb;\n"
"	rgb.w = 0;\n"
"	if( s == 0 ) { // achromatic (grey)\n"
"		rgb.x = rgb.y = rgb.z = v;\n"
"		return rgb;\n"
"	}\n"
"	h /= 60;			// sector 0 to 5\n"
"	i = floor( h );\n"
"	f = h - i;			// factorial part of h\n"
"	p = v * ( 1 - s );\n"
"	q = v * ( 1 - s * f );\n"
"	t = v * ( 1 - s * ( 1 - f ) );\n"
"	switch( i ) {\n"
"		case 0:\n"
"			rgb.x = v;\n"
"			rgb.y = t;\n"
"			rgb.z = p;\n"
"			break;\n"
"		case 1:\n"
"			rgb.x = q;\n"
"			rgb.y = v;\n"
"			rgb.z = p;\n"
"			break;\n"
"		case 2:\n"
"			rgb.x = p;\n"
"			rgb.y = v;\n"
"			rgb.z = t;\n"
"			break;\n"
"		case 3:\n"
"			rgb.x = p;\n"
"			rgb.y = q;\n"
"			rgb.z = v;\n"
"			break;\n"
"		case 4:\n"
"			rgb.x = t;\n"
"			rgb.y = p;\n"
"			rgb.z = v;\n"
"			break;\n"
"		default:		// case 5:\n"
"			rgb.x = v;\n"
"			rgb.y = p;\n"
"			rgb.z = q;\n"
"			break;\n"
"	}\n"
"	return rgb;\n"
"}\n"
"\n"
"//a function to read an OpenGL texture pixel when using Snapdragon's Android OpenCL implementation\n"
"float GL_to_CL(uint val) {\n"
"	if (BUGGY_CL_GL) {\n"
"		if (val >= 14340) return round(0.1245790*val - 1658.44);	//>=128\n"
"		if (val >= 13316) return round(0.0622869*val - 765.408);	//>=64\n"
"		if (val >= 12292) return round(0.0311424*val - 350.800);	//>=32\n"
"		if (val >= 11268) return round(0.0155702*val - 159.443);	//>=16\n"
"	\n"
"		float v = (float) val;\n"
"		return round(0.0000000000000125922*pow(v,4.f) - 0.00000000026729*pow(v,3.f) + 0.00000198135*pow(v,2.f) - 0.00496681*v - 0.0000808829);\n"
"	}\n"
"	else return (float)val;\n"
"}\n";