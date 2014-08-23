// hdr.cpp (HDR)
// Copyright (c) 2014, Amir Chohan,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <exception>
#include <stdexcept>
#include <sys/stat.h>
#include <stdlib.h>
#include <sys/types.h>
#include <dirent.h>

#include <jpeglib.h>
#include <SDL2/SDL_image.h>

#include "ReinhardGlobal.h"
#include "ReinhardLocal.h"
#include "GradDom.h"
#include "HistEq.h"

#define PIXEL_RANGE 255
#define NUM_CHANNELS 4

using namespace hdr;
using namespace std;

struct _options_ {
	map<string, Filter*> filters;
	map<string, unsigned> methods;

	_options_() {
		filters["histEq"] = new HistEq();
		filters["reinhardGlobal"] = new ReinhardGlobal();
		filters["reinhardLocal"] = new ReinhardLocal();
		filters["gradDom"] = new GradDom();

		methods["reference"] = METHOD_REFERENCE;
		methods["opencl"] = METHOD_OPENCL;
	}
} Options;


void clinfo();
void printUsage();
int updateStatus(const char *format, va_list args);
void checkError(const char* message, int err);
bool is_dir(const char* path);
bool hasEnding (string const &fullString, string const &ending);
Image readJPG(const char* filePath);
void writeJPG(Image &image, const char* filePath);


int main(int argc, char *argv[]) {
	Filter::Params params;
	Filter *filter = NULL;
	unsigned method = METHOD_NONE;
	bool recomputeMapping = false;
	bool verifyOutput = false;
	std::string image_path("../test_images/lena-300x300.jpg");
	std::vector<bool> whichKernelsToRun(1+MAX_NUM_KERNELS, true); // kernel indexing starts from 1

	// Parse arguments
	for (int i = 1; i < argc; i++) {
		if (!filter && (Options.filters.find(argv[i]) != Options.filters.end())) {		//tonemap filter
			filter = Options.filters[argv[i]];
		}
		else if (method == METHOD_NONE && Options.methods.find(argv[i]) != Options.methods.end()) {
			method = Options.methods[argv[i]];		//implementation method
		}
		else if (!strcmp(argv[i], "-cldevice")) {	//run on the given device
			++i;
			if (i >= argc) {
				cout << "Platform/device index required with -cldevice." << endl;
				exit(1);
			}
			char *next;
			params.platformIndex = strtoul(argv[i], &next, 10);
			if (strlen(next) == 0 || next[0] != ':') {
				cout << "Invalid platform/device index." << endl;
				exit(1);
			}
			params.deviceIndex = strtoul(++next, &next, 10);
			if (strlen(next) != 0) {
				cout << "Invalid platform/device index." << endl;
				exit(1);
			}
		}
		else if (!strcmp(argv[i], "-kernels")) {	// select which kernels to run
			whichKernelsToRun.assign(1+MAX_NUM_KERNELS, false); // reset if the flag is provided
			++i;
			if (i >= argc) {
				cout << "Comma-separated list of kernel indices required with -kernels." << endl;
				exit(1);
			}
			unsigned long kernel_idx;
			char *next_idx = argv[i];
			while (kernel_idx = strtoul(next_idx, &next_idx, 10)) {
				if (kernel_idx <= MAX_NUM_KERNELS) { // last valid vector element has index MAX_NUM_KERNELS
					whichKernelsToRun[kernel_idx] = true;
					if (*next_idx == ',') {
						++next_idx;
					}
				} else {
					cout << "Invalid kernel index." << endl;
					exit(1);
				}
			}
		}
		else if (!strcmp(argv[i], "-image")) {	//apply filter on the given image
			++i;
			if (i >= argc) {
				cout << "Invalid image path with -image." << endl;
				exit(1);
			}
			image_path = argv[i];
		}		
		else if (!strcmp(argv[i], "-clinfo")) {
			clinfo();
			exit(0);
		}
		else if (!strcmp(argv[i], "-verify")) {
			verifyOutput = true;
		}
		else if (!strcmp(argv[i], "-remap")) {
			recomputeMapping = true;
		}
	}

	if (filter == NULL || method == METHOD_NONE) {	//invalid arguments
		printUsage();
		exit(1);
	}

	Image input = readJPG(image_path.c_str());

	// Run filter
	filter->setStatusCallback(updateStatus);
	Image output = {(uchar*) calloc(input.width*input.height*NUM_CHANNELS, sizeof(uchar)), input.width, input.height};

	std::cout << "--------------------------------Tonemapping using " << filter->getName() << std::endl;

	filter->setImageSize(input.width, input.height);
	switch (method)
	{
		case METHOD_REFERENCE:
			filter->runReference(input.data, output.data);
			break;
		case METHOD_OPENCL:
			filter->setupOpenCL(NULL, params);
			filter->runOpenCL(input.data, output.data, whichKernelsToRun, recomputeMapping, verifyOutput);
			filter->cleanupOpenCL();
			break;
		default:
			assert(false && "Invalid method.");
	}


	//Save the file
	int lastindex;
	if (is_dir(image_path.c_str()))	image_path = image_path.substr(0, image_path.find_last_of("/"));
	else image_path = image_path.substr(0, image_path.find_last_of("."));

	string image_name = image_path.substr(image_path.find_last_of("/")+1, 100);
	string output_path = "../output_images/" + image_name + "_";
	output_path = output_path + filter->getName() + ".jpg";

	writeJPG(output, output_path.c_str());

	return 0;
}

Image readJPG(const char* filePath) {
	SDL_Surface *input = IMG_Load(filePath);
	if (!input) throw std::runtime_error("Problem opening input file");
 
 	uchar* udata = (uchar*) input->pixels;
  	uchar* data = (uchar*) calloc(NUM_CHANNELS*(input->w * input->h), sizeof(uchar));

	for (int y = 0; y < input->h; y++) {
		for (int x = 0; x < input->w; x++) {
			for (int j=0; j<3; j++) {
 		 		data[(x + y*input->w)*NUM_CHANNELS + j] = (uchar)udata[(x + y*input->w)*3 + j];
 		 	}
 		 	data[(x + y*input->w)*NUM_CHANNELS + 3] = 0;
 		 }
 	}

 	Image image = {data, input->w, input->h};

 	free(input);

	return image;
}

void writeJPG(Image &img, const char* filePath) {
	FILE *outfile  = fopen(filePath, "wb");

	if (!outfile) throw std::runtime_error("Problem opening output file");
 
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr       jerr;
 
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, outfile);

	cinfo.image_width      = img.width;
	cinfo.image_height     = img.height;
	cinfo.input_components = 3;
	cinfo.in_color_space   = JCS_RGB;

	jpeg_set_defaults(&cinfo);
	/*set the quality [0..100]  */
	jpeg_set_quality (&cinfo, 75, true);
	jpeg_start_compress(&cinfo, true);

	uchar* charImageData = (uchar*) calloc(3*img.width*img.height, sizeof(uchar));
	int2 pos;
	int2 img_size = (int2){img.width, img.height};
	for (pos.y = 0; pos.y < img.height; pos.y++) {
		for (pos.x = 0; pos.x < img.width; pos.x++) {
			for (int i=0; i < 3; i++)
				charImageData[(pos.x + pos.y*img.width)*3 + i] = getPixel(img.data, img_size, pos, i);
		}
	}

	JSAMPROW row_pointer;          /* pointer to a single row */
 	while (cinfo.next_scanline < cinfo.image_height) {
		row_pointer = (JSAMPROW) &charImageData[cinfo.next_scanline*cinfo.input_components*img.width];
		jpeg_write_scanlines(&cinfo, &row_pointer, 1);
	} 
	jpeg_finish_compress(&cinfo);
}



void clinfo() {
#define MAX_PLATFORMS 8
#define MAX_DEVICES   8
#define MAX_NAME    256
	cl_uint numPlatforms, numDevices;
	cl_platform_id platforms[MAX_PLATFORMS];
	cl_device_id devices[MAX_DEVICES];
	char name[MAX_NAME];
	cl_int err;

	err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &numPlatforms);
	checkError("Error retrieving platforms\n", err);
	if (numPlatforms == 0) {
		cout << "No platforms found." << endl;
		return;
	}

	for (int p = 0; p < numPlatforms; p++) {
		clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, MAX_NAME, name, NULL);
		cout << endl << "Platform " << p << ": " << name << endl;

		err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
						 MAX_DEVICES, devices, &numDevices);
		checkError("Error retrieving devices\n", err);

		if (numDevices == 0) {
			cout << "No devices found." << endl;
			continue;
		}
		for (int d = 0; d < numDevices; d++) {
			clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_NAME, name, NULL);
			cout << "-> Device " << d << ": " << name << endl;
		}
	}
	cout << endl;
}


void printUsage() {
	cout << endl << "Usage: hdr FILTER METHOD [-image PATH] [-cldevice P:D] [-kernels index0,index1,...] [-verify]";
	cout << endl << "       hdr -clinfo" << endl;

	cout << endl << "Where FILTER is one of:" << endl;
	map<string, Filter*>::iterator fItr;
	for (fItr = Options.filters.begin(); fItr != Options.filters.end(); fItr++) {
		cout << "\t" << fItr->first << endl;
	}

	cout << endl << "Where METHOD is one of:" << endl;
	map<string, unsigned int>::iterator mItr;
	for (mItr = Options.methods.begin(); mItr != Options.methods.end(); mItr++) {
		cout << "\t" << mItr->first << endl;
	}

	cout << '\n'
	<< "If specifying an OpenCL device with \'-cldevice\',\n"
	<< "P and D correspond to the platform and device\n"
	<< "indices reported by running \'-clinfo\'."
	<< endl;

	cout << '\n'
	<< "When specifying which OpenCL kernels to run with\n"
	<< "\'-kernels\', kernels are still run in program order,\n"
	<< "not in order their indices are listed. Unless all\n"
	<< "the kernels all listed, verification is likely to fail."
	<< endl;

	cout << '\n'
	<< "Verification of output against reference implementation\n"
	<< "is performed only when \'-verify\' is specified."
	<< endl;

	cout << endl;
}


int updateStatus(const char *format, va_list args) {
	vprintf(format, args);
	printf("\n");
	return 0;
}

void checkError(const char* message, int err) {
	if (err != CL_SUCCESS) {
		printf("%s %d\n", message, err);
		exit(1);
	}
}

bool is_dir(const char* path) {
	struct stat buf;
	stat(path, &buf);
	return S_ISDIR(buf.st_mode);
}

bool hasEnding (string const &fullString, string const &ending) {
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
	} else {
		return false;
	}
}
