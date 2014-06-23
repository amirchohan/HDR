# Android.mk (HDR)
# Copyright (c) 2014, Amir Chohan,
# University of Bristol. All rights reserved.
#
# This program is provided under a three-clause BSD license. For full
# license terms please see the LICENSE file distributed with this
# source code.

LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

SRC_PATH := ../../src

LOCAL_CFLAGS    += -I$(SRC_PATH) -g -Wno-deprecated-declarations
LOCAL_CFLAGS    += -DSHOW_REFERENCE_PROGRESS=1
LOCAL_CFLAGS += -fopenmp
LOCAL_LDFLAGS += -fopenmp
LOCAL_MODULE    := hdr
LOCAL_SRC_FILES := hdr.cpp \
	$(SRC_PATH)/Filter.cpp \
	$(SRC_PATH)/HistEq.cpp \
	$(SRC_PATH)/GradDom.cpp \
	$(SRC_PATH)/ReinhardLocal.cpp \
	$(SRC_PATH)/ReinhardGlobal.cpp

LOCAL_LDLIBS := -landroid -llog -ljnigraphics -lOpenCL -lEGL -lGLESv2
LOCAL_STATIC_LIBRARIES := android_native_app_glue

include $(BUILD_SHARED_LIBRARY)
$(call import-module,android/native_app_glue)