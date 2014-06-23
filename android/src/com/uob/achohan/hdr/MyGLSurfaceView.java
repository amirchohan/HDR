// MyGLSurfaceView.java (HDR)
// Copyright (c) 2014, Amir Chohan,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.

package com.uob.achohan.hdr;

import android.content.Context;
import android.graphics.Point;
import android.hardware.Camera.Size;
import android.opengl.GLSurfaceView;
import android.util.AttributeSet;
import android.view.SurfaceHolder;

/**
 * A view container where OpenGL ES graphics can be drawn on screen.
 * This view can also be used to capture touch events, such as a user
 * interacting with drawn objects.
 */
public class MyGLSurfaceView extends GLSurfaceView {

	private final MyGLRenderer mRenderer;

	public MyGLSurfaceView(Context context, AttributeSet attrs) {
	   super(context, attrs);
		mRenderer = new MyGLRenderer(this);
		setEGLContextClientVersion (2);
		setRenderer(mRenderer);
		setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
	}

	public void surfaceCreated ( SurfaceHolder holder ) {
		super.surfaceCreated ( holder );
	}

	public void surfaceDestroyed ( SurfaceHolder holder ) {
		mRenderer.close();
		super.surfaceDestroyed ( holder );
	}

	public void surfaceChanged ( SurfaceHolder holder, int format, int w, int h ) {
		super.surfaceChanged ( holder, format, w, h );
	}

	public void setDisplayDim(Point displayDim) {
		mRenderer.setDisplayDim(displayDim);
	}

	public void setHDR(int option) {
		switch(option) {
			case 0:
				mRenderer.setHDR(true);
				break;
			case 1:
				mRenderer.setHDR(false);
				break;
		}
	}
}
