// MyGLRenderer.java (HDR)
// Copyright (c) 2014, Amir Chohan,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.

package com.uob.achohan.hdr;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.List;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;
import javax.microedition.khronos.egl.EGLSurface;
import javax.microedition.khronos.opengles.GL10;

import android.graphics.Point;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.Camera.Size;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.util.Log;

/**
 * Provides drawing instructions for a GLSurfaceView object. This class
 * must override the OpenGL ES drawing lifecycle methods:
 * <ul>
 *   <li>{@link android.opengl.GLSurfaceView.Renderer#onSurfaceCreated}</li>
 *   <li>{@link android.opengl.GLSurfaceView.Renderer#onDrawFrame}</li>
 *   <li>{@link android.opengl.GLSurfaceView.Renderer#onSurfaceChanged}</li>
 * </ul>
 */
public class MyGLRenderer implements GLSurfaceView.Renderer, SurfaceTexture.OnFrameAvailableListener {

	private final String TAG = "hdr";
	
	private final String vss =
		"attribute vec2 vPosition;\n" +
		"attribute vec2 vTexCoord;\n" +
		"varying vec2 texCoord;\n" +
		"void main() {\n" +
		"  texCoord = vTexCoord;\n" +
		"  gl_Position = vec4 ( vPosition.x, vPosition.y, 0.0, 1.0 );\n" +
		"}";

	private final String camera_fss =
		"#extension GL_OES_EGL_image_external : require\n" +
		"precision mediump float;\n" +
		"uniform samplerExternalOES sTexture;\n" +
		"varying vec2 texCoord;\n" +
		"void main() {\n" +
		"  gl_FragColor = texture2D(sTexture, texCoord);\n" +
		"  //gl_FragColor = vec4(0, 0.75, 0, 1);\n" +
		"}";

	private final String texture_fss =
		"precision mediump float;\n" +
		"uniform sampler2D sTexture;\n" +
		"varying vec2 texCoord;\n" +
		"void main() {\n" +
		"  gl_FragColor = texture2D(sTexture, texCoord);\n" +
		"}";



	private int[] hTex;
	private int[] glTextures;
	private int display_texture;
	private IntBuffer targetFramebuffer;
	private FloatBuffer vertexCoord;
	private FloatBuffer cameraTexCoord;
	private FloatBuffer openclTexCoord;
	private int hProgram;
	private int displayTextureProgram;

	private Camera mCamera;
	private SurfaceTexture mSTexture;
	private boolean mUpdateST = false;
	private ByteBuffer cameraBuffer;


	private MyGLSurfaceView mView;

	EGL10 mEgl;
	EGLDisplay mEglDisplay;
	EGLSurface mEglSurface;
	EGLConfig mEglConfig;
	EGLContext mEglContext;


	//switch between HDR and Raw camera feed
	private boolean process_HDR;

	//Sizes
	private Point camera_res;	//resolution of the camera image being captured
	private Point display_dim;	//dimensions of the Android display. OpenGL viewport should be set to this

	//information about recomputing intervals
	private float recomputeInterval = 0.5f;	//how often to recompute mappings in seconds
	private long lastRecomputeTime = System.currentTimeMillis();	//last time when mappings were computed
	private boolean recomputeMapping=true;	//whether to compute mappings at next frame or not
	
	//computing FPS
	private int frames = 0;	//number of frames in the second
	private long lastFPScomputeTime = System.nanoTime();	//time since last FPS was computed

	MyGLRenderer (MyGLSurfaceView view) {
		mView = view;
		float[] vertexCoordTmp = {
			-1.0f, -1.0f, 0.0f,
			-1.0f,  1.0f, 0.0f,
			 1.0f, -1.0f, 0.0f,
			 1.0f,  1.0f, 0.0f};
		float[] textureCoordTmp = {
			 0.0f, 1.0f,
			 0.0f, 0.0f,
			 1.0f, 1.0f,
			 1.0f, 0.0f };
		float[] openclCoordTmp = {
			 0.0f, 0.0f,
			 0.0f, 1.0f,
			 1.0f, 0.0f,
			 1.0f, 1.0f
		};

		vertexCoord = ByteBuffer.allocateDirect(12*4).order(ByteOrder.nativeOrder()).asFloatBuffer();
		vertexCoord.put(vertexCoordTmp);
		vertexCoord.position(0);
		cameraTexCoord = ByteBuffer.allocateDirect(8*4).order(ByteOrder.nativeOrder()).asFloatBuffer();
		cameraTexCoord.put(textureCoordTmp);
		cameraTexCoord.position(0);
		openclTexCoord = ByteBuffer.allocateDirect(8*4).order(ByteOrder.nativeOrder()).asFloatBuffer();
		openclTexCoord.put(openclCoordTmp);
		openclTexCoord.position(0);
	}


	public void onSurfaceCreated (GL10 unused, EGLConfig config) {
		mCamera = Camera.open();

		Camera.Parameters param = mCamera.getParameters();
		/*List<Size> psize = param.getSupportedPictureSizes();	//get supported picture sizes
		if ( psize.size() > 0 ) {
			for (int i = 0; i < psize.size(); i++ ) {
				Log.d(TAG, psize.get(i).width + "x" + psize.get(i).height);
			}
		}*/
		camera_res = new Point(1280, 720);
		param.setPictureSize(camera_res.x, camera_res.y);
		mCamera.setParameters(param);

		Log.d(TAG, "Camera Resoultion: " + mCamera.getParameters().getPictureSize().width + "x" + mCamera.getParameters().getPictureSize().height);

		initTex();

		mSTexture = new SurfaceTexture (hTex[0]);
		mSTexture.setOnFrameAvailableListener(this);
		try {
			mCamera.setPreviewTexture(mSTexture);
		}
		catch ( IOException ioe ) {
		}

		initCL(camera_res.x, camera_res.y, glTextures[0], glTextures[1]);

		GLES20.glClearColor (1.0f, 1.0f, 0.0f, 1.0f);
		hProgram = loadShader(vss, camera_fss);
		displayTextureProgram = loadShader(vss, texture_fss);

		cameraBuffer = ByteBuffer.allocate(camera_res.y*camera_res.x*4);
	}

	public void onDrawFrame ( GL10 unused ) {
		int displayTexture=0;
		cameraToTexture();
		displayTexture = applyHDRonTexture();
		renderFromTexture(displayTexture);
		logFrame();
	}


	public void onSurfaceChanged ( GL10 unused, int width, int height ) {
		GLES20.glViewport( 0, 0, width, height);
		Camera.Parameters param = mCamera.getParameters();
		List<Size> psize = param.getSupportedPreviewSizes();
		if ( psize.size() > 0 ) {
			int i;
			for ( i = 0; i < psize.size(); i++ ) {
				if ( psize.get(i).width < width || psize.get(i).height < height ) break;
			}
			if ( i > 0 ) i--;
			param.setPreviewSize(psize.get(i).width, psize.get(i).height);
		}
		mCamera.setParameters(param);
		mCamera.startPreview();
	}


	//converts information from the Camera Texture to an OpenGL 2D texture
	private int cameraToTexture() {
		GLES20.glClear( GLES20.GL_COLOR_BUFFER_BIT );
		
		GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, targetFramebuffer.get(0));
		int fbret = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
		if (fbret != GLES20.GL_FRAMEBUFFER_COMPLETE) {
		  Log.d(TAG, "unable to bind fbo" + fbret);
		}
		GLES20.glViewport(0, 0, camera_res.x, camera_res.y);
		GLES20.glClear( GLES20.GL_COLOR_BUFFER_BIT );

		synchronized(this) {
			if (mUpdateST) {
				mSTexture.updateTexImage();
				mUpdateST = false;
			}
		}

		GLES20.glUseProgram(hProgram);

		int ph = GLES20.glGetAttribLocation(hProgram, "vPosition");
		int tch = GLES20.glGetAttribLocation (hProgram, "vTexCoord");
		int th = GLES20.glGetUniformLocation (hProgram, "sTexture");

		GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
		GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, hTex[0]);
		GLES20.glUniform1i(th, 0);

		GLES20.glVertexAttribPointer(ph, 2, GLES20.GL_FLOAT, false, 4*3, vertexCoord);
		GLES20.glVertexAttribPointer(tch, 2, GLES20.GL_FLOAT, false, 4*2, cameraTexCoord);
		GLES20.glEnableVertexAttribArray(ph);
		GLES20.glEnableVertexAttribArray(tch);

		GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
		GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, glTextures[0]);
		GLES20.glGenerateMipmap(GLES20.GL_TEXTURE_2D);
		GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);

		GLES20.glFinish();

		return glTextures[0];
	}

	//applies HDR on the texture
	private int applyHDRonTexture() {
		if (process_HDR) {
			processFrame(recomputeMapping);
			return glTextures[1];
		}
		return glTextures[0];
	}

	//displays the result of OpenCL filters
	private void renderFromTexture(int displayTexture) {
		GLES20.glViewport(0, 0, display_dim.x, display_dim.y);

		GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
		GLES20.glUseProgram(displayTextureProgram);

		int ph = GLES20.glGetAttribLocation(displayTextureProgram, "vPosition");
		int tch = GLES20.glGetAttribLocation(displayTextureProgram, "vTexCoord");
		int th = GLES20.glGetUniformLocation(displayTextureProgram, "sTexture");

		GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
		GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, displayTexture);
		GLES20.glUniform1i(th, 0);

		GLES20.glVertexAttribPointer(ph, 2, GLES20.GL_FLOAT, false, 4*3, vertexCoord);
		GLES20.glVertexAttribPointer(tch, 2, GLES20.GL_FLOAT, true, 4*2, openclTexCoord);
		GLES20.glEnableVertexAttribArray(ph);
		GLES20.glEnableVertexAttribArray(tch);
		GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
		GLES20.glFinish();
	}

	//initialise all the required textures
	private void initTex() {
		hTex = new int[1];
		glTextures = new int[2];
		GLES20.glGenTextures ( 1, hTex, 0 );
		GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, hTex[0]);
		GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
		GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
		GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
		GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);

		GLES20.glGenTextures ( 2, glTextures, 0 );
		GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, glTextures[0]);
		GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MIN_FILTER,GLES20.GL_LINEAR_MIPMAP_NEAREST);
		GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MAG_FILTER,GLES20.GL_LINEAR);
		GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, camera_res.x, camera_res.y, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
		GLES20.glGenerateMipmap(GLES20.GL_TEXTURE_2D);
		GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);

		GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, glTextures[1]);
		GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MIN_FILTER,GLES20.GL_LINEAR);
		GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MAG_FILTER,GLES20.GL_LINEAR);
		GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, camera_res.x, camera_res.y, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
		GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);
		
		targetFramebuffer = IntBuffer.allocate(1);
		GLES20.glGenFramebuffers(1, targetFramebuffer);
		GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, targetFramebuffer.get(0));
		GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, glTextures[0], 0);
		GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
	}


	public synchronized void onFrameAvailable ( SurfaceTexture st ) {
		mUpdateST = true;
		mView.requestRender();
	}

	private static int loadShader ( String vertex_shader, String fragment_shader ) {
		int vshader = GLES20.glCreateShader(GLES20.GL_VERTEX_SHADER);
		GLES20.glShaderSource(vshader, vertex_shader);
		GLES20.glCompileShader(vshader);
		int[] compiled = new int[1];
		GLES20.glGetShaderiv(vshader, GLES20.GL_COMPILE_STATUS, compiled, 0);
		if (compiled[0] == 0) {
			Log.e("Shader", "Could not compile vshader");
			Log.v("Shader", "Could not compile vshader:"+GLES20.glGetShaderInfoLog(vshader));
			GLES20.glDeleteShader(vshader);
			vshader = 0;
		}

		int fshader = GLES20.glCreateShader(GLES20.GL_FRAGMENT_SHADER);
		GLES20.glShaderSource(fshader, fragment_shader);
		GLES20.glCompileShader(fshader);
		GLES20.glGetShaderiv(fshader, GLES20.GL_COMPILE_STATUS, compiled, 0);
		if (compiled[0] == 0) {
			Log.e("Shader", "Could not compile fshader");
			Log.v("Shader", "Could not compile fshader:"+GLES20.glGetShaderInfoLog(fshader));
			GLES20.glDeleteShader(fshader);
			fshader = 0;
		}

		int program = GLES20.glCreateProgram();
		GLES20.glAttachShader(program, vshader);
		GLES20.glAttachShader(program, fshader);
		GLES20.glLinkProgram(program);
			 
		return program;
	}

	private void deleteTex() {
		GLES20.glDeleteTextures (1, hTex, 0);
	}

	public void close() {
		mUpdateST = false;
		mSTexture.release();
		mCamera.stopPreview();
		mCamera.release();
		killCL();
		deleteTex();
	}


	public void logFrame() {
		//check whether to recompute mappings at next frame or not
		if(System.currentTimeMillis() - lastRecomputeTime >= recomputeInterval*1000) {
			recomputeMapping = true;
			lastRecomputeTime = System.currentTimeMillis();
		}
		else recomputeMapping = false;

		//work out FPS
		frames++;
		if(System.nanoTime() - lastFPScomputeTime >= 1000000000) {
			Log.d(TAG, "FPS: " + frames);
			frames = 0;
			lastFPScomputeTime = System.nanoTime();
		}
	}



	//setters
	public void setDisplayDim(Point displayDim) {
		display_dim = displayDim;
	}
	public void setHDR(boolean process) {
		process_HDR = process;
	}



	/////////////////////////////////////////////////////////////////////////////////////////JNI
	public void updateStatus(String text) {
		Log.d(TAG, text);
	}

	public static native void initCL(int width, int height, int input_texid, int output_texid);
	public static native void processFrame(boolean recomputeMapping);
	public static native void killCL();


	static {
		System.loadLibrary("hdr");
	}
}