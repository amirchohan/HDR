// HDR.java (HDR)
// Copyright (c) 2014, Amir Chohan,
// University of Bristol. All rights reserved.
//
// This program is provided under a three-clause BSD license. For full
// license terms please see the LICENSE file distributed with this
// source code.

package com.uob.achohan.hdr;

import java.util.ArrayList;
import java.util.List;

import android.app.Activity;
import android.content.Context;
import android.hardware.Camera;
import android.graphics.Point;
import android.hardware.Camera.Size;
import android.os.Bundle;
import android.os.PowerManager;
import android.os.PowerManager.WakeLock;
import android.support.v4.widget.DrawerLayout;
import android.util.Log;
import android.view.Display;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.Toast;

public class HDR extends Activity {

	private MyGLSurfaceView mView;
	private WakeLock mWL;

	private final String TAG = "hdr";

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		mWL = ((PowerManager)getSystemService (Context.POWER_SERVICE)).newWakeLock(PowerManager.FULL_WAKE_LOCK, "WakeLock");
		mWL.acquire();
		setContentView(R.layout.main);

		Display display = getWindowManager().getDefaultDisplay();
		Point displayDim = new Point();
		display.getSize(displayDim);

		final DrawerLayout drawer = (DrawerLayout)findViewById(R.id.drawer_layout);
		final ListView navList = (ListView) findViewById(R.id.drawer);
		List<DrawerItem> dataList = new ArrayList<DrawerItem>();

		dataList.add(new DrawerItem("HDR", new String[]{"On", "Off"}));

		CustomDrawerAdapter adapter = new CustomDrawerAdapter(this, R.layout.custom_drawer_item, dataList);
		navList.setAdapter(adapter);
		navList.setOnItemClickListener(new ListView.OnItemClickListener(){
			@Override
			public void onItemClick(AdapterView<?> parent, View view, final int pos,long id){
				drawer.setDrawerListener( new DrawerLayout.SimpleDrawerListener(){
					@Override
					public void onDrawerClosed(View drawerView){
						super.onDrawerClosed(drawerView);
					}
				});
				drawer.closeDrawer(navList);
			}
    	});

		mView = (MyGLSurfaceView) findViewById(R.id.surfaceviewclass);
		mView.setDisplayDim(displayDim);
	}
	
	@Override
	protected void onPause() {
		if ( mWL.isHeld() )	mWL.release();
		mView.onPause();
		super.onPause();
	}
			
	@Override
	protected void onResume() {
		super.onResume();
		mView.onResume();
		mWL.acquire();
	}


	public void changeConfig(int item, int option) {
		Log.d(TAG, "item " + item + " pos " + option);
		if (item == 0) mView.setHDR(option);
	}
}