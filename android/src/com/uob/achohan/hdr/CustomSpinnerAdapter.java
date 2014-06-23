package com.uob.achohan.hdr;

import java.util.List;

import android.app.Activity;
import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;

public class CustomSpinnerAdapter extends ArrayAdapter<String> {

	Context context;
	int layoutResID;
	String[] spinnerData;
	
	public CustomSpinnerAdapter(Context context, int layoutResourceID, String[] spinnerDataList) {
		super(context, layoutResourceID, spinnerDataList);
		
		this.context=context;
		this.layoutResID=layoutResourceID;
		this.spinnerData=spinnerDataList;
	}

	@Override
	public View getDropDownView(int position, View convertView, ViewGroup parent) {
		// TODO Auto-generated method stub
		return getCustomView(position, convertView, parent);
	}

	@Override
	public View getView(int position, View convertView, ViewGroup parent) {
		// TODO Auto-generated method stub
		return getCustomView(position, convertView, parent);
	}


	public View getCustomView(int position, View convertView, ViewGroup parent) {
		View row=convertView;
		TextView optionName;
		
		if(row==null) {
			LayoutInflater inflater=((Activity)context).getLayoutInflater();
			row=inflater.inflate(layoutResID, parent, false);

			optionName = (TextView)row.findViewById(R.id.text_main_name);
			row.setTag(optionName);
		}
		else optionName = (TextView)row.getTag();
		
		String spinnerItem = spinnerData[position];
		optionName.setText(spinnerItem);
		
		return row;
	}

}
