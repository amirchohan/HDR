package com.uob.achohan.hdr;

import java.util.ArrayList;
import java.util.List;


import android.app.Activity;
import android.content.Context;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.AdapterView.OnItemSelectedListener;

public class CustomDrawerAdapter extends ArrayAdapter<DrawerItem> {

	Context context;
	List<DrawerItem> drawerItemList;
	int layoutResID;

	public CustomDrawerAdapter(Context context, int layoutResourceID, List<DrawerItem> listItems) {
		super(context, layoutResourceID, listItems);
		this.context = context;
		this.drawerItemList = listItems;
		this.layoutResID = layoutResourceID;
	}

	@Override
	public View getView(final int position, View convertView, ViewGroup parent) {
		// TODO Auto-generated method stub

		DrawerItemHolder drawerHolder;
		View view = convertView;

		if (view == null) {
			LayoutInflater inflater = ((Activity) context).getLayoutInflater();
			drawerHolder = new DrawerItemHolder();
			view = inflater.inflate(layoutResID, parent, false);

			drawerHolder.spinnerLayout = (LinearLayout) view.findViewById(R.layout.custom_drawer_item);

			drawerHolder.itemName = (TextView) view.findViewById(R.id.drawerText);
			drawerHolder.spinner = (Spinner) view.findViewById(R.id.drawerSpinner);

			view.setTag(drawerHolder);
		} else drawerHolder = (DrawerItemHolder) view.getTag();

		DrawerItem dItem = (DrawerItem) this.drawerItemList.get(position);

		CustomSpinnerAdapter adapter = new CustomSpinnerAdapter(context, R.layout.custom_spinner_item, dItem.getOptions());
		drawerHolder.spinner.setAdapter(adapter);
		drawerHolder.itemName.setText(dItem.getItemName());
		drawerHolder.spinner.setOnItemSelectedListener(new OnItemSelectedListener() {
			@Override
			public void onItemSelected(AdapterView<?> arg0,	View arg1, int spinner_item, long arg3) {
				selectedItem(position, spinner_item);
			}
			@Override
			public void onNothingSelected(AdapterView<?> arg0) {
			}
		});
		setDefaultValues(drawerHolder.spinner, position);

		return view;
	}

	private void selectedItem(int property_modified, int chosen_option) {
		((HDR) context).changeConfig(property_modified, chosen_option);
	}

	private void setDefaultValues(Spinner spinner, int position) {
		if (position == 0) spinner.setSelection(0);
		if (position == 1) spinner.setSelection(0);
		if (position == 2) spinner.setSelection(0);
		if (position == 3) spinner.setSelection(0);
		if (position == 4) spinner.setSelection(2);
	}

	private static class DrawerItemHolder {
		TextView itemName;
		LinearLayout spinnerLayout;
		Spinner spinner;
	}
}