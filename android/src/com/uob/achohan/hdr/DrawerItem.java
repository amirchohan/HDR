package com.uob.achohan.hdr;

public class DrawerItem {

	String item;
	String[] options;

	public DrawerItem(String itemName, String[] choices) {
		item = itemName;
		options = choices;
	}

	public String getItemName() {
		return item;
	}

	public String[] getOptions() {
		return options;
	}
}
