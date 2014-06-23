#!/bin/bash

# Any subsequent commands which fail will cause the shell script to exit immediately
set -e

#create a directory to store the ndk and sdk
sudo mkdir /opt/android/

#create a working directory
mkdir tmp_hdr
cd tmp_hdr

#download and extract the latest version of Android NDK
wget "$(wget -q -O - http://developer.android.com/tools/sdk/ndk/index.html | egrep 'linux-x86_64' | egrep -o "(http(s)?://){1}[^'\"]+")"
sudo tar vxjf android-ndk* -C /opt/android/
sudo mv /opt/android/android-ndk* /opt/android/ndk

#download and extract the latest version of Android SDK
wget "$(wget -q -O - https://developer.android.com/sdk/index.html#download | egrep 'linux-x86_64' | egrep -o "(http(s)?://){1}[^'\"]+")"
sudo unzip adt-bundle* adt-bundle*/sdk* -d /opt/android/
sudo mv /opt/android/adt-bundle*/sdk /opt/android/sdk

#delete the working directory
cd ..
rm -r tmp_hdr

#set the environment variables
echo PATH=\"$PATH:/opt/android/sdk/platform-tools:/opt/android/sdk/tools:/opt/android/ndk\" | sudo tee /etc/environment

#install ant
sudo apt-get install ant

#copy the OpenCL library
sudo cp ../jni/libOpenCL.so /opt/android/ndk/platforms/
for x in $(ls -l | grep ^d | awk '{print $9}');
	do sudo ln -s /opt/android/ndk/platforms/libOpenCL.so /opt/android/ndk/platforms/$x/arch-arm/usr/lib/libOpenCL.so ;
done;