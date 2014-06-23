#!/bin/bash

set -e
# Any subsequent commands which fail will cause the shell script to exit immediately

sudo su <<EOF

# Stop app and clear log
adb shell am force-stop com.uob.achohan.hdr

# Initialize project
android update project -p . -t android-19

# Build and install
ant debug install

adb logcat -c

# Run app (all one line)
adb shell am start -n com.uob.achohan.hdr/com.uob.achohan.hdr.HDR

# Display log (Ctrl-C when finished)
adb logcat | grep "D/hdr\|F/libc\|E/libEGL\|W/Adreno-ES20"

EOF