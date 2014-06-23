#!/bin/bash

set -e
# Any subsequent commands which fail will cause the shell script to exit immediately

sudo su <<EOF

# Clean previous install
ant uninstall
ant clean

EOF