#!/bin/bash

kernels="histEq reinhardGlobal reinhardLocal gradDom"

for name in $kernels
do
    IN=$name.cl
    OUT=$name.h

    if [ $IN -ot $OUT ]
    then
        echo "Skipping generation of OpenCL $name kernel"
        continue
    fi
    echo "Generating OpenCL $name kernel"

    echo "const char *"$name"_kernel =" >$OUT
    sed -e 's/\\/\\\\/g;s/"/\\"/g;s/  /\\t/g;s/^/"/;s/$/\\n"/' $IN >>$OUT
    if [ $? -ne 0 ]
    then
        exit 1
    fi
    echo ";" >>$OUT
done
