#!/bin/bash

options=""
programfile="../*.cu"
testfiles=$(find . -name "test_*.cpp")

constfile="test_const.cu"
for testfile in $testfiles; do

    if [[ $(basename "$testfile") == "$constfile" ]]; then
        continue
    fi

    nvcc $testfile $constfile $programfile $options
    a.exe
done


