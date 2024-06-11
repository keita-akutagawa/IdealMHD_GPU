#!/bin/bash

options="-lgtest -lgtest_main -pthread"
programfile="../*.cpp"
testfiles=$(find . -name "test_*.cpp")

constfile="test_const.cpp"
for testfile in $testfiles; do

    if [[ $(basename "$testfile") == "$constfile" ]]; then
        continue
    fi

    g++ $testfile $constfile $programfile $options
    ./a.out
done


