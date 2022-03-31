#!/bin/bash

mkdir data

wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -xvzf lfw.tgz
rm lfw.tgz
mv lfw data/input

wget https://smplverse.s3.us-east-2.amazonaws.com/smpls.tar.gz
tar -xvzf smpls.tar.gz
rm smpls.tar.gz
mv smpls data/
