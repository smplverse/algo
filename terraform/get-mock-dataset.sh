#!/bin/bash

wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -xvzf lfw.tgz
rm lfw.tgz
mv lfw data/input
