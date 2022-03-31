#!/bin/bash

wget https://dist.ipfs.io/go-ipfs/v0.12.0/go-ipfs_v0.12.0_linux-amd64.tar.gz

tar -xvzf go-ipfs_v0.12.0_linux-amd64.tar.gz

cd go-ipfs
sudo bash install.sh
cd ..

rm -rf go-ipfs_v0.12.0_linux-amd64.tar.gz go-ipfs
