#!/bin/bash

sudo apt-get update \
  && sudo apt-get install -y \
  apt-transport-https curl

curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

cat <<EOF
| sudo tee /etc/apt/sources.list.d/kubernetes.list 
deb https://apt.kubernetes.io/ kubernetes-xenial main 
EOF

sudo apt-get update \
  && sudo apt-get install -y \
  kubelet kubectl kubeadm
  
sudo kubeadm init --pod-network-cidr=192.168.0.0/16

kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml

kubectl taint nodes --all node-role.kubernetes.io/master-
