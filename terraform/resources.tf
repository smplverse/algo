resource "aws_instance" "smplverse_instance" {
  ami           = var.DL_AMI
  instance_type = "g4dn.xlarge"
  count         = 1
  key_name      = "smplverse_key"
  depends_on = [
    aws_security_group.smplverse_security_group
  ]
  vpc_security_group_ids = [
    aws_security_group.smplverse_security_group.id
  ]

  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("~/.ssh/smplverse")
    timeout     = "4m"
    host        = self.public_ip
    password    = var.SSH_PASSWORD
  }

  # install docker-compose
  provisioner "remote-exec" {
    inline = [
      <<EOT
      sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
      EOT,
      "sudo chmod +x /usr/local/bin/docker-compose",
      <<EOT
      sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
      EOT,
    ]
  }

  # install k8s
  provisioner "remote-exec" {
    inline = [
      <<EOT
      sudo apt-get update \
      && sudo apt-get install -y \
        apt-transport-https curl
      EOT,
      <<EOT
      curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
      EOT,
      <<EOT
      cat <<EOF | sudo tee /etc/apt/sources.list.d/kubernetes.list 
      deb https://apt.kubernetes.io/ kubernetes-xenial main EOF
      EOT,
      <<EOT
      sudo apt-get update \
      && sudo apt-get install -y -q kubelet kubectl kubeadm \
      && sudo kubeadm init --pod-network-cidr=192.168.0.0/16
      EOT,
      <<EOT
      kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml
      EOT,
      <<EOT
      kubectl taint nodes --all node-role.kubernetes.io/master-
      EOT
    ]
  }
}

resource "aws_security_group" "smplverse_security_group" {
  egress {
    cidr_blocks = ["0.0.0.0/0"]
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
  }

  ingress {
    cidr_blocks = ["0.0.0.0/0"]
    protocol    = "tcp"
    from_port   = 22
    to_port     = 22
  }
}

resource "aws_key_pair" "smplverse_key" {
  key_name   = "smplverse_key"
  public_key = file("~/.ssh/smplverse.pub")
}
