packer {
  required_plugins {
    amazon = {
      version = ">= 0.0.1"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

source "amazon-ebs" "smplverse_ami" {
  ami_name      = "smplverse_ami"
  region        = "us-east-2"
  source_ami    = "ami-0fd4336c5e2ccb32d"
  instance_type = "g4dn.xlarge"
  ssh_username  = "ubuntu"
}

variable "GITHUB_ACCESS_TOKEN" {
  type      = string
  sensitive = true
  default   = "${env("GITHUB_ACCESS_TOKEN")}"
}

variable "SMPLVERSE_HASH" {
  type = string
  sensitive = true
  default = "${env("SMPLVERSE_HASH")}"
}

build {
  sources = ["amazon-ebs.smplverse_ami"]

  provisioner "shell" {
    inline = [
      "git clone https://piotrostr:${var.GITHUB_ACCESS_TOKEN}@github.com/piotrostr/smplverse",
    ]
  }

  provisioner "shell" {
    scripts = [
      "./setup-compose.sh",
      "./get-mock-dataset.sh",
      "./setup-ipfs.sh"
    ]
  }

  provisioner "shell" {
    inline = [
      "cd smplverse && ipfs get -o smpls ${var.SMPLVERSE_HASH}",
    ]
  }
}
