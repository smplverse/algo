resource "aws_instance" "smplverse_instance" {
  ami           = var.DL_AMI
  instance_type = "g4dn.xlarge"
  count         = 1
  key_name      = "smplverse"

  vpc_security_group_ids = [aws_security_group.smplverse_sg.id]

  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("~/.ssh/smplverse")
    timeout     = "10s"
  }
}

resource "aws_security_group" "smplverse_sg" {
  egress {
    cidr_blocks      = ["0.0.0.0/0"]
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
  }

  ingress {
    cidr_blocks      = ["0.0.0.0/0"]
    protocol         = "tcp"
    from_port        = 22
    to_port          = 22
  }
}

resource "aws_key_pair" "smplverse_key" {
  key_name = "smplverse"
  public_key = file("~/.ssh/smplverse.pub")
}
