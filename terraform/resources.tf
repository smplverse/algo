resource "aws_instance" "foo" {
  ami           = var.DL_AMI
  instance_type = "g4dn.xlarge"
  count         = 1
  key_name      = "smplverse_key"

  vpc_security_group_ids = [aws_security_group.sg.id]

  connection {
    type        = "ssh"
    user        = "ubuntu"
    host        = self.public_ip
    private_key = file("~/.ssh/smplverse")
  }
}

resource "aws_security_group" "sg" {
  ingress {
      protocol  = "tcp"
      from_port = 22
      to_port   = 22
  }

  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
}
