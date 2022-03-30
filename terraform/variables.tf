variable "AWS_ACCESS_KEY" {
  type = string
  sensitive = true
}

variable "AWS_SECRET_KEY" {
  type = string
  sensitive = true
}

variable "SSH_PASSWORD" {
  type = string
  sensitive = true
}

variable "DL_AMI" {
  type = string
  sensitive = false
  default = "ami-0fd4336c5e2ccb32d"
}
