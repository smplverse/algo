### tech stack:

frontend:

- typescript
- react (nextjs or gatsby depending on amount of static content to be supplied)
- emotion css
- ethers.js

backend:

- python
- pytorch + tensorrt engines for accelerated inference
- fastapi
- web3.py

nft contract:

- typescript
- hardhat
- waffle with mocha for unit-tests
- opensea integration

devops:

- both fe and be deployed in clusters on aws/linode
- if end up going with linode will also use an nginx container for load balancing, aws has its custom elastic lb
- containerized with docker
- orchestrated with kubernetes/docker-swarm
  managed with terraform
