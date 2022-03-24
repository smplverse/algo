# SMPLVERSE

---

### workflow:

after entering the mint page the user can connect their wallet and mint an nft

one nft gives the user the right to open their webcam and select a frame to be used to match them an nft that resembles the frame the most, where sending the frame is to be a part of another transaction which is irreversible

upon sending the frame, one of the images is to be bound to the given user and taken out of the pool

if the transaction is successful, the nft is locked irreversibly, in case of the transaction being reverted or cancelled, the user has the right to re-try capturing the right frame and send it over again

one player might mint multiple nft's, as there are less and less nfts in the pool the combinations tend to match the players' appearances less

### stack:

the frontend has to have a standard wallet connection mechanism (will use the tool developed by uniswap and available open-source choosing the ethers.js provider)

the contract shall contain a standard ERC721 base with the counter instead of enumerable to minimize the gas needed per mint

what is more, there shall be a mapping of uint nft-index to uint image-index to account for any images that have been matched

the image matching shall occur instantly after minting to favour the users on first come first serve basis, so that the first minters have the greatest likelihood of landing a piece that matches their appearance best

after the mint transaction completes, the address can send the frame from webcam to the backend server powered by a cluster of gpu-powered machines (likely g4dn) which will perform the matching using an ai model yet to be chosen based on accuracy and performance

the api has to have an authentication system that enables only the user that has minted to send over a frame and it has middleware that checks the contract whether a given address has already provided a frame

the above functionality could be implemented using signature verification with a bearer token being assigned after successful

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
