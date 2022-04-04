### stack:

the frontend has to have a standard wallet connection mechanism (will use the tool developed by uniswap and available open-source choosing the ethers.js provider)

the contract shall contain a standard ERC721 base with the counter instead of enumerable to minimize the gas needed per mint

what is more, there shall be a mapping of uint nft-index to uint image-index to account for any images that have been matched

the image matching shall occur instantly after minting to favour the users on first come first serve basis, so that the first minters have the greatest likelihood of landing a piece that matches their appearance best

after the mint transaction completes, the address can send the frame from webcam to the backend server powered by a cluster of gpu-powered machines (likely g4dn) which will perform the matching using an ai model yet to be chosen based on accuracy and performance

the api has to have an authentication system that enables only the user that has minted to send over a frame and it has middleware that checks the contract whether a given address has already provided a frame

the above functionality could be implemented using signature verification with a bearer token being assigned after successful
