### Notes and TODOs (mostly TODOs)

to save time and ensure every one of the smpls faces gets detected

the smpls can have their faces pre-located and the vector representations pre-generated once the final 7667 is chosen

store hash of the image uploaded in the address mapping for given indices

metadata:

- json img uri, take the token index, confidence (distance)
- describe the smpls with 3-5 words using some other model

  https://3wordsproject.com/metadata/270.json
  https://etherscan.io/address/0xa8f9b843c449c5d93a58400f8586599b8f336dbb#code
  https://etherscan.io/address/0x517e643f53eb3622fd2c3a12c6bfde5e7bc8d5ca#code

storage

- s3
- ipfs (pinata, infura)

contract:

channel season 0 etherscan
nuxui contract also etherscan

- erc721 non-enumerable
- image hash
- mint is a ticket
- second transaction for file upload, getting a smpl match
- store img uploaded hash in address mapping
- flattened, use ascii header
- name "Smplverse"
- symbol "SMPL"
- total supply 7667
- mint price 0.07 eth, not changeable
- payable
  - withdraw: 20% 80%
- ownable
- multiple mint, azuki.com/erc721a
- max mint: 5
- toggle cap (we only do 1k)
