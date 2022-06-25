<h1 align="center">SMPLVERSE</h1>
<br/>
<div align="center">
  <img src="https://user-images.githubusercontent.com/63755291/161608061-4df6089e-f263-490b-ba9c-286b739c0bc3.png" width="300" />
</div>
<br />

### frontend

- address bob
- bob has a mint token
- bob approves an image
- bob sends tx [userImagehash, tokenId]
- if tx reverts - alert exception, otherwise continue by sending below to server
- allow the user to download their image upload?

```
HTTP/1.1 POST
{
  address
  tokenId
  image
}
```

### server

- stores the smpls representations
- retrieve the tx details
- save the uploaded image only till the smpl is assigned
- assert that:
  - postRequest.uploadedImageHash == tx.uploadedImageHash
  - postRequest.tokenId == tx.tokenId
  - !postRequest.tokenId doesnt have a smpl assigned
- perform match
- assign smplId to tokenId in our mapping
- make smpl public under given tokenId (metadata api)

the user can now see their smpl on opensea/metamask, and the contract is
storing the hash of an image sent

### Contents

- [Algorithm](https://github.com/piotrostr/smplverse/tree/main/docs/algorithm.md)
- [Models](https://github.com/piotrostr/smplverse/tree/main/docs/models.md)
- [Workflow](https://github.com/piotrostr/smplverse/tree/main/docs/workflow.md)
- [Tech Stack](https://github.com/piotrostr/smplverse/tree/main/docs/tech-stack.md)
- [Stack](https://github.com/piotrostr/smplverse/tree/main/docs/tech-stack.md)

### Usage

```
main.py [-h] [--headless] [--model MODEL] [--dataset DATASET]
        [--make-embeddings]

Runs the matcher on chosen dataset against smplverse pieces

optional arguments:
  -h, --help         show this help message and exit
  --headless         include flag to skip displaying the images
  --model MODEL      the model to use, either resnet100 or vggface2
  --dataset DATASET  ibug_faces or famous_people
  --make-embeddings  include to create embeddings with the given model
```
