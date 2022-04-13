<h1 align="center">SMPLVERSE</h1>
<br/>
<div align="center">
  <img src="https://user-images.githubusercontent.com/63755291/161608061-4df6089e-f263-490b-ba9c-286b739c0bc3.png" width="300" />
</div>
<br />

### Contents:

- [ Algorithm ](https://github.com/piotrostr/smplverse/tree/main/docs/algorithm.md)
- [ Models ](https://github.com/piotrostr/smplverse/tree/main/docs/models.md)
- [ Workflow ](https://github.com/piotrostr/smplverse/tree/main/docs/workflow.md)
- [ Tech Stack ](https://github.com/piotrostr/smplverse/tree/main/docs/tech-stack.md)
- [ Stack ](https://github.com/piotrostr/smplverse/tree/main/docs/tech-stack.md)

### Usage:

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

### TODO

try the model here:
[ MagFace ](https://github.com/IrvingMeng/MagFace)

and buffalo from here:
[ Insightface ](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md#1-face-recognition-models)
