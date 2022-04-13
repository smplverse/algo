### algorithm

actual algo is quite simple:

```
choose:
    face, smpl
    metric { cosine, euclidean, euclidean_l2 }
    face_detection_model
    face_recognition_model

func preprocess(face):
    face := resize_and_apply_padding(face, recognition_model.input_shape)
    face := normalize(face)
    face := expand_dims(face)

for { face, smpl } as img:
    face := face_detection_model(img)
    face := preprocess(face)
    representation := face_recognition_model(face)

minimize:
    distance := find_distance(face_representation, smpl_representation)
```

#### to note:

- resizing and padding is to ensure there are no deformations
- depending on face recognition model there could be
  different normalization techniques
