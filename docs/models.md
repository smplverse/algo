| **backend** | success rate | time elapsed | time per image |
| ----------- | :----------: | :----------: | :------------: |
| opencv      |    0.417     |    24.46s    |    0.0244s     |
| ssd         |    0.913     |    24.29s    |    0.0243s     |
| dlib        |    0.649     |    90.42s    |    0.0904s     |
| mtcnn       |    0.941     |   289.77s    |    0.2896s     |
| retinaface  |    0.987     |   134.98s    |    0.1348s     |
| mediapipe   |    0.907     |    4.95s     |    0.0049s     |

| **model**  | model parameters | gpu memory requirement | weights size |
| ---------- | :--------------: | :--------------------: | :----------: |
| DeepFace   |   102.4129 MM    |       0.0992 GB        | 390.6741 MB  |
| Facenet512 |    23.4974 MM    |       0.0628 GB        |  89.6356 MB  |
| DeepID     |    0.3951 MM     |       0.0008 GB        |  1.5071 MB   |
| ArcFace    |    34.1652 MM    |       0.1151 GB        | 130.3298 MB  |
| Facenet    |    22.8081 MM    |       0.0621 GB        |  87.0062 MB  |
| OpenFace   |    3.7433 MM     |       0.0127 GB        |  14.2795 MB  |
| VGG-Face   |   145.0029 MM    |       0.2277 GB        | 553.1421 MB  |
