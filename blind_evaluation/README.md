# Blind evaluation

To estimate different methods with the best mask and skeleton, generated 30 videos are randomly placed in a 3x2 grid with 6 methods labeled from A to F:

* hrnetbu_maskrcnn

* openpose_maskrcnn

* udp_maskrcnn

* hrnetbu_solov2

* openpose_solov2

* udp_solov2

Note: The above 6 methods are named by "pose estimation method" + "instant segmentation method".

## How to run

Download this project

```
git clone https://github.com/xuxu50007/previous_projects.git

cd blind_evaluation

python blindeval.py
```

#### Note:
Due to space limit，I only uploaded the same temp.mp4 for each video folder "blindeval/skegho...".

#### Fig.1 TXT file table with original video name, new output video name, and used method list [A, B, C, D, E, F]
![1633119876192](https://user-images.githubusercontent.com/91754487/135682106-b67254d6-2757-4c22-84d4-78cf2cd63572.jpg)
