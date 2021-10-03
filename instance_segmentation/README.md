## Environment
The code is developed using python 3.6.9 on Ubuntu 18.04. NVIDIA GPUs are needed. 

### Requirements

- Linux (Windows is not officially supported)
- Python 3.5+
- PyTorch 1.1 or higher (>=1.5 is not tested)
- CUDA 9.0 or higher
- NCCL 2
- GCC 4.9 or higher
- [mmcv 0.2.16](https://github.com/open-mmlab/mmcv/tree/v0.2.16)

Have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04 and CentOS 7.2
- CUDA: 9.0/9.2/10.0/10.1
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
- GCC(G++): 4.9/5.3/5.4/7.3

## Quick start
### Installation
1. Clone this repo as ${SEG_ROOT}
  
   ``` 
   git clone https://github.com/xuxu50007/previous_projects.git
   cd ${SEG_ROOT}
   ```
   
2. Install build requirements and then install this project:
   
   ```
   pip install torch==1.4.0 torchvision==0.5.0 
   pip install mmcv==0.2.16

   pip install -r requirements/build.txt
   pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
   pip install -v -e .
   ```

Note:

1. The version of torch and torchvision is according to cuda version 10.1.

2. If there is a ModuleNotFoundError: No module named 'skbuild', running `pip install scikit-build`.

3. `pip install -v -e .` will only install the minimum runtime requirements.


## Datasets

```
mkdir data
```

Your directory tree should look like this:

```
${SEG_ROOT}
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

## Models

```
mkdir checkpoints
```
The pretrained models on COCO are stored under the folder "checkpoints".

Model | Multi-scale training | Testing time / im | AP (minival) | Link
--- |:---:|:---:|:---:|:---:
SOLO_R50_1x | No | 77ms | 32.9 | [download](https://cloudstor.aarnet.edu.au/plus/s/nTOgDldI4dvDrPs/download)
SOLO_R50_3x | Yes | 77ms |  35.8 | [download](https://cloudstor.aarnet.edu.au/plus/s/x4Fb4XQ0OmkBvaQ/download)
SOLO_R101_3x | Yes | 86ms |  37.1 | [download](https://cloudstor.aarnet.edu.au/plus/s/WxOFQzHhhKQGxDG/download)
Decoupled_SOLO_R50_1x | No | 85ms | 33.9 | [download](https://cloudstor.aarnet.edu.au/plus/s/RcQyLrZQeeS6JIy/download)
Decoupled_SOLO_R50_3x | Yes | 85ms | 36.4 | [download](https://cloudstor.aarnet.edu.au/plus/s/dXz11J672ax0Z1Q/download)
Decoupled_SOLO_R101_3x | Yes | 92ms | 37.9 | [download](https://cloudstor.aarnet.edu.au/plus/s/BRhKBimVmdFDI9o/download)
SOLOv2_R50_1x | No | 54ms | 34.8 | [download](https://cloudstor.aarnet.edu.au/plus/s/DvjgeaPCarKZoVL/download)
SOLOv2_R50_3x | Yes | 54ms | 37.5 | [download](https://cloudstor.aarnet.edu.au/plus/s/nkxN1FipqkbfoKX/download)
SOLOv2_R101_3x | Yes | 66ms | 39.1 | [download](https://cloudstor.aarnet.edu.au/plus/s/61WDqq67tbw1sdw/download)
SOLOv2_R101_DCN_3x | Yes | 97ms | 41.4 | [download](https://cloudstor.aarnet.edu.au/plus/s/4ePTr9mQeOpw0RZ/download)
SOLOv2_X101_DCN_3x | Yes | 169ms | 42.4 | [download](https://cloudstor.aarnet.edu.au/plus/s/KV9PevGeV8r4Tzj/download)

**Light-weight models:**

Model | Multi-scale training | Testing time / im | AP (minival) | Link
--- |:---:|:---:|:---:|:---:
Decoupled_SOLO_Light_R50_3x | Yes | 29ms | 33.0 | [download](https://cloudstor.aarnet.edu.au/plus/s/d0zuZgCnAjeYvod/download)
Decoupled_SOLO_Light_DCN_R50_3x | Yes | 36ms | 35.0 | [download](https://cloudstor.aarnet.edu.au/plus/s/QvWhOTmCA5pFj6E/download)
SOLOv2_Light_448_R18_3x | Yes | 19ms | 29.6 | [download](https://cloudstor.aarnet.edu.au/plus/s/HwHys05haPvNyAY/download)
SOLOv2_Light_448_R34_3x | Yes | 20ms | 32.0 | [download](https://cloudstor.aarnet.edu.au/plus/s/QLQpXg9ny7sNA6X/download)
SOLOv2_Light_448_R50_3x | Yes | 24ms | 33.7 | [download](https://cloudstor.aarnet.edu.au/plus/s/cn1jABtVJwsbb2G/download)
SOLOv2_Light_512_DCN_R50_3x | Yes | 34ms | 36.4 | [download](https://cloudstor.aarnet.edu.au/plus/s/pndBdr1kGOU2iHO/download)


## Usage

### A quick demo

   ```
   config=configs/solov2/solov2_x101_dcn_fpn_8gpu_3x.py
   model=checkpoints/SOLOv2_X101_DCN_3x.pth
   python demo/inference_demo_yan.py --config $config --model $model --inputvideo multi_people.mp4 --outputzip out.zip --scoreth 0.5 --onlyperson
   ```
Note:

1. Put all the files: inference_demo_yan.py, generate_pkl.py and multi_people.mp4(Due to the space limit, I did not upload the .mp4 file) into the folder "demo". And the result pkl files are all under the folder 'pkl_files'.
2. If you want to test a simple image, just comment the part of "Load a video" from Line 50, uncomment the part of "Test a single image" from Line 76. And the result pkl file is under the folder "demo".

### Train with multiple GPUs
    tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}

    Example: 
    tools/dist_train.sh configs/solov2/solov2_x101_dcn_fpn_8gpu_3x.py  8

### Train with single GPU
    python tools/train.py ${CONFIG_FILE}
    
    Example:
    python tools/train.py configs/solov2/solov2_x101_dcn_fpn_8gpu_3x.py

### Testing
    # multi-gpu testing
    tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}  --show --out  ${OUTPUT_FILE} --eval segm
    
    Example: 
    tools/dist_test.sh configs/solov2/solov2_x101_dcn_fpn_8gpu_3x.py checkpoints/SOLOv2_X101_DCN_3x.pth  8  --show --out results_solo.pkl --eval segm

    # single-gpu testing
    python tools/test_ins.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --out  ${OUTPUT_FILE} --eval segm
    
    Example: 
    python tools/test_ins.py configs/solov2/solov2_x101_dcn_fpn_8gpu_3x.py  checkpoints/SOLOv2_X101_DCN_3x.pth --show --out  results_solo.pkl --eval segm


### Visualization

    python tools/test_ins_vis.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --save_dir  ${SAVE_DIR}
    
    Example: 
    python tools/test_ins_vis.py configs/solo/solo_r50_fpn_8gpu_1x.py  SOLO_R50_1x.pth --show --save_dir  work_dirs/vis_solo
