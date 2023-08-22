# CEMR

Code repository for the paper:

**Cross-Domain Multi-Level Refinements for 3D Human Reconstruction in Wild Videos**

## Description
We focus on reconstructing human mesh from out-of-domain videos. In our experiments, we train a source model on Human 3.6M. To produce accurate human mesh on out-of-domain frames, we optimize the BaseModel on target frames via MiRRo at time. Below are the comparison results between BaseModel and the adapted model on the videos with various camera parameters, motion, etc.

## Get Started

MiRRo has been implemented and tested on Ubuntu 18.04 with python = 3.6.

Install required packages:

```bash
conda create -n mirro-env python=3.6
conda activate mirro-env
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install -r requirements.txt
install spacepy following https://spacepy.github.io/install_linux.html
```

Download required file from [File 1](https://drive.google.com/file/d/1_4GhHaiNIu2aidVwMBvbdcdGd2vgy-gR/view?usp=sharing) and [File 2](https://drive.google.com/file/d/1uekfFsWnLcKdrT6CxZ9zFQFy_ySdDaXK/view?usp=sharing). After unzipping files, rename `File 1` to `data` (ensuring you do not overwrite `gmm_08.pkl` in `./data`) and move the files in `File 2` to `data/retrieval_res`. Finally, they should look like this:
```
|-- data
|   |--dataset_extras
|   |   |--3dpw_0_0.npz
|   |   |--3dpw_0_1.npz
|   |   |--...
|   |--retrieval_res
|   |   |--...
|   |--smpl
|   |   |--...
|   |--spin_data
|   |   |--gmm_08.pkl
|   |--basemodel.pt
|   |--J_regressor_extra.npy
|   |--J_regressor_h36m.npy
|   |--smpl_mean_params.npz
```

Download Human3.6M using this [tool](https://github.com/kotaro-inoue/human3.6m_downloader), and then extract images by:
```
python process_data.py --dataset h36m
```
We train h36m on [VIBE](https://github.com/mkocabas/VIBE) and [HMR](https://github.com/akanazawa/hmr) and put the checkpoint message into our MiRRo model.

---
## Running on the 3DPW
Download the [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) dataset. Then edit `PW3D_ROOT` in the config.py.
Then, run:
```bash
bash run_on_3dpw.sh
```

#### Results on 3DPW

| Method                                                       | Protocol | PA-MPJPE |  MPJPE   |   PVE    |
| :----------------------------------------------------------- | :------: | :------: | :------: | :------: |
| [SPIN](https://github.com/nkolot/SPIN)                       |   #PS    |   59.2   |   96.9   |  135.1   |
| [PARE](https://github.com/mkocabas/PARE)                     |   #PS    |   46.4   |   79.1   |   94.2   |
| [Mesh Graphormer](https://github.com/microsoft/MeshGraphormer) |   #PS    |   45.6   |   74.7   |   87.7   |
| CEMR (Ours)                                               |   #PS    | **34.6** | **52.1** | **67.9** |


## Acknowledgement
We borrow some code from [VIBE](https://github.com/mkocabas/VIBE) and [DynaBOA](https://github.com/syguan96/DynaBOA). [VideoMAE](https://github.com/MCG-NJU/VideoMAE) is used to capture pre-trained representations. [Learn2learn](https://github.com/learnables/learn2learn) is also used to implement multi-level optimization.
