# Different Embedding Techniques on DSNeRF

[**Project**](https://www.cs.cmu.edu/~dsnerf/) | [**Paper**](https://arxiv.org/abs/2107.02791)

This project analyses the impact of different Fourier Feature mappings (Positional Encoding, Gauss Mapping) and the SIREN and SINONE networks on the rendering quality of DSNeRF.
The code base was taken from https://github.com/dunbar12138/DSNeRF.
The files run_nerf.py and run_nerf_helpers.py (implement Gauss mapping) were modified.
The files SSIMGenerator.py (calculate SSIM metric), SirenDsnerf.py (Implement SIREN and SINONE Networks) were added.

## Results

Positional Encoding        |  Gauss Mapping            | SINONE                    |  SIREN         |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
<img src="resources/Horns_Basic.mp4"  width="100" />  |  <img src="resources/Horns_Gauss.mp4"  width="100" /> | <img src="resources/Horns_SINONE.mp4"  width="100" /> | <img src="resources/Horns_SIREN.mp4"  width="100" />


## Quick Start

### Data
[Datasets](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7) for all experiments.

### Train and run the network:
1. Install the requirements from requirements.txt.
2. Use [COLMAP](https://github.com/colmap/colmap) to generate the camera poses and sparse depth information of a downloaded dataset.
3. Use python `imgs2poses.py <your_scenedir>` to generate the poses and sparse point cloud.
4. Create a config file like the examples in `./configs` and store the relative path to the data.
5. Train the network  with `python run_nerf.py --config configs/<config_file.txt>`. This will create a directory in `./logs` with checkpoints and rendering examples.
6. Test the network with `python run_nerf.py --config configs/<config_file.txt> --render_only`.
