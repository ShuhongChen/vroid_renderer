


# vroid_renderer

This repo converts and renders the 3D datasets introduced in [PAniC-3D: Stylized Single-view 3D Reconstruction from Portraits of Anime Characters](https://github.com/ShuhongChen/panic3d-anime-reconstruction).  As described in that repo, these scripts will add to `./_data/lustrous`


## setup

Make a copy of `./_env/machine_config.bashrc.template` to `./_env/machine_config.bashrc`, and set `$PROJECT_DN` to the absolute path of this repository folder.  The other variables are optional.

This project requires docker with a GPU.  Run these lines from the project directory to pull the image and enter a container; note these are bash scripts inside the `./make` folder, not `make` commands.  Alternatively, you can build the docker image yourself.

    make/docker_pull
    make/shell_docker
    # OR
    make/docker_build
    make/shell_docker


## vroid-dataset

The [vroid-dataset](https://github.com/ShuhongChen/vroid-dataset) should have downloaded folders of `.vrm` with their metadata to `./_data/lustrous/raw/vroid/[0-9]/*`.  This script renders those to `./_data/lustrous/renders/rutileE/`

    # run render script
    python3 -m _scripts.render_all_vroid_rutileE


## animerecon-benchmark

The [animerecon-benchmark](https://github.com/ShuhongChen/animerecon-benchmark) should have downloaded compressed files to `./_data/lustrous/raw/[genshin,hololive]`.  Decompress all these files to a temp directory; each file becomes a directory carrying a `.pmx` MMD model.  Using a [DSSc converter](https://drive.google.com/drive/folders/1Zpt9x_OlGALi-o-TdvBPzUPcvTc7zpuV?usp=share_link), go to `PMX-to-VRM > Batch`, and select `./_data/lustrous/raw/dssc/dssc_mapping_daredemoE.txt`.  This should convert and put files to `./_data/lustrous/raw/dssc/[genshin,hololive]/*.vrm`.  The following script renders the `.vrm` files to `./_data/lustrous/renders/daredemoE/`

    # run render script
    python3 -m _scripts.render_all_animerecon_daredemoE

(Thanks to [Softmind Ltd.](https://www.softmind.tech/) for sharing their [DanSingSing converter](https://vtuber.itch.io/dssconverter), and Geng Lin for adding the batch function)


## citing

If you use our repo, please cite our work:

    @inproceedings{chen2023panic3d,
        title={PAniC-3D: Stylized Single-view 3D Reconstruction from Portraits of Anime Characters},
        author={Chen, Shuhong and Zhang, Kevin and Shi, Yichun and Wang, Heng and Zhu, Yiheng and Song, Guoxian and An, Sizhe and Kristjansson, Janus and Yang, Xiao and Matthias Zwicker},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2023}
    }


