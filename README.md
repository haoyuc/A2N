# A2N

This repository is an PyTorch implementation of the paper

"Attention in Attention Network for Image Super-Resolution" [[arXiv]](https://arxiv.org/abs/2104.09497)


Visual results in the paper are availble at [Google Drive](https://drive.google.com/file/d/1SCO2t3HeNsyofREmflsDjF1AKOHBAaRQ/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1iIw9dzeKZTvgIxSEL8K3Qw) (password: 7t74). 

## Test

```
python main.py  --scale 4 --pre_train ./experiment/model/aan_x4.pt --chop --test_only
```
If you use CPU, please add "--cpu".


## Tran 

### Training data preparation 

  1. Download DIV2K training data from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).
  2. Specify `'--dir_data'` in option.py based on the data path. 

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Training

```
python main.py --scale 4 
```




## Citation

If you have any question or suggestion, welcome to email me [at here](mailto:haoyuchen@link.cuhk.edu.cn).

If you find our work helpful in your resarch or work, please cite the following papers.

```
@misc{chen2021attention,
      title={Attention in Attention Network for Image Super-Resolution}, 
      author={Haoyu Chen and Jinjin Gu and Zhi Zhang},
      year={2021},
      eprint={2104.09497},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Acknowledgements

This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [PAN](https://github.com/zhaohengyuan1/PAN). We thank the authors for sharing their codes.
