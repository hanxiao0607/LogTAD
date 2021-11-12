# LogTAD: Unsupervised Cross-system Log Anomaly Detection via Domain Adaptation
A Pytorch implementation of [LogTAD](https://dl.acm.org/doi/abs/10.1145/3459637.3482209).

## Configuration
- Ubuntu 20.04
- NVIDIA driver 460.73.01 
- CUDA 11.2
- Python 3.9
- PyTorch 1.9.0

## Installation
This code requires the packages listed in requirements.txt.
A virtual environment is recommended to run this code

On macOS and Linux:  
```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
deactivate
```
Reference: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

## Instructions
LogTAD and other baseline models are implemented on [BGL](https://github.com/logpai/loghub/tree/master/BGL) and [Thunderbird](https://github.com/logpai/loghub/tree/master/Thunderbird) datasets

Clone the template project, replacing ``my-project`` with the name of the project you are creating:

        git clone https://github.com/hanxiao0607/LogTAD.git my-project
        cd my-project

Run and test:

        python3 main_LogTAD.py

## Citation
```
@inproceedings{han2021unsupervised,
  title={Unsupervised Cross-system Log Anomaly Detection via Domain Adaptation},
  author={Han, Xiao and Yuan, Shuhan},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={3068--3072},
  year={2021}
}
```
