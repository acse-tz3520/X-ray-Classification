# X-RAY-Classifier: Solution to classify a dataset of X-Ray lung images

[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)

X-RAY-Classifier is a python machine learning package to implement the solution to classify a dataset of X-Ray lung images.

## Table of Contents

- [X-RAY-Classifier](#about-x-ray-classifier)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)


## About X-RAY-Classifier

X-RAY-Classifier is a solution to [X-RAY Classification](https://www.kaggle.com/c/acse4-ml-2020). The goal of this Kaggle competition is to classify a dataset of X-Ray lung images into 4 classes. These package include machine learning modules to data processing and model training with `Pytorch`. Moreover, `X-RAY-Classifier.ipynb` includes all functional parts, as well as user guidence, process of tuning parameters and strategy to select the final submission. The final output is the prediction result in `csv` format as the submission file for the competition.


## Installation

To install X-RAY-Classifier, run the following commands:

```
# get the code
git clone https://github.com/acse-2020/acse-4-x-ray-classification-softmax.git
cd acse-4-x-ray-classification-softmax

# install pre-requisites
pip install -r requirements.txt
```

### Data Downloading

To download the data, first go to [kaggle](https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the download of `kaggle.json`, a file containing your API credentials which you must upload to colab. Then you must move this file to your root directory and request a forced update of kaggle in pip to download your data correctly. Simply run:

```
bash download_data.sh
```


## Usage

Use `python main.py` in `acse_softmax` to train model and use `--help` option to see all the parameters that can be changed.

```bash
usage: Covid_CT_Classifier_Softmax [-h] [-version] [-s S] [-lr LR] [-m M] [-bs BS] [-ts TS] [-e E]

optional arguments:
  -h, --help  show this help message and exit
  -version    show program's version number and exit
  -s S        seed
  -lr LR      learning rate
  -m M        momentum
  -bs BS      batch size
  -ts TS      test batch size
  -e E        epoch

```

Or you can choose to use `X-RAY-Classifier.ipynb`, which includes all functional parts, as well as user guidence, process of tuning parameters and strategy to select the final submission.


## Documentation

To get documentation of X-RAY-Classifier, open `index.html` in `docs/html` after installation.


## Contributing

Feel free to dive in! [Open an issue](https://github.com/acse-2020/acse-4-x-ray-classification-softmax/issues/new) or submit PRs.

### Contributors

This project exists thanks to all the people who contributed.


## License

[MIT](LICENSE) © acse-2020 group Softmax
