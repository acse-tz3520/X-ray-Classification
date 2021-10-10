mkdir ~/.kaggle
cp ./kaggle.json ~/.kaggle/kaggle.json
pip install --upgrade --force-reinstall --no-deps kaggle
kaggle competitions download -c acse4-ml-2020
unzip acse4-ml-2020.zip