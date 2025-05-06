kaggle datasets download -d datamunge/sign-language-mnist -p tmp_smnist
unzip tmp_smnist/sign-language-mnist.zip -d tmp_smnist
python tools/prepare_signmnist.py --csv tmp_smnist/sign_mnist_train/sign_mnist_train.csv --out data/signmnist/train
python tools/prepare_signmnist.py --csv tmp_smnist/sign_mnist_test/sign_mnist_test.csv --out data/signmnist/test
rm -r tmp_smnist
