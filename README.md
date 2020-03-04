# PYY






## Dependencies

1. Python3
2. Numpy
3. Matplotlib
4. Tensorflow

Sample codes require these libraries. Please install them on your environment.



## How to Use

- train MNIST model
  ```python
  $ python train.py -d mnist -e 10 -m cnn_1
  ```

- train CIFAR-10 model
  ```python
  $ python train.py -d cifar10 -e 50 -m cnn_2
  ```
  
- targeted attack on MNIST
  ```python
  $ python attack.py -a PYY -d mnist -e 0.3 -m cnn_1 -t --start 0 --stop 10
  ```
  If you delete -t option, attack on untargeted.

- targeted attack on CIFAR-10
  ```python
  $ python attack.py -a PYY -d cifar10 -e 0.05 -m cnn_2 -t --start 0 --stop 10
  ```
  If you delete -t option, attack on untargeted.

If you run with more data, increase the number of the part of ```--stop 10``` by yourself please.



## License

Some of the codes including this source use [gongzhitaao/tensorflow-adversarial](https://github.com/gongzhitaao/tensorflow-adversarial).

