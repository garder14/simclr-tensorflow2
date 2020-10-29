# SimCLR - TensorFlow 2

This is an unofficial implementation of [SimCLR](https://arxiv.org/pdf/2002.05709.pdf) for self-supervised image representation learning on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

## Results



## Software installation

Clone this repository:

```bash
git clone https://github.com/garder14/simclr-tensorflow2.git
cd erfnet-tensorflow2/
```

Install the dependencies:

```bash
conda create -n tf-gpu tensorflow-gpu cudatoolkit=10.1
conda activate tf-gpu
pip install tensorflow_addons==0.10.0
```

## Pretraining (representation learning)

```bash
python pretraining.py --encoder resnet18 --num_epochs 300 --batch_size 512 --temperature 0.5
```

## Linear evaluation

```bash
python linearevaluation.py --encoder resnet18 --encoder_weights f300.h5
```

## References

* [T. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", 2020](https://arxiv.org/pdf/2002.05709.pdf)
