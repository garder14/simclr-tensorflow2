# SimCLR - TensorFlow 2

This is an unofficial implementation of ["A Simple Framework for Contrastive Learning of Visual Representations"](https://arxiv.org/pdf/2002.05709.pdf) (SimCLR) for self-supervised representation learning on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

<img src="https://i.imgur.com/G2vt50c.png" width="85%">

## Results

The linear evaluation accuracy of a ResNet-18 or ResNet-34 encoder pretrained for 100, 200 or 300 epochs is shown below.

|           | 100 epochs | 200 epochs | 300 epochs |
|:---------:|:----------:|:----------:|:----------:|
| **ResNet-18** |   81.65%   |   85.83%   |   86.90%   |
| **ResNet-34** |   82.92%   |   87.56%   |   88.69%   |

## Software installation

Clone this repository:

```bash
git clone https://github.com/garder14/simclr-tensorflow2.git
cd simclr-tensorflow2/
```

Install the dependencies:

```bash
conda create -n tf-gpu tensorflow-gpu cudatoolkit=10.1
conda activate tf-gpu
pip install tensorflow_addons==0.10.0
```

## Pretraining (representation learning)
 
To pretrain a ResNet-18 base encoder for 300 epochs (weights saved every 100 epochs), try the following command:

```bash
python pretraining.py --encoder resnet18 --num_epochs 300 --batch_size 512 --temperature 0.5
```

It takes around 15 hours on a Tesla P100 GPU. 

## Linear evaluation

To evaluate the quality of representations extracted by a certain base encoder, try the following command:

```bash
python linearevaluation.py --encoder resnet18 --encoder_weights f300.h5
```

It takes around 1 hour on a Tesla P100 GPU.

## References

* [Ting Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", 2020](https://arxiv.org/pdf/2002.05709.pdf)
