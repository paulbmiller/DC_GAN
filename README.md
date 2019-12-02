# DC_GAN
## Description
A Deep Convolutional Generative Adversarial Network which generates MNIST images to train a discriminator. The generator takes a 100-dim tensor of random noise as input and outputs MNIST digits.

## Model architecture
These model diagrams have been made using http://alexlenail.me/NN-SVG/. Diagram for the discriminator:
![Discriminator](https://raw.githubusercontent.com/paulbmiller/DC_GAN/master/github/Discriminator.PNG)

Diagram for the generator:
![Generator](https://raw.githubusercontent.com/paulbmiller/DC_GAN/master/github/Generator.PNG)

## Samples of generated images
Generated images over a batch of 32 images after 10 epochs:
![](https://raw.githubusercontent.com/paulbmiller/DC_GAN/master/results/250_epochs/112089633_10.png)
Generated images over a batch of 32 images after 20 epochs:
![](https://raw.githubusercontent.com/paulbmiller/DC_GAN/master/results/250_epochs/356870242_20.png)
Generated images over a batch of 32 images after 50 epochs:
![](https://raw.githubusercontent.com/paulbmiller/DC_GAN/master/results/250_epochs/380560714_50.png)
Generated images over a batch of 32 images after 100 epochs:
![](https://raw.githubusercontent.com/paulbmiller/DC_GAN/master/results/250_epochs/368795332_100.png)
Generated images over a batch of 32 images after 250 epochs:
![](https://raw.githubusercontent.com/paulbmiller/DC_GAN/master/results/250_epochs/327074246_250.png)