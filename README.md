# LimitAug

This is the official repository of our paper "Towards robust music source separation on loud commercial music" in ISMIR 2022.

Specifically, this repo contains the code for LimitAug data augmentation for training of robust music source separation algorithms.

If you are interested in the musdb-XL evaluation dataset that we also proposed in our paper, check the repo below.

[musdb-XL](https://github.com/jeonchangbin49/musdb-XL)


# TL;DR
![image](https://user-images.githubusercontent.com/60494498/187076031-2ae09405-bef1-4c75-a9c8-dd2d1e9d540d.png)


# Motivation (LimitAug)
Real world 'mastering-finished' music is heavily compressed and loud. How to consider it while training music source separation network? LimitAug starts from the simple idea that just use a limiter when making training examples so that the input data can be loud and compreseed like real world data!


# Methods
Total of 4 methods that we experimented in our paper are implemented in limitaug.py. 

(1) Linear gain increasing of training example

(2) LimitAug - using a limiter in training data sampling process

(3) Loudness normalization of training examples

(4) LimitAug then loudness normalization of training examples

In our experiments, we found all of the methods were helpful for generalization of the music source separation network, but if you have to choose only one, we strongly recommend to use the method (4), LimitAug then loud-norm for vocals and other stems, and the method (3) loud-norm for drums and bass stems. We expect that the usage of both methods at the same time while training will be also helpful (randomly choose to use method (3) or (4)).


# How to use LimitAug data augmentation
Our code (limitaug.py) only contains the training dataloader implemented with [Numpy](https://numpy.org) and [PyTorch](https://pytorch.org). Loudness calculation was done by [Pyloundnorm](https://github.com/csteinmetz1/pyloudnorm), and we used a limiter implemented in [Pedalboard](https://github.com/spotify/pedalboard). We did not contain full network training code because there are numbers of great implementations for training of music source separation network, such as [Open-unmix](https://github.com/sigsep/open-unmix-pytorch), [Demucs](https://github.com/facebookresearch/demucs), [TFC-TDF-U-Net](https://github.com/ws-choi/ISMIR2020_U_Nets_SVS), [D3Net](https://github.com/sony/ai-research-code/tree/master/d3net/music-source-separation) etc. Just simply refer and replace the dataloader in your project with our implementation.

# Notes
1) Q: In section 5.3 and Table 6, why the method (3) loud-norm performs better than the method (4) LimitAug + loud-norm on bass and drums?

  A : Honestly, I couldn't figure out the exact scientific reasons for this. In my subjective guess as a music engineer, dynamic range compression comes from a limiter is pretty much indistinguishable from the compression applied to each bass and drums stem. That is, there is no use to forcibly make compressed training examples for these stems.
  
  As opposed to this, compression comes from a limiter for each vocals and others stem are quite hard to achieve from the compression for respective stems. I think '[Dua Lipa - Don't start now](https://youtu.be/oygrmJFKYZY)' is a great example for this. In the chorus part, you can hear such 'wo-omph' sound in vocals. This sound generally cannot be obtained by compression triggerd by the vocal source itself. It is usually achieved by triggering compressor from the other source (for example, kick or whole drums, this method is called 'Side-chain compression'. Thank you for reviwer #2 for giving the opinion on the side-chain compression.) or the entire mixture (this is exactly how the LimitAug works.). Therefore, forcibly making these examples in training by using LimitAug + loud-norm was of help on vocals, and so does others stem.


# Acknowledgements
We appreciate Zafar Rafii, the author of musdb, for allowing us to reprocess the original musdb data. We thank Antoine Liutkus, also the author of musdb, for giving the creative suggestion on the distribution of our proposed datasets. We are grateful to Ben Sangbae Chon, Keunwoo Choi, and Hyeongi Moon from [GaudioLab, Inc](https://www.gaudiolab.com). for their fruitful discussions on our proposed methods. Last but not least, we thank Donmoon Lee, Juheon Lee, Jaejun Lee, Junghyun Koo, and Sungho Lee for helpful feedbacks.

# References
[1] Stöter, Fabian-Robert, et al. Open-Unmix for PyTorch. (https://github.com/sigsep/open-unmix-pytorch)
