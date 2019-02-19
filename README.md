# Overview

This proof-of-concept generates output vectors for each input image. The targets for regression are derived from GloVe vectors for the human-readable labels of each class. 

This repository includes a model that trained for 14 epochs; this cutoff was arbitrarily selected in order to free up the GPU for other work, so there's probably still some improvement in store. The model has a top-1 validation accuracy of about 60.3%.

# Training Results

Some sample results showing how classifications evolved through training for a few different images:


Epoch | The Good | The Bad | The Weird
----- | -------- | ------- | ---------
 | ![Red-Breasted Merganser Bird](/ImgPredictions/0/37_red-breasted merganser bird:potter's wheel.png) | ![Submarine](/ImgPredictions/0/3_submarine:potter's wheel.png) | ![American Lobster](/ImgPredictions/0/6_American lobster:potter's wheel.png) 
0 | potter's wheel | potter's wheel | potter's wheel
1 | great egret bird | yellow lady's slipper orchid | yellow lady's slipper orchid
2 | red-breasted merganser bird | hand plane | yellow lady's slipper orchid
3 | black swan | potter's wheel | red king crab
4 | red-breasted merganser bird | assault rifle | red king crab
5 | red-breasted merganser bird | potter's wheel | American lobster
6 | red-breasted merganser bird | potter's wheel | cowboy boot
7 | red-breasted merganser bird | potter's wheel | American lobster
8 | red-breasted merganser bird | potter's wheel | American lobster
9 | red-breasted merganser bird | potter's wheel | cowboy boot
10 | red-breasted merganser bird | potter's wheel | cowboy boot
11 | red-breasted merganser bird | potter's wheel | potter's wheel
12 | red-breasted merganser bird | potter's wheel | cowboy boot
13 | red-breasted merganser bird | potter's wheel | American black bear
14 | red-breasted merganser bird | potter's wheel | American lobster

One thing worth noting is that "potter's wheel" is the closest label to an all-zero vector, which probably accounts for its frequency as a wrong answer, especially at the start. I don't have a good answer for "yellow lady's slipper orchid".

The validation accuracy over training epochs looks like this:

![Validation Accuracy](/MiscImages/Figure_1.png)

Note that only 49792 images were used in validating each epoch, and the indexes were randomized each epoch so each epoch got judged on a slightly different set. For a proof-of-concept hobby project this doesn't bother me.

# Details

`Main.py` runs the actual training process. All command line arguments are required except for a pretrained model to load. If none is provided, randomly-initialized weights are used.

The specific labels used to train the provided model are a slightly augmented variant of the labels in [imagenet-simple-labels](https://github.com/anishathalye/imagenet-simple-labels). In an extremely subjective process, additional words were added to labels that seemed like they could be clearer (for example, changing "brambling" to "brambling bird").

A decoder is included for mapping a vector to a class, but this isn't involved in training. Because this decoder is (relatively speaking) very slow, it is not used to assess accuracy during training - a separate script, `TestEpochModels.py` evaluates accuracy on the validation set.

When training starts, this script also selects 50 random validation images. At the start of each epoch, it will save each image with its current, decoded label so you can see how it's doing. The format is:

`[sample directory]/[epoch]/[image index]_[true label]:[predicted label].png`
