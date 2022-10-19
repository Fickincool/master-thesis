# Denoising of cryo-electron tomograms and its effect on cellular membrane segmentation performance.

This is the repo for my Master Thesis from my M. Sc. in Mathematics at TUM.

I think it makes sense to share this in case anyone is interested in the subject. I hope someone can learn a little bit from what I did. It would be super exciting if someone would want to develop the denoising algorithms I worked on a little further.

First part of the Readme is a small description of my work, followed by installation of packages. Then I describe some details on the implementation. Afterwards, some of my results are described. And finally possible next steps for anyone interested.

TODO: add examples of usage.

## TODO: abstract

## Installation 

[Install Miniconda](https://docs.conda.io/en/latest/miniconda.html), then add some channels thet are important for conda installation by running:

`conda config --add channels pytorch simpleitk anaconda conda-forge`

Afterwards run

`conda create --name <yourEnvNameHere> --file master-thesis/requirements_S2SdDenoising.txt`

Finally, install some code I developed for my use case:

`cd cryoS2Sdrop; pip install --editable ./`
`cd tomoSegmentPipeline; pip install --editable ./`


## TODO: implementation


## TODO: results

## TODO: next steps