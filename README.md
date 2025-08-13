# TokYy Framework 

TokYy framework is built with PyTorch, and is used to create, train and analyse machine learning models. The framework takes advantage of high modularity and complex training monitoring and analyisis capabilities.

## Prerequisites

Libraries required:

> Install PyTorch [here]( https://pytorch.org/get-started/locally/ ). <br>
> Torchvision `pip install torchvision` <br>
> H5py `pip install h5py` <br>
> Numpy `pip install numpy` <br>
> Matplotlib `pip install matplotlib` <br>
> Opencv `pip install python-opencv` <br>
> Pillow  `pip install pillow`  <br>

Terminal integration:

> On linux, open the shell configuration folder (e.g., if `$SHELL` outputs `bin/bash`, open `~/.bashrc`), and add the following two lines. <br>
> `alias tokyystar = "$path/to/tokyy/tokyystar` <br>
> `alias tokyystar = "$path/to/tokyy/tokyyplot` <br>

## Training a model using CLI

> Create a directory `DIR` and move `tokyY` inside. Now any any extra file inide `DIR` can be run using the commnad `tokyystar`.

`tokyystar` arguments:

> First argument should aslways be the name (without .py extension) of the file to run inside `DIR`. <br>
> `-h` for description of the rest of the arguments.
> `--checkpoint-dir` sets the checkpoint saving directory. Defaults to `DIR/tokyy/checkpoints`. <br>
> `--checkpont-name` sets the checkpoint saving name. Defaults to `default.pt`. Recommended to use `.pt`, `.pth,` or `.ckpt` as extension. <br>
> `--arch` sets the architecture. Defaults to `cbam` which is a U-Net With Residual Blocks and CBAM. <br>
> `--no-ask-before` disables required user input before loading the dataset and starting the training. <br>
> `--batch-size`
> `--input-image` for vision models. Takes two integers separated by space.
> `--accum-steps` sets accumulation steps.

## Plotting model analytics

> Inside `DIR/tokyy/results` are multiple subdirs where running `tokyyplot` saves models analytics.

`tokyyplot` arguments:

> `--checkpoint-dir` sets the checkpoint loading directory. Defaults to `DIR/tokyy/checkpoints`. <br>
> `--checkpont-name` sets the checkpoint loading name. <br>
> `--loss` saves train, validation, and test losses across all epochs inside `results/losses`. <br>
> `--_loss` saves the sub-losses that sum up to the final one inside `test`, `val`, and `traing` subdirs of `results/_losses` <br>
> `--metric` saves the metrics inside `results/metrics` <br>
> `--lr` saves the learning rate inside `results/learning_rates`. Useful when using a scheduler. <br>
> `--pred` (if NYU-Depth-V2) is installed locally) saves eight predictions with input, prediction, and ground truth. Helpful for comparing multiple models on same eight predictions. <br>














