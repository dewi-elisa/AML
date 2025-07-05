# Testing the robustness of PredRNN

Code adjusted from [the original PredRNN paper](https://github.com/thuml/predrnn-pytorch).

See the [mnist dataset examples](/moving-mnist-example) for .npz files of augmented data.

The different kinds of noise can also be used with the --noise parameter in the command line.

Results of the data augmentations can be found [here](/results_data_augmentation/). It consists of 10 examples of each noise type and a text file with the test metrics. The examples consist of the 10 input frames (gt1-gt10), the actual output (gt11-gt20) and the predicted output(pd11-pd20).

Use the [finetuning file](/finetuning.py) for finetuning. 

See the [.sh file](/finetuning.sh) for the full finetuning script.

The trained models can be found in the [checkpoints folder for finetuning](/checkpoints_finetuning/mnist_predrnn_v2/).

Results from the fine tuning can be found [here](/results_finetuning/).