# Testing the robustness of PredRNN
Code adjusted from [the original PredRNN paper](https://github.com/thuml/predrnn-pytorch).

## Data augmentation
See the [mnist dataset examples](/moving-mnist-example) for .npz files of augmented data.
The different kinds of noise can also be used with the --noise parameter in the command line.
The noise parameter uses the [noise file](/core/utils/noise.py) for calculating the noise in the [mnist dataset](/core/data_provider/mnist.py).

Results of the data augmentations can be found [here](/results_data_augmentation/). 
It consists of 10 examples of each noise type and a text file with the test metrics.
The examples consist of the 10 input frames (gt1-gt10), the actual output (gt11-gt20) and the predicted output(pd11-pd20).

## Fine-tuning
Use the [fine-tuning file](/fine-tuning.py) for fine-tuning. See the [.sh file](/fine-tuning.sh) for the full fine-tuning script used for the report.

The trained models can be found in the [checkpoints folder for fine-tuning](/checkpoints_fine-tuning/mnist_predrnn_v2/). 
Results from the fine-tuning can be found [here](/results_fine-tuning/).