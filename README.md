# Light U-net
Introducing depthwise separable convolution into U-net in different ways.



<div align=center><img width="700" height="500" src="general_network_architecture.png" alt="General network architecture for U-net, Light U-net and Lighter U-net"/></div>

**Different models:**

- **U-net:** “Conv set 1” and “Conv set 2” are regular 3 x 3 convolutions.

- **Light U-net:** “Conv set 1” corresponds to regular 3 x 3 convolution while “Conv set 2” corresponds to depthwise separable convolution (C = 1).

- **Lighter U-net @C:** Both “Conv set 1” and “Conv set 2” are intermediate modules with parameter C, where C \in {1; 2; 4; 8; 16; 32; 64; 128} and C = 1 represents depthwise
separable convolutions. For the first layer with 3 channels, set C to 3 (except when C = 1). For other layers, choose the minimum value of C and the input channel
number to implement the intermediate modules.

Note that both “Conv set 1” and “Conv set 2” are applied with Batch Normalization and ReLU activation function.

&nbsp;

**Original paper: Light U-net for lesion segmentation in ultrasound images.**






## Information of the Pretrained models:
The model weights trained on our ultrasound image data are uploaded in "pretrained_models" folder, and their performances on our test data are summarized as below:

| Model                 | number of parameters    | pretrained model size     |  Dice Coefficient | 95% Hausdorff Distance|
| ----------            | :-----------:  | :-----------: | :-----------: | :-----------: |
| U-net                 | 17,267,393     | 65.9 MB       |0.929          |13.980         |
| Light U-net           | 11,000,257     | 42.0 MB       |0.927          |14.039         |
| Lighter U-net @128    | 7,944,661      | 30.4 MB       |0.928          |14.072         |
| Lighter U-net @1      | 1,983,583      | 7.67 MB       |0.918          | 15.926        |


&nbsp;

**Note:** our ultrasound image data has electronic calipers imposed by radiologists during the patient treatment to mark the tumor location, so the pretrained weights may work very badly on your ultrasound image data without such calipers. 


## Usage:

**Train the models:** set the hyperparameters in "parser_args.py" file, including the selected model, the data path et al., then run the following code to train the model.

```python train.py```

**Predict:** 

```python predict.py```

## Aknowledgements:
https://github.com/milesial/Pytorch-UNet

O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,”
in International Conference on Medical image computing and computer-assisted intervention. Springer, 2015, pp. 234–241.




## Contact me

Email: yingpingleee@126.com
