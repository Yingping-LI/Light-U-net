Light U-net for lesion segmentation in ultrasound images.
=================

## The repository consists: 
1) code: Our Light U-net and Lighter U-net@C code for image segmentation.
2) pretrained_models: Pretrained models on our ultrasound image data.


## 1.Code for Light U-net and Lighter U-net@C.

### 1.1 Different models:


<div align=center><img width="700" height="500" src="general_network_architecture.png" alt="General network architecture for U-net, Light U-net and Lighter U-net"/></div>

**Introducing depthwise separable convolution into U-net in different ways:**

- **U-net:** “Conv set 1” and “Conv set 2” are regular 3 x 3 convolutions.

- **Light U-net:** “Conv set 1” corresponds to regular 3 x 3 convolution while “Conv set 2” corresponds to depthwise separable convolution (C = 1).

- **Lighter U-net @C:** Both “Conv set 1” and “Conv set 2” are intermediate modules with parameter C, where C \in {1; 2; 4; 8; 16; 32; 64; 128} and C = 1 represents depthwise
separable convolutions. For the first layer with 3 channels, set C to 3 (except when C = 1). For other layers, choose the minimum value of C and the input channel
number to implement the intermediate modules.

Note that both “Conv set 1” and “Conv set 2” are applied with Batch Normalization and ReLU activation function.




### 2.2 Usage of the code:

Set the hyperparameters in "parser_args.py" file, including the selected model, the data path et al., then run the following code to train the model and predict the segmentation.

**Train the models:** 

```python train.py```

**Predict:** 

```python predict.py```



## 2. Information of the Pretrained models:

### 2.1 Data used for training the network:
Ultrasound images with tumor lesions marked by eletronic calipers (blue cross in the images) by the radiologists, see more details in paper [5-6].

Each patient has two ultrasound images taken in 2 mutual-orthogonal directions. We have 208 patients as training data, 70 patients as validation data.
The results summarized in the following table are reported on the test data which consists of 70 patients.

The following images show two examples of the dataset. In each example, the blue cross is the eletronic calipers imposed by radiologists. The contour in red and in greeen correspond to the ground truth and the predicted lesion segmentation by our proposed Lighter U-net@128, separately. Dice Score (DC) is shown in upper right corner.
<center class="half">
    <img src="Patient295_img2.png" width="500"/><img src="Patient307_img2.png" width="500"/>
</center>


### 2.2 Summary of the pretrained models:

The model weights trained on ultrasound image data described in section 2.1 are uploaded in "pretrained_models" folder, and their performances on the test dataset are summarized as below:

| Model                 | number of parameters    | pretrained model size     |  Dice Coefficient | 95% Hausdorff Distance|
| ----------            | :-----------:  | :-----------: | :-----------: | :-----------: |
| U-net                 | 17,267,393     | 65.9 MB       |0.929          |13.980         |
| Light U-net           | 11,000,257     | 42.0 MB       |0.927          |14.039         |
| Lighter U-net @128    | 7,944,661      | 30.4 MB       |0.928          |14.072         |
| Lighter U-net @1      | 1,983,583      | 7.67 MB       |0.918          | 15.926        |






# References:
[1]  https://github.com/milesial/Pytorch-UNet

[2] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,”
in International Conference on Medical image computing and computer-assisted intervention. Springer, 2015, pp. 234–241.

[3] F. Chollet, “Xception: Deep learning with depthwise separable convolutions,” in Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition (CVPR 2017), 2017, pp. 1251–1258.

[4] L. Sifre, “Rigid-motion scattering for image classification,” PhD Thesis, 2014, https://www.di.ens.fr/data/publications/papers/phd_sifre.pdf.

[5] N. Lassau, L. Chapotot, B. Benatsou, and et al., “Standardization of dynamic contrast-enhanced ultrasound
for the evaluation of antiangiogenic therapies: the french multicenter support for innovative and expensive
techniques study,” Investigative Radiology, vol. 47, no. 12, pp. 711–716, 2012.

[6] N. Lassau, J. Bonastre, M. Kind, and et al., “Validation of dynamic contrast-enhanced ultrasound in predicting
outcomes of antiangiogenic therapy for solid tumors: the french multicenter support for innovative and expensive
techniques study,” Investigative Radiology, vol. 49, no. 12, pp. 794, 2014.




# Authors:
Yingping LI: yingpingleee@126.com;

Emilie Chouzenoux: emilie.chouzenoux@centralesupelec.fr;

Benoit Charmettant;

Baya Benatsou;

Jean-Philippe Lamarque;

Nathalie Lassau
