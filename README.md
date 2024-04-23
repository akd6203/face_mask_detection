# Face Mask Detection using Python

![Face Mask Detection](https://github.com/akd6203/face_mask_detection/blob/main/faceMask.png)

This project implements a deep learning model for detecting whether a person is wearing a face mask or not. The model is trained using Python and the Keras library with a pre-trained VGG16 architecture.

## Installation

To run this project locally, you need to follow these steps:

1. **Clone the Repository**: First, clone the repository to your local machine using Git:
   ```bash
   git clone https://github.com/akd6203/face_mask_detection.git
   ```

2. **Navigate to Project Directory**: Move into the cloned repository directory:
   ```bash
   cd face_mask_detection
   ```

3. **Install Dependencies**: Install the required Python dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The face mask dataset used in this project can be found [here](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset). It contains over 12,000 images of individuals with and without masks. The dataset has already been preprocessed and split into training, testing, and validation sets. Various transformations, such as rescaling, rotation, and shifting, are applied to the images to prevent overfitting.

## Model Architecture

The model architecture is based on the VGG16 convolutional neural network (CNN), which is a popular choice for image classification tasks. The pre-trained VGG16 model is loaded and modified by adding custom fully connected layers for fine-tuning. The last few layers of the VGG16 model are replaced with dense layers for binary classification (with mask / without mask).

### Methods and Techniques

1. **Optimizer: Adam**: The Adam optimizer is chosen for optimizing the model's weights during training. Adam is an adaptive learning rate optimization algorithm that combines the advantages of two other popular optimizers, RMSprop and AdaGrad. It is known for its fast convergence and good performance on a wide range of deep learning tasks. The default learning rate for Adam is set to 0.001, which is considered a suitable value for many classification problems.

2. **Activation Function: ReLU (Rectified Linear Unit)**: Rectified Linear Unit (ReLU) is used as the activation function for the hidden layers of the neural network. ReLU activation function introduces non-linearity to the model, allowing it to learn complex patterns and relationships in the data more effectively. ReLU function returns the input directly if it is positive, otherwise, it returns zero. This property helps in preventing the vanishing gradient problem and accelerates the convergence of the training process.

3. **Loss Function: Categorical Cross-Entropy**: Categorical Cross-Entropy loss function is selected for this project since it is suitable for multi-class classification problems like image classification, where each input can belong to one class. The categorical cross-entropy loss measures the discrepancy between the actual probability distribution of classes and the predicted probability distribution. By penalizing the model more heavily for incorrect predictions, it encourages the model to make more accurate predictions during training.

4. **Transfer Learning: VGG16**: Transfer learning is employed using the VGG16 (Visual Geometry Group 16) architecture as the base model. VGG16 is a convolutional neural network architecture known for its simplicity and effectiveness in image classification tasks. By leveraging the pre-trained weights of the VGG16 model trained on the ImageNet dataset, the model can quickly learn to detect features relevant to face mask detection, thereby reducing the need for extensive training data.

5. **Custom Layers**: Additional custom fully connected layers are added on top of the VGG16 base model to adapt it for the specific task of face mask detection. These custom layers allow the model to learn high-level features and make predictions based on the detected features from the input images.

## Training

The model is trained using the `fit_generator` function provided by Keras. Early stopping is implemented as a callback to prevent overfitting. The Adam optimizer is used with a learning rate of 0.001, and the categorical cross-entropy loss function is employed. The training parameters include 5 epochs and a batch size of 16.

## Testing

Once the model is trained, it is tested on a separate set of images to evaluate its performance. Individual images, as well as batches of images, are processed to make predictions on whether each person is wearing a mask or not.

## Repository Link

You can find the complete code and resources for this project in the [Face Mask Detection Repository](https://github.com/akd6203/face_mask_detection).
