# Celebrities_Classification

##Chosen Model:
The chosen model is a Convolutional Neural Network (CNN) for the classification of celebrity images.

##Model Architecture:
1. Input Layer: Convolutional layer with 32 filters of size (3,3) and ReLU activation.
2. MaxPooling layer with pool size (2,2).
3. Convolutional layer with 48 filters of size (3,3) and ReLU activation.
4. MaxPooling layer with pool size (2,2).
5. Dropout layer with a dropout rate of 0.5.
6. Flatten layer to convert 2D features to a 1D vector.
7. Fully connected Dense layer with 512 units and ReLU activation.
8. Fully connected Dense layer with 128 units and ReLU activation.
9. Output layer with 10 units and softmax activation for multi-class classification.

##Compilation:
- Optimizer: Adam
- Loss Function: Sparse categorical crossentropy
- Metrics: Accuracy

##Data Preparation:
- The code loads images of celebrities (Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, Virat Kohli) from the specified directory.
- Images are resized to (128, 128) pixels.
- Data and corresponding labels are created, and the dataset is normalized.

##Data Augmentation:
- Data augmentation is applied using `ImageDataGenerator` from TensorFlow on the training dataset. Augmentation includes rotation, width and height shifts, shear, zoom, and horizontal flip.


##Model Training:
- The model is trained using the training dataset with 200 epochs and a batch size of 128.
- Learning rate scheduling is implemented with a custom scheduler function.

##Training Results:
- Training accuracy, validation accuracy, training loss, and validation loss are plotted and saved as images.
- The training process and results are printed to the console.

##Model Evaluation:
- The trained model is evaluated on the test dataset, and accuracy is printed.
- The trained model is evaluated on the test set, resulting in an accuracy of 79.41%.

##Model Prediction:
- The model predicts labels for the test dataset.
- The actual and predicted labels are saved in a CSV file.

##Critical Findings:
- The model's accuracy and loss over epochs are visualized.
- The trained model is evaluated on the test set, and accuracy is reported.
- Predictions on the test set are saved to a CSV file for further analysis.


