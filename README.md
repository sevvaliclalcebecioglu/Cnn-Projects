# _CNN Image Classification Projects_

This repository contains **four different image classification projects** developed using **deep learning and Convolutional Neural Networks (CNNs)**. Each project includes its **own dataset**, **model architecture**, and **user interface**. The projects utilize **transfer learning** and **Streamlit-based interactive applications**.

---

## _Selected Projects and Datasets_
1. **Fruit & Vegetable Classification** – [HuggingFace Space](https://huggingface.co/spaces/sevvaliclal/FruitOrVegetablesModel)  
   **Dataset:** [kritikseth/fruit-and-vegetable-image-recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)

![fruits_vegetables](assets/fruits_vegetables.gif)

2. **Date Fruit Classification** – [HuggingFace Space](https://huggingface.co/spaces/sevvaliclal/DateFruitModel)  
   **Dataset:** [wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment](https://www.kaggle.com/datasets/wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment)

![date_fruit](assets/date_fruit.gif)

3. **Grapevine Disease Classification** – [HuggingFace Space](https://huggingface.co/spaces/sevvaliclal/GrapevineDiseaseModel)  
   **Dataset:** [rm1000/augmented-grape-disease-detection-dataset](https://www.kaggle.com/datasets/rm1000/augmented-grape-disease-detection-dataset)

![grape](assets/üzüm.gif)

4. **Rice Classification** – [HuggingFace Space](https://huggingface.co/spaces/sevvaliclal/rice-species-predictor)  
   **Dataset:** [muratkokludataset/rice-image-dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)

![rice](assets/rice.gif)

---

## _Libraries Used and Their Purpose_

The following libraries were used across the projects:

- **Python 3.x**: Main programming language for coding, data processing, and model management.  
- **TensorFlow / Keras**: For defining, training, and predicting deep learning models.  
- **Streamlit**: To create user-friendly interactive web interfaces.  
- **PIL (Pillow)**: Loading, resizing, and transforming images.  
- **NumPy**: Numerical computations, matrix operations, and data preparation.  
- **OpenCV (cv2)**: Image processing and some data manipulations.  
- **scikit-learn**: Calculating model performance metrics (accuracy, precision, recall, F1-score).  
- **Kagglehub**: Automatic dataset downloading and usage.  
- **ImageDataGenerator**: Image normalization, augmentation, and preparation of train/validation sets.

---

## _Data Processing Steps_

1. **Image Resizing**: All images were resized to **224x224 pixels** to match the CNN input size.  
2. **Normalization**: Pixel values scaled to `[0,1]` for faster and stable learning.  
3. **RGBA → RGB Conversion**: Removed alpha channel for consistency with RGB models.  
4. **Data Split and Augmentation**:  
   - Train/validation split **80% / 20%**.  
   - `ImageDataGenerator` used for data augmentation (rotation, flipping, etc.) in some projects.

---

## _Modeling Approach_

### _Transfer Learning Models_

All projects used **VGG16 pre-trained models**:

- **Base Model**: `VGG16(weights='imagenet', include_top=False)`  
- **Flatten Layer**: Converts feature maps to 1D vector.  
- **Dense Layer**: 1024 neurons with ReLU activation.  
- **Output Layer**: Number of classes neurons with softmax activation.  
- **Compile**: Loss = `categorical_crossentropy`, Optimizer = `Adam`, Metrics = `accuracy`.  

> This approach allows **high accuracy with limited data** via transfer learning.

### _Regularization_

- Dropout and data augmentation used to prevent overfitting.  
- Validation sets were used to monitor generalization performance.

---

## _User Interface_

Each project contains a **Streamlit-based interactive UI**:

- **Image Display**: Shows the selected image to the user.  
- **Prediction**: User selects an option via dropdown or radio button.  
- **Result**: Immediate feedback on the prediction.  
- **Game Mode**: Random images for user to guess (e.g., 4 grape leaves or 6 rice grains).

---

## _Project Summaries_

### _1. Fruit & Vegetable Classification_
- **Goal**: Classify whether an image is a fruit or a vegetable.  
- **Dataset**: `kritikseth/fruit-and-vegetable-image-recognition`  
- **Model**: VGG16 + Flatten + Dense(1024) + Dense(classes)  
- **UI**: Random image selection with user prediction and instant feedback  
- **HuggingFace Space**: [Link](https://huggingface.co/spaces/sevvaliclal/FruitOrVegetablesModel)

### _2. Date Fruit Classification_
- **Goal**: Classify date fruit types  
- **Dataset**: `wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment`  
- **Model**: VGG16 + Dense(1024) + Dense(9)  
- **UI**: Dropdown selection with random images  
- **HuggingFace Space**: [Link](https://huggingface.co/spaces/sevvaliclal/DateFruitModel)

### _3. Grapevine Disease Classification_
- **Goal**: Classify grape leaf diseases  
- **Dataset**: `rm1000/augmented-grape-disease-detection-dataset`  
- **Model**: VGG16 + Dense(1024) + Dense(4)  
- **UI**: 4-image selection with user prediction  
- **HuggingFace Space**: [Link](https://huggingface.co/spaces/sevvaliclal/GrapevineDiseaseModel)

### _4. Rice Classification_
- **Goal**: Classify rice species  
- **Dataset**: `muratkokludataset/rice-image-dataset`  
- **Model**: VGG16 + Dense(1024) + Dense(5)  
- **UI**: File upload + random image game mode  
- **HuggingFace Space**: [Link](https://huggingface.co/spaces/sevvaliclal/rice-species-predictor)

---

## _Model Evaluation_

- **Metrics**: Accuracy, Precision, Recall, F1-score  
- **Train / Validation**: Performance logs printed to console after training  
- **Class-wise Analysis**: Detailed performance for each class

---

## _Usage Instructions_

1. Run the project in terminal:
```bash
streamlit run <project_file_name>.py
