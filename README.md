# Text Generator Based on NEXT Character Prediction Using MLP 🔡✨

Welcome to Text Generator Based on NEXT Character Prediction Using MLP! 🚀 This project explores the exciting world of neural networks for text generation, using Multi-Layer Perceptrons (MLP) to predict the next character in a sequence. Whether you're a Shakespeare enthusiast or just love experimenting with AI, this tool lets you dive deep into the magic of text generation.

## 🌟 Features

- **Streamlit Web App**: A sleek, user-friendly interface to interact with the model.
- **Model Selection**: Switch between trained models and configure hyperparameters like block size and embedding dimensions.
- **Dynamic Text Generation**: Generate text on-the-fly based on user-provided input.
- **Multiple Datasets**: Trained on various datasets, including a Shakespeare-specific corpus for poetic flair.
- **Interactive Training Insights**: Delve into training details through dedicated Jupyter notebooks.

---

## Streamlit Application Link:-
[Text Generation Model](https://next-character-predictor-using-mlp.streamlit.app/)

## Table of Contents

- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Training process](#training-process)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contributors](#contributors)

## Project Overview

The primary focus of this project was to implement and fine-tune a vanilla neural network architecture for text generation. The model was designed to predict subsequent characters based on preceding sequences of characters. The model's parameters are adjustable, allowing for experimentation with different embedding sizes and block sizes to achieve optimal performance.

## Model Architecture

The text generation model employs a basic neural network architecture, characterized by the following key parameters:

- **Embedding Size:** The embedding size is a crucial hyperparameter in a neural network, especially for text data. It refers to the dimensionality of the vector space in which words or characters are represented. Higher embedding sizes allow the model to capture more nuanced relationships and features of the input data. For example, an embedding size of 60 means each character is represented as a 60-dimensional vector, capturing various aspects of its meaning and context within the text. Adjusting the embedding size can significantly impact the model's ability to learn and generate coherent text.

- **Block Size:** The block size refers to the length of the input sequence that the model uses to predict the next character. It determines how much context the model considers when making predictions. For instance, a block size of 100 means the model looks at the preceding 100 characters to predict the next one. Larger block sizes provide more context, which can improve prediction accuracy, but also increase computational complexity. Finding the right balance between block size and model performance is essential for effective text generation.

The neural network architecture consists of the following layers:
1. **Embedding Layer:** The embedding layer converts the input characters into dense vectors of fixed size (embedding size). This layer helps the model learn the relationships between characters and their context within the text data.

2. **Multi-Layer-Perceptron (MLP):** The MLP layer processes the embedded input sequences and extracts relevant features to predict the next character. It consists of multiple fully connected layers with activation functions to capture complex patterns in the data.

## Training Process

The model was trained using the following steps:

1. **Data Preparation:** The text data was divided into sequences of fixed block size. Each sequence was used as input to the model, with the corresponding next character serving as the target output.

2. **Training Loop:** The model was trained over multiple epochs, with each epoch consisting of a forward pass (to calculate the predicted next character) and a backward pass (to update the model weights based on the prediction error).

3. **Loss Function:** The model's performance was measured using cross-entropy loss, which quantifies the difference between the predicted character probabilities and the actual target characters.

4. **Optimization:** The Adam optimizer was used to minimize the loss function and update the model weights iteratively.

## Dependencies

To run the project, you need to install the following dependencies:

- Python 3.7+
- NumPy
- PyTorch
- Matplotlib
- Streamlit

You can install the necessary packages using the following command:

```bash
pip install numpy torch matplotlib streamlit
```

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/Zeenu03/Text-Generator-based-upon-next-character-prediction-from-MLP.git
    ```

2. Navigate to the project directory:

    ```bash
    cd text-generation-model
    ```

3. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

4. Customize the model parameters (embedding size and block size) using the Streamlit interface and start generating text.

---

## 🏗️ Project Structure

```plaintext
📁 Text-Generator-Based-on-NEXT-Character-Prediction-USING-MLP
├── app.py                # Streamlit app for user interaction
├── model.ipynb           # Jupyter notebook for model training
├── shakespear_model.ipynb # Notebook for Shakespeare-specific model
├── models/               # Directory of pre-trained model files
├── text files/           # Training datasets (text files)
├── README.md             # This sexy readme ✨
├── requirements.txt      # Dependencies list
└── SECURITY.md           # Security-related documentation
```

---

## 📊 Key Components

### **Streamlit Interface**
- Select your dataset, configure model parameters, and input seed text.
- View real-time generated text and adjust settings for experimentation.

### **Model Training**
- **Notebook**: Use `model.ipynb` for general datasets or `shakespear_model.ipynb` for the Shakespeare dataset.
- **Optimization**: Models are trained with MLP architecture and a cross-entropy loss function.

### **Text Generation**
The `generate_text` function predicts the next character based on the input sequence, using the model's learned patterns.

---


## 🌍 Datasets and Models
We provide pre-trained models and datasets for instant exploration, including:
- **Shakespeare Corpus**: Dive into poetic text generation.
- **Custom Datasets**: Use your own text files for personalized models.

---

## Results

The training loss for different configurations of embedding size and block size is visualized below:

<img src="Training_loss.png" alt="Training Loss" width="600"/>

*Example of generated text with different model configurations:*

- **Model 11 (Block Size = 10, Embedding Size = 60)**
    ```
    Generated Text: two years ago i wrote about what i called "a huge, unexploited opportunity in startup founders hate most about fundraising take less time, not more. with a classic fixed size round as a legitimate, thei startups more pliable in negotiations, since they'r
    ```

- **Model 12 (Block Size = 10, Embedding Size = 150)**
    ```
    Generated Text: two years ago i wrote about what i calle will becom? mere time gives investors. the former is obviously a better predictor of success. [9] some of the randomness is concealed by the fact that investments than they were a year ago. and meanwhile the past
    ```

## Conclusion

This project successfully demonstrates the capability of a basic neural network to generate coherent text based on preceding sequences. By fine-tuning the model's parameters, we achieved notable improvements in text generation quality. Further experimentation with more advanced architectures and larger datasets could yield even better results.