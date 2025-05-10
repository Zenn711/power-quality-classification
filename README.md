# Power Quality Classification Using Deep Learning

## Overview
This project implements a deep learning-based approach to classify power quality disturbances in electrical systems. It utilizes a 2D Convolutional Neural Network (CNN) to analyze Short-Time Fourier Transform (STFT) representations of voltage signals. The model is designed to identify seven types of disturbances: normal, voltage sag, swell, harmonics, transients, notches, and interruptions. The dataset is sourced from IEEE DataPort, and the implementation is done in Python using Keras, optimized for execution in a Jupyter Notebook environment.

## Background
Power quality is a pivotal factor in ensuring the reliability of modern electrical grids. Disturbances such as voltage sags, swells, or harmonics can lead to equipment malfunctions, operational downtime, and significant economic losses. Traditional methods for detecting these anomalies often lack the scalability and precision required for real-time applications. This project leverages deep learning, specifically CNNs, to automate the detection and classification of power quality disturbances by processing time-frequency STFT data, offering a robust and efficient solution.

## Dataset
The dataset is obtained from IEEE DataPort and consists of voltage signal data transformed into STFT representations. It includes the following seven classes:

- **0: Normal** - No disturbances
- **1: Sag** - Voltage drop
- **2: Swell** - Voltage rise
- **3: Harmonics** - Waveform distortion
- **4: Transient** - Sudden short-term changes
- **5: Notch** - Brief voltage dips
- **6: Interruption** - Complete voltage loss

The data is stored in `.mat` files and preprocessed into a format compatible with the 2D CNN input requirements.

## Model Architecture
The classification model is a 2D CNN with the following structure:

- **Conv2D**: 32 filters, 3x3 kernel, ReLU activation
- **Squeeze-and-Excitation Block**: Attention mechanism to emphasize relevant features
- **MaxPooling2D**: 2x2 pool size
- **Conv2D**: 64 filters, 3x3 kernel, ReLU activation
- **Squeeze-and-Excitation Block**: Second attention layer
- **MaxPooling2D**: 2x2 pool size
- **Flatten**: Converts 2D feature maps to 1D
- **Dense**: 128 units, ReLU activation
- **Dropout**: 0.5 rate to prevent overfitting
- **Dense**: 7 units, Softmax activation for multi-class output

The model is compiled using categorical cross-entropy loss and the Adam optimizer. Training parameters include a batch size of 20 and a maximum of 50 epochs, with early stopping triggered by stagnant validation loss.

## Improvements
Several enhancements have been incorporated to optimize performance:

- **Data Augmentation**: Frequency and time masking applied to STFT data to simulate real-world variations and improve model generalization.
- **Early Stopping**: Training stops if validation loss does not decrease for 5 consecutive epochs, mitigating overfitting.
- **Class Weighting**: Computed weights address class imbalance in the dataset, ensuring equitable learning across all disturbance types.
- **Attention Mechanism**: Squeeze-and-Excitation blocks enhance feature extraction by focusing on critical signal components.

## Requirements
The following Python libraries are required:

- `scipy>=1.7.3`
- `numpy>=1.19.5`
- `scikit-learn>=1.0.2`
- `matplotlib>=3.4.3`
- `seaborn>=0.11.2`
- `tensorflow>=2.6.0`

Install dependencies using:

```

pip install -r requirements.txt

```

## Usage
To run the project:

1. **Download the Dataset**: Obtain the dataset from IEEE DataPort and place it in the project directory.
2. **Install Dependencies**: Execute `pip install -r requirements.txt` to set up the environment.
3. **Run the Notebook**: Launch the Jupyter Notebook and execute the cells to train and evaluate the model.

## Expected Results
Post-training evaluation includes:

- **Metrics**: Accuracy, precision, recall, and F1-score per class.
- **Confusion Matrix**: Visualizes classification performance across all classes.
- **Training History**: Plots of accuracy and loss over epochs.
- **Classification Report**: Detailed breakdown of model performance.

## Benefits
This system offers significant advantages for power utility companies (e.g., PLN):

- Real-time monitoring and classification of power quality disturbances.
- Proactive anomaly detection to minimize downtime and equipment damage.
- Scalable solution for large-scale electrical grid management.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are encouraged! Please submit issues or pull requests for bug fixes, enhancements, or new features via GitHub.

## Contact

For questions, feedback, or suggestions:
* Email Muhammad Harits at **[haritsnaufal479@gmail.com](mailto:haritsnaufal479@gmail.com)**
* Or open an issue on the GitHub repo
