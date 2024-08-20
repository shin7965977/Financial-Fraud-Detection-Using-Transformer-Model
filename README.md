# Financial Fraud Detection Using Transformer Model

## Authors
- Chen-Yu Liu
- Deepshika

## Background
This project is an application of deep learning knowledge acquired in the classroom to the real-world problem of financial fraud detection. Notably, this is our first attempt at implementing such a model, marking a significant milestone in our journey of learning and applying deep learning techniques.

## Data Source
The dataset used for this project can be found on Kaggle: [Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data).

## Objective
The primary objective of this project is to develop a model that can successfully detect fraudulent transactions with high accuracy, leveraging the power of deep learning.

## Technical Details
- **Model Architecture**: Transformer-based model
- **Data Processing**: Included steps such as data cleaning, normalization, and feature engineering.
- **Training**: The model was trained on the provided dataset with a focus on optimizing the detection of fraudulent transactions.
- **Evaluation Metrics**: The performance of the model was evaluated using precision, recall, F1-score, and accuracy.

## Challenges
One of the primary challenges encountered during this project was the extensive training time required for the model. Additionally, despite the complexity of the model, the overall performance was lower than expected, particularly in detecting the minority class of fraudulent transactions.

## Results

| Metric          | Class 0 (Non-Fraud) | Class 1 (Fraud) | Macro Avg | Weighted Avg |
|-----------------|---------------------|-----------------|-----------|--------------|
| **Precision**   | 1.00                | 0.44            | 0.72      | 1.00         |
| **Recall**      | 1.00                | 0.76            | 0.88      | 1.00         |
| **F1-Score**    | 1.00                | 0.56            | 0.78      | 1.00         |
| **Support**     | 110715              | 429             | 111144    | 111144       |

### Confusion Matrix

|                 | Predicted Class 0 | Predicted Class 1 |
|-----------------|-------------------|-------------------|
| **Actual Class 0** | 110296            | 419               |
| **Actual Class 1** | 103               | 326               |

The results indicate that while the model performs exceptionally well in identifying non-fraudulent transactions, it struggles with accurately identifying fraudulent ones, as shown by the lower precision and F1-score for Class 1.

## Areas for Improvement and Future Work
- **Model Optimization**: To address the long training times and improve performance, further optimization of the model architecture and hyperparameters is necessary.
- **Data Augmentation**: Exploring data augmentation techniques to better balance the classes and improve the model's ability to detect fraudulent transactions.
- **Alternative Models**: Considering alternative deep learning models or even ensemble methods to enhance detection capabilities.
- **Feature Engineering**: Additional feature engineering could be explored to extract more relevant features that may improve the model's performance.

## Lessons Learned
Through this project, we gained hands-on experience with deep learning models, particularly in the context of financial fraud detection. We learned about the complexities involved in training and optimizing models and the importance of addressing class imbalance in datasets.

## Data Visualization in Power BI
In addition to the model implementation, we utilized Power BI to visualize the dataset, providing a deeper understanding of the data's characteristics. The Power BI file includes the following visualizations:
1. **Pie Chart**: Represents the distribution of transactions across different categories.
2. **Bar Chart**: Shows the frequency of transactions by merchant or category.
3. **Scatter Chart**: Displays the relationship between transaction amount and other features.
4. **Line Chart**: Illustrates trends in transaction data over time.

These visualizations help in identifying patterns and anomalies in the data, which are crucial for understanding and improving the model.

## Requirements
To replicate this project, the following libraries and packages are required:
- Python 3.x
- TensorFlow / PyTorch (depending on the model implementation)
- scikit-learn
- Pandas
- NumPy
- Matplotlib
- Power BI (for data visualization)

Please ensure that you have the above dependencies installed in your environment before running the code.
