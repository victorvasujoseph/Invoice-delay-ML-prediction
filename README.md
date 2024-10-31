# Invoice Payment Delay Prediction

**Author:** Victor Vasu Joseph


## Executive Summary

This project focuses on predicting invoice payment delays based on customer behavior and transaction history. By identifying customers likely to delay payments, businesses can improve cash flow management, streamline accounts receivable, and implement proactive collection strategies.

## Rationale

Delayed payments are a significant challenge in managing cash flow for businesses. Predicting payment delays allows companies to take preemptive measures, prioritize follow-ups, and reduce the risk associated with late payments. This project leverages machine learning to offer predictive insights, assisting finance teams in efficient accounts receivable management.

## Research Question

Can we accurately predict whether an invoice will be paid on time based on customer behavior and transaction characteristics?

## Data Sources

The data includes the following features:
- **person_age**: Age of the customer
- **person_income**: Annual income of the customer
- **person_home_ownership**: Home ownership status (e.g., Rent, Own, Mortgage)
- **person_emp_length**: Employment length in years
- **loan_intent**: Purpose or type of transaction related to the invoice
- **loan_grade**: Customer credit grade
- **loan_amnt**: Invoice amount
- **loan_int_rate**: Interest rate (if applicable)
- **loan_status**: Target variable indicating whether the invoice is paid on time (0) or delayed (1)
- **loan_percent_income**: Invoice amount as a percentage of customer income
- **cb_person_default_on_file**: Historical default information of the customer
- **cb_person_cred_hist_length**: Customer's credit history length

## Methodology

The project follows these steps:
1. **Data Preprocessing**: Handling missing values, encoding categorical features, and scaling numeric variables.
2. **Model Training**:
   - **Random Forest**: Trained with hyperparameter tuning using GridSearchCV to optimize performance.
   - **Neural Network**: Trained using Keras with a structured architecture, using dropout layers to prevent overfitting.
3. **Evaluation**:
   - **Random Forest**: Evaluated with a confusion matrix, classification report, and accuracy score.
   - **Neural Network**: Evaluated using accuracy and loss curves, and tested on a hold-out set to assess generalization.


### Project Structure

- **Data Preprocessing and EDA**: Handles missing values, feature encoding, and scaling.
- **Model Training and Tuning**: Builds and tunes both the Random Forest and Neural Network models.
- **Evaluation and Analysis**: Provides insights into model performance using confusion matrices, accuracy/loss plots, and classification metrics.

---

#### Repository Structure

- **data**: Contains the raw and processed dataset files.
- **result**: Visualizations and plots used in the README and reports.
- **README.md**: Documentation (you are here).
- **InvoiceDelay.ipynb**: The main Jupyter notebook with all analysis and model code.

## Step-by-Step Code Explanation

### 1. Import Libraries
- **pandas, numpy**: Used for data manipulation and numerical operations.
- **seaborn, matplotlib.pyplot**: For creating visualizations to explore data distributions and correlations.
- **train_test_split, GridSearchCV**: From `sklearn.model_selection`, used for splitting the dataset and performing hyperparameter tuning via grid search.
- **StandardScaler, LabelEncoder, SimpleImputer**: From `sklearn.preprocessing`, used for feature scaling, encoding categorical features, and imputing missing values.
- **RandomForestClassifier**: A tree-based model to predict loan status.
- **classification_report, confusion_matrix**: Used to evaluate the classification model.
- **Sequential, Dense, Dropout**: Layers from `tensorflow.keras` for building the neural network.
- **EarlyStopping**: A callback to stop training when validation loss doesn’t improve, preventing overfitting.
- **tensorflow (tf)**: To set up and train the neural network model.

## 2. Load and Explore the Dataset
- **Load the Data**: Using `pd.read_csv()`, the dataset is loaded and stored in `df`.
- **Initial Overview**: Display the first few rows, data types, missing values, and summary statistics using `df.head()`, `df.info()`, and `df.isnull().sum()`. This helps in understanding the structure of the dataset and identifying missing values or irregularities.

## 3. Data Visualization
- **Distribution Plots**: A loop is used to create distribution plots for each numerical feature to understand their distribution (e.g., normal, skewed).
- **Target Distribution Plot**: A count plot for the target variable `payment_status` shows the distribution between on-time and delayed payments.
- **Correlation Heatmap**: For numerical features, a heatmap is generated to visually inspect the correlation between features, helping identify multicollinearity and potential feature interactions.

## 4. Data Preprocessing
- **Missing Value Handling**: Columns with more than 30% missing values are dropped. For remaining missing values, numerical columns are filled with the mean, and categorical columns with the most frequent values.
- **Encoding Categorical Variables**:
  - **Binary Encoding**: Features like `historical_default` are encoded using `LabelEncoder`.
  - **One-Hot Encoding**: Multi-category features like `home_ownership`, `purchase_intent`, and `credit_grade` are encoded with `pd.get_dummies()`.
- **Feature Scaling**: Standardize numerical columns with `StandardScaler` to normalize the data, which helps neural network training.

## 5. Feature Engineering
- **Column Renaming**: Renaming columns for clarity and ease of understanding.
- **New Feature**: A new feature, `invoice_to_income_ratio`, is calculated as the ratio of loan amount to annual income, which can be insightful for risk assessment.

## 6. Train-Test Split
- **Define Features and Target**: `X` represents the features, and `y` is the target variable (`payment_status`).
- **Train-Test Split**: The dataset is split into training and testing sets with an 80-20 split using `train_test_split()`. This provides separate data for model training and evaluation.

## 7. Random Forest Model with Hyperparameter Tuning
- **Parameter Grid**: Define a parameter grid for `RandomForestClassifier` (e.g., `n_estimators`, `max_depth`).
- **GridSearchCV**: Use `GridSearchCV` to perform 5-fold cross-validation across multiple parameter combinations to find the best hyperparameters based on accuracy.
- **Evaluate Performance**: Print the best parameters and cross-validation score, followed by a confusion matrix and classification report on the test set.

## 8. Feature Importance Visualization
- **Feature Importance Plot**: Using `best_model.feature_importances_`, the importance of each feature in the Random Forest model is displayed in a bar plot, helping in understanding which features contribute most to the predictions.

## 9. Neural Network Model
- **Define Neural Network Architecture**: Use `Sequential` with three hidden layers and dropout for regularization to reduce overfitting.
  - **Input Layer**: First layer with 64 neurons and ReLU activation.
  - **Hidden Layers**: Additional layers with dropout for regularization.
  - **Output Layer**: A single neuron with a sigmoid activation, suitable for binary classification.
- **Compile the Model**: Use the Adam optimizer, binary cross-entropy loss, and track accuracy as the evaluation metric.
- **Early Stopping**: Set up `EarlyStopping` with patience to halt training if the validation loss doesn’t improve, minimizing overfitting.

## 10. Train the Neural Network
- **Training the Model**: Train the neural network on `X_train` and `y_train` for 30 epochs with validation split and batch size of 32.
- **Monitor Training**: The `history` object stores training and validation accuracy and loss across epochs for visualization.

## 11. Neural Network Evaluation
- **Evaluate on Test Set**: Print final test accuracy to assess the model’s generalization.
- **Training History Plot**: Plot both accuracy and loss for training and validation sets, allowing visual assessment of the model's training progress and potential overfitting or underfitting signs.


## Results

The model performs well on the dataset, showing high accuracy and balanced generalization between training and validation data. Here's a breakdown of the results:

![result_data](/result/4.png)
![result_data](/result/5.png)

### Model Performance Summary

### 1. **Random Forest Classifier**

- **Best Parameters**:
  - `max_depth`: 20
  - `min_samples_leaf`: 1
  - `min_samples_split`: 2
  - `n_estimators`: 100
- **Best Cross-Validation Score**: 0.7028
- **Confusion Matrix**:


- **True Negatives (984)**: Correctly identified non-delayed payments.
- **False Positives (19)**: Predicted delay, but it was on-time.
- **False Negatives (98)**: Predicted on-time, but it was delayed.
- **True Positives (203)**: Correctly identified delayed payments.

- **Classification Report**:
- **Precision**: 0.91 for both non-delayed and delayed payments.
- **Recall**:
  - 0.98 for non-delayed payments.
  - 0.67 for delayed payments, indicating moderate recall on delayed payments.
- **F1-Score**: 0.86 (macro average).
- **Accuracy**: 0.91.

### 2. **Gradient Boosting Classifier**

- **Best Parameters**:
- `learning_rate`: 0.1
- `max_depth`: 5
- `n_estimators`: 100
- **Best Cross-Validation Score**: 0.7116
- **Confusion Matrix**:

- **Classification Report**:
- **Precision**: 0.91 for non-delayed and 0.92 for delayed payments.
- **Recall**:
  - 0.98 for non-delayed payments.
  - 0.69 for delayed payments, showing an improvement over Random Forest.
- **F1-Score**: 0.87 (macro average).
- **Accuracy**: 0.91.

### 3. **Neural Network Model**
- **Training and Validation Accuracy**:
- Converged around 90-91% over 30 epochs.
- **Training and Validation Loss**:
- Gradual decrease with a slightly higher validation loss, showing good generalization.
- **Test Results**:
- **Accuracy**: 0.8919
- **Recall**: 0.6113 for delayed payments, indicating it struggles slightly with recall for this class compared to the tree models.

## Conclusion
For this dataset, **Gradient Boosting** is the preferred model as it provides the best balance between precision, recall, and F1-score for delayed payments. **Random Forest** also performs well, though slightly lower in recall for delayed payments. The **Neural Network** model, while achieving high accuracy, could benefit from further tuning, especially to improve recall on delayed payments.

## Project Structure
- **data**: Contains the raw and processed dataset files.
- **results**: Visualizations and plots used in the README and analysis.
- **notebooks**: Jupyter notebooks for data analysis, model training, and evaluation.
- **README.md**: Project summary and model performance overview (you are here).

### Next Steps

For further enhancement:
- Experiment with more sophisticated ensemble models like XGBoost.
- Fine-tune the neural network model architecture for better performance.
- Explore ensemble stacking methods to combine model predictions for improved accuracy.

---

## Credits
Data for this project was sourced from [Kaggle's Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset).

