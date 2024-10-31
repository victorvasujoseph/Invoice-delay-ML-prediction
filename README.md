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

- **person_age**: Age of the customer (e.g., 24, 27, 30).
- **person_income**: Annual income of the customer, in dollars (e.g., 28,000, 64,000).
- **person_home_ownership**: Type of home ownership (e.g., OWN, RENT, MORTGAGE).
- **person_emp_length**: Employment length in years, indicating the stability of employment (e.g., 6.0, 0.0, 10.0).
- **loan_intent**: Purpose of the loan, such as HOMEIMPROVEMENT, PERSONAL, EDUCATION, DEBTCONSOLIDATION, and MEDICAL.
- **loan_grade**: Credit grade assigned to the customer (e.g., A, B, C, D, E).
- **loan_amnt**: Amount of the loan or invoice, in dollars (e.g., 10,000, 13,000, 16,000).
- **loan_int_rate**: Interest rate for the loan; some values may be missing.
- **loan_status**: Payment status, where `0` indicates on-time payment and `1` indicates a delay.
- **loan_percent_income**: Percentage of the loan amount relative to the customer’s income (e.g., 0.36, 0.16).
- **cb_person_default_on_file**: Binary indicator showing whether the customer has a history of defaults (`N` for no, `Y` for yes).
- **cb_person_cred_hist_length**: Length of the customer’s credit history, in years (e.g., 2, 10, 6).

### Data Summary

- **Rows**: 6,516 entries, with various columns containing some missing values.
- **Columns**: 12 columns (5 numerical, 3 floating-point, 4 categorical)
  
This dataset provides a comprehensive view of customer demographics, transaction information, and payment behaviors, which form the basis for predicting invoice delays.

## Project Workflow

The project follows these main steps:

### 1. Data Preprocessing
   - **Missing Values:** Columns with significant missing values were dropped, and imputation techniques were applied to handle minor gaps.
   - **Feature Engineering:** New features were created, such as the invoice-to-income ratio, to add predictive value.
   - **Column Renaming:** Columns were renamed for better readability and consistency.
   - **Encoding Categorical Variables:** Categorical features were transformed into a format suitable for machine learning models using binary and multi-category encoding.

### 2. Exploratory Data Analysis (EDA)
   - **Distribution Analysis:** Visualized distributions for numerical features to understand their patterns and identify potential outliers.
   - **Count Plots for Categorical Variables:** Examined distributions of categorical data, providing insights into customer demographics and behaviors.
   - **Correlation Analysis:** Generated a correlation heatmap to identify relationships among numerical features, aiding in feature selection.

### 3. Model Training and Evaluation
   - **Algorithms Used:** Trained models including Random Forest, Gradient Boosting, and a Neural Network (using Keras) to predict payment delays.
   - **Hyperparameter Tuning:** Optimized model parameters using grid search to improve accuracy.
   - **Evaluation Metrics:** Measured performance using metrics such as accuracy, precision, recall, F1-score, and a confusion matrix.

### 4. Model Selection
   - **Feature Importance:** Assessed feature importance to determine which variables had the greatest impact on predicting delays.
   - **Performance Comparison:** Selected the model with the best combination of metrics, establishing it as the primary prediction tool.


### Repository Structure

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

### 2. Load and Explore the Dataset
- **Load the Data**: Using `pd.read_csv()`, the dataset is loaded and stored in `df`.
- **Initial Overview**: Display the first few rows, data types, missing values, and summary statistics using `df.head()`, `df.info()`, and `df.isnull().sum()`. This helps in understanding the structure of the dataset and identifying missing values or irregularities.

### 3. Data Visualization
- **Distribution Plots**: A loop is used to create distribution plots for each numerical feature to understand their distribution (e.g., normal, skewed).
- **Target Distribution Plot**: A count plot for the target variable `payment_status` shows the distribution between on-time and delayed payments.
- **Correlation Heatmap**: For numerical features, a heatmap is generated to visually inspect the correlation between features, helping identify multicollinearity and potential feature interactions.

### 4. Data Preprocessing
- **Missing Value Handling**: Columns with more than 30% missing values are dropped. For remaining missing values, numerical columns are filled with the mean, and categorical columns with the most frequent values.
- **Encoding Categorical Variables**:
  - **Binary Encoding**: Features like `historical_default` are encoded using `LabelEncoder`.
  - **One-Hot Encoding**: Multi-category features like `home_ownership`, `purchase_intent`, and `credit_grade` are encoded with `pd.get_dummies()`.
- **Feature Scaling**: Standardize numerical columns with `StandardScaler` to normalize the data, which helps neural network training.

### 5. Feature Engineering
- **Column Renaming**: Renaming columns for clarity and ease of understanding.
- **New Feature**: A new feature, `invoice_to_income_ratio`, is calculated as the ratio of loan amount to annual income, which can be insightful for risk assessment.

### 6. Train-Test Split
- **Define Features and Target**: `X` represents the features, and `y` is the target variable (`payment_status`).
- **Train-Test Split**: The dataset is split into training and testing sets with an 80-20 split using `train_test_split()`. This provides separate data for model training and evaluation.

### 7. Random Forest Model with Hyperparameter Tuning
- **Parameter Grid**: Define a parameter grid for `RandomForestClassifier` (e.g., `n_estimators`, `max_depth`).
- **GridSearchCV**: Use `GridSearchCV` to perform 5-fold cross-validation across multiple parameter combinations to find the best hyperparameters based on accuracy.
- **Evaluate Performance**: Print the best parameters and cross-validation score, followed by a confusion matrix and classification report on the test set.

### 8. Feature Importance Visualization
- **Feature Importance Plot**: Using `best_model.feature_importances_`, the importance of each feature in the Random Forest model is displayed in a bar plot, helping in understanding which features contribute most to the predictions.

### 9. Neural Network Model
- **Define Neural Network Architecture**: Use `Sequential` with three hidden layers and dropout for regularization to reduce overfitting.
  - **Input Layer**: First layer with 64 neurons and ReLU activation.
  - **Hidden Layers**: Additional layers with dropout for regularization.
  - **Output Layer**: A single neuron with a sigmoid activation, suitable for binary classification.
- **Compile the Model**: Use the Adam optimizer, binary cross-entropy loss, and track accuracy as the evaluation metric.
- **Early Stopping**: Set up `EarlyStopping` with patience to halt training if the validation loss doesn’t improve, minimizing overfitting.

### 10. Train the Neural Network
- **Training the Model**: Train the neural network on `X_train` and `y_train` for 30 epochs with validation split and batch size of 32.
- **Monitor Training**: The `history` object stores training and validation accuracy and loss across epochs for visualization.

### 11. Neural Network Evaluation
- **Evaluate on Test Set**: Print final test accuracy to assess the model’s generalization.
- **Training History Plot**: Plot both accuracy and loss for training and validation sets, allowing visual assessment of the model's training progress and potential overfitting or underfitting signs.

## Results

The model performs well on the dataset, showing high accuracy and balanced generalization between training and validation data. Here's a breakdown of the results:

### Neural Network Model
- **Accuracy**: The neural network achieves over 90% accuracy on both training and validation sets, showing effective learning without overfitting.
- **Loss**: Both training and validation loss decrease steadily, indicating good convergence and model generalization over 30 epochs.

![result_data](/result/4.png)

### Random Forest Model
- **Best Parameters**: `max_depth=20`, `min_samples_leaf=1`, `min_samples_split=2`, `n_estimators=100`
- **Cross-Validation Score**: 0.7028
- **Confusion Matrix**:
  - True Positives (0): 984
  - False Positives (0): 19
  - False Negatives (1): 98
  - True Negatives (1): 203
- **Classification Report**:
  - Precision: 91% (for both classes)
  - Recall: 98% (Class 0), 67% (Class 1)
  - F1 Score: 94% (Class 0), 78% (Class 1)
  - **Overall Accuracy**: 91%

### Gradient Boosting Model
- **Best Parameters**: `learning_rate=0.1`, `max_depth=5`, `n_estimators=100`
- **Cross-Validation Score**: 0.7116
- **Confusion Matrix**:
  - True Positives (0): 984
  - False Positives (0): 19
  - False Negatives (1): 92
  - True Negatives (1): 209
- **Classification Report**:
  - Precision: 91% (Class 0), 92% (Class 1)
  - Recall: 98% (Class 0), 69% (Class 1)
  - F1 Score: 95% (Class 0), 79% (Class 1)
  - **Overall Accuracy**: 91%

![result_data](/result/5.png)

### Summary
Both the Random Forest and Gradient Boosting models achieved 91% accuracy, with Gradient Boosting showing a slightly higher recall and F1 score for delayed payments. The neural network model also displayed high accuracy and a smooth learning curve, suggesting it generalizes well.

These results indicate that the models are effective in predicting invoice payment delays, with strong accuracy and precision across both classes.

## Technologies Used

- **Python Libraries:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib
- **Machine Learning Models:** Random Forest, Gradient Boosting, Neural Network (Keras)
- **Jupyter Notebook:** For interactive analysis, visualization, and model development

### Next Steps

For further enhancement:
- Experiment with more sophisticated ensemble models like XGBoost.
- Fine-tune the neural network model architecture for better performance.
- Explore ensemble stacking methods to combine model predictions for improved accuracy.

---

## Credits
Data for this project was sourced from [Kaggle's Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset).

