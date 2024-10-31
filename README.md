# Invoice Payment Delay Prediction

**Author:** Victor Vasu Joseph


### Executive Summary

This project focuses on predicting invoice payment delays based on customer behavior and transaction history. By identifying customers likely to delay payments, businesses can improve cash flow management, streamline accounts receivable, and implement proactive collection strategies.

### Rationale

Delayed payments are a significant challenge in managing cash flow for businesses. Predicting payment delays allows companies to take preemptive measures, prioritize follow-ups, and reduce the risk associated with late payments. This project leverages machine learning to offer predictive insights, assisting finance teams in efficient accounts receivable management.

### Research Question

Can we accurately predict whether an invoice will be paid on time based on customer behavior and transaction characteristics?

### Data Sources

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

### Methodology

The project follows these steps:
1. **Data Preprocessing**: Handling missing values, encoding categorical features, and scaling numeric variables.
2. **Model Training**:
   - **Random Forest**: Trained with hyperparameter tuning using GridSearchCV to optimize performance.
   - **Neural Network**: Trained using Keras with a structured architecture, using dropout layers to prevent overfitting.
3. **Evaluation**:
   - **Random Forest**: Evaluated with a confusion matrix, classification report, and accuracy score.
   - **Neural Network**: Evaluated using accuracy and loss curves, and tested on a hold-out set to assess generalization.


### Step-by-Step Code Explanation

#### 1. Import Libraries:

- **pandas, numpy**: For data manipulation and numerical operations.
- **train_test_split, StandardScaler**: To split the dataset and scale features for better neural network performance.
- **RandomForestClassifier, GridSearchCV**: For training and hyperparameter tuning of the Random Forest model.
- **Sequential, Dense, Dropout**: For building and training the neural network.
- **matplotlib, seaborn**: For data visualization.

#### 2. Load and Explore the Dataset:

- Load the dataset using pd.read_csv() and display the first few rows using df.head().
- Print an overview of the dataset, including the column data types, any missing values, and basic statistics. This helps identify the structure of the data and spot any columns that might need cleaning.

#### 3. Data Preprocessing:

- **Handle Missing Values**: Fill missing values in numerical columns like person_emp_length and loan_int_rate with the median, a common technique for handling missing data in ML.
- **Encoding Categorical Variables**: Convert categorical columns, such as person_home_ownership, loan_intent, loan_grade, and cb_person_default_on_file, into numerical format using one-hot encoding (pd.get_dummies). This is necessary as machine learning models work best with numerical data.
- **Feature Scaling**: Standardize the numerical features using StandardScaler. This helps neural networks train more efficiently.

#### 4. Feature Selection and Target Variable:

- Define the target variable y (loan_status) and feature matrix X.
- Perform a train-test split, typically with a 80-20 ratio, for training and evaluation.

#### 5. Random Forest Model with Hyperparameter Tuning:

- Set up a RandomForestClassifier to understand the baseline performance and identify feature importance.
- Use GridSearchCV for hyperparameter tuning, specifying a grid of parameters like n_estimators, max_depth, and min_samples_split.
- Run the grid search with cross-validation, allowing the model to automatically select the best parameters based on performance.

#### 6. Evaluate Random Forest Performance:

- Print the best parameters and the cross-validation score obtained from GridSearchCV.
- Calculate and print the confusion matrix and classification report, which give insight into precision, recall, and F1-score for each class.

#### 7. Neural Network Model:

- **Define Architecture**: Use Keras’ Sequential model with fully connected (Dense) layers. The final layer has a single neuron with a sigmoid activation, ideal for binary classification.
- **Compile the Model**: Specify loss (binary_crossentropy), optimizer (adam), and evaluation metric (accuracy).
- **Train the Model**: Use model.fit() with a set number of epochs (e.g., 30), batch size, and split the data further into training and validation.

#### 8. Plot Training Progress:

Plot the model accuracy and loss for both training and validation sets. This allows us to monitor the training process and check for overfitting or underfitting.

#### 9. Neural Network Evaluation:

Evaluate the model on the test set to see its accuracy and loss after training.
Print out the test accuracy for final performance evaluation.

![result_data](/result/2.png)

### Results

The model performs well on the dataset, showing high accuracy and balanced generalization between training and validation data. Here's a breakdown of the results:

## Model Performance Summary

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

## Credits
Data for this project was sourced from [Kaggle's Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset).

