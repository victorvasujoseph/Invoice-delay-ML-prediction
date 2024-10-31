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

### Results

- **Random Forest Model**: Achieved high accuracy in predicting invoice delays, with clear separation between timely and delayed payments in the confusion matrix.
- **Neural Network Model**: Converged well with consistent training and validation accuracies around 90%, showing that the model generalizes well with minimal overfitting.

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

### Repository Structure

- **notebooks**:
  - [01_Data_Preprocessing_and_EDA.ipynb](#) - Initial data cleaning, preprocessing, and exploratory data analysis.
  - [02_Random_Forest_Model.ipynb](#) - Training and optimizing the Random Forest model.
  - [03_Neural_Network_Model.ipynb](#) - Training and evaluation of the neural network model.
- **data**: Contains the raw and processed dataset files.
- **images**: Visualizations and plots used in the README and reports.
- **README.md**: Documentation (you are here).



