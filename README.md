# Microsoft Cyber Security Incident Project

****Overview****

This project involves building a machine learning model to predict outcomes for a cybersecurity incident dataset provided by Microsoft. The goal is to analyze the data, perform exploratory data analysis (EDA), handle missing values, and find patterns that help build an effective predictive model.

**The following steps were performed:**

- EDA (Exploratory Data Analysis)
- Handling Missing Values
- Finding Correlations
- Hypothesis Testing (Chi-Square Test)
- Data Visualization (Heatmaps)
- Model Building using Random Forest, KNN, and Decision Tree
- Model Evaluation and Hyperparameter Tuning
- Prediction on Test Data

**Tools & Technologies Used**

- Programming Language: Python
- Libraries:
  - Pandas: For data manipulation and handling missing values.
  - Matplotlib/Seaborn: For data visualization and correlation heatmaps.
  - Scikit-Learn: For machine learning algorithms, cross-validation, and hyperparameter tuning.

**Dataset**

The dataset contains several features related to cybersecurity incidents, such as:

- OrgId: Organization identifier
- IncidentId: Unique identifier for the incident
- AlertId, DetectorId, AlertTitle, etc.

The train.csv dataset was used to train and tune the models, and the test.csv dataset was used for final predictions.


**Data Preprocessing**
 - Handling Missing Values: Imputed missing values using appropriate methods (e.g., mean, median, mode).
 - Finding Correlations: Explored relationships between features using correlation matrices.
 - Hypothesis Testing (Chi-Square Test): Applied the chi-square test to test for independence between categorical variables.
 - Heatmap: Visualized the correlations between features using a heatmap.

**Model Building and Evaluation**
The following algorithms were used to build models:

- Random Forest
- K-Nearest Neighbors (KNN)
- Decision Tree

**Model Performance**

- Random Forest: Achieved the best accuracy on the training data with 86%.
- KNN: Provided a moderate accuracy.
- Decision Tree: Had higher variance and risk of overfitting.

**Hyperparameter Tuning**
- Cross-Validation: 5-fold cross-validation was performed to validate the model's performance across different splits of the dataset.
- Grid Search: Hyperparameters were fine-tuned using Grid Search, optimizing parameters like n_estimators, max_depth, min_samples_split, and min_samples_leaf.

**Final Model Prediction**
After tuning the Random Forest model, the test.csv dataset was used for final predictions. The accuracy on the test set was 70%, reflecting good generalization, though there is room for improvement.

**Key Results**
- Random Forest Accuracy (Train): 86%
- Random Forest Accuracy (Test): 70%
- Best Model: Random Forest
- Cross-Validation: Used to verify the consistency of model performance.
- Grid Search: Performed to fine-tune the hyperparameters for optimal performance.

**Conclusion**
The Random Forest model provided the best performance on this cybersecurity dataset, with an accuracy of 86% on the training set and 70% on the test set. The project also highlights the importance of EDA, handling missing values, and using hyperparameter tuning to achieve optimal results.

**Future Work**
- Further improve model performance by exploring advanced techniques like Ensemble Methods.
- Explore feature engineering to derive new, relevant features.
- Experiment with additional algorithms like XGBoost or LightGBM for better performance.
