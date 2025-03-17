Problem Statement:
The goal of this project is to perform sentiment analysis on text data related to mental health. By applying Natural Language Processing (NLP) techniques and machine learning classification models, this project aims to identify and classify sentiments that could indicate mental health concerns, aiding in early detection and analysis.

Libraries Used:

    pandas: Data manipulation and analysis
    numpy: Numerical computations
    matplotlib: Data visualization
    seaborn: Statistical data visualization
    nltk: Natural Language Processing (stopword removal and text preprocessing)
    scikit-learn (sklearn):
    → Model building and evaluation (Logistic Regression, Random Forest)
    → Preprocessing (LabelEncoder, TF-IDF Vectorizer, Stratified K-Fold)
    → Metrics (Accuracy, ROC-AUC, Classification Report)
    imbalanced-learn (imblearn): Handling class imbalance using SMOTE (Synthetic Minority Oversampling Technique)

Project Steps:

    Step 1: Data Preprocessing
    Handled missing values, removed duplicates, dropped irrelevant columns, cleaned textual data by removing special characters, stopwords, and performed tokenization.

    Step 2: Exploratory Data Analysis (EDA)
    Visualized class distribution using count plots to understand imbalance in the target variable.

    Step 3: Feature Engineering & Vectorization
    Applied TF-IDF Vectorization to convert cleaned text into numerical format suitable for machine learning models.

    Step 4: Handling Imbalanced Data
    Used SMOTE to balance the dataset and address class imbalance issues for better model performance.

    Step 5: Model Training and Evaluation
    Trained Logistic Regression (with class weights) and Random Forest Classifier.
    Performed Stratified K-Fold Cross Validation to evaluate model stability.
    Evaluated models using Accuracy, ROC-AUC, and Classification Reports.

    Step 6: Visualization of Model Performance
    Plotted ROC curves to assess model performance and visualize class-wise ROC-AUC scores.
