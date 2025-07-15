# AI-Career-Recommender-Based-on-Resume

#Problem Statement
This project aims to develop an AI-based system that can analyze resume text and recommend the most suitable career field. Using natural language processing and machine learning techniques, the system will classify resumes into predefined job categories such as Data Science, HR, or Web Development. The goal is to assist students and professionals in identifying the right career path based on their skills and experiences.

#Tech Stack
- Python 3.7+
- Jupyter Notebook / Google Colab
- Libraries: `pandas`, `nltk`, `scikit-learn`, `matplotlib`

  #Algorithms Used
- Logistic Regression  
- Naive Bayes  
- Random Forest Classifier  
- Support Vector Machine (SVM)

  #Dataset
  Dataset link: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset

  #Project Workflow
1. Upload and load resume dataset
2. Clean and preprocess resume text (remove stopwords, punctuation, etc.)
3. Convert text to numerical features using TF-IDF
4. Encode career categories using LabelEncoder
5. Split data into training and testing sets
6. Train multiple ML models
7. Evaluate performance using accuracy, F1-score, precision, and confusion matrix
8. Predict career field for a new input resume

   #Results (Best Model: Random Forest)

| Metric     | Score     |
|------------|-----------|
| Accuracy   | 92.3%     |
| Precision  | 92.0%     |
| F1-Score   | 91.8%     |

#Future Scope

- Use transformer models like BERT for deeper text understanding  
- Add skill-gap suggestions based on resume and target role  
- Build a web interface using Streamlit or Flask  
- Support for multilingual resumes
