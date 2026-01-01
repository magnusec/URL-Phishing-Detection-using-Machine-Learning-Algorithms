# Phishing URL Detection using Machine Learning Algorithms


## Problem Statement
Phishing attacks use deceptive URLs to trick users into revealing sensitive information such as login credentials, banking details, and personal data. Traditional blacklist-based approaches fail to detect newly generated phishing URLs. This project aims to build a machine learning–based phishing URL detection system that classifies URLs as **Phishing** or **Legitimate** using structural and lexical URL features.

This work is the inspiration for an IEEE research paper on phishing URL detection using classical machine learning algorithms. The research paper served as a **methodological and evaluation reference**, and **entire implementation, experimentation, dataset handling, model training, and web deployment were carried out** as part of this project. This project uses other 2 datasets than in the paper due to more realistic data that helps in detecting phishing URL's more efficiently.

https://ieeexplore.ieee.org/document/11208974



## Project Overview
This project implements and compares multiple machine learning models to detect phishing URLs. The models are trained on multiple datasets and evaluated using accuracy, precision, recall, and F1-score. The best-performing model is deployed using a Flask web application that allows users to input a URL and receive a real-time phishing prediction.

Dataset 3 was selected for deployment as it provides better real-world generalization due to its size, diversity, and feature distribution.




## Project Flow
1. Dataset collection and inspection
2. Data preprocessing and feature selection
3. Train–test split with stratification
4. Model training and evaluation
5. Metrics comparison across models and datasets
6. Flask-based web application deployment





## Final Dataset Used for Deployment
- **Dataset 3 (web-page-phishing.csv)**
- Reasons for selection:
  - Larger dataset size
  - Better feature diversity
  - Improved generalization
  - More realistic phishing patterns
 
    
- **Dataset Links -**
- Dataset 1 - https://www.kaggle.com/datasets/amj464/phishing
- Dataset 2 - https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning
- Dataset 3 - https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset
  



## Repository Structure


phishing-url-detection/
- `comparisons/` – metrics CSVs and dataset comparison results
- `data/` – datasets used for training & testing
- `flask_app/` – Flask web application (UI + backend)
- `models/` – trained model files (.pkl) and links
- `src/` – ML training, evaluation, preprocessing code
- `requirements.txt` – project dependencies





## Model Files
The trained Random Forest model for Dataset 3 is large in size and exceeds GitHub’s file size limits.  
Therefore, it is hosted externally, and a download link is provided inside the `models/` directory.
However Dataset 1 & 2 models are available for direct download in `models/`




## Technologies Used
- Python 3
- scikit-learn
- pandas
- numpy
- Flask
- joblib
- HTML / CSS (for frontend)



## Machine Learning Models Used
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes

Random Forest achieved the best overall performance and was selected as the final model for deployment. It achieved the accuracy of 0.9775, 0.9749 and 0.8814 or Datasets 1, 2 & 3 respectively.



## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Accuracy alone was not considered sufficient due to the security-sensitive nature of phishing detection.



## Flask Web Application
The Flask application allows users to:
- Enter a URL
- Predict whether it is **Phishing** or **Legitimate**
- View results instantly
- Log checked URLs automatically
- Export prediction logs as CSV




## How to run project

1. Create and activate a virtual environment:
`python3 -m venv venv`
`source venv/bin/activate`

2. Install required dependencies:
`pip install -r requirements.txt`

3. Run the Flask web application:
`cd flask_app`
`python app.py`

4. Follow the link

5. Enter a URL in the input field to check whether it is Phishing or Legitimate.





## Future Enhancements
- Real-time browser extension
- Deep learning–based models
- Cloud deployment

