# 🧠 Resume Analyzer (Prototype)

This project is a **simple resume analyzer prototype** that uses NLP and machine learning techniques to match candidate resumes with a given job description.
Originally part of an assignment, this project was discontinued due to its complexity and is now used as a personal exploration tool.

## 🚀 Features

- Cleans and preprocesses resume text using spaCy
- Converts text to TF-IDF vectors
- Computes similarity scores between resumes and job descriptions
- Visualizes:
  - Distribution of job roles
  - Most common words in resumes

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- spaCy (for lemmatization and stopword removal)
- scikit-learn (TF-IDF & cosine similarity)
- Matplotlib (for plotting)

## 📄 Sample Input

The program uses mock resume data like:

```python
"Experienced Data Scientist skilled in Python, Machine Learning, and NLP."
