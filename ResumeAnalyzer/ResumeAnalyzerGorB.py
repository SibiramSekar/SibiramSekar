import numpy as np
import pandas as pd
import spacy
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


nlp = spacy.load("en_core_web_md")


def extract_experience(text):
    match = re.search(r"(\d+)\s*years?", text)
    return int(match.group(1)) if match else 0  


def extract_skills(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "SKILL"]]
    return ", ".join(skills) if skills else "No skills found"


data = {
    'Resume_Text': [
        "Software engineer with 5 years of Python, SQL, and ML experience.",
        "Recent graduate with no work experience.",
        "Data scientist skilled in deep learning, NLP, and big data analysis.",
        "High school student looking for an entry-level job.",
        "Marketing specialist with 10 years of experience in digital advertising.",
        "Engineer with Java and cloud computing expertise."
    ],
    'Label': [1, 0, 1, 0, 1, 1]  
}

df = pd.DataFrame(data)


df["Years_Experience"] = df["Resume_Text"].apply(extract_experience)
df["Skills"] = df["Resume_Text"].apply(extract_skills)


vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(df['Resume_Text']).toarray()
y = np.array(df['Label'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


def analyze_resume():
    user_resume = input("\nEnter the resume description: ")
    
    extracted_experience = extract_experience(user_resume)
    extracted_skills = extract_skills(user_resume)
    
    print("\nExtracted Information:")
    print(f"Years of Experience: {extracted_experience}")
    print(f"Skills: {extracted_skills}")
    
    
    vectorized_resume = vectorizer.transform([user_resume]).toarray()
    prediction = model.predict(vectorized_resume)
    
    result = "Good Resume ✅" if prediction[0] == 1 else "Bad Resume ❌"
    print(f"\nResume Evaluation: {result}")


analyze_resume()
