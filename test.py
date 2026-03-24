import pickle

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Test resume
resume = ["I know python, pandas, sql and data analysis"]

# Convert to vector
resume_vector = vectorizer.transform(resume)

# Predict
prediction = model.predict(resume_vector)

print("Predicted Role:", prediction[0])

