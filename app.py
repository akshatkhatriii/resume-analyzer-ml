from flask import Flask, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Skill requirements
skills_required = {
    "Data Analyst": ["python", "sql", "excel"],
    "Backend Developer": ["java", "spring", "api"],
    "Frontend Developer": ["html", "css", "javascript"],
    "ML Engineer": ["python", "machine learning"]
}

@app.route("/")
def home():
    return '''
    <html>
    <head>
        <title>Resume Analyzer</title>
        <style>
            body {
                font-family: Arial;
                background: linear-gradient(to right, #1e3c72, #2a5298);
                color: white;
                text-align: center;
                padding: 50px;
            }
            textarea {
                width: 60%;
                height: 150px;
                padding: 10px;
                border-radius: 10px;
                border: none;
                font-size: 16px;
            }
            button {
                margin-top: 20px;
                padding: 10px 20px;
                font-size: 16px;
                border: none;
                border-radius: 5px;
                background-color: #00c6ff;
                color: white;
                cursor: pointer;
            }
            button:hover {
                background-color: #0072ff;
            }
        </style>
    </head>
    <body>
        <h1>🚀 Smart Resume Analyzer</h1>
        <p>Paste your resume and get role prediction + skill gaps</p>
        <form method="POST" action="/predict">
            <textarea name="resume" placeholder="Paste your resume here"></textarea><br>
            <button type="submit">Analyze Resume</button>
        </form>
    </body>
    </html>
    '''

@app.route("/predict", methods=["POST"])
def predict():
    resume = request.form["resume"]
    vector = vectorizer.transform([resume])
    prediction = model.predict(vector)[0]

    resume_lower = resume.lower()
    required_skills = skills_required.get(prediction, [])
    missing_skills = [skill for skill in required_skills if skill not in resume_lower]

    return f"""
    <h3>Predicted Role: {prediction}</h3>
    <h4>Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}</h4>
    """

if __name__ == "__main__":
    import os

port = int(os.environ.get("PORT", 10000))
app.run(host="0.0.0.0", port=port)


