import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Function to preprocess text (lowercase, remove punctuation)
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation
    return text


# Function to load training data from a text file
def load_training_data(file_path):
    X_train = []
    y_train = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if "|" in line:
                question, response = line.strip().split("|", 1)
                X_train.append(preprocess(question))  # Preprocess questions
                y_train.append(response)
    return X_train, y_train


# Load training data
X_train, y_train = load_training_data("training_data.txt")

# Create a pipeline with TF-IDF Vectorizer and Logistic Regression
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000))  # Increased iterations for convergence
])

# Train the model
pipeline.fit(X_train, y_train)

# Chatbot loop with debugging
print("Chatbot: Hello! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Chatbot: Goodbye!")
        break

    processed_input = preprocess(user_input)

    # Predict with probabilities for debugging purposes
    probabilities = pipeline.predict_proba([processed_input])[0]
    classes = pipeline.classes_

    # Display probabilities for each intent
    #Comment out if you don't want this to be in the terminal
    #print("DEBUG - Probabilities:")
    #for cls, prob in zip(classes, probabilities):
    #    print(f"{cls}: {prob:.2f}")

    # Predict the best response
    response = pipeline.predict([processed_input])[0]
    print(f"Chatbot: {response}")