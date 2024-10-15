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
    X_train = [] #User Input
    y_train = [] #Chatbot Output
    #This will scan the txt file with the input and outputs to know which is which and assign which one goes with the input and output
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if "|" in line:
                question, response = line.strip().split("|", 1)
                X_train.append(preprocess(question))  # Preprocess questions
                y_train.append(response)
    return X_train, y_train


# Load training data. This will train both the X and the Y so the chatbot knows fully which is which in terms of input and output.
#Mess with the training_data.txt file to add more input and responses.
X_train, y_train = load_training_data("training_data.txt")

# Create a pipeline with TF-IDF Vectorizer and Logistic Regression. Don't worry about this code its a really condensed version of Logistic Regression that the library condsed so its easier to load
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000))  # Increased iterations for convergence
])

# Train the model. This now will make a AI of sorts and the ChatBot now has full knowledge of the data that it was given
pipeline.fit(X_train, y_train)

# Chatbot loop with debugging. This while loop will keep the program running after each prompt and made a feature where if the user inputs the string quit it will quit the program.
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

    # Display probabilities for each intent. This shows the probability of each output. This does not need to be here just fun to see if you are interested in how close some outputs would be to be used compared to others
    #Comment out if you don't want this to be in the terminal
    #print("DEBUG - Probabilities:")
    #for cls, prob in zip(classes, probabilities):
    #    print(f"{cls}: {prob:.2f}")

    # Predict the best response. This will take the higested probability response and print it after the user has entered an input.
    response = pipeline.predict([processed_input])[0]
    print(f"Chatbot: {response}")