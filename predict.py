""" Example of prediction from CLI or user input """
import sys
import gzip
import pickle
import os

CLASSES = {
    0: "negative",
    4: "positive"
}

def load_model(model_filename):
    """ Load model from file """
    print("Loading the model...")
    if not os.path.isfile(model_filename):
        raise FileNotFoundError(f"Model file not found: {model_filename}")

    try:
        with gzip.open(model_filename, 'rb') as fmodel:
            model = pickle.load(fmodel)
    except Exception as ex:
        raise IOError(f"Couldn't load model: {ex}")

    return model

def predict(model, text):
    """ Predict class given model and input (text) """
    print("Extracting features...")
    x_vector = model.vectorizer.transform([text])
    y_predicted = model.predict(x_vector)
    return CLASSES.get(y_predicted[0], "unknown")

def main(argv):
    """ Predict the sentiment of the given text """
    if len(argv) < 2:
        text = input("Enter your sentence: ")
    else:
        text = argv[1]

    model_filename = "data/model.dat.gz"

    model = load_model(model_filename)
    result = predict(model, text)
    print(f"Predicted sentiment: {result}")

if __name__ == '__main__':
    main(sys.argv)