import os
import nltk

# Ensure NLTK resources are downloaded


from flask import Flask, request, render_template, jsonify
import random

import text_classifier

nltk.download('punkt')
nltk.download('punkt_tab')


def create_app():
    """
    App Creation factor
    :return: flask app ready to be deployed
    """
    app = Flask(__name__)

    logprior, loglikelihood = load_model()

    @app.route('/', methods=['GET'])
    def main():
        return render_template('index.html')

    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    @app.route('/predict', methods=['POST'])
    def predict_review():
        # Get the review from the form data
        review_text = request.json.get('review')

        classification = text_classifier.naive_bayes_predict(
            review_text, logprior, loglikelihood)

        if classification == 0:
            mood = "happy"
            prediction = "Awesome! Glad you enjoyed it !"
        else:
            # Randomly choose a negative sentiment: annoyed or sad
            options = [0, 1]
            weights = [0.8, 0.2]

            chosen_option = random.choices(options, weights=weights, k=1)[0]

            if chosen_option == 0:
                mood = "annoyed"
                prediction = "Sorry you didnt enjoy that movie..."
            else:
                mood = "angry"
                prediction = "I agree. Lets demand a refund!"

        # return render_template(
        #     'index.html', prediction=prediction, mood=mood)
        return jsonify({'prediction': prediction, 'mood': mood})

    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port)

    return app

def load_model():
    """
    Load model and return the values needed nicely
    :return: logprior and loglikelihood
    """
    model = text_classifier.load_model("movie_sentiment_model_parameters.json")
    logprior = model['logprior']
    loglikelihood = model['loglikelihood']
    return logprior, loglikelihood

app = create_app()