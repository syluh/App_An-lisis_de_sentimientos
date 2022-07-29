from flask import Flask, render_template, request
import pickle
import re


def clean_text(text):
    text = re.sub(r'@[A-Za-z09]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'_', ' ', text)
    text = re.sub(r'\d', '', text)
    return text


app = Flask(__name__)
model_knn = pickle.load(open('models/knn/classifier_KNN.sav', 'rb'))
model_nb = pickle.load(open('models/nb/classifier_naive_bayes.sav', 'rb'))
model_svm = pickle.load(open('models/svm/classifier_svm.sav', 'rb'))
count_vector = pickle.load(open('models/svm/count_vectorSVM.sav', 'rb'))


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    text = clean_text(request.form['text'])
    data = count_vector.transform([text])
    prediction_knn = model_knn.predict(data)
    prediction_nb = model_nb.predict(data)
    prediction_svm = model_svm.predict(data)

    context = {
        'prediction_knn': prediction_knn,
        'prediction_nb': prediction_nb,
        'prediction_svm': prediction_svm
    }
    
    
    return render_template('index.html', **context)

if __name__ == '__main__':
    app.run(debug=True, port=5000)