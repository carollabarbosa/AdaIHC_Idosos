from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Corrigindo o nome do arquivo do vetor e do modelo
vector = pickle.load(open("finalized_vectorizer.pkl", 'rb'))
model = pickle.load(open("modelo_regressao_logistica.pkl", 'rb'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        news = str(request.form['news'])
        print(news)
        
        # Alterando de `verificar` para `predict`
        predict = model.predict(vector.transform([news]))[0]
        print(predict)
        
        # Exibindo o resultado da previsão na página
        return render_template("prediction.html", prediction_text=" Notícia é -> {}".format(predict))
    else:
        return render_template("prediction.html")
    
@app.route('/projeto')
def projeto():
    return render_template("projeto.html")

# Certifique-se de que o servidor Flask é iniciado
if __name__ == '__main__':
    app.run(debug=True)
