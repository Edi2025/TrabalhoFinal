from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from waitress import serve

# Criação do objeto Flask
app = Flask(__name__)
api = Api(app, version='1.0', title='API de Predição', description='Uma API para predição de textos')

# Carregar o modelo e o vetorizador
model_loaded = joblib.load('modelo_classificacao.pkl')
vectorizer_loaded = joblib.load('tfidf_vectorizer.pkl')

texto_model = api.model('TextoModel', {
    'texto': fields.String(required=True, description='Texto a ser analisado', example="Insira seu texto aqui")
})

@api.route('/predict')
class Predict(Resource):
    @api.doc('predict_text')
    @api.expect(texto_model)
    def post(self):
        '''Recebe texto e retorna a predição'''
        data = api.payload  # Automáticamente pega e valida o JSON de entrada
        texto = data['texto']
        texto_transformado = vectorizer_loaded.transform([texto])
        resultado = model_loaded.predict(texto_transformado)
        return {'predicao': resultado[0]}

@app.route('/')
def home():
    return "API de Predição de Texto. Use /predict para fazer uma predição via POST."

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)
