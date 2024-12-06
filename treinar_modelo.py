from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle


dados_treinamento = [
    {"texto": "O cliente não pagou a dívida", "tipo_acao": "Ação de Cobrança"},
    {"texto": "Contratação de serviços não pagos", "tipo_acao": "Ação de Cobrança"},
    {"texto": "Contestação de divórcio", "tipo_acao": "Ação de Divórcio"},
    
]


X = [dado["texto"] for dado in dados_treinamento]
y = [dado["tipo_acao"] for dado in dados_treinamento]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),  
    ("modelo", LogisticRegression())  
])


pipeline.fit(X_train, y_train)


with open("modelo_juridico.pkl", "wb") as file:
    pickle.dump(pipeline, file)

print("Modelo treinado e salvo com sucesso!")
