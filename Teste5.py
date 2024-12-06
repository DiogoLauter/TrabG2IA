
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def criar_dados_ficticios(arquivo="dados_treinamento.json"):
    dados = [
        {
            "tipo_acao": "Ação de Cobrança",
            "descricao": "atraso de pagamento em contrato de prestação de serviços",
            "fundamentos": ["Art. 389 CC", "Art. 395 CC", "Súmula 54 STJ"]
        },
        {
            "tipo_acao": "Ação de Danos Morais",
            "descricao": "falha na prestação de serviço que causou dano moral",
            "fundamentos": ["Art. 927 CC", "Art. 186 CC"]
        },
        {
            "tipo_acao": "Ação de Despejo",
            "descricao": "inadimplência no pagamento do aluguel por mais de três meses",
            "fundamentos": ["Art. 59 da Lei do Inquilinato", "Art. 9 da Lei do Inquilinato"]
        },
        {
            "tipo_acao": "Ação de Divórcio",
            "descricao": "divórcio litigioso com disputa de guarda",
            "fundamentos": ["Art. 226 CF", "Art. 1.634 CC"]
        },
        {
            "tipo_acao": "Ação de Execução",
            "descricao": "cobrança judicial de dívida garantida por título executivo",
            "fundamentos": ["Art. 783 CPC", "Art. 784 CPC"]
        }
    ]
    with open(arquivo, "w", encoding="utf-8") as file:
        json.dump(dados, file, ensure_ascii=False, indent=4)


def carregar_dados(arquivo="dados_treinamento.json"):
    with open(arquivo, "r", encoding="utf-8") as file:
        return json.load(file)


def preparar_dados(dados):
    textos = [f"{item['tipo_acao']} {item['descricao']}" for item in dados]
    fundamentos = [", ".join(item["fundamentos"]) for item in dados]
    return textos, fundamentos


def treinar_modelo(textos, fundamentos):
    x_train, x_test, y_train, y_test = train_test_split(textos, fundamentos, test_size=0.2, random_state=42)

    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier())
    ])
    
    
    pipeline.fit(x_train, y_train)
    print("Modelo treinado!")
    
    
    y_pred = pipeline.predict(x_test)
    print(classification_report(y_test, y_pred))
    return pipeline


criar_dados_ficticios()  
dados = carregar_dados("dados_treinamento.json")
textos, fundamentos = preparar_dados(dados)
modelo = treinar_modelo(textos, fundamentos)


novo_caso = "Ação de Cobrança atraso no pagamento de serviços de consultoria"
sugestao = modelo.predict([novo_caso])
print("Fundamentos sugeridos:", sugestao)
