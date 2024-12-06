from sklearn.pipeline import Pipeline
import pickle 
from docx import Document
from jinja2 import Template


def carregar_modelo(caminho="C:\\Users\\diogo\\OneDrive\\Desktop\\Diogo Lauter\\Faculdade\\IA\\G2\\modelo_juridico.pkl"):
    with open(caminho, "rb") as file:
        return pickle.load(file)


def prever_fundamentos(modelo, tipo_acao, descricao_caso):
    texto = f"{tipo_acao} {descricao_caso}"
    predicao = modelo.predict([texto])
    return predicao[0]


def gerar_peticao_civil_com_fundamentos_sugeridos(dados, fundamentos_sugeridos):
    modelo_peticao = """
    Excelentíssimo(a) Senhor(a) Doutor(a) Juiz(a) de Direito da ___ Vara Cível da Comarca de {{ comarca }}

    {{ espaco }}

    {{ cliente.nome_completo }}, nacionalidade {{ cliente.nacionalidade }}, {{ cliente.estado_civil }}, profissional de ocupação {{ cliente.ocupacao }}, residente e domiciliado à {{ cliente.endereco }}, vem, respeitosamente, à presença de Vossa Excelência, por meio de seu advogado que esta subscreve, com endereço profissional em {{ advogado.endereco }}, com fundamento nos seguintes fundamentos jurídicos: {{ fundamentos }}, propor a presente

    {{ tipo_peticao }}

    {{ espaco }}

    Pelos fatos e fundamentos a seguir expostos:

    {{ espaco }}

    1. DOS FATOS

    No dia {{ dados_caso.data }}, o requerente celebrou um contrato com o requerido {{ dados_caso.requerido_nome }}, residente em {{ dados_caso.requerido_endereco }}, referente a {{ dados_caso.descricao_contrato }}. Entretanto, {{ dados_caso.descricao_problema }}.

    {{ espaco }}

    2. DO DIREITO

    Fundamentação Jurídica:
    - {{ fundamentos }}

    {{ espaco }}

    3. DOS PEDIDOS

    Diante do exposto, requer:

    {{ pedidos }}

    Nestes termos,
    Pede deferimento.

    {{ espaco }}

    Local e Data: {{ local_data }}

    Assinatura: _____________________________

    Nome do Advogado: {{ advogado.nome }}
    OAB: {{ advogado.oab }}
    """

    
    template = Template(modelo_peticao)
    return template.render(
        comarca=dados["comarca"],
        espaco="\n" * 2,
        cliente=dados["cliente"],
        advogado=dados["advogado"],
        fundamentos=fundamentos_sugeridos,
        tipo_peticao=dados["tipo_peticao"],
        dados_caso=dados["dados_caso"],
        pedidos=dados["pedidos"],
        local_data=dados["local_data"]
    )


def salvar_peticao_em_word(peticao_texto, nome_arquivo="Peticao_Civil_Gerada.docx"):
    doc = Document()
    for linha in peticao_texto.split("\n"):
        doc.add_paragraph(linha)
    caminho_arquivo = f"C:\\Users\\diogo\\OneDrive\\Desktop\\Diogo Lauter\\Faculdade\\IA\\G2\\{nome_arquivo}"
    doc.save(caminho_arquivo)
    return caminho_arquivo


dados_exemplo_civil = {
    "comarca": "São Paulo",
    "cliente": {
        "nome_completo": "João da Silva",
        "nacionalidade": "brasileiro",
        "estado_civil": "solteiro",
        "ocupacao": "vendedor autônomo",
        "endereco": "Rua Exemplo, 123, São Paulo - SP"
    },
    "advogado": {
        "nome": "Dr. Carlos Almeida",
        "oab": "SP123456",
        "endereco": "Av. das Advogadas, 456, São Paulo - SP"
    },
    "tipo_peticao": "Ação de Cobrança",
    "dados_caso": {
        "data": "10 de janeiro de 2024",
        "requerido_nome": "Maria Souza",
        "requerido_endereco": "Rua Outra, 456, São Paulo - SP",
        "descricao_contrato": "a prestação de serviços de pintura residencial",
        "descricao_problema": "a requerida não realizou o pagamento acordado de R$ 2.000,00, vencido em 01 de fevereiro de 2024"
    },
    "pedidos": "a condenação do requerido ao pagamento da quantia de R$ 2.000,00, acrescida de correção monetária e juros legais, bem como as custas processuais e honorários advocatícios.",
    "local_data": "São Paulo, 02 de dezembro de 2024"
}


modelo_treinado = carregar_modelo()
descricao_caso = dados_exemplo_civil["dados_caso"]["descricao_problema"]
fundamentos_sugeridos = prever_fundamentos(modelo_treinado, dados_exemplo_civil["tipo_peticao"], descricao_caso)


peticao_gerada = gerar_peticao_civil_com_fundamentos_sugeridos(dados_exemplo_civil, fundamentos_sugeridos)
print(peticao_gerada)


caminho_arquivo = salvar_peticao_em_word(peticao_gerada)
print(f"Arquivo salvo em: {caminho_arquivo}")
