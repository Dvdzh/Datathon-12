     
import yfinance as yf 
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def technical_analysis(company_name , start_date , end_date):
    ticker = company_name  
    # Remplacez par le symbole de votre actif

    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.dropna()

    print(data['Close'])
    data['MA50'] = data['Close'].rolling(window=50).mean()

    # Prédire le prix du lendemain
    data['Target'] = data['Close'].shift(-1)  
    data.dropna(inplace=True)

    # Variables indépendantes (X) et dépendantes (y)
    X = data[['Open', 'High', 'Low', 'Volume', 'MA50']]
    y = data['Target']
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner le modèle
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Variables
    n_days = 30  # Nombre de jours à prédire
    predicted_prices = []  # Liste pour stocker les prix prévus
    
    # Dernière valeur connue
    
    last_close_price = data['Close'].iloc[-1]
    # Prévisions pour chaque jour du mois
    for day in range(n_days):
        future_data = pd.DataFrame({
            'Open': [last_close_price],
            'High': [last_close_price],
            'Low': [last_close_price],
            'Volume': [0],  # Volume hypothétique
            'MA50': [data['MA50'].iloc[-1]]  # Moyenne mobile actuelle
        })
        # Faire la prédiction pour le jour actuel
        predicted_price = model.predict(future_data)[0]  # Supposons que la prédiction soit un tableau

        # Ajouter la prédiction à la liste
        predicted_prices.append(predicted_price)

        # Mettre à jour le dernier prix pour la prochaine itération
        last_close_price = predicted_price

        # Calculer la moyenne des prix prévus
    average_predicted_price = np.mean(predicted_prices)
        
    marge = average_predicted_price - np.array(data['Close'].iloc[-1]) 
    return marge

# ------------------------------------------------------
# Streamlit
# Knowledge Bases for Amazon Bedrock and LangChain 🦜️🔗
# ------------------------------------------------------

import boto3
import logging

from typing import List, Dict
from pydantic import BaseModel
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_aws import ChatBedrock
from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

import yfinance as yf 
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------------------------------------------
# Log level

logging.getLogger().setLevel(logging.ERROR) # reduce log level


# ------------------------------------------------------
# Streamlit

import streamlit as st

# Streamlit Chat Message History
history = StreamlitChatMessageHistory(key="chat_messages")

# Page title
st.set_page_config(page_title='FinApp - Assistant Financier', page_icon="🦜️")

# Main app layout

# Beautifull CSS
# def load_css():
#     with open("styles.css", "r") as f:
#         css = f"<style>{f.read()}</style>"
#         st.markdown(css, unsafe_allow_html=True)
# load_css()

welcome_message = "Bienvenue ! Je suis FinApp, votre assistant financier. Comment puis-je vous aider ?"

# Clear Chat History function
def clear_chat_history():
    history.clear()
    st.session_state.messages = [{"role": "assistant", "content": welcome_message}]



# Function to add user message to chat history
def button_prompt(selected_company, selected_indicators, selected_period):
    time_periods = {
        "1d": "1 jour",
        "5d": "5 jours",
        "1mo": "1 mois",
        "3mo": "3 mois",
        "6mo": "6 mois",
        "1y": "1 an",
        "2y": "2 ans",
        "5y": "5 ans",
        "10y": "10 ans",
        "ytd": "Début de cette année",
        "max": "Maximum disponible"
    }

    period = time_periods[selected_period]
    
    user_message = {
        "role": "user",
        "content": f"Fais les graphiques des indicateurs {', '.join(selected_indicators)} de la compagnie {selected_company} sur la période de {period}"
    }
    st.session_state.messages.append(user_message)
    st.chat_message("user").write(user_message["content"])

# Function to fetch financial data
def fetch_financial_data(symbol, period, selected_indicators):
    # Financial Indicators Selection
    history_indicators = ["Open", "High",	"Low",	"Close",	"Volume"]
    
    quaterly_indicators = [
        "Total Revenue", "Gross Profit", "Operating Income", "EBITDA", "EBIT", 
        "Net Income", "Normalized Income", "Diluted EPS", "Operating Expense", "Research And Development",
        "Total Expenses", "Net Interest Income","Pretax Income"]
    data = {}
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=period)
        quarterly = ticker.quarterly_financials

        for indicator in selected_indicators:
            if indicator in history_indicators:
                data[indicator] = history[indicator]
            elif indicator in quaterly_indicators:
                data[indicator] = quarterly.loc[indicator]
        return data
    
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données pour {symbol}: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# Function to plot financial data with parameters
def plot_financial_data(company_symbol, selected_indicators, selected_period):

    # Financial Indicators Selection
    history_indicators = ["Open", "High",	"Low",	"Close",	"Volume"]
    
    quaterly_indicators = [
        "Total Revenue", "Gross Profit", "Operating Income", "EBITDA", "EBIT", 
        "Net Income", "Normalized Income", "Diluted EPS", "Operating Expense", "Research And Development",
        "Total Expenses", "Net Interest Income","Pretax Income"]
    
    if company_symbol != None and len(selected_indicators) > 0:
        time_periods = {
            "1d": "1 jour",
            "5d": "5 jours",
            "1mo": "1 mois",
            "3mo": "3 mois",
            "6mo": "6 mois",
            "1y": "1 an",
            "2y": "2 ans",
            "5y": "5 ans",
            "10y": "10 ans",
            "ytd": "Début de cette année",
            "max": "Maximum disponible"
         }
        
        period = time_periods[selected_period]
        
        # Fetch data with a loading indicator
        with st.spinner("Récupération des données financières..."):
            financial_data = fetch_financial_data(company_symbol, selected_period, selected_indicators)
            num_indicators = len(selected_indicators)
            
        # Set up subplot layout based on the number of indicators
        if num_indicators == 1:
            fig, axs = plt.subplots(1, 1, figsize=(10, 6))
            axs = [axs]  # Convert single subplot to a list for uniform handling
        elif num_indicators == 2:
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        elif num_indicators == 3:
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            axs[3].remove()  # Remove the empty fourth subplot
            axs = axs.flatten()[:3]  # Flatten and limit to 3 subplots
        elif num_indicators == 4:
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            axs = axs.flatten()
        else:
            st.warning("Le nombre d'indicateurs sélectionnés dépasse la limite de 4.")
            return

        # Plot each indicator
        for i, indicator in enumerate(selected_indicators):
            if indicator in quaterly_indicators:
                axs[i].plot(financial_data[indicator].index, financial_data[indicator].values, marker='o')                
                axs[i].set_title(f"Indicator: {indicator}")
                axs[i].set_xlabel("Date")
                axs[i].set_ylabel("Valeur")
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                axs[i].tick_params(axis='x', rotation=45)
                axs[i].legend()
            else:
                axs[i].plot(financial_data[indicator].index, financial_data[indicator].values)                
                axs[i].set_title(f"Indicator: {indicator}")
                axs[i].set_xlabel("Date")
                axs[i].set_ylabel("Valeur")
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                axs[i].tick_params(axis='x', rotation=45)
                axs[i].legend()           
        plt.tight_layout()

        # Add assistant message with the plot (without rendering it twice)
        assistant_message = {"role": "assistant", "content": f"Indicateurs financiers de {selected_company}\
                                sur la période de {period}. Certains indicateurs sont disponibles seulement sur une période limitée", "fig": fig}
        st.session_state.messages.append(assistant_message)

        # Render the plot
        st.chat_message("assistant").write(assistant_message["content"])
        st.pyplot(fig)
    else:
        # Handle case when no company or indicators are selected
        assistant_message = {"role": "assistant", "content": "Veuillez choisir une compagnie et au moins un indicateur financier pour voir les graphiques."}
        st.session_state.messages.append(assistant_message)
        st.chat_message("assistant").write(assistant_message["content"])

with st.sidebar:
    # Sidebar Title
    st.title('Analyse rapide des KPIs')

    # Company Selection
    companies = ["Aucune", "Couche-Tard", "Empire", "Loblaws", "Métro", "Canadian National Railway", "CPKC", "AltaGas", "Fortis", "Hydro One", "Bell Canada", "Cogeco", "Québecor", "Rogers", "Telus"]
    company_symbols = [None, "ATD.TO", "EMP-A.TO", "L.TO", "MRU.TO", "CNR.TO", "CP.TO", "ALA.TO", "FTS.TO", "H.TO", "BCE.TO", "CGO.TO", "QBR-B.TO", "RCI-B.TO", "T.TO"]

    selected_company = st.selectbox("Choisissez une compagnie:", companies)
    company_symbol = company_symbols[companies.index(selected_company)]

    # Financial Indicators Selection
    history_indicators = ["Open", "High",	"Low",	"Close",	"Volume"]
    
    quaterly_indicators = [
        "Total Revenue", "Gross Profit", "Operating Income", "EBITDA", "EBIT", 
        "Net Income", "Normalized Income", "Diluted EPS", "Operating Expense", "Research And Development",
        "Total Expenses", "Net Interest Income","Pretax Income"]

    selected_indicators = st.multiselect("Choisissez des indicateurs financiers:",
                                         history_indicators + quaterly_indicators,
                                         max_selections=4)

    # Selection for Time Periods
    time_periods = {
        "1d": "1 jour",
        "5d": "5 jours",
        "1mo": "1 mois",
        "3mo": "3 mois",
        "6mo": "6 mois",
        "1y": "1 an",
        "2y": "2 ans",
        "5y": "5 ans",
        "10y": "10 ans",
        "ytd": "Début de cette année",
        "max": "Maximum disponible"
    }
    # Show selectbox with human-readable values but store the key
    selected_period = st.selectbox(
        "Choisissez la période:",
        options=list(time_periods.keys()),
        format_func=lambda x: time_periods[x]
    )

    # Plot Button
    plot_button = st.button("Faire les graphiques")

    # Streaming Toggle (changed to a checkbox)
    streaming_on = st.checkbox('Streaming')

    # Clear Chat History Button
    st.button('Clear Chat History', on_click=clear_chat_history)

    # Divider
    st.divider()

    # History Logs
    st.write("History Logs")
    st.write(history.messages)


# ------------------------------------------------------
# Amazon Bedrock - settings

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
)

model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

model_kwargs =  { 
    "max_tokens": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

# ------------------------------------------------------
# LangChain - RAG chain with chat history

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."
         "Si tu n'as pas la réponse dans le contexte, dis que tu ne sais pas. Tu es un analyste financier, lorsque tu réponds, cite des chiffres et prudent dans tes analyses:\n {context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Création des ChatPromptTemplate
# prompt_identification = ChatPromptTemplate.from_messages(
#     [
#         ("system", "Analyse la phrase suivante pour identifier le ticker de l'entreprise et la période d'investissement. "
#          "Retourne le résultat sous la forme 'ticker, nombre d'années', où la durée est exprimée en nombre suivi de 'y'. "
#          "Si le ticker n'est pas clairement identifiable ou s'il y a une ambiguïté dans la période d'investissement, retourne 'NOT FOUND'. "
#          "Assure-toi que le ticker est correctement extrait et que la durée est bien formatée."
#          "Si l'utilisateur dit 'je veux investir dans l'entreprise Metro sur 5 ans', tu dois extraire 'MRU.TO,5y'"),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{question}"),
#     ]
# )

prompt_identification = (
        "Analyse la phrase suivante pour identifier le ticker de l'entreprise et la période d'investissement. "
        "Retourne le résultat sous la forme 'ticker, nombre d'années', où la durée est exprimée en nombre suivi de 'y'. "
        "Si le ticker n'est pas clairement identifiable ou s'il y a une ambiguïté dans la période d'investissement, retourne 'NOT FOUND'. "
        "Assure-toi que le ticker est correctement extrait et que la durée est bien formatée."
        "Si l'utilisateur dit 'je veux investir dans l'entreprise Metro sur 5 ans', tu dois renvoyer uniquement 'MRU.TO,5y'"
    ) 

prompt_fondamental_analysis = PromptTemplate(
    input_variables=["company_name", "context"],
    template=("Tu es un analyste financier avec accès à une base de connaissance pour réaliser une analyse fondamentale. "
              "Si l'utilisateur demande une analyse fondamentale de couche tard, utilise ce context:{context}."
              "Sinon utilise ce que tu sais et précise que tu ne t'appuis pas sur la base de connaissance "
              "Question: Réalise une analyse fondamentale de l'entreprise {company_name}"),
)


# prompt_extract_fondamental = ChatPromptTemplate.from_messages(
#     [
#         ("system", "Identifie les informations suivantes dans le prochain prompt : 'chiffre d'affaire', 'bénéfice net', 'marge nette', 'ratio dette/capital', "
#          "'ratio de liquidité', 'ratio de solvabilité', 'ratio de rentabilité', 'ratio de croissance' dans la phrase qui suit et renvoie les sous la forme (..., ..., etc)."),
#         ("human", "{question}"),
#     ]
# )

prompt_technical_analysis = ChatPromptTemplate.from_messages(
    [
        ("system", "Tu as la valeur de marge de {company_name} sur 2 * {n_year} ans."
         "Tu as accès à la valeur de projection que l'on fait pour cette analyse technique."
         "Cette valeur provient de la projection de la tendance de marché à court terme."
         "L'algorithme d'apprentissage par arbre de décision est utilisé pour analyser les données historiques de prix d'un actif et prévoir les mouvements futurs des prix."
         "Il vise à fournir aux analystes financiers un outil permettant de prendre des décisions éclairées sur les investissements."
         "Explique comment cette valeur est calculée et comment elle peut être utilisée pour prendre des décisions d'investissement."),
        ("human", "{question}"),
    ]
)
prompt_decision_tree_explanation = ChatPromptTemplate.from_messages(
    [
        ("system", "Explique le but de l'algorithme d'apprentissage par arbre de décision dans le contexte de la finance. "
         "Cet algorithme est utilisé pour analyser les données historiques de prix d'un actif et prévoir les mouvements futurs des prix. "
         "Il vise à fournir aux analystes financiers un outil permettant de prendre des décisions éclairées sur les investissements. "
         "L'objectif principal est de prédire les tendances du marché à court terme (comme les 30 prochains jours) en utilisant des modèles de machine learning. "
         "L'algorithme utilise des étapes telles que le calcul de la moyenne mobile, la préparation des données, l'entraînement du modèle et la comparaison des prix prévus à la valeur actuelle de l'actif. "
         "Détaille comment l'algorithme aide à identifier les opportunités d'achat ou de vente et à anticiper les fluctuations du marché."),
        ("human", "{question}"),
    ]
)

prompt_investment_recommendation = ChatPromptTemplate.from_messages(
    [
        ("system", "Sur la base de l'analyse fondamentale et technique, recommande une action d'investissement pour l'entreprise {company_name} sur {n_year} ans. "
         "Il faut absolument que la réponse se termine par une phrase équivalente à 'Nous recommandons d'acheter/vendre/rester neutre sur cette action.'"
         "L'analyse technique est {marge}, l'analyse fondamentale donne : {response_fondamental}."),
        ("human", "{question}"),
    ]
)

# Amazon Bedrock - KnowledgeBase Retriever 
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="BEFHFLTIB7", # 👈 Set your Knowledge base ID
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 20}},
)

model = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

chain = (
    RunnableParallel({
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    })
    .assign(response = prompt | model | StrOutputParser())
    .pick(["response", "context"])
)

# Streamlit Chat Message History
history = StreamlitChatMessageHistory(key="chat_messages")

# Chain with History
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="question",
    history_messages_key="history",
    output_messages_key="response",
)

# Fonction pour créer une chaîne avec un prompt spécifique
def create_chain_with_prompt(prompt_template):
    chain = (
        RunnableParallel({
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        })
        .assign(response = prompt_template | model | StrOutputParser())
        .pick(["response", "context"])
    )

    return RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key="response",
    )

# Initialiser l'historique du chat si nécessaire
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

# Fonction pour gérer les prompts séquentiels
def handle_prompts_sequentially(prompt_text):

    # Ajouter le message de l'utilisateur à l'historique
    st.session_state["chat_messages"].append({"role": "user", "content": prompt_text})

    # # 1. Identification de l'entreprise et de la période d'investissement
    # # chain_with_history = create_chain_with_prompt(prompt_identification)
    # chain = RunnableWithMessageHistory(
    #     RunnableParallel({
    #         "context": itemgetter("question") | retriever,
    #         "question": itemgetter("question"),
    #         "history": itemgetter("history"),
    #     })
    #     .assign(response = prompt_identification | model | StrOutputParser())
    #     .pick(["response", "context"]),
    #     lambda session_id: history,
    #     input_messages_key="question",
    #     history_messages_key="history",
    #     output_messages_key="response",
    # )   
    
    # response_identification = chain.invoke(
    #     {"question": prompt_text, "history": st.session_state["chat_messages"]},
    #     config
    # )
    # Créez le template de prompt
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template=(
        "Analyse la phrase suivante pour identifier le ticker de l'entreprise et la période d'investissement. "
        "Si l'utilisateur dit 'je veux investir dans l'entreprise Metro sur 5 ans', tu dois renvoyer uniquement 'MRU.TO,5y'"
        "Si l'utilisateur dit 'je veux investir dans l'entreprise google sur 2 ans', tu dois renvoyer uniquement 'GOOG,2y'"
        "Assure-toi que le ticker est correctement extrait et que la durée est bien formatée."
        "Retourne le résultat sous la forme 'ticker, nombre d'années', où la durée est exprimée en nombre suivi de 'y'. "
        "Si le ticker n'est pas clairement identifiable ou s'il y a une ambiguïté dans la période d'investissement, retourne 'NOT FOUND'. "
        "Here is the input: {question}"
        )
    )

    # Générez la réponse en passant le prompt formaté au modèle
    formatted_prompt = prompt_template.format(question=prompt_text)
    response = model.invoke(formatted_prompt)
    print("AAAA", response.content)
    st.session_state["chat_messages"].append({"role": "assistant", "content": response.content})
    
    if len(response.content.split(',')) != 2:
        
        # Créez le template de prompt
        prompt_template = PromptTemplate(
            input_variables=["question"],
            template=("You are a financial advisor, you can analyse Metro and many canadians companies"
                      "Such as Loblaws, Couche tard and Dollarama. "
                      "Here is a simple prompt: {question}."
                      "Answer the question in the context of the Canadian market and the retail sector."
                      "Tell the user to provide more context or clarify the question if needed."
                      "If needed try to use the ticker of the company in the answer."
                      "Insist on using ticker instead of the name of the company."
            )
        )
    
        # Générez la réponse en passant le prompt formaté au modèle
        formatted_prompt = prompt_template.format(question=prompt_text)
        response = model.invoke(formatted_prompt)

        return response.content
    
    else:
        pass
    ticker, n_year = response.content.split(',')

    stock = yf.Ticker(ticker)
    company_name = stock.info.get('longName', 'Unknown')
    if company_name == 'Unknown':
        company_name = stock.info.get('shortName', 'Unknown')
    if company_name == 'Unknown':
        raise ValueError("L'entreprise n'a pas été trouvée, veuillez précisez l'entreprise.")
    print(f"COMPANY NAME  : {company_name}")
    
    with st.expander("Résultats intermédiaires - Identification"):
        st.write("Ticker de l'entreprise:", ticker)
        st.write("Période d'investissement:", n_year)
        st.write("Nom de l'entreprise:", company_name)
        
    # 2. Analyse fondamentale
    chain = RunnableWithMessageHistory(
    RunnableParallel({
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "history": itemgetter("history"),
        "company_name": itemgetter("company_name"),
        "n_year": itemgetter("n_year"),
    })
    .assign(response=prompt_fondamental_analysis | model | StrOutputParser())
    .pick(["response", "context"]),
    lambda session_id: history,
    input_messages_key="question",
    history_messages_key="history",
    output_messages_key="response",
    )

    response_fondamental = chain.invoke(
        {
            "company_name": company_name,
            "n_year": n_year,
            "question": f"Effectue une analyse fondamentale de l'entreprise {company_name}, incluant le 'chiffre d'affaires', "
                    "'bénéfice net', 'marge nette', 'ratio dette/capital', 'ratio de liquidité', 'ratio de solvabilité', "
                    "'ratio de rentabilité', et 'ratio de croissance'.",
            "history": st.session_state["chat_messages"],
        },
        config
    )


    
    st.session_state["chat_messages"].append({"role": "assistant", "content": response_fondamental['response']})
    with st.expander("Résultats intermédiaires - Analyse Fondamentale"):
        st.write(response_fondamental['response'])
        
    # 4. Explication de l'algorithme d'apprentissage par arbre de décision
    # chain_with_history = create_chain_with_prompt(prompt_decision_tree_explanation)
    # response_decision_tree = chain_with_history.invoke(
    #     {"question": "Explique le but de l'algorithme d'apprentissage par arbre de décision dans le contexte de la finance.", "history": st.session_state["chat_messages"]},
    #     config
    # )
    # st.session_state["chat_messages"].append({"role": "assistant", "content": response_decision_tree['response']})
    # with st.expander("Résultats intermédiaires - Explication de l'Arbre de Décision"):
    #     st.write(response_decision_tree['response'])
        
    # on lance l'analyse technique
    marge = technical_analysis(ticker , '2021-01-01', '2021-12-31')
    
    # Créez le template de prompt
    chain = RunnableWithMessageHistory(
        RunnableParallel({
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
            "company_name": itemgetter("company_name"),  # Ajout de company_name
            "n_year": itemgetter("n_year"),              # Ajout de n_year
        })
        .assign(response=prompt_technical_analysis | model | StrOutputParser())
        .pick(["response", "context"]),
        lambda session_id: history,
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key="response",
    )
    # chain_with_history = create_chain_with_prompt(prompt_investment_recommendation)
    response_technical = chain.invoke(
        {"question": f"Sur la base de l'analyse technique effectué sur l'entreprise {ticker} qui résulte en {marge}",
            "history": st.session_state["chat_messages"],
            "company_name": company_name,
            "n_year": n_year},
        config
    )
    st.session_state["chat_messages"].append({"role": "assistant", "content": response_technical['response']}) 
    with st.expander("Résultats intermédiaires - Analyse technique"):
        st.write(response_technical['response'])
    
    # 5. Recommandation d'action d'investissement
    chain = RunnableWithMessageHistory(
        RunnableParallel({
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
            "company_name": itemgetter("company_name"),  # Ajout de company_name
            "n_year": itemgetter("n_year"),              # Ajout de n_year
            "marge": itemgetter("marge"),
            "response_fondamental": itemgetter("response_fondamental"),
        })
        .assign(response=prompt_investment_recommendation | model | StrOutputParser())
        .pick(["response", "context"]),
        lambda session_id: history,
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key="response",
    )
    # chain_with_history = create_chain_with_prompt(prompt_investment_recommendation)
    response_investment = chain.invoke(
        {"question": f"Sur la base de l'analyse fondamentale et technique, recommande une action d'investissement pour l'entreprise {ticker} sur {n_year} ans.", 
         "history": st.session_state["chat_messages"], 
         "company_name": ticker, 
         "n_year": n_year,
         "marge": str(marge),
         "response_fondamental": response_fondamental['response'],},
        config
    )
    st.session_state["chat_messages"].append({"role": "assistant", "content": response_investment['response']})
        
    return response_investment['response']
# ------------------------------------------------------
# Pydantic data model and helper function for Citations

class Citation(BaseModel):
    page_content: str
    metadata: Dict

def extract_citations(response: List[Dict]) -> List[Citation]:
    return [Citation(page_content=doc.page_content, metadata=doc.metadata) for doc in response]

# ------------------------------------------------------
# S3 Presigned URL

def create_presigned_url(bucket_name: str, object_name: str, expiration: int = 300) -> str:
    """Generate a presigned URL to share an S3 object"""
    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except NoCredentialsError:
        st.error("AWS credentials not available")
        return ""
    return response

def parse_s3_uri(uri: str) -> tuple:
    """Parse S3 URI to extract bucket and key"""
    parts = uri.replace("s3://", "").split("/")
    bucket = parts[0]
    key = "/".join(parts[1:])
    return bucket, key

# ------------------------------------------------------
# Streamlit

# import streamlit as st

# # Page title
# st.set_page_config(page_title='Knowledge Bases for Amazon Bedrock and LangChain 🦜️🔗')

# # Clear Chat History function
# def clear_chat_history():
#     history.clear()
#     st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# with st.sidebar:
#     st.title('Knowledge Bases for Amazon Bedrock and LangChain 🦜️🔗')
#     streaming_on = st.toggle('Streaming')
#     st.button('Clear Chat History', on_click=clear_chat_history)
#     st.divider()
#     st.write("History Logs")
#     st.write(history.messages)

# Initialize session state for messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat Input - User Prompt 
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    config = {"configurable": {"session_id": "any"}}
    
    if streaming_on:
        # Chain - Stream
        # with st.chat_message("assistant"):
        #     placeholder = st.empty()
        #     full_response = ''
        #     for chunk in chain_with_history.stream(
        #         {"question" : prompt, "history" : history},
        #         config
        #     ):
        #         if 'response' in chunk:
        #             full_response += chunk['response']
        #             placeholder.markdown(full_response)
        #         else:
        #             full_context = chunk['context']
        #     placeholder.markdown(full_response)
        #     # Citations with S3 pre-signed URL
        #     citations = extract_citations(full_context)
        #     with st.expander("Show source details >"):
        #         for citation in citations:
        #             st.write("Page Content:", citation.page_content)
        #             s3_uri = citation.metadata['location']['s3Location']['uri']
        #             bucket, key = parse_s3_uri(s3_uri)
        #             presigned_url = create_presigned_url(bucket, key)
        #             if presigned_url:
        #                 st.markdown(f"Source: [{s3_uri}]({presigned_url})")
        #             else:
        #                 st.write(f"Source: {s3_uri} (Presigned URL generation failed)")
        #             st.write("Score:", citation.metadata['score'])
        #     # session_state append
        #     st.session_state.messages.append({"role": "assistant", "content": full_response})
        pass
    else:
        # Chain - Invoke
        with st.chat_message("assistant"):
            response = handle_prompts_sequentially(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Citations with S3 pre-signed URL
            # citations = extract_citations(response['context'])
            # with st.expander("Show source details >"):
            #     for citation in citations:
            #         st.write("Page Content:", citation.page_content)
            #         s3_uri = citation.metadata['location']['s3Location']['uri']
            #         bucket, key = parse_s3_uri(s3_uri)
            #         presigned_url = create_presigned_url(bucket, key)
            #         if presigned_url:
            #             st.markdown(f"Source: [{s3_uri}]({presigned_url})")
            #         else:
            #             st.write(f"Source: {s3_uri} (Presigned URL generation failed)")
            #         st.write("Score:", citation.metadata['score'])
            # session_state append
            # st.session_state.messages.append({"role": "assistant", "content": response['response']})
       

if plot_button:
    # Add user message for plotting request
    button_prompt(selected_company, selected_indicators, selected_period)
    # Plot financial data
    plot_financial_data(company_symbol, selected_indicators, selected_period)

