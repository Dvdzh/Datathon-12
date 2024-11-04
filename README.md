
# Projet d'Analyse Financière et de Prédiction

## Présentation du projet
Ce projet d'analyse financière est conçu pour évaluer les entreprises dans un but d'investissement. Il oriente un chatbot à réaliser une analyse fondamentale basée uniquement sur sa base de données et une analyse technique reposant sur un calcul expliqué par la suite. L'objectif est de former une synthèse permettant de recommander l'achat ou non du titre analysé.



## Installation

### Prérequis
Assurez-vous que les éléments suivants sont installés et configurés sur votre machine :

- **Python** : Veillez à utiliser Python version 3.7 ou ultérieure. Vous pouvez vérifier votre version de Python en utilisant la commande suivante :
  ```bash
  python --version
  python3 --version # si python3 est le bon alias
  ```

- **Identifiants AWS** : Pour permettre au projet d'accéder aux services AWS, vous devez configurer vos identifiants AWS. Vous pouvez le faire en exportant les variables d'environnement suivantes :
  ```bash
  export AWS_ACCESS_KEY_ID=your_access_key_id
  export AWS_SECRET_ACCESS_KEY=your_secret_access_key
  export AWS_REAWS_SESSION_TOKEN=your_session_token
  ```

  Ces identifiants sont nécessaires pour utiliser des fonctionnalités telles que `ChatBedrock` et `AmazonKnowledgeBasesRetriever`.

### Étapes d'installation

```bash
# 1. Clone le repository
git clone https://github.com/Dvdzh/Datathon-12
cd Datathon

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate   # Sur Unix/MacOS
# .\venv\Scripts\activate    # Sur Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Lancer une démo

Pour exécuter une démonstration du projet, utilisez le script principal (par exemple, `main.py`) :

```bash
python -m streamlit run main.py
python3 -m streamlit run main.py # si python3 est le bon alias 
```

## Exemples d'utilisation :

Pour réaliser une analyse d'entreprise, suivez ces exemples :

1. **Pour une analyse générale de Google sur une période de 3 ans**, tapez par exemple:
`Analyse moi Google pour une perspective de 2 ans`
2. **Pour une analyse basée sur des documents financiers, tapez :**, tapez par exemple:
`Analyse-moi Couche-Tard pour une perspective de 2 ans.`
> **Remarque :** Pour le moment, seuls les documents financiers de Couche-Tard sont disponibles dans le système.
3. **Pour afficher des graphiques financiers**, sélectionnez l'entreprise, les indicateurs à afficher, puis la période de temps souhaitée. 