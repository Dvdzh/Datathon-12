
# Projet d'Analyse Financière et de Prédiction

## Présentation du projet
Ce projet a pour objectif de créer un outil d'analyse et de prédiction des données financières. Il combine l'utilisation de bibliothèques d'apprentissage automatique telles que `scikit-learn`, la récupération de données financières en temps réel via `yfinance`, et des visualisations graphiques avec `matplotlib`. De plus, le projet intègre des fonctionnalités de prompts conversationnels basées sur les services AWS, telles que `ChatBedrock` et `AmazonKnowledgeBasesRetriever`.

## Installation

### Prérequis
Assurez-vous d'avoir Python installé sur votre machine (version 3.7 ou ultérieure).

### Étapes d'installation

```bash
# 1. Clone le repository
git clone https://github.com/cy-4/Datathon.git
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
```

## Utilisation des fonctionnalités avancées

Pour utiliser les fonctionnalités basées sur AWS, assurez-vous que vos identifiants AWS sont correctement configurés dans votre environnement. Vous pouvez le faire en utilisant `aws configure` ou en configurant les variables d'environnement `AWS_ACCESS_KEY_ID` et `AWS_SECRET_ACCESS_KEY`.

---

### Note
Pour des détails supplémentaires sur les configurations avancées et les cas d'utilisation, veuillez consulter la documentation complète dans le répertoire `docs`.

## Exemples d'utilisation :

Pour réaliser une analyse d'entreprise, suivez ces exemples :

1. **Pour une analyse générale de Google sur une période de 3 ans**, tapez par exemple:
`Analyse moi Google pour une perspective de 2 ans`
2. **Pour une analyse basée sur des documents financiers, tapez :**, tapez par exemple:
`Analyse-moi Couche-Tard pour une perspective de 2 ans.`
> **Remarque :** Pour le moment, seuls les documents financiers de Couche-Tard sont disponibles dans le système.
3. **Pour afficher des graphiques financiers**, sélectionnez l'entreprise, les indicateurs à afficher, puis la période de temps souhaitée. 