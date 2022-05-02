import pandas as pd
import numpy as np

import streamlit as st

st.set_page_config(
    page_title="Rapport de Projet Bank_Py",
    page_icon="*",
    layout="wide",)

st.title ("Projet Bank_Py")
st.markdown("--- ")
st.subheader('Datascientest : Formation Continue Datascientist Octobre 2021')

st.markdown("--- ")

st.markdown("*   Ndeye-Yacine FALL")
st.markdown("*   Louis PAVIE")
st.markdown("*   Karima TOUMI")

st.markdown('## 1) Contexte')
st.markdown("--- ")


st.markdown("L'objectif de ce projet *fil rouge* consiste à prédire le succès d'une campagne de Marketing bancaire visant à la souscription d'un contrat à terme.")
 
st.markdown("Le support de ce projet est un *dataset* contenant les données d'une campagne de télémarketing réalisée auprès de ses clients par une banque Portugaise entre Mai 2008 et Novembre 2010.")

st.markdown("Il existe plusieurs versions disponibles de ce dataset, de tailles et de compositions différentes. Parmi les 6 versions trouvées, nous avons préselectionnées les 2 versions contenant le plus de lignes (> 40 000 lignes), et finalement choisi la version avec données économiques (21 colonnes) qui nous paraissait plus intéressante que la version purement données bancaires (17 colonnes).")
 
st.markdown("Pour mener à bien ce projet, notre démarche s'articulera en trois temps:") 
st.markdown(" * Nous effectuerons d'abord une analyse visuelle et statistique des données clients et du lien avec la variable cible (Y) souscription au dépôt à terme.")

st.markdown(" * Dans un deuxième temps, nous utiliserons les techniques du *Machine Learning* pour essayer de déterminer à l'avance si un client va souscrire ou non au dépot à terme proposé.")

st.markdown(" * Enfin à l'aide des techniques d'interprétabilité des modèles, nous tenterons d'expliquer à l'échelle d'un individu pourquoi il est plus susceptible de souscrire ou non.")

st.markdown(" **Références** : Données publiques utilisables dans le cadre de la recherche et l'éducation.")

st.markdown(" [Moro et al., 2011] *S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing.* ")



st.markdown('## 2) Exploration des données')
st.markdown("--- ")

df=pd.read_csv("bank-additional-full.csv",sep=";")

df = df.drop_duplicates(keep = 'first')

st.dataframe(df.head())

#st.write(df.info())

df['Y_num'] = df['y']
df['Y_num'].replace({'no': 0}, inplace=True)
df['Y_num'].replace({'yes': 1}, inplace=True)

df['pdays'].value_counts()
df['pdays'].replace({999: -1}, inplace=True)

df['duration'] = df['duration'] / 60

st.write(df.describe().transpose().round())

st.markdown("* L'age moyen du portefeuille est de 40 ans")
st.markdown("* 50% du portefeuille a un age compris entre 32 ans et 47 ans.")
st.markdown("* La durée du contact est en moyenne de 4 minutes et 18 secondes.")
st.markdown("* Pour la campagne en cours, chaque client a été contacté en moyenne entre 2 et 3 fois.")
st.markdown("* 50% des clients ont été contactés entre 1 ou 3 fois.")
st.markdown("* Le nombre de contact maximum qui est de 56 nous semble aberrant.")
st.markdown("* Vu les statistiques, la variable pdays ne semble pas exploitable, nous pourrons la retirer par la suite.")
st.markdown("* La majorité des clients est contacté pour la première fois lors de cette campagne.")
st.markdown("* Sur la période d'observation, les indicateurs économiques sont relativement stables hormis l'Euribor et la variation du taux d'emploi.")

import matplotlib.pyplot as plt
import seaborn as sns



if st.checkbox("--> Afficher la visualisation de la variable Age"):
    st.markdown("#### Visualisation de la variable 'Age'")
    fig, ax = plt.subplots(figsize=(15,15))
    ax.hist(df['age'], rwidth=0.9, bins=15)
    plt.xlabel('Ages')
    plt.ylabel("Fréquence")
    plt.xticks([15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95])
    plt.title("Distribution de la variable 'Age'")
    st.pyplot(fig)
    
    st.markdown("Cette visualisation de la distribution de la variable 'age' permet de constater que la population ciblée est majoritairement 'jeune' et d'age inférieur à 60 ans.")


if st.checkbox("--> Afficher la visualisation de la variable Duration"):
    st.markdown("#### Visualisation de la variable 'Duration'")
    fig, ax = plt.subplots(figsize=(20,15))

    ax.hist(df['duration'], rwidth=0.9, bins=60)
    plt.xlabel('Durée en minutes du dernier contact')
    plt.ylabel("Fréquence")
    plt.xticks([0,1,3,6,12,20,24,30,40,60])
    plt.title("Distribution de la variable 'Duration'")
    st.pyplot(fig)
    st.markdown("Nous pouvons constater que : ")
    st.markdown("* la majorité des appels durent entre 1 et 6 minutes")
    st.markdown("* les appels de moins de 1 minutes sont improductifs")
    st.markdown("* les appels entre 3 et 12 minutes sont souvent productifs")
    st.markdown("* les appels entre 12 et 24 minutes, sont très productifs, mais peu nombreux.")


st.markdown("/n")    
st.markdown("#### Visualisation de la Heatmap")
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(df.corr(), ax=ax, cmap='coolwarm', annot=True)       
st.pyplot(fig)
st.markdown("La matrice de corrélation entre les variables quantitatives permet de constater de manière générale des corrélations assez faibles entre les variables.")
st.markdown("Toutefois nous pouvons constater des corrélations assez fortes pour les variables économiques entre elles, ce qui est logique.")
st.markdown("En ce qui concerne la corrélation avec la variable cible, c'est la durée de l'appel qui obtient le coefficient de corrélation le plus important (0,41).")
st.markdown("Suivi par le nombre de jours depuis le dernier contact (0,28) et le nombre de contact (0,23).")
st.markdown("Les coefficients de corrélations négatif entre la variable cible et les variables économiques s'expliquent par l'aspect 'refuge' du dépôt à terme.")

#fig = plt.subplots(figsize=(15,15))
#sns.displot(df['age'], kde= True)
#st.pyplot(fig)