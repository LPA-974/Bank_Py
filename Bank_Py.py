import pandas as pd
import numpy as np

import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns


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

st.markdown('### Visualisations graphiques')
st.markdown("--- ")

if st.checkbox("--> Afficher la visualisation de la variable Months"):
    st.markdown("#### Visualisation de la variable 'Month'")
    st.markdown(" ")
    fig, ax = plt.subplots(figsize=(15,15))
    ax.hist(df['month'], rwidth=0.9, bins=15)
    plt.xlabel('Month')
    plt.ylabel("Fréquence")
    plt.title("Distribution de la variable 'Month'")
    st.pyplot(fig)
    
    st.markdown("La répartition par jours de la semaine est assez homogène, cela se retrouve dans les proportions de souscriptions. Remarque, nous pouvons constater qu'il n'y a pas de démarchage le week-end. Notre reconstitution de la série chronologique se fera donc uniquement sur année-mois.")
    
if st.checkbox("--> Afficher la visualisation de la variable Day_of_Weeks"):
    st.markdown("#### Visualisation de la variable 'Day_of_Weeks'")
    st.markdown(" ")
    fig, ax = plt.subplots(figsize=(15,15))
    ax.hist(df['day_of_week'], rwidth=0.9, bins=15)
    plt.xlabel('day_of_week')
    plt.ylabel("Fréquence")
    plt.title("Distribution de la variable 'Day_of_Weeks'")
    st.pyplot(fig)
    
    st.markdown("Nous constatons une répartition inégale des Mois, suite à un recouvrement, la campagne durant plus de un an. L'absence d'une variable date (avec l'année) rend difficile une analyse plus précise du phénomène sans retravailler les données pour reconstitution d'une série chronologique explicite.")
    
if st.checkbox("--> Afficher la visualisation du Nombre d'entretiens et de la conversion au fil du temps"):
    st.markdown("Nombre d'entretien et du nombre de conversion au fil du temps")
    st.image('./Var_Interview.jpg')

    
    st.markdown("Taux de conversion au fil du temps")
    st.image('./Var_Tx_Conv.jpg')

    st.markdown("Nous pouvons constater un effet d'apprentissage certain au cours du temps. En effet le taux de conversion augmente significativement et en même temps le nombre d'appels diminue drastiquement.")
    

    
if st.checkbox("--> Afficher la visualisation de la variable Age"):
    st.markdown("#### Visualisation de la variable 'Age'")
    st.image('./Var_Dist_Age.jpg')

    
    st.markdown("Cette visualisation de la distribution de la variable 'age' permet de constater que la population ciblée est                   majoritairement 'jeune' et d'age inférieur à 60 ans.")

    st.markdown("Age moyen des clients au fil du temps")
    st.image('./Var_Age_Mean.jpg')

if st.checkbox("--> Afficher la visualisation de la variable Duration"):
    st.markdown("#### Visualisation de la variable 'Duration'")
    st.image('./Var_Dist_Duration.jpg')

    st.markdown("Nous pouvons constater que : ")
    st.markdown("* la majorité des appels durent entre 1 et 6 minutes")
    st.markdown("* les appels de moins de 1 minutes sont improductifs")
    st.markdown("* les appels entre 3 et 12 minutes sont souvent productifs")
    st.markdown("* les appels entre 12 et 24 minutes, sont très productifs, mais peu nombreux.")

    
    st.markdown("Durée moyenne des entretiens au fil du temps")
    st.image('./Var_Duration.jpg')    

if st.checkbox("--> Afficher la visualisation de la Heatmap"):
    st.markdown("#### Visualisation de la Heatmap")
    st.markdown(" ")
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(df.corr(), ax=ax, cmap='coolwarm', annot=True)       
    st.pyplot(fig)
    st.markdown("La matrice de corrélation entre les variables quantitatives permet de constater de manière générale des corrélations assez     faibles entre les variables.")
    st.markdown("Toutefois nous pouvons constater des corrélations assez fortes pour les variables économiques entre elles, ce qui est         logique.")
    st.markdown("En ce qui concerne la corrélation avec la variable cible, c'est la durée de l'appel qui obtient le coefficient de             corrélation le plus important (0,41).")
    st.markdown("Suivi par le nombre de jours depuis le dernier contact (0,28) et le nombre de contact (0,23).")
    st.markdown("Les coefficients de corrélations négatif entre la variable cible et les variables économiques s'expliquent par l'aspect       'refuge' du dépôt à terme.")

if st.checkbox("--> Afficher la visualisation de la variable Campaign"):
    st.markdown("#### Visualisation de la variable 'Campaign'")
    st.image('./Var_Dist_Campaign.jpg')

    st.markdown("Les personnes qui ont souscrits sont près de 50% à le faire au premier contact, cependant, les personnes qui ont été contactées plusieurs fois au cours de cette campagne sont encore susceptibles de souscrire même si cette tendance diminue après le troisiemme contact et devient négligeable après le cinquième contact (effet de lassitude ?).")

if st.checkbox("--> Afficher la visualisation de la variable PDays"):
    st.markdown("#### Visualisation de la variable 'Pdays'")
    st.image('./Var_Dist_Pdays.jpg')

    st.markdown("La majorité des non souscriptions (98,5%) se produisent pour des personnes qui non pas été recontactées après une campagne antérieure, c'est à dire pour des personnes primo-contactées. Par contre pour les souscriptions, elles sont réalisées à 79% par des primo-contactées, puis le reste des souscriptions par les personnes au fur et à mesure des recontacts mêmes si les volumes représentés sont très faibles.")

    
if st.checkbox("--> Afficher la visualisation de la variable Previous"):
    st.markdown("#### Visualisation de la variable 'Previous'")
    st.image('./Var_Dist_Previous.jpg')

    st.markdown("La trés grande majorité des personnes n'ont pas été contactées avant la campagne en cours (previous = 0). Cependant, les personnes qui ont été recontactées sont susceptibles de souscrire même si cette tendance diminue après le troisiemme contact (effet de lassitude ?).")
    