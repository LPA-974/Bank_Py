# #  Projet Bank_Py
# ---
# 
# Datascientest : Formation Continue Datascientist Octobre 2021
# 
# *   Ndeye-Yacine FALL
# *   Louis PAVIE
# *   Karima TOUMI
# 
# 

# ## 1) Contexte
# 
# 
# ---
# 
# 

# L'objectif de ce projet *fil rouge* consiste à prédire le succès d'une campagne de Marketing bancaire visant à la souscription d'un contrat à terme.
# 
# Le support de ce projet est un *dataset* contenant les données d'une campagne de télémarketing réalisée auprès de ses clients par une banque Portugaise entre Mai 2008 et Novembre 2010.
# Il existe plusieurs versions disponibles de ce dataset, de tailles et de compositions différentes. Parmi les 6 versions trouvées, nous avons préselectionnées les 2 versions contenant le plus de lignes (> 40 000 lignes), et finalement choisi la version avec données économiques (21 colonnes) qui nous paraissait plus intéressante que la version purement données bancaires (17 colonnes).
# 
# Pour mener à bien ce projet, notre démarche s'articulera en trois temps: 
# * Nous effectuerons d'abord une analyse visuelle et statistique des données clients et du lien avec la variable cible (Y) souscription au dépôt à terme.
# 
# * Dans un deuxième temps, nous utiliserons les techniques du *Machine Learning* pour essayer de déterminer à l'avance si un client va souscrire ou non au dépot à terme proposé.
# 
# * Enfin à l'aide des techniques d'interprétabilité des modèles, nous tenterons d'expliquer à l'échelle d'un individu pourquoi il est plus susceptible de souscrire ou non.
# 
# **Références** : Données publiques utilisables dans le cadre de la recherche et l'éducation.
# 
# [Moro et al., 2011] *S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing.*

# ## 2) Configuration
# 
# 
# ---
# 
# 

# Cette étape recouvre les importations de bibliothèques nécessaires au bon fonctionnement du notebook

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st

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

df['Y_num'] = df['y']
df['Y_num'].replace({'no': 0}, inplace=True)
df['Y_num'].replace({'yes': 1}, inplace=True)

df['pdays'].value_counts()
df['pdays'].replace({999: -1}, inplace=True)

st.write(df.describe().transpose().round())

st.markdown("* L'age moyen du portefeuille est de 40 ans.")
st.markdown("* 50% du portefeuille a un age compris entre 32 ans et 47 ans.")
st.markdown("* La durée du contact est en moyenne de 4 minutes et 18 secondes.")
st.markdown("* Pour la campagne en cours, chaque client a été contacté en moyenne entre 2 et 3 fois.")
st.markdown("* 50% des clients ont été contactés entre 1 ou 3 fois.")
st.markdown("* Le nombre de contact maximum qui est de 56 nous semble aberrant.")
st.markdown("* Vu les statistiques, la variable pdays ne semble pas exploitable, nous pourrons la retirer par la suite.")
st.markdown("* La majorité des clients est contacté pour la première fois lors de cette campagne.")
st.markdown("* Sur la période d'observation, les indicateurs économiques sont relativement stables hormis l'Euribor et la variation du taux d'emploi.")

st.markdown("Distribution de la variable 'Age'")
st.bar_chart(df['age'])

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plt.xlabel('Ages')
plt.ylabel("Fréquence")
plt.title("Distribution de la variable 'Age', en fonction du résultat (Y/N) de souscription")
ax.hist(df['age'], bins=20)
st.pyplot(fig)