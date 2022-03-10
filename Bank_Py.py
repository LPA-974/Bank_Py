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
