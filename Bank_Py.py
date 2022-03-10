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

st.markdown("*   Ndeye-Yacine FALL
st.markdown("*   Louis PAVIE")
st.markdown("*   Karima TOUMI")

st.markdown('## 1) Contexte')
st.markdown("--- ")
# In[2]:


# Pour fonctionner à partir de Google Colab
#from google.colab import drive
#drive.mount('/content/drive')


# ## 3) Exploration des données
# 
# 
# ---
# 

# ### A. Chargement des données depuis le Dataset
# 
# 

# In[3]:


# pour fonctionner à partir de la plateforme CAAS
#df=pd.read_csv("/content/bank-additional-full.csv",sep=";")

# Pour fonctionner en local
df=pd.read_csv("bank-additional-full.csv",sep=";")

# Pour fonctionner à partir de Google Colab
#df=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Bank_Py /bank-additional-full.csv",sep=";")


# ### B. Découverte du DataSet

# In[4]:


print('Shape :', df.shape)
df.head()


# In[11]:


df.tail()


# In[5]:


df.info()


# * Le Dataset se compose de 41188 lignes pour 21 colonnes.
# * Pas de colonne avec des valeurs non renseignées.
# * Pas de colonne identifiant.
# * Pas de colonne date.
# 
# Le document [Bank_Py-Rapport_Exploration_Données-VF.xlsx] contient les informations utiles à la compréhension des données de ce dataset.

# Recherche de doublons parmi les lignes du dataset.

# In[6]:


df.duplicated().sum()


# 
# 
# *   Présence de 12 lignes en doublons
# *   Suppression de ces lignes en doublons
# 

# In[7]:


df = df.drop_duplicates(keep = 'first')


# Création d'un clone *'numérique'* de la variable cible pour les besoins de la matrice de corrélation.

# In[8]:


df['Y_num'] = df['y']
df['Y_num'].replace({'no': 0}, inplace=True)
df['Y_num'].replace({'yes': 1}, inplace=True)
df['Y_num'].value_counts()


# In[9]:


df['pdays'].value_counts()
df['pdays'].replace({999: -1}, inplace=True)


# ### C. Analyse visuelle et statistique

# 
# #### Variables Quantitatives
# 

# ##### Analyse de la distribution des variables quantitatives

# In[10]:


df.describe().round()


# * L'age moyen du portefeuille est de 40 ans.
# * 50% du portefeuille a un age compris entre 32 ans et 47 ans.
# * La durée du contact est en moyenne de 4 minutes et 18 secondes.
# * Pour la campagne en cours, chaque client a été contacté en moyenne entre 2 et 3 fois.
# * 50% des clients ont été contactés entre 1 ou 3 fois.
# * Le nombre de contact maximum qui est de 56 nous semble aberrant.
# * Vu les statistiques, la variable pdays ne semble pas exploitable, nous pourrons la retirer par la suite.
# * La majorité des clients est contacté pour la première fois lors de cette campagne.
# * Sur la période d'observation, les indicateurs économiques sont relativement stables hormis l'Euribor et la variation du taux d'emploi.

# ##### Visualisation via 'heatmap' de la corrélation entre les variables quantitatives

# In[11]:


cor = df.corr()

fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(cor, annot=True, ax=ax, cmap='coolwarm')
plt.title('Matrice de Corrélation des variables quantitatives');


# La matrice de corrélation entre les variables quantitatives permet de constater de manière générale des corrélations assez faibles entre les variables. Toutefois nous pouvons constater des corrélations assez fortes pour les variables économiques entre elles, ce qui est logique.
# 
# En ce qui concerne la corrélation avec la variable cible, c'est la durée de l'appel qui obtient le coefficient de corrélation le plus important (0,41).
# Suivi par le nombre de jours depuis le dernier contact (0,28) et le nombre de contact (0,23).
# Les coefficients de corrélations négatif entre la variable cible et les variables économiques s'expliquent par l'aspect 'refuge' du dépôt à terme.  

# ##### Visualisation de la distribution de la variable 'Age', en fonction du résultat (Y/N) de souscription.

# In[12]:


sns.displot(data=df, x="age", hue="y", kde= True, height=15)
plt.xlabel('Ages')
plt.ylabel("Fréquence")
plt.title("Distribution de la variable 'Age', en fonction du résultat (Y/N) de souscription");


# Cette visualisation de la distribution de la variable 'age' permet de constater que la population ciblée est majoritairement 'jeune' et d'age inférieur à 60 ans.
# Nous pouvons constater aussi que la population d'age supérieur à 60 ans est beaucoup plus apétente à la souscrition du produit dépôt à terme.  

# ##### Visualisation de la distribution de la variable 'Duration', en fonction du résultat (Y/N) de souscription.

# In[13]:


sns.displot(data=df, x="duration", hue="y", kde= True, height=15)
# découpage en plages 'minutes compatibles' des durées en secondes 
plt.xticks([0,60,180,360,720,1200,1440,1800,2400,3600])
plt.xlabel('Durée en secondes du dernier contact')
plt.ylabel("Fréquence")
plt.title("Distribution de la variable 'Duration', en fonction du résultat (Y/N) de souscription");


# Nous pouvons constater que : 
# * la majorité des appels durent entre 1 et 6 minutes
# * les appels de moins de 1 minutes sont improductifs
# * les appels entre 3 et 12 minutes sont souvent productifs
# * les appels entre 12 et 24 minutes, sont très productifs, mais peu nombreux. 

# ##### Visualisation de la distribution de la variable 'Campaign', en fonction du résultat (Y/N) de souscription.

# In[14]:


sns.catplot(x='y', y='campaign', kind='violin', data=df, height=15)
plt.yticks([0,1,2,3,4,5,6,7,8,9,10,15,20,25])
plt.xlabel('Résultat de souscription Y/N')
plt.ylabel("Nombre de contacts pendant la campagne en cours)")
plt.title("Distribution de la variable 'Campaign', en fonction du résultat (Y/N) de souscription");


# In[15]:


pd.crosstab(df.y, df.campaign, normalize = 0)


# Les personnes qui ont souscrits sont près de 50% à le faire au premier contact, cependant, les personnes qui ont été contactées plusieurs foisau cours de cette campagne sont encore susceptibles de souscrire même si cette tendance diminue après le troisiemme contact et devient négligeable après le cinquième contact (effet de lassitude ?).

# ##### Visualisation de la variable 'pdays', en fonction du résultat (Y/N) de souscription.

# In[16]:


sns.catplot(x='y', y='pdays', kind='violin', data=df, height=15)
plt.yticks([-1,0,1,2,3,4,5,6,7,8,9,10,15,20,25])
plt.xlabel('Résultat de souscription Y/N')
plt.ylabel("Nombre de jours depuis le dernier appel (-1 = jamais)")
plt.title("Distribution de la variable 'pdays', en fonction du résultat (Y/N) de souscription");


# In[17]:


pd.crosstab(df.y, df.pdays, normalize = 0)


# La majorité des non souscriptions (98,5%) se produisent pour des personnes qui non pas été recontactées après une campagne antérieure, c'est à dire pour des personnes primo-contactées. Par contre pour les souscriptions, elles sont réalisées à 79% par des primo-contactées, puis le reste des souscriptions par les personnes au fur et à mesure des recontacts mêmes si les volumes représentés sont très faibles.  

# ##### Visualisation de la variable 'previous', en fonction du résultat (Y/N) de souscription.

# In[18]:


sns.catplot(x='y', y='previous', kind='violin', data=df, height=15)

plt.xlabel('Résultat de souscription Y/N')
plt.ylabel("Nombre de contacts avant cette campagne pour ce client")
plt.title("Distribution de la variable 'previous', en fonction du résultat (Y/N) de souscription");


# In[19]:


pd.crosstab(df.y, df.previous, normalize = 0)


# La trés grande majorité des personnes n'ont pas été contactées avant la campagne en cours (previous = 0). Cependant, les personnes qui ont été recontactées sont susceptibles de souscrire même si cette tendance diminue après le troisiemme contact (effet de lassitude ?).

# #### Variables Qualitatives

# ##### Visualisation de la variable 'Job', en fonction du résultat (Y/N) de souscription

# In[20]:


plt.figure(figsize = (15, 15))
sns.countplot(x='job', data=df, hue = 'y')
plt.xlabel('Métiers des personnes contactées')
plt.ylabel("Nombre d'occurences")
plt.title("Variable 'Job', en fonction du résultat (Y/N) de souscription");


# In[21]:


pd.crosstab(df.y, df.job, normalize = 0)


# Les employés de bureau, les ouvriers et les techniciens sont les plus représentés parmi les personnes contactées, et c'est parmi leurs rangs que l'on retrouve la plus forte proportion de ceux qui ont souscrit (employés de bureau 29%, techniciens 16% et ouvriers 14%).   

# In[22]:


pd.crosstab(df.y, df.job, normalize = 1)


# Par contre quand nous étudions la proportion de souscription par modalité, nous le trio étudiants (31%), retraités (25%), et chômeurs (14%) ressort en tête, ce qui pourrait sembler contre-intuitif mais peut s'expliquer par un meilleur ciblage dans ces catégories et sans doute aussi par un effet de nombre pour les catégories les plus représentées.

# ##### Visualisation de la variable 'Marital', en fonction du résultat (Y/N) de souscription

# In[23]:


plt.figure(figsize = (15, 15))
sns.countplot(x='marital', data=df, hue = 'y')
plt.xlabel('Statut marital des personnes contactées')
plt.ylabel("Nombre d'occurences")
plt.title("Variable 'Marital', en fonction du résultat (Y/N) de souscription");


# In[24]:


pd.crosstab(df.y, df.marital, normalize = 0)


# Les personnes marriées sont les plus représentés parmi les personnes contactées, et c'est parmi leurs rangs que l'on retrouve la plus forte proportion de ceux qui ont souscrit (54,5%), suivi par les célibataires (35%).

# In[25]:


pd.crosstab(df.y, df.marital, normalize = 1)


# Par contre quand nous étudions la proportion de souscription par modalité, ce sont les célibataires qui soucrivent le plus (14%).

# ##### Visualisation de la variable 'Education', en fonction du résultat (Y/N) de souscription

# In[26]:


plt.figure(figsize = (15, 15))
sns.countplot(x='education', data=df, hue = 'y')
plt.xlabel("Niveau d'études des personnes contactées")
plt.ylabel("Nombre d'occurences")
plt.title("Variable 'Education', en fonction du résultat (Y/N) de souscription");


# In[27]:


pd.crosstab(df.y, df.education, normalize = 0)


# Les personnes les plus représentés parmi les personnes contactées sont celles qui ont un diplome universitaire, suivi de celles qui suivi leur scolarité dans une 'grande école'. C'est parmi leurs rangs que l'on retrouve la plus forte proportion de ceux qui ont souscrit (36%) et (22%).

# In[28]:


pd.crosstab(df.y, df.education, normalize = 1)


# 22% des illettrés souscrivent un dépôt à terme, ce bon score est à rapporter au trés faible effectif de ceux-ci, et n'est pas significatif. 

# ##### Visualisation de la variable 'Default', en fonction du résultat (Y/N) de souscription

# In[29]:


plt.figure(figsize = (15, 15))
sns.countplot(x='default', data=df, hue = 'y')
plt.xlabel("Préexistence d'un défaut de paiement")
plt.ylabel("Nombre d'occurences")
plt.title("Variable 'default', en fonction du résultat (Y/N) de souscription");


# In[30]:


pd.crosstab(df.y, df.default, normalize = 0)


# Sans surprise la trés grande majorité des personnes contactées n'ont pas connues de défaut de paiement. Et aucune des quelques personnes ayant un défaut de paiement n'a souscrit (clause exclusive ?).

# ##### Visualisation de la variable 'Housing', en fonction du résultat (Y/N) de souscription

# In[31]:


plt.figure(figsize = (15, 15))
sns.countplot(x='housing', data=df, hue = 'y')
plt.xlabel("Préexistence d'un prêt Habitat")
plt.ylabel("Nombre d'occurences")
plt.title("Variable 'Housing', en fonction du résultat (Y/N) de souscription");


# In[32]:


pd.crosstab(df.y, df.housing, normalize = 0)


# Nous constatons une légère pré-éminence des détenteurs de prêt Habitat, tant en nombre de personnes qu'en proportion de ceux qui ont souscrits (54%).

# ##### Visualisation de la variable 'Loan', en fonction du résultat (Y/N) de souscription

# In[33]:


plt.figure(figsize = (15, 15))
sns.countplot(x='loan', data=df, hue = 'y')
plt.xlabel("Préexistence d'un prêt personnel")
plt.ylabel("Nombre d'occurences")
plt.title("Variable 'Loan', en fonction du résultat (Y/N) de souscription");


# In[34]:


pd.crosstab(df.y, df.loan, normalize = 0)


# La très grande majorité des personnes contactées n'avaient pas souscrit de prêt personnel, et c'est parmi eux que se retrouvent l'écrasante majorité de ceux qui ont souscrits (83%). Cependant la répartition homogène no/yes entre les trois modalités indique que cette variable est sans impact sur le résultat.  

# ##### Visualisation de la variable 'Contact', en fonction du résultat (Y/N) de souscription

# In[35]:


plt.figure(figsize = (15, 15))
sns.countplot(x='contact', data=df, hue = 'y')
plt.xlabel("Canal de contact")
plt.ylabel("Nombre d'occurences")
plt.title("Variable 'Contact', en fonction du résultat (Y/N) de souscription");


# In[36]:


pd.crosstab(df.y, df.contact, normalize = 0)


# Une majorité des personnes contactées l'ont été via téléphone mobile, et c'est parmi eux que se retrouvent l'écrasante majorité de ceux qui ont souscrits (83%).  

# ##### Visualisation des variables 'Month' et 'Day of Week', en fonction du résultat (Y/N) de souscription

# In[37]:


plt.figure(figsize = (15, 15))
sns.countplot(x='month', data=df, hue = 'y')
plt.xlabel("Mois de contact")
plt.ylabel("Nombre d'occurences")
plt.title("Variable 'Month', en fonction du résultat (Y/N) de souscription");


# Nous constatons une répartition inégale des Mois, suite à un recouvrement, la campagne durant plus de un an. L'absence d'une variable date (avec l'année) rend difficile une analyse plus précise du phénomène.

# In[38]:


plt.figure(figsize = (15, 15))
sns.countplot(x='day_of_week', data=df, hue = 'y')
plt.xlabel("Jours de la semaine")
plt.ylabel("Nombre d'occurences")
plt.title("Variable 'day_of_week', en fonction du résultat (Y/N) de souscription");


# La répartition par jours de la semaine est assez homogène, cela se retrouve dans les proportions de souscriptions. Remarque, nous pouvons constater qu'il n'y a pas de démarchage le week-end.

# ##### Visualisation de la variable 'Poutcome', en fonction du résultat (Y/N) de souscription

# In[39]:


plt.figure(figsize = (15, 15))
sns.countplot(x='poutcome', data=df, hue = 'y')
plt.xlabel("Résultats précédents")
plt.ylabel("Nombre d'occurences")
plt.title("Variable 'Poutcome', en fonction du résultat (Y/N) de souscription");


# In[40]:


pd.crosstab(df.y, df.poutcome, normalize = 0)


# La trés grande majorité des personnes contactées l'ont été pour la première fois (nonexistent), et c'est parmi eux que nous retrouvons la plus forte proportion de ceux qui ont souscrits (67%).

# In[41]:


pd.crosstab(df.y, df.poutcome, normalize = 1)


# 
# 
# Nous remarquons par ailleurs que 35% des personnes qui avaient déjà souscrit un dépôt à terme, n'ont pas souhaitées en souscrire un nouveau, à l'occasion de cette campagne.

# ##### Visualisation de la variable cible 'Y' résultat de la souscription

# In[42]:


plt.figure(figsize = (15, 15))
sns.countplot(x='y', data=df)
plt.xlabel("Résultats de la campagne de souscription")
plt.ylabel("Nombre d'occurences")
plt.title("Variable cible 'y', résultat de la souscription");


# In[43]:


df['y'].value_counts(normalize = True)


# Le résultat de la campagne de télémarketing bancaire est plutôt bon avec 11% de souscription de dépôt à terme.
# Remarque, pour les étapes de modélisations à suivre nous devrons certainement retravailler la taille de l'échantillon (undersampling ?) pour ré-équilibrer les 2 modalités.

# #### Tests Statistiques

# Nous étudierons ici les varaibles qualitatives, pour les variables quantitative on se reportera utilement à la matrice de corrélation préalablement étudiée ('heatmap')

# ##### Tests du Chi2 et de V de Cramer pour les variables qualitatives

# In[44]:


def V_Cramer(table, N):
    stat_chi2 = chi2_contingency(table)[0]
    k = table.shape[0]
    r = table.shape[1]
    phi = max(0,(stat_chi2/N)-((k-1)*(r-1)/(N-1)))
    k_corr = k - (np.square(k-1)/(N-1))
    r_corr = r - (np.square(r-1)/(N-1))
    return np.sqrt(phi/min(k_corr - 1,r_corr - 1))

df_quali = df[['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week','poutcome']]
colonne = (df_quali.columns)

for index, element in enumerate (colonne):
        table = pd.crosstab(df_quali[element],df['y'])
        display(element,V_Cramer(table, df_quali.shape[0]))


# Les V Cramer ne sont pas très élevés, nous pouvons en déduire que la corrélation est faible sans pour autant être négligeable surtout pour la variable 'poutcome' (0,32) et la variable 'month' (0,27). Enfin, nous pouvons confirmer que la variable 'loan' n'est pas corrélée au résultat (cf. analyse visuelle). 

# ##### Test ANOVA entre la variable qualitative 'Education' et la variable quantitative 'Duration'

# Notre hypothèse (H0) est que le niveau d'études n'influe pas sur la durée de l'entretien. C'est à dire que les variable 'Education' et 'Duration' sont indépendantes.

# In[45]:



sns.displot(data=df[df['duration'] < 1500], x="duration", kde= True,hue = 'education', height=15 )
plt.xticks([0,60,180,360,720,1200,1440])
plt.xlabel("Durée du contact en seconde")
plt.ylabel("Nombre de clients")
plt.title("Modalité et distribution de la variable duration selon le niveau d'études"); 


# In[46]:


result = statsmodels.formula.api.ols('duration ~ education', data=df).fit()
table = statsmodels.api.stats.anova_lm(result)
display(table)


# Le graphique nous montre que la distribution de la durée du contact semble corrélée au niveau d'éducation du client.
# 
# En faisant le test ANOVA, la p-value (6%) est supérieure de peu à la référence généralement admise (5%). Ce qui ne nous permettrait pas de rejeter l'hypothèse d'indépendance.
# 
# Cependant dans le cas de marketing de masse, comme c'est le cas ici, il est d'usage d'augmenter ce seuil jusqu'à une valeur de 10%. En considérant ce nouveau seuil, nous pouvons donc rejeter l'hypothèse d'indépendance de ces deux variables.
# Nous affirmerons donc que le niveau d'études du prospect à un effet statistique significatif sur la durée de l'entretien.
# 
# Il semble que plus le niveau d'éducation est élevé plus la durée du contact est élevé. Ceci peut s'expliquer par une capacité accrue à demander des détails techniques sur les conditions du dépôt à terme.
# 

# ## 4) Préprocessing
# 
# ---
# 

# ### A. Travail sur les variables Qualitatives
# Nous allons ici transformer les variables Qualitatives en variables Quantitatives. Pour les variables Qualitatives sans notion de progression, nous utiliserons la 'dummification'.

# In[47]:


df = df.join(pd.get_dummies(df.job, prefix='JT'))


# In[48]:


df = df.join(pd.get_dummies(df.marital, prefix='MS'))


# In[49]:


df = df.join(pd.get_dummies(df.education, prefix='EL'))


# In[50]:


df = df.join(pd.get_dummies(df.contact, prefix='CO'))


# In[51]:


df = df.join(pd.get_dummies(df.poutcome, prefix='PO'))


# In[52]:


df = df.join(pd.get_dummies(df.default, prefix='DF'))


# In[53]:


df = df.join(pd.get_dummies(df.housing, prefix='HO'))


# In[54]:


df = df.join(pd.get_dummies(df.loan, prefix='LO'))


# In[55]:


df.head()


# Pour les variables Qualitatives avec notion d'ordre, nous les transformerons en suite croissante de 1 à n.

# In[56]:


df['Month_num'] = df['month']
df['Month_num'].replace({'jan': 1}, inplace=True)
df['Month_num'].replace({'feb': 2}, inplace=True)
df['Month_num'].replace({'mar': 3}, inplace=True)
df['Month_num'].replace({'apr': 4}, inplace=True)
df['Month_num'].replace({'may': 5}, inplace=True)
df['Month_num'].replace({'jun': 6}, inplace=True)
df['Month_num'].replace({'jul': 7}, inplace=True)
df['Month_num'].replace({'aug': 8}, inplace=True)
df['Month_num'].replace({'sep': 9}, inplace=True)
df['Month_num'].replace({'oct': 10}, inplace=True)
df['Month_num'].replace({'nov': 11}, inplace=True)
df['Month_num'].replace({'dec': 12}, inplace=True)
df['Month_num'].value_counts()


# In[57]:


df['Day_of_Week_num'] = df['day_of_week']
df['Day_of_Week_num'].replace({'mon': 1}, inplace=True)
df['Day_of_Week_num'].replace({'tue': 2}, inplace=True)
df['Day_of_Week_num'].replace({'wed': 3}, inplace=True)
df['Day_of_Week_num'].replace({'thu': 4}, inplace=True)
df['Day_of_Week_num'].replace({'fri': 5}, inplace=True)
df['Day_of_Week_num'].value_counts()


# Pour vérification, je relance la matrice de corrélation.

# In[58]:


cor = df.corr()

fig, ax = plt.subplots(figsize=(50,50))
sns.heatmap(cor, annot=True, ax=ax, cmap='coolwarm')
plt.title('Matrice de Corrélation des variables quantitatives');


# ### B. Split et Scaling
# Deux Datasets seront constitués 'Target' et 'Data'.
# Ces Dataset seront ensuite répartis entre base d'apprentissage et base test. On retient une répartation 80/20 dans un premier temps.
# Ces variables seront ensuite centrées réduites.

# In[59]:


Target = df['Y_num']

Data = df.drop (['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y', 'Y_num',
                ], axis = 1)

Data.head()


# In[60]:


X_train, X_test, y_train, y_test=train_test_split(Data, Target, test_size=0.2, random_state=974)


# In[61]:


scaler=preprocessing.StandardScaler().fit(X_train)
X_train_scaled=scaler.transform(X_train)

scaler=preprocessing.StandardScaler().fit(X_test)
X_test_scaled=scaler.transform(X_test)
 


# ### C. Equilibrage du jeu de données
# Comme vu précédemment notre base de données est déséquilibrée. Nous utiliserons la technique d'oversampling SMOTE pour rééquilibrer les données. Cette technique fonctionne en augmentant le nombre d'observations de la classe minoritaire afin d'arriver à un ratio classe minoritaire/classe majoritaire satisfaisant.

# In[62]:


rOs = RandomOverSampler()
X_ro, y_ro = rOs.fit_resample(X_train_scaled, y_train)
print('Classes échantillon oversampled :', dict(pd.Series(y_ro).value_counts()))


# In[63]:


## undersampling aléatoire (RandomUnderSampler)
rUs = RandomUnderSampler()
X_ru, y_ru = rUs.fit_resample(X_train_scaled, y_train)
print('Classes échantillon undersampled :', dict(pd.Series(y_ru).value_counts()))


# ## 5) Modélisations
# 
# ---
# 

# ### A. Régression Logistique

# In[64]:


# Création du classifieur et construction du modèle sur les données d'entraînement
clf = linear_model.LogisticRegression(C=1.0)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
print(cm)

print(classification_report(y_test, y_pred))
print(clf.score(X_test_scaled, y_test))


# Avec oversampling

# In[65]:


# Création du classifieur et construction du modèle sur les données d'entraînement
clf = linear_model.LogisticRegression(C=1.0)
clf.fit(X_ro, y_ro)

y_pred = clf.predict(X_test_scaled)

cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
print(cm)

print(classification_report_imbalanced(y_test, y_pred))
#print(classification_report(y_test, y_pred))
print(clf.score(X_test_scaled, y_test))


# Avec undersampling

# In[66]:


# Création du classifieur et construction du modèle sur les données d'entraînement
clf = linear_model.LogisticRegression(C=1.0)
clf.fit(X_ru, y_ru)

y_pred = clf.predict(X_test_scaled)

cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
print(cm)

print(classification_report_imbalanced(y_test, y_pred))
#print(classification_report(y_test, y_pred))
print(clf.score(X_test_scaled, y_test))


# ### B. KNN

# In[67]:


### modéle : K-plus proches voisins avec n_neighbors : 2 à 40

# Sélection des hyperparamètres sur l'échantillon d’apprentissage par validation croisée
knn = neighbors.KNeighborsClassifier()
parametres = {'n_neighbors': range(2,41)}
grid_knn = GridSearchCV(estimator=knn, param_grid=parametres)
#swapper les 2 lignes suivantes en cas d'oversampling
#grid = grid_knn.fit(X_ro, y_ro)
grid = grid_knn.fit(X_train_scaled, y_train)
print(pd.DataFrame.from_dict(grid.cv_results_).loc[:,['params', 'mean_test_score']])
# les hyperparamètres retenus
print(grid_knn.best_params_)
# modèle appliqué à l'ensemble de test, et matrice de confusion
y_pred = grid_knn.predict(X_test_scaled)
cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
print(cm)
# score du modèle sur ce dernier
#print(classification_report_imbalanced(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(grid.score(X_test_scaled, y_test))


# Avec oversampling

# In[68]:


### modéle : K-plus proches voisins avec n_neighbors : 2 à 40

# Sélection des hyperparamètres sur l'échantillon d’apprentissage par validation croisée
knn = neighbors.KNeighborsClassifier()
parametres = {'n_neighbors': range(2,41)}
grid_knn = GridSearchCV(estimator=knn, param_grid=parametres)
#swapper les 2 lignes suivantes en cas d'oversampling
grid = grid_knn.fit(X_ro, y_ro)
#grid = grid_knn.fit(X_train_scaled, y_train)
print(pd.DataFrame.from_dict(grid.cv_results_).loc[:,['params', 'mean_test_score']])
# les hyperparamètres retenus
print(grid_knn.best_params_)
# modèle appliqué à l'ensemble de test, et matrice de confusion
y_pred = grid_knn.predict(X_test_scaled)
cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
print(cm)
# score du modèle sur ce dernier
print(classification_report_imbalanced(y_test, y_pred))
#print(classification_report(y_test, y_pred))
print(grid.score(X_test_scaled, y_test))


# Avec undersampling

# In[69]:


### modéle : K-plus proches voisins avec n_neighbors : 2 à 40

# Sélection des hyperparamètres sur l'échantillon d’apprentissage par validation croisée
knn = neighbors.KNeighborsClassifier()
parametres = {'n_neighbors': range(2,41)}
grid_knn = GridSearchCV(estimator=knn, param_grid=parametres)
#swapper les 2 lignes suivantes en cas d'undersampling
grid = grid_knn.fit(X_ru, y_ru)
#grid = grid_knn.fit(X_train_scaled, y_train)
print(pd.DataFrame.from_dict(grid.cv_results_).loc[:,['params', 'mean_test_score']])
# les hyperparamètres retenus
print(grid_knn.best_params_)
# modèle appliqué à l'ensemble de test, et matrice de confusion
y_pred = grid_knn.predict(X_test_scaled)
cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
print(cm)
# score du modèle sur ce dernier
print(classification_report_imbalanced(y_test, y_pred))
#print(classification_report(y_test, y_pred))
print(grid.score(X_test_scaled, y_test))


# ### C. SVM

# In[70]:


### modele : SVM avec kernel : 'linear', 'sigmoid', 'rbf' et C : 0.1, 1, 10, 20
# Sélection des hyperparamètres sur l'échantillon d’apprentissage par validation croisée
clf = svm.SVC()
parametres = {'C':[0.1,1,10,20], 'kernel':['linear', 'sigmoid', 'rbf']}

grid_clf = GridSearchCV(estimator=clf, param_grid=parametres)

#swapper les 2 lignes suivantes en cas d'oversampling
#grid = grid_clf.fit(X_ro, y_ro)
grid = grid_clf.fit(X_train_scaled, y_train)

print(pd.DataFrame.from_dict(grid.cv_results_).loc[:,['params', 'mean_test_score']])

# les hyperparamètres retenus
print(grid_clf.best_params_)

# modèle appliqué à l'ensemble de test, et  la matrice de confusion
y_pred = grid_clf.predict(X_test_scaled)
cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
print(cm)

# score du modèle sur ce dernier
#print(classification_report_imbalanced(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(grid.score(X_test_scaled, y_test))


# Avec oversampling

# In[71]:


### modele : SVM avec kernel : 'linear', 'sigmoid', 'rbf' et C : 0.1, 1, 10, 20
# Sélection des hyperparamètres sur l'échantillon d’apprentissage par validation croisée
clf = svm.SVC()
parametres = {'C':[0.1,1,10,20], 'kernel':['linear', 'sigmoid', 'rbf']}

grid_clf = GridSearchCV(estimator=clf, param_grid=parametres)

#swapper les 2 lignes suivantes en cas d'oversampling
grid = grid_clf.fit(X_ro, y_ro)
#grid = grid_clf.fit(X_train_scaled, y_train)

print(pd.DataFrame.from_dict(grid.cv_results_).loc[:,['params', 'mean_test_score']])

# les hyperparamètres retenus
print(grid_clf.best_params_)

# modèle appliqué à l'ensemble de test, et  la matrice de confusion
y_pred = grid_clf.predict(X_test_scaled)
cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
print(cm)

# score du modèle sur ce dernier
print(classification_report_imbalanced(y_test, y_pred))
#print(classification_report(y_test, y_pred))

print(grid.score(X_test_scaled, y_test))


# Avec undersampling

# In[72]:


### modele : SVM avec kernel : 'linear', 'sigmoid', 'rbf' et C : 0.1, 1, 10, 20
# Sélection des hyperparamètres sur l'échantillon d’apprentissage par validation croisée
clf = svm.SVC()
parametres = {'C':[0.1,1,10,20], 'kernel':['linear', 'sigmoid', 'rbf']}

grid_clf = GridSearchCV(estimator=clf, param_grid=parametres)

#swapper les 2 lignes suivantes en cas d'undersampling
grid = grid_clf.fit(X_ru, y_ru)
#grid = grid_clf.fit(X_train_scaled, y_train)

print(pd.DataFrame.from_dict(grid.cv_results_).loc[:,['params', 'mean_test_score']])

# les hyperparamètres retenus
print(grid_clf.best_params_)

# modèle appliqué à l'ensemble de test, et  la matrice de confusion
y_pred = grid_clf.predict(X_test_scaled)
cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
print(cm)

# score du modèle sur ce dernier
print(classification_report_imbalanced(y_test, y_pred))
#print(classification_report(y_test, y_pred))

print(grid.score(X_test_scaled, y_test))


# ### D. XGBoost

# Avec undersampling

# In[73]:


xgb=XGBClassifier()
#X,y  = X_train_scaled, y_train
X,y  = X_ru, y_ru

#swapper les 2 lignes suivantes en cas d'oversampling

#xgb.fit(X_ro, y_ro)

#xgb.fit(X_train_scaled, y_train)

xgb.fit(X, y)

# modèle appliqué à l'ensemble de test, et  la matrice de confusion

y_pred=xgb.predict(X_test_scaled)

cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])

print(cm)

#score du modèle sur ce dernier

print(classification_report_imbalanced(y_test, y_pred))

print(classification_report(y_test, y_pred))


# Avec oversampling

# In[74]:


xgb=XGBClassifier()

#X,y  = X_train_scaled, y_train

X,y  = X_ro, y_ro

#swapper les 2 lignes suivantes en cas d'oversampling

#xgb.fit(X_ro, y_ro)

#xgb.fit(X_train_scaled, y_train)

xgb.fit(X, y)

# modèle appliqué à l'ensemble de test, et  la matrice de confusion

y_pred=xgb.predict(X_test_scaled)

cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])

print(cm)

#score du modèle sur ce dernier

print(classification_report_imbalanced(y_test, y_pred))

print(classification_report(y_test, y_pred))


# ### E.Random Forest

# Avec undersampling

# In[75]:


# Création du classificateur et construction du modèle sur les données d'entraînement
clf = ensemble.RandomForestClassifier(n_jobs=-1, random_state=321)

#X,y  = X_train_scaled, y_train

X,y  = X_ru, y_ru

clf.fit(X, y)

y_pred = clf.predict(X_test_scaled)

cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])

print(cm)

print(classification_report_imbalanced(y_test, y_pred))

print(classification_report(y_test, y_pred))


# Avec oversampling

# In[76]:


# Création du classificateur et construction du modèle sur les données d'entraînement
clf = ensemble.RandomForestClassifier(n_jobs=-1, random_state=321)

#X,y  = X_train_scaled, y_train

X,y  = X_ro, y_ro

clf.fit(X, y)

y_pred = clf.predict(X_test_scaled)

cm = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])

print(cm)

print(classification_report_imbalanced(y_test, y_pred))

print(classification_report(y_test, y_pred))


# ### F. AdaBoost

# Avec undersampling

# In[77]:


#X,y  = X_train_scaled, y_train

X,y  = X_ru, y_ru

dtc = DecisionTreeClassifier(max_depth=5)
dtc.fit(X, y)

ac = AdaBoostClassifier(base_estimator=dtc, n_estimators=400)
ac.fit(X, y)

y_pred = ac.predict(X_test_scaled)

cm = pd.crosstab(y_test, y_pred)

print(cm)

print(classification_report_imbalanced(y_test, y_pred))

print(classification_report(y_test, y_pred))


# Avec oversampling

# In[78]:


#X,y  = X_train_scaled, y_train

X,y  = X_ro, y_ro

dtc = DecisionTreeClassifier(max_depth=5)
dtc.fit(X, y)

ac = AdaBoostClassifier(base_estimator=dtc, n_estimators=400)
ac.fit(X, y)

y_pred = ac.predict(X_test_scaled)

cm = pd.crosstab(y_test, y_pred)

print(cm)

print(classification_report_imbalanced(y_test, y_pred))

print(classification_report(y_test, y_pred))


# ### G. Voting

# Avec undersampling

# In[79]:


#X,y  = X_train_scaled, y_train

X,y  = X_ru, y_ru

clf1 = neighbors.KNeighborsClassifier(n_neighbors=3)
clf2 = ensemble.RandomForestClassifier(random_state=123)
clf3 = linear_model.LogisticRegression(max_iter=1000)

vclf = ensemble.VotingClassifier(estimators=[('knn', clf1), ('rf', clf2), ('lr', clf3)], voting='hard')

cv3 = KFold(n_splits=3, random_state=111, shuffle=True)


for clf, label in zip([clf1, clf2, clf3, vclf], ['KNN', 'Random Forest', 'Logistic Regression', 'Voting Classifier']):
    scores = cross_validate(clf, X, y, cv=cv3, scoring=['accuracy','f1'])
    print("[%s]: \n Accuracy: %0.2f (+/- %0.2f)" % (label, scores['test_accuracy'].mean(), scores['test_accuracy'].std()),
          "F1 score: %0.2f (+/- %0.2f)" % (scores['test_f1'].mean(), scores['test_f1'].std()))


# Avec oversampling

# In[80]:


#X,y  = X_train_scaled, y_train

X,y  = X_ro, y_ro

clf1 = neighbors.KNeighborsClassifier(n_neighbors=3)
clf2 = ensemble.RandomForestClassifier(random_state=123)
clf3 = linear_model.LogisticRegression(max_iter=1000)

vclf = ensemble.VotingClassifier(estimators=[('knn', clf1), ('rf', clf2), ('lr', clf3)], voting='hard')

cv3 = KFold(n_splits=3, random_state=111, shuffle=True)


for clf, label in zip([clf1, clf2, clf3, vclf], ['KNN', 'Random Forest', 'Logistic Regression', 'Voting Classifier']):
    scores = cross_validate(clf, X, y, cv=cv3, scoring=['accuracy','f1'])
    print("[%s]: \n Accuracy: %0.2f (+/- %0.2f)" % (label, scores['test_accuracy'].mean(), scores['test_accuracy'].std()),
          "F1 score: %0.2f (+/- %0.2f)" % (scores['test_f1'].mean(), scores['test_f1'].std()))


# In[ ]:




