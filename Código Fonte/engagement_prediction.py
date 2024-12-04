#Imports das bibliotecas necessárias e do dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score, confusion_matrix


def converterUnidade(value): # Como boa parte dos valores das colunas são strings, preciso converter para numérico para uma melhor manipulação dos dados.
    if isinstance(value, str): # Se for uma string
        
        if 'b' in value:
            return float(value.replace('b', '')) * 1000000000
        
        elif 'm' in value:
            return float(value.replace('m', '')) * 1000000
        
        elif 'k' in value:
            return float(value.replace('k', '')) * 1000
        
        else:
            return float(value.replace('%', ''))
    
    return value # Se já for numeric

sns.set(style= 'whitegrid')

url = 'https://drive.google.com/uc?export=download&id=1PEVjCKqK2MlTxBno0d32qpK5AldMrJ_X'
influencers_df = pd.read_csv(url)

relevant_columns = ['influence_score', 'posts', 'followers', 'avg_likes', 'new_post_avg_like', 'total_likes', '60_day_eng_rate'] # Foi retirada a coluna "country", pois 31% das linhas do dataset possuem valores nulos, o que acarretaria num dataframe com pouco valor. Já "channel_info" e "rank" não foram consideradas variáveis relevantes para a análise.

influencers_df_clean = influencers_df[relevant_columns]
for column in influencers_df_clean:
    influencers_df_clean[column] = influencers_df_clean[column].apply(converterUnidade)
influencers_df_clean = influencers_df_clean.dropna()


# PLOTANDO AS RELAÇÕES DAS COLUNAS X's COM A COLUNA Y ('60_day_eng_rate')

plt.figure(figsize=(14, 8)) # As dimensões das figuras dos gráficos

for i, column in enumerate(influencers_df_clean.columns[:-1], 1): # Para construir o layout dos gráficos de dispersão
    plt.subplot(3, 2, i) #Define o grid dos gráficos
    sns.scatterplot(data=influencers_df_clean, x = column, y = '60_day_eng_rate')
    plt.title(f'Relação {column} x Taxa de Engajamento')
    plt.xlabel(column)
    plt.ylabel('60_day_eng_rate')
    

plt.tight_layout() # Para que os gráficos não se sobreponham
plt.show()


# DIVISÃO DOS DADOS E TREINAMENTO

xInfluencers = influencers_df_clean.drop('60_day_eng_rate', axis=1)
yInfluencers = influencers_df_clean['60_day_eng_rate']

xTrain, xTest, yTrain, yTest = train_test_split(xInfluencers, yInfluencers, test_size=0.2, random_state=42) # 80% para treinamento e 20% para teste

model = LinearRegression()
model.fit(xTrain, yTrain)

# ANALISANDO O DESEMPENHO DO MODELO

yPrevisto = model.predict(xTest)
MSE_Influencers = mean_squared_error(yTest, yPrevisto) # RMSE DA TAXA DE ERRO
r2_Influencers = r2_score(yTest, yPrevisto) # R² do modelo
#valCruzada_scores = cross_val_score(model, xInfluencers, yInfluencers, cv=5, scoring='neg_mean_squared_error')

print(f'RMSE: {MSE_Influencers ** 0.5:.2f}') # 0.69 (Quanto menor, melhor)
print(f'R²: {r2_Influencers:.2f}') # 0.92 (Quanto mais próximo de 1, melhor)
#print(f'Média dos scores de cross-validation: {-valCruzada_scores.mean() ** 0.5:.2f}')

# visualizar os resultados

plt.figure(figsize=(12,8))
plt.scatter(yTest, yPrevisto)
plt.plot([yTest.min(), yTest.max()], [yTest.min(), yTest.max()], 'k--', lw=4)
plt.xlabel('Real')
plt.ylabel('Previsão')
plt.title('Real vs Previsão')
plt.show()

