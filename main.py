import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

# Título da aplicação
st.title("Mineração de Regras de Associação - MovieLens")

# Upload de arquivos
st.sidebar.header("Carregue seus dados")
uploaded_movies = st.sidebar.file_uploader("Carregar arquivo de filmes (movies.csv)", type=["csv"])
uploaded_ratings = st.sidebar.file_uploader("Carregar arquivo de avaliações", type=["csv"])

if uploaded_movies and uploaded_ratings:
# Carregar o dataset de avaliações
    ratings = pd.read_csv('ml-latest-small/ratings.csv')

    # Filtrar para incluir apenas avaliações com nota >= 4 (considerando "gostaram")
    ratings = ratings[ratings['rating'] >= 4.0]

    # Criar um dataframe onde cada linha representa os filmes avaliados por um usuário
    user_movie_matrix = ratings.groupby('userId')['movieId'].apply(list).reset_index()
    user_movie_matrix.columns = ['userId', 'movies']

    # Exibir as primeiras transações
    #print(user_movie_matrix.head())

    # Converter lista de transações para formato adequado
    te = TransactionEncoder()
    te_matrix = te.fit(user_movie_matrix['movies']).transform(user_movie_matrix['movies'])
    df_te = pd.DataFrame(te_matrix, columns=te.columns_)

    # Aplicar Apriori para encontrar itemsets frequentes
    frequent_itemsets = apriori(df_te, min_support=0.02, use_colnames=True, max_len=2)

    # Gerar regras de associação
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=200)
    rules_sorted = rules.sort_values(by=['support', 'conviction', 'lift'], ascending=[False, False, False])

    # Exibir regras
    #print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    st.write("**Regras de associação geradas:**")
    st.write(rules_sorted)

