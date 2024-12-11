import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

# Título da aplicação
st.title("Regras de Associação - Aplicado a Sugestões de Filmes")

# Carregar os datasets
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Filtrar para incluir apenas avaliações com nota >= 4 (considerando "gostaram")
ratings = ratings[ratings['rating'] >= 4.0]

# Criar um dataframe onde cada linha representa os filmes avaliados por um usuário
user_movie_matrix = ratings.groupby('userId')['movieId'].apply(list).reset_index()
user_movie_matrix.columns = ['userId', 'movies']

# Converter lista de transações para formato adequado
te = TransactionEncoder()
te_matrix = te.fit(user_movie_matrix['movies']).transform(user_movie_matrix['movies'])
df_te = pd.DataFrame(te_matrix, columns=te.columns_)

# Aplicar Apriori para encontrar itemsets frequentes
frequent_itemsets = apriori(df_te, min_support=0.02, use_colnames=True, max_len=2)

# Gerar regras de associação
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=200)

# Mapear os IDs de filmes para títulos
movie_map = dict(zip(movies['movieId'], movies['title']))

# Função para substituir IDs pelos títulos nos antecedentes e consequentes
def map_movie_titles(row):
    antecedents = [movie_map[int(movie)] for movie in row['antecedents']]
    consequents = [movie_map[int(movie)] for movie in row['consequents']]
    return pd.Series({
        'antecedents_names': ', '.join(antecedents),
        'consequents_names': ', '.join(consequents)
    })

# Aplicar a substituição de IDs pelos títulos
rules[['antecedents_names', 'consequents_names']] = rules.apply(map_movie_titles, axis=1)

# Campo para o usuário digitar o nome de um filme
movie_name = st.text_input("Digite o nome de um filme para buscar regras de associação:")

if movie_name:
    # Verificar se o filme está no dataset
    matching_movies = movies[movies['title'].str.contains(movie_name, case=False, na=False)]

    if not matching_movies.empty:
        # Obter o ID do filme correspondente
        movie_ids = matching_movies['movieId'].tolist()

        # Filtrar regras que possuem o filme como antecedente
        filtered_rules = rules[
            rules['antecedents'].apply(lambda x: any(int(movie) in movie_ids for movie in x))
        ]

        # Remover linhas que não sejam do filme no antecedente
        filtered_rules = filtered_rules[
            filtered_rules['antecedents_names'].str.contains(movie_name, case=False, na=False)
        ]

        # Exibir as regras relacionadas
        if not filtered_rules.empty:
            st.write(f"Regras de associação relacionadas ao filme '{movie_name}':")
            st.write(filtered_rules[['antecedents_names', 'consequents_names', 'support', 'confidence', 'lift', 'conviction']])
        else:
            st.write(f"Não foram encontradas regras de associação com o filme '{movie_name}' nos antecedentes.")
    else:
        st.write(f"Filme '{movie_name}' não encontrado no dataset.")
else:
    st.write("Digite o nome de um filme para começar a busca.")
