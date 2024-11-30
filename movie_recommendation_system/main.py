# Importando as dependências 
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Obtendo os dados do arquivo csv
movies_data = pd.read_csv('C:/Users/user/Documents/machine_learning_python/movie_recommendation_system/movies.csv')
# print(movies_data.head())

selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director'] 

# Convertendo os dados em vetores
vectorizer = TfidfVectorizer()
feature_vectores = vectorizer.fit_transform(combined_features)

# Obtendo as simildaridades usando cosine_similarity
similarity = cosine_similarity(feature_vectores)

# Obtendo o nome do filme pelo usuário
movie_name = input("Digite o nome do seu filme favorito: ")

# Criando uma lista com todos os títulos dos filmes presentes no dataset
all_tiles_list = movies_data['title'].tolist()

find_close_match_titles = difflib.get_close_matches(movie_name, all_tiles_list)

close_match = find_close_match_titles[0]
index_of_close_match = movies_data[movies_data.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_close_match]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)

# Retornando os nomes dos filmes com maior similaridade baseado no index
print('Filmes sugeridos para você:  \n')

i = 1

for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index==index]['title'].values[0]
    if (i < 11):
        print(i, '.', title_from_index)
        i += 1