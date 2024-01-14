import surprise
from surprise import Reader, Dataset
import pandas as pd
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split
import requests
def get_movie_cover(api_key, movie_title):
    # Search for the movie ID using TMDb API
    search_url = f'https://api.themoviedb.org/3/search/movie'
    search_params = {'api_key': api_key, 'query': movie_title}
    search_response = requests.get(search_url, params=search_params)
    search_results = search_response.json()

    # Check if the search was successful and get the movie ID
    if 'results' in search_results and search_results['results']:
        movie_id = search_results['results'][0]['id']

        # Get details for the specific movie using its ID
        details_url = f'https://api.themoviedb.org/3/movie/{movie_id}'
        details_params = {'api_key': api_key}
        details_response = requests.get(details_url, params=details_params)
        details_result = details_response.json()

        # Get the poster path (cover image) for the movie
        poster_path = details_result.get('poster_path')

        if poster_path:
            # Construct the full image URL
            base_url = 'https://image.tmdb.org/t/p/original'
            image_url = f'{base_url}{poster_path}'
            return image_url
        else:
            print('Poster not available for this movie.')
    else:
        print('Movie not found.')
reader = Reader(rating_scale=(1, 5))
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')
ratings_tags_merged = pd.merge(ratings, tags, on=['userId', 'movieId'], how='outer')
merged_data = pd.merge(ratings_tags_merged, movies, on='movieId')

merged_data_sorted = merged_data.sort_values(by='userId', ascending=True)
columns_to_drop = ['timestamp_x', 'timestamp_y', 'tag']

merged_data_sorted.drop(columns=columns_to_drop, inplace=True)

merged_data_sorted = merged_data_sorted.dropna(subset=['userId', 'movieId', 'rating'])
print("hello")
print(merged_data_sorted)
merged_data_sorted['userId'] = merged_data_sorted['userId'].astype(int)
merged_data_sorted['movieId'] = merged_data_sorted['movieId'].astype(int)
merged_data_sorted['rating'] = merged_data_sorted['rating'].astype(int)
data = Dataset.load_from_df(merged_data_sorted[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25)
algo = SVD(n_factors=100, reg_all=0.02, lr_all=0.005, biased=False)
algo.fit(trainset)
user_id = 5
user_ratings = merged_data_sorted[merged_data_sorted['userId'] == user_id][['movieId', 'rating']]
print("User Ratings:")
print(user_ratings)
movies_rated_by_user = user_ratings['movieId'].tolist()
all_movie_ids = data.build_full_trainset().all_items()
movies_to_predict = list(set(all_movie_ids) - set(movies_rated_by_user))
print("Movies to Predict:")
print(movies_to_predict)
user_predictions = [algo.predict(user_id, movie_id) for movie_id in movies_to_predict]
estimated_ratings = [prediction.est for prediction in user_predictions]
average_est_rating = sum(estimated_ratings) / len(estimated_ratings)
print("Average Estimated Rating:", average_est_rating)
print("Predictions for Individual Movies:")
for prediction in user_predictions:
    print(f"Movie ID: {prediction.iid}, Estimated Rating: {prediction.est}")

user_predictions_sorted = sorted(user_predictions, key=lambda x: x.est, reverse=True)
top_n = 5
for i in range(min(top_n, len(user_predictions_sorted))):
    movie = user_predictions_sorted[i]
    movie_name = movies[movies['movieId'] == movie.iid]['title'].values[0]
api_key = 'YOUR_API_KEY'
movie_title = 'Inception'
cover_image_url = get_movie_cover(api_key, movie_title)

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    # You can customize this function to render your HTML file
    return render_template('Website.html')

if __name__ == '__main__':
    app.run(debug=True)