from flask import Flask, render_template, request
import surprise
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)

# Function to get movie cover image URL (you can keep this function)
# Load data and train SVD model (you can keep this code)
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

@app.route('/', methods=['GET', 'POST'])
def home():
    user_id = None
    recommendations = []

    if request.method == 'POST':
        user_id = request.form.get('user_input')
        user_ratings = merged_data_sorted[merged_data_sorted['userId'] == int(user_id)][['movieId', 'rating']]
        movies_rated_by_user = user_ratings['movieId'].tolist()
        all_movie_ids = data.build_full_trainset().all_items()
        movies_to_predict = list(set(all_movie_ids) - set(movies_rated_by_user))
        user_predictions = [algo.predict(int(user_id), movie_id) for movie_id in movies_to_predict]
        user_predictions_sorted = sorted(user_predictions, key=lambda x: x.est, reverse=True)
        top_n = 5
        for i in range(min(top_n, len(user_predictions_sorted))):
            movie = user_predictions_sorted[i]
            movie_name = movies[movies['movieId'] == movie.iid]['title'].values[0]
            recommendations.append(f"Movie Name: {movie_name}, Estimated Rating: {movie.est}")

    return render_template('index.html', user_id=user_id, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)