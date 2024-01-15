from flask import Flask, render_template, request
import surprise
from surprise import Reader, Dataset, SVD, dump
from surprise.model_selection import train_test_split
import pandas as pd

app = Flask(__name__, template_folder='templates')
reader = Reader(rating_scale=(1, 5))
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
merged_data_sorted = pd.read_parquet('smalldata.parquet')
data = Dataset.load_from_df(merged_data_sorted[['userId', 'movieId', 'rating']], reader)
loaded_data = dump.load('algorithm.pkl')
algo, trainset = loaded_data[0], loaded_data[1]
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
