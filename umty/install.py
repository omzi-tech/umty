import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the dataset
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# Step 2: Merge both datasets
data = pd.merge(ratings, movies, on='movieId')

# Step 3: Create a pivot table (users × movies)
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')

# Step 4: Fill missing values with 0
user_movie_matrix = user_movie_matrix.fillna(0)

# Step 5: Compute similarity between users
similarity = cosine_similarity(user_movie_matrix)
print("✅ User similarity matrix calculated successfully.")

# Step 6: Recommend movies for a target user
def get_recommendations(user_id, num_recommendations=5):
    user_index = user_id - 1
    sim_scores = similarity[user_index]

    # Create a Series of similarity values indexed by userId
    sim_scores_series = pd.Series(sim_scores, index=user_movie_matrix.index)

    # Exclude the target user
    sim_scores_series = sim_scores_series.drop(user_id, errors='ignore')

    # Get top 5 similar users
    top_users = sim_scores_series.sort_values(ascending=False).head(5).index

    # Get ratings from these users
    similar_users_ratings = user_movie_matrix.loc[top_users]

    # Compute a weighted average of ratings
    weighted_ratings = similar_users_ratings.T.dot(sim_scores_series[top_users])
    normalization = sim_scores_series[top_users].sum()

    # Avoid division by zero
    if normalization == 0:
        normalization = 1

    weighted_avg = weighted_ratings / normalization

    # Get movies the target user has not rated yet
    user_ratings = user_movie_matrix.loc[user_id]
    unwatched = user_ratings[user_ratings == 0].index

    recommendations = weighted_avg[unwatched].sort_values(ascending=False).head(num_recommendations)
    return recommendations


# Step 9: Movie-to-Movie Recommendations
def get_similar_movies(movie_title, num_recommendations=5):
    # Transpose the matrix (movies x users)
    movie_user_matrix = user_movie_matrix.T

    # Compute similarity between movies
    movie_similarity = cosine_similarity(movie_user_matrix)

    # Create a DataFrame for easy lookup
    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=movie_user_matrix.index,
        columns=movie_user_matrix.index
    )

    if movie_title not in movie_similarity_df.columns:
        return f"❌ Movie '{movie_title}' not found in dataset."

    # Sort by similarity
    similar_movies = movie_similarity_df[movie_title].sort_values(ascending=False)

    # Exclude the movie itself and show top N
    similar_movies = similar_movies.iloc[1:num_recommendations+1]

    return similar_movies


# Example: Recommend movies for user 1
print("\n🎬 Recommended Movies for User 1:")
print(get_recommendations(1))

# Example: Recommend similar movies
print("\n🎥 Because you liked 'Toy Story (1995)', you might also like:")
print(get_similar_movies("Toy Story (1995)"))
