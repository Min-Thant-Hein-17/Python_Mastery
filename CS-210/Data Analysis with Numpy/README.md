# Data Analysis with Numpy 

# Exercise: Analyzing Movie Ratings Data

Suppose you are working for a streaming platform and have access to a dataset containing information about user ratings for different movies. Your task is to analyze the data to gain insights about movie popularity and viewer preferences.

## Dataset (Generate Random Data):
1. `movie_ids`: Array of unique movie IDs (e.g., 500 movies).
2. `ratings`: Array of movie ratings from 1 to 5 (whole numbers, 1 being lowest and 5 highest).
3. `views`: Array representing the number of times each movie was watched (values between 100 and 10,000).
4. `genres`: Array of genres for each movie (5 unique genres: Action, Comedy, Drama, Thriller, Sci-Fi).

Generate these arrays using NumPy's random functions.

## Tasks:

### 1. Data Preparation:
* Calculate the total rating score for each movie (`total_rating_score = ratings * views`).
* Classify each movie based on its average rating:
  * Low-rated: less than 2.5.
  * Medium-rated: between 2.5 and 4.
  * High-rated: more than 4.

### 2. Data Analysis:
* Find the average view count for each genre.
* Calculate the total view count for each rating category (low-rated, medium-rated, high-rated).
* Find the top 5 most-watched movies.
* Determine the percentage of movies in each rating category.

### 3. Advanced Analysis:
* Calculate the standard deviation of ratings in each genre.
* Find the most common genre among the top 10 most-watched movies.

## Expected Outcomes:
* Insights on which genres attract the most views.
* Distribution of movies based on rating categories.
* Identification of top-performing movies and common preferences among top-viewed movies.
