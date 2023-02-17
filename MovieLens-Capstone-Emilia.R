###################################################################################
# INTRODUCTION
###################################################################################

# The purpose of this code is to build a recommendation system based on the MovieLens dataset available on the <http://grouplens.org> website. 
# We will use the 10M version of the MovieLens dataset.  
# This dataset consists of the ratings given by different users to various movies. 
# We will see that each row represents the rating given by one user to one movie with details about the genre of that movie and the timestamp of the moment of rating. 
# Our objective will be to develop a prediction algorithm capable of predicting the rating that a certain user will give to a particular movie.  

###################################################################################
# ANALYSIS
###################################################################################

########
#STEP 1 - COLLECTING & PREPARING THE DATA
########

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library (knitr)
library(recommenderlab)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

# downloading the file
dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# creating the ratings & movies files
ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")


# Creating the edx dataset and the final hold-out test set which will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in the final hold-out test set are also in the edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Save the resulting files for future use - movielens, edx and final_holdout_test so we can use them for our analysis file
save(movielens, file = "movielens.rda")
save(edx, file = "edx.rda")
save(final_holdout_test, file = "final_holdout_test.rda")

# We will now load and use the edx dataset for our next steps. 
# To load the edx and the final holdout test files in RStudio after closing and reopening the R session:
load("edx.rda")
load("final_holdout_test.rda")


########
#STEP 2 - EXPLORING THE TRAIN DATASET (edx)
########

# Let's first do some exploratory data analysis to better understand the challenge that we have ahead
# Let's start by seeing the dimension and structure of our edx dataset:
dim(edx)
str(edx)

# Let's see the number of unique users and the number of unique movies that are rated in the edx data set
edx %>% summarize (unique_users = n_distinct(userId), unique_movies = n_distinct(movieId)) %>% kable()

# Let's now see the actual number of movies rated for each of the ratings in the edx data set
# either as a table:
table(edx$rating)

# or as a histogram:
hist(edx$rating)

# If we plot the distribution of movies that received a rating we'll notice that some movies are rated more often than others
edx %>% count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies with Ratings")

# If we plot the distribution of users that have rated movies we'll notice, again, that some users give ratings to more movies than other users
edx %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users with Ratings")


# Let's see the top 5 movies by number of ratings
top_movies <- edx %>% group_by(title) %>%  summarise(n_ratings = n()) %>% top_n(5)
top_movies

# Let's now show a sample of users rating these top 5 movies
library(knitr)
edx %>%
  filter(userId %in% c(12:20)) %>% 
  filter(title %in% top_movies$title) %>% 
  select(userId, title, rating) %>% 
  spread(title, rating) %>% kable()


# we notice that not all users rate all these top movies

# In fact, to see how sparse the ratings are among users let's build a matrix with a random sample of 100 movies and 100 users with yellow indicating the rating given by a user to that movie
users <- sample(unique(edx$userId), 100)
sample_matrix<- edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users", main = "Movies rated by a sample of 100 users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

# The challenge with this recommendation system is that each outcome has a different set of predictors.
# For example, to predict the rating for movie i by user u we will need to consider how user u is rating other movies
# similar to movie i, how movie i is rated by other users similar to user u and how often movie i is rated overall - we have already seen that some movies get rated a lot while others are barely rated.
# Therefore, in predicting the rating of the movie i by user u we will need to consider the entire matrix, i.e. all movies and users. 


###########################
# STEP 3 - PREPARING THE TRAIN DATASET (edx) 
###########################

# In order to train and test different models we'll use the edx dataset and split it further into train and test datasets
# The edx test set will be 10% of the edx data. This test dataset will be used to test the performance of the different models that we'll build
# After we chose the best model, we will test it on the movielens final holdout test dataset

set.seed(1, sample.kind="Rounding") 

test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-test_index,]
temp_test <- edx[test_index,]

# Make sure userId and movieId in the edx test set are also in edx train set
edx_test <- temp_test %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Add rows removed from edx hold-out test set back into edx train set
removed <- anti_join(temp_test, edx_test)
edx_train <- rbind(edx_train, removed)

rm(removed, test_index, temp_test)

str(edx_train)

#########################  NOTE  ###########################
# In order for the pdf to render from the rmd file we need to create a more manageable sample of the edx dataset.
# Below is the code to create a sample dataset of the edx file which will include only the movies that have been rated at least 100 times
# and the users that have rated at least 50 movies

movieIds_to_keep <- edx %>% count(movieId) %>% filter(n>=100) %>% pull(movieId)
userIds_to_keep <- edx %>% count(userId) %>% filter (n>=50) %>% pull(userId)
edx_sample <- edx %>% filter(userId %in% userIds_to_keep, movieId %in% movieIds_to_keep)

# Split that sample into train and test data set.
set.seed(1, sample.kind="Rounding")
sampletest_index <- createDataPartition(y = edx_sample$rating, times=1, p = 0.2, list = FALSE)
edx_sample_train <- edx_sample[-sampletest_index, ]
temp_sample_test <- edx_sample[sampletest_index,]

# Make sure userId and movieId in the edx_sample_test set are also in edx_sample train set
edx_sample_test <- temp_sample_test %>% 
  semi_join(edx_sample_train, by = "movieId") %>%
  semi_join(edx_sample_train, by = "userId")

# Add rows removed from edx_sample_test back into edx_sample_train dataset
removed <- anti_join(temp_sample_test, edx_sample_test)
edx_sample_train <- rbind(edx_sample_train, removed)

rm(removed, sampletest_index, temp_sample_test)

########################  END OF NOTE ###################

# Let's now create a matrix from the edx_train dataset which will have user IDs as rows, movie IDs as columns and values as ratings for each user/movie combination
# Please note that in the rmd file we'll use the edx_sample_train dataset to create the same matrix since the entire edx_train dataset is too big for the pdf to render
y <- select(edx_train, movieId, userId, rating)  %>% 
  pivot_wider(names_from = movieId, values_from = rating) 
class(y)
rnames <- y$userId
y <- as.matrix(y[,-1])
rownames(y) <- rnames

dim(y)



# Let's see the first 5 rows and columns of this matrix which has a row for all the ratings given by each userId to each of the movies:
y[1:5, 1:5]


# To compare the different models we will use the Root Mean Squared Error (RMSE) as the loss function. 
# Let's build a function first that calculates this RMSE

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
  }

############
# STEP 4: BUILDING AND EVALUATING DIFFERENT MODELS
############


## FIRST MODEL: average rating for all movies, regardless of user or movie
##############

# In this very basic model we predict that the rating for any movie will be the average rating accross all movies
# In this case, the variation of ratings from movie to movie will be explained by a random error coming from a user/movie combination

# Let's calculate the average rating for all movies in the edx_train dataset:
avg <- mean(y, na.rm = TRUE)
avg

# If we predict all unknown ratings as the avg rating of all movies then the RMSE will be:
avg_rmse <- RMSE (edx_test$rating, avg)
avg_rmse

# We need to get close to 0.8649 though, so we definitely need to find better ways to improve our model
# Let's create a table with the RMSE obtained from each of our models and record the result of our first model:
rmse_models <- data.frame (model = "A simple average", RMSE = avg_rmse)
rmse_models


## SECOND MODEL: taking into account a movie effect 
###############

# We've already noticed that some movies get rated higher than others (e.g. blockbusters)
# In this model we predict that the rating of each movie is made of the average rating of all movies plus a movie effect:
# movie rating = average rating for all movies + movie effect
# This movie effect is the one explaining the differences in ratings between movies
# To estimate the movie effect we can use least squares by fitting a linear regression model on the edx_train dataset
# fit <- lm(rating ~ as.factor(movieId), data = edx_train)
# but because we have thousands of movies and thus thousands of movie effects this linear model will take time to run 

# Instead, by looking at the movie rating equation above, we can tell that the least squares estimate of the movie effect is actually the average of the difference between the rating of each movie and the average rating for all movies:
# movie effect estimate = avg (movie rating - average rating for all movies)
movie_effect <- colMeans (y - avg, na.rm = TRUE)

# Let's create a histogram to see the distribution of these movie effects (m_effect):
qplot(movie_effect, bins=10, color = I("gray"))

# We can see that the movie effect varies from -2.5 to 1.5 which means that if we add this movie effect to the average rating for all movies we obtain the rating for each movie
# So a movie effect of 1.5 means a maximum rating (1.5 + 3.5 = 5)

# Let's now build a data frame to show the movie effects for each movie 
fit_movie_effect <- data.frame (movieId = as.integer(colnames(y)), average = avg, movie_effect = movie_effect)

head(fit_movie_effect)

# and lets join this movie effect dataframe with the edx_holdout_test dataset so we can calculate our prediction and compare it easily with the actual rating in the holdout test dataset:
movie_rmse <- left_join(edx_test, fit_movie_effect, by = "movieId") %>%
  mutate(prediction = average + movie_effect) %>% 
  summarize(rmse = RMSE(rating,prediction))
movie_rmse

# Let's add this to our rmse_models data frame for an easy comparison:
rmse_models <- rbind(rmse_models, list("A movie effect model",movie_rmse ))
rmse_models


## THIRD MODEL: taking into account the movie plus a user effect 
##############

# We've already seen that the same movie gets rated differently by different users which means that the user preferences need to be taken into consideration as well.
# Let's improve our model by also adding the user effect to the movie effect.
# Let's first calculate the average rating accross all movies for each user in the edx_train dataset:
user_avg_rating <- rowMeans(y, na.rm = TRUE)

# Let's plot this user effect in a histogram:
qplot(user_avg_rating, bins = 20, color = I ("gray"))

# This user effect allows us to predict that a user who usually gives movies lower ratings is also expected to give a lower rating to a highly rated movie
# We can use least squares again to estimate the user effect by employing a linear model like this:
# fit <- lm(rating ~ as.factor(movieId) + as.factor(userId), data = edx_train) but this will crash R because of the size of the dataset
# Instead, we can estimate an approximation of the user effect by calculating the average of the difference between the movie rating, the average rating of all movies and  the movie effect, like this:
user_effect <- rowMeans(sweep(y-avg, 2, movie_effect), na.rm = TRUE)

# Let's now build a data frame to show the user effects for each user:
fit_user_effect <- data.frame(userId = as.integer(rownames(y)), user_effect = user_effect)

# and let's join this user effect dataframe with the edx_holdout_test dataset and the movie effect dataframe so we can calculate our prediction and compare it easily with the actual rating in the holdout test dataset:
user_rmse <- left_join(edx_test, fit_movie_effect, by = "movieId") %>% 
  left_join(fit_user_effect, by = "userId") %>%
  mutate(prediction = average + movie_effect + user_effect) %>% 
  summarize(rmse = RMSE(rating,prediction))
user_rmse

# Let's add the user_rmse to our rmse_models dataframe:
rmse_models <- rbind(rmse_models, list("A movie plus user effect model", user_rmse))
rmse_models


## FOURTH MODEL: Applying regularization to the movie effect model
###############

# We know already that some movies get more rated than others.
# For the movies that get lots of rating, when calculating the movie effect, averaging these ratings will give us a fairly good estimate of the movie effect.
# However, when a movie has only one or two ratings the movie effect will only reflect those few ratings which means that it's more likely to have larger estimates of the movie effect which will cause larger errors.
# Large errors in the movie effect will increase our RMSE. 
# Therefore, we need to give a weight or a penalty to each movie effect depending on how many ratings a movie has. 
# The more ratings a movie has the more weight we will give to the movie effect and the lower the penalty will be.
# Regularization allows us to penalize large estimates that are built based on small sample sizes.

# We will apply the penalty to the movie effect by dividing the movie effect to the sum between the number of samples and a penalty number.
# We'll choose the penalty number using cross validation:

penalties <- seq(from = 0, to= 10, by = 0.1)
count_ratings <- colSums(!is.na(y))
fit_movie_effect$count_ratings <- count_ratings
sum_movie_effect <- colSums(y-avg, na.rm = TRUE)

rmse_penalty <- sapply(penalties, function(penalty){
  movie_effect <- sum_movie_effect/(count_ratings + penalty)
  fit_movie_effect$movie_effect <- movie_effect
  left_join(edx_test, fit_movie_effect, by = "movieId") %>%
    mutate(prediction = average + movie_effect) %>% 
    summarize(rmse = RMSE(rating,prediction)) %>% 
    pull(rmse)
  })

# Let's plot these penalties to see which of these values minimizes the RMSE:
qplot(penalties, rmse_penalty, geom = "point")

# The penalty number that minimizes the RMSE is:
penalty <- penalties[which.min(rmse_penalty)]
penalty

# So we found that the penalty that will minimize the rmse for our movie effect is 1.6
# Let's apply this penalty to the movie effect model to see if this improves our rmse:
fit_movie_effect$movie_effect_reg <- sum_movie_effect/(count_ratings + penalty)
movie_rmse_reg <- left_join(edx_test, fit_movie_effect, by = "movieId") %>%
  mutate(prediction = average + movie_effect_reg) %>% 
  summarize(rmse = RMSE(rating,prediction))
movie_rmse_reg

# Let's add the rmse for movie effect with regularization to the rmse_models:
rmse_models <- rbind(rmse_models, list("A movie effect model with regularization", movie_rmse_reg))
rmse_models


# FIFTH MODEL: Applying regularization to the movie and user effect model
#############

# We just saw that applying regularization to the movie effect model only diminished very little the original movie effect model.
# Let's now use the regularization of the movie effect model to the movie and user effect model to see if our RMSE improves

# Let's estimate again the user effect, this time using the regularized movie effect:
user_rmse_reg <- left_join(edx_test, fit_movie_effect, by = "movieId") %>% 
  left_join(fit_user_effect, by = "userId") %>%
  mutate(prediction = average + movie_effect_reg + user_effect) %>% 
  summarize(rmse = RMSE(rating,prediction))
user_rmse_reg

# Let's add this improved rmse to our rmse_models dataframe for comparison:
rmse_models <- rbind(rmse_models, list("A movie + user effect model with regularization", user_rmse_reg))
rmse_models

# SIXTH MODEL: Adding a genres effect

# If we analyze the ratings by genre we can see that genre plays a role in ratings as well.
# If we plot the average ratings by genre for movies with more than 1000 ratings for example
# we notice that some genres get rated consistently lower, on average, compared to other genres.

edx %>% group_by(genres) %>%
  summarize(n = n(), avg_rating = mean(rating)) %>%
  filter(n >= 1000) %>% 
  mutate(genres = reorder(genres, avg_rating)) %>%
  ggplot(aes(x = genres, y = avg_rating )) + 
  geom_point() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Let's create a model in which we explain the differences in ratings between movies through a genre effect. 
# In this model we predict that the rating of each movie is comprised of the average rating of all movies plus a genre effect:
# movie rating = average rating for all movies + genre effect
genre_effect <- edx_train %>% 
  group_by(genres) %>%
  summarize(genre_effect = mean(rating)-avg) %>% 
  mutate(genres = genres)
head(genre_effect, n=5)

# Let's now add this genre effect to the user and movie with regularization model (i.e.fifth model that we just built) and test it against the edx_test dataset.  
genres_rmse <- left_join(edx_test, fit_movie_effect, by = "movieId") %>% 
  left_join(fit_user_effect, by = "userId") %>%
  left_join(genre_effect, by = "genres") %>% 
  mutate(prediction = avg + movie_effect_reg + user_effect + genre_effect) %>% 
  summarize(rmse = RMSE(rating,prediction))
genres_rmse

# Unfortunately, the genre effect didn't manage to further minimize the error on the test dataset.

# Let's add the genres_rmse to our rmse_modes data frame for an easy comparison:
rmse_models <- rbind(rmse_models, list("A movie + genre + user effect model with regularization", 
                                       genres_rmse))
rmse_models %>% kable()

# We can now see that our best model is the one that takes into account both movie and user effect with regularization
# as the RMSE for this model is 0.869

#################
# RESULTS  - TESTING THE RETAINED MODEL ON THE final_holdout_test DATASET
################

load("final_holdout_test.rda")

rmse_final <- left_join(final_holdout_test, fit_movie_effect, by = "movieId") %>% 
  left_join(fit_user_effect, by = "userId") %>% 
  mutate(prediction = avg + movie_effect_reg + user_effect) %>% 
  summarize(rmse = RMSE(rating,prediction))
rmse_final



##############
# CONCLUSION
##############



# In general, we know that movies tend to be rated higher closer to the launch date of the movie since the fans of a particular genre are usually the ones rating first that movie. 
# The longer the time passes the fewer the chances for the users to be influenced by the attention a movie gets in the media and online.
# Perhaps an improved model could also take into account the time factor and look at the time lapsed from the launch of a movie to the time of the rating.

# However, the best next step is actually to use the recommenderlab package which has been designed espcially for building recommendation models.
# This package contains several recommendation algorithms to choose from. 
# In our case, the most relevant, in my opinion, would be to use the collaborative filtering recommender systems, starting with the item-based collaborative filtering model first. This model contains a similarity function that calculates a similarity matrix between  movies based on their genre and the ratings received. 
# We could also look at applying the user-based collaborative filtering model in which the algorithm calculates the similarity between users.  




