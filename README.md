# Capstone
This is a report on the steps taken to build a recommendation system based on the MovieLens dataset available on the [*grouplens*](http://grouplens.org) website.   

This dataset consists of the ratings given by different users to various movies. My objective is to develop a prediction algorithm capable of predicting the rating that a certain user will give to a particular movie.  

The method used to judge the accuracy of our prediction algorithm will be the residual mean squared error (RMSE) that will be applied on a test set - a dataset the we have not used when building our algorithm but for which we already know the actual movie ratings from different users. The RMSE will show us the typical error we make when predicting a movie rating. My objective is to reach a RMSE lower than 0.865, the lower the better.  

The approach to build this recommendation system takes into account several steps:  
- first we start with the average rating of all movies and consider this as the baseline.  
- further we take into account the fact that some movies are better than others and assume a movie effect which we add to the baseline.  
- next we observe that the same movie gets rated differently by different users so we assume a user effect that we will add to the movie effect and the baseline.  
- then we notice that some movies don't have many ratings which increases our chances of error in predicting the rating for those movies. Therefore, we will use regularization which assigns a weight/penalty number to minimize this error when the number of ratings for a movie is low. We use cross validation to identify the best candidate for this penalty number.  
- we also observe that certain genres get rated consistently lower than other genres which indicates that there might also be a genre effect that we could take into account.  
- in the end, we will combine all the effects above (i.e. baseline + movie effect + regularization + user effect + genre effect) to develop a final model and test the predictions of this model on the final holdout test dataset. 
