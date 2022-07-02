##################################
# Creating edx set, validation set
##################################

# Package installations
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")

# Load libraries
library(tidyverse)
library(caret)
library(data.table)
library(kableExtra)
library(lubridate)

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

################
# Data analysis
################

# Dimension of edx
dimedx <- dim(edx) 
dimedx

# Dimension of validation
dimval <- dim(validation) 
dimval
head(edx) #top 6 rows
n <- edx %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId)) #number of users and movies
n

# Number of ratings per rating
edx %>%
  group_by(rating) %>%
  summarize(n = n()) %>%
  ggplot(aes(rating, n)) +
  geom_bar(stat = "identity") +
  xlab("Rating") +
  ylab("Count") 

# Mean of all the ratings
avg <- round(mean(edx$rating), digits = 2) 
avg

# Distribution of average movie rating
edx %>% 
  group_by(movieId) %>%
  summarize(ratings = mean(rating)) %>%
  ggplot(aes(ratings)) +
  geom_histogram(bins = 30, color = "black") +
  xlab("Average rating per movie") +
  ylab("Count")

# Mean of average movie rating
avg_m <- edx %>% 
  group_by(movieId) %>%
  summarize(ratings = mean(rating)) %>%
  summarize(mean(ratings)) %>% round(digits = 2)
avg_m

# Distribution of average user rating
edx %>% 
  group_by(userId) %>% 
  summarize(ratings = mean(rating)) %>% 
  ggplot(aes(ratings)) + 
  geom_histogram(bins = 30, color = "black") +
  xlab("Average rating per user") +
  ylab("Count")

# # Mean of average user rating
avg_u <- edx %>% 
  group_by(userId) %>% 
  summarize(ratings = mean(rating)) %>% 
  summarize(mean(ratings)) %>% round(digits = 2)
avg_u

##############################
# Creating train and test sets
##############################

# 20 percent of data goes to test
set.seed(1994)

test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE) 
train_set <- edx[-test_index,]
test_set <- edx[test_index,]
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")


###############################
# Function for RMSE calculation
###############################

# Loss function to evaluate goodness of prediction
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


#############
# Naive model
#############

# Average of train set ratings is equivalent to predicted ratings
mu <- mean(train_set$rating) 
mu

# RMSE for this model
naive_rmse <- RMSE(test_set$rating, mu)

# Incorporation of this result in a styled table
rmse_results <- tibble(Method = "Just the average", RMSE = naive_rmse)
rmse_results %>% knitr::kable() %>%
  kable_styling(bootstrap_options = "bordered", full_width = FALSE, position = "center") %>%
  kable_styling(bootstrap_options = "striped" , full_width = FALSE, position = "center") %>%
  kable_styling(latex_options = "hold_position") %>%
  row_spec(0, bold = TRUE)


####################
# Movie effect model
####################

# Calculation of b_i and prediction of ratings
b_i <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
predicted_ratings <- test_set %>% 
  left_join(b_i, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  .$pred

# RMSE for this model
model_1_rmse <- RMSE(predicted_ratings, test_set$rating)

# Incorporation of this result in the results table
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Movie Effect Model",
                                 RMSE = model_1_rmse ))
rmse_results %>% knitr::kable() %>%
  kable_styling(bootstrap_options = "bordered", full_width = FALSE, position = "center") %>%
  kable_styling(bootstrap_options = "striped" , full_width = FALSE, position = "center") %>%
  kable_styling(latex_options = "hold_position") %>%
  row_spec(0, bold = TRUE)


############################
# Movie + user effects model 
############################

# Calculation of b_u and prediction of ratings
b_u <- train_set %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
predicted_ratings <- test_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

# RMSE for this model
model_2_rmse <- RMSE(predicted_ratings, test_set$rating)

# Incorporation of this result in the results table
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Movie + User Effects Model",  
                                 RMSE = model_2_rmse ))
rmse_results %>% knitr::kable() %>%
  kable_styling(bootstrap_options = "bordered", full_width = FALSE, position = "center") %>%
  kable_styling(bootstrap_options = "striped" , full_width = FALSE, position = "center") %>%
  kable_styling(latex_options = "hold_position") %>%
  row_spec(0, bold = TRUE)


#######################################
# Creating 5 folds for cross-validation
#######################################

# Clear environment
rm(train_set, test_set, test_index, b_i, b_u) 

# Fold creation with 20 percent of the data for test for every fold
set.seed(1994)

folds_index <- createFolds(y = edx$rating, k = 5)

fold1_train <-edx[-folds_index[[1]],]
fold1_test <- edx[folds_index[[1]],]
fold1_test <- fold1_test %>% 
  semi_join(fold1_train, by = "movieId") %>%
  semi_join(fold1_train, by = "userId")

fold2_train <-edx[-folds_index[[2]],]
fold2_test <- edx[folds_index[[2]],]
fold2_test <- fold2_test  %>% 
  semi_join(fold2_train, by = "movieId") %>%
  semi_join(fold2_train, by = "userId")

fold3_train <-edx[-folds_index[[3]],]
fold3_test <- edx[folds_index[[3]],]
fold3_test <- fold3_test  %>% 
  semi_join(fold3_train, by = "movieId") %>%
  semi_join(fold3_train, by = "userId")

fold4_train <-edx[-folds_index[[4]],]
fold4_test <- edx[folds_index[[4]],]
fold4_test <- fold4_test  %>% 
  semi_join(fold4_train, by = "movieId") %>%
  semi_join(fold4_train, by = "userId")

fold5_train <-edx[-folds_index[[5]],]
fold5_test <- edx[folds_index[[5]],]
fold5_test <- fold5_test  %>% 
  semi_join(fold5_train, by = "movieId") %>%
  semi_join(fold5_train, by = "userId")


##############################################
# Regularization on movie + user effects model
##############################################

lambdas <- seq(3, 6, 0.1)
rmses <- sapply(lambdas, function(l){
  #fold1
  mu <- mean(fold1_train$rating)
  b_i <- fold1_train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- fold1_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- fold1_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId")  %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  rmse1 <- RMSE(predicted_ratings, fold1_test$rating)
  #fold2
  mu <- mean(fold2_train$rating)
  b_i <- fold2_train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- fold2_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- fold2_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  rmse2 <- RMSE(predicted_ratings, fold2_test$rating)
  #fold3
  mu <- mean(fold3_train$rating)
  b_i <- fold3_train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- fold3_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- fold3_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  rmse3 <- RMSE(predicted_ratings, fold3_test$rating)
  #fold4
  mu <- mean(fold4_train$rating)
  b_i <- fold4_train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- fold4_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- fold4_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  rmse4 <- RMSE(predicted_ratings, fold4_test$rating)
  #fold5
  mu <- mean(fold5_train$rating)
  b_i <- fold5_train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- fold5_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- fold5_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  rmse5 <- RMSE(predicted_ratings, fold5_test$rating)
  #combination
  return((rmse1+rmse2+rmse3+rmse4+rmse5)/5)
})

# Plot of RMSE versus lambda
qplot(lambdas, rmses, xlab = "Lambda", ylab = "RMSE", geom = c("point", "line"))

# Value of the optimal lambda
bestlambda <- lambdas[which.min(rmses)]
bestlambda

# RMSE for this model
model_3_rmse <- min(rmses)

# Incorporation of this result in the results table
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Regularized Movie + User Effects Model (CV)",  
                                 RMSE = model_3_rmse))
rmse_results %>% knitr::kable() %>%
  kable_styling(bootstrap_options = "bordered", full_width = FALSE, position = "center") %>%
  kable_styling(bootstrap_options = "striped" , full_width = FALSE, position = "center") %>%
  kable_styling(latex_options = "hold_position") %>%
  row_spec(0, bold = TRUE)


##################################
# Addition of a rating date effect
##################################

# Transformation of timestamp into a year-month-day format and rounding by week
edx <- edx %>%
  mutate(date = ymd(round_date(as_datetime(timestamp), unit = "week"))) %>%
  select(-timestamp) # remove timestamp column

# Visualization of rating versus rating date
edx %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  xlab("Date (rounded by week)") +
  ylab("Mean rating per week") +
  geom_hline(yintercept = mean(edx$rating))

# Visualization of rating count versus rating date
edx %>%
  group_by(date) %>%
  summarize(n = n()) %>%
  ggplot(aes(date, n)) +
  geom_bar(stat = "identity") +
  xlab("Date (rounded by week)") +
  ylab("Count")

# Creating 5 folds including date
set.seed(1994)

folds_index <- createFolds(y = edx$rating, k = 5)

fold1_train <-edx[-folds_index[[1]],]
fold1_test <- edx[folds_index[[1]],]
fold1_test <- fold1_test %>% 
  semi_join(fold1_train, by = "movieId") %>%
  semi_join(fold1_train, by = "userId") %>%
  semi_join(fold1_train, by = "date")

fold2_train <-edx[-folds_index[[2]],]
fold2_test <- edx[folds_index[[2]],]
fold2_test <- fold2_test  %>% 
  semi_join(fold2_train, by = "movieId") %>%
  semi_join(fold2_train, by = "userId") %>%
  semi_join(fold2_train, by = "date")

fold3_train <-edx[-folds_index[[3]],]
fold3_test <- edx[folds_index[[3]],]
fold3_test <- fold3_test  %>% 
  semi_join(fold3_train, by = "movieId") %>%
  semi_join(fold3_train, by = "userId") %>%
  semi_join(fold3_train, by = "date")

fold4_train <-edx[-folds_index[[4]],]
fold4_test <- edx[folds_index[[4]],]
fold4_test <- fold4_test  %>% 
  semi_join(fold4_train, by = "movieId") %>%
  semi_join(fold4_train, by = "userId") %>%
  semi_join(fold4_train, by = "date")

fold5_train <-edx[-folds_index[[5]],]
fold5_test <- edx[folds_index[[5]],]
fold5_test <- fold5_test  %>% 
  semi_join(fold5_train, by = "movieId") %>%
  semi_join(fold5_train, by = "userId") %>%
  semi_join(fold5_train, by = "date")

# Loess smoothing of rating by date; tuning of span and degree parameters
set.seed(1994)
l <- bestlambda # optimal lambda
spans <- c(seq(0.1,1,0.1), 2, 5, 10)
degs <- 1:2
params <- expand.grid(s = spans, d = degs) # grid of parameters

rmses <- apply(params,1,function(param){
  s <- param[1]
  d <- param[2]
  #fold1
  mu <- mean(fold1_train$rating)
  b_d <- fold1_train %>%
    group_by(date) %>%
    summarize(rating = mean(rating - mu)) %>%
    mutate(b_d = predict(loess(rating ~ as.numeric(date), data = ., span = s, degree = d))) %>%
    select(-rating)
  b_i <- fold1_train %>%
    left_join(b_d, by = "date") %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - b_d)/(n()+l))
  b_u <- fold1_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_i, by= "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_d - b_i)/(n()+l))
  predicted_ratings <- fold1_test %>% 
    left_join(b_d, by = "date") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId")  %>%
    mutate(pred = mu + b_d + b_i + b_u) %>%
    .$pred
  rmse1 <- RMSE(predicted_ratings, fold1_test$rating)
  #fold2
  mu <- mean(fold2_train$rating)
  b_d <- fold2_train %>%
    group_by(date) %>%
    summarize(rating = mean(rating - mu)) %>%
    mutate(b_d = predict(loess(rating ~ as.numeric(date), data = ., span = s, degree = d))) %>%
    select(-rating)
  b_i <- fold2_train %>%
    left_join(b_d, by = "date") %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - b_d)/(n()+l))
  b_u <- fold2_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_i, by= "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_d - b_i)/(n()+l))
  predicted_ratings <- fold2_test %>% 
    left_join(b_d, by = "date") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId")  %>%
    mutate(pred = mu + b_d + b_i + b_u) %>%
    .$pred
  rmse2 <- RMSE(predicted_ratings, fold2_test$rating)
  #fold3
  mu <- mean(fold3_train$rating)
  b_d <- fold3_train %>%
    group_by(date) %>%
    summarize(rating = mean(rating - mu)) %>%
    mutate(b_d = predict(loess(rating ~ as.numeric(date), data = ., span = s, degree = d))) %>%
    select(-rating)
  b_i <- fold3_train %>%
    left_join(b_d, by = "date") %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - b_d)/(n()+l))
  b_u <- fold3_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_i, by= "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_d - b_i)/(n()+l))
  predicted_ratings <- fold3_test %>% 
    left_join(b_d, by = "date") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId")  %>%
    mutate(pred = mu + b_d + b_i + b_u) %>%
    .$pred
  rmse3 <- RMSE(predicted_ratings, fold3_test$rating)
  #fold4
  mu <- mean(fold4_train$rating)
  b_d <- fold4_train %>%
    group_by(date) %>%
    summarize(rating = mean(rating - mu)) %>%
    mutate(b_d = predict(loess(rating ~ as.numeric(date), data = ., span = s, degree = d))) %>%
    select(-rating)
  b_i <- fold4_train %>%
    left_join(b_d, by = "date") %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - b_d)/(n()+l))
  b_u <- fold4_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_i, by= "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_d - b_i)/(n()+l))
  predicted_ratings <- fold4_test %>% 
    left_join(b_d, by = "date") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId")  %>%
    mutate(pred = mu + b_d + b_i + b_u) %>%
    .$pred
  rmse4 <- RMSE(predicted_ratings, fold4_test$rating)
  #fold5
  mu <- mean(fold5_train$rating)
  b_d <- fold5_train %>%
    group_by(date) %>%
    summarize(rating = mean(rating - mu)) %>%
    mutate(b_d = predict(loess(rating ~ as.numeric(date), data = ., span = s, degree = d))) %>%
    select(-rating)
  b_i <- fold5_train %>%
    left_join(b_d, by = "date") %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - b_d)/(n()+l))
  b_u <- fold5_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_i, by= "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_d - b_i)/(n()+l))
  predicted_ratings <- fold5_test %>% 
    left_join(b_d, by = "date") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId")  %>%
    mutate(pred = mu + b_d + b_i + b_u) %>%
    .$pred
  rmse5 <- RMSE(predicted_ratings, fold5_test$rating)
  #combination
  return((rmse1+rmse2+rmse3+rmse4+rmse5)/5)
})

# Plot of RMSE versus span with degree 1 and 2
qplot(params$s, rmses, col = as.factor(params$d), xlab = "Span",
      ylab = "RMSE", geom = c("point", "line"), log = "x") +
  labs(colour = "Degree")

# Optimal loess parameters
params_date <- params[which.min(rmses),]
params_date

# Visualization of the loess fit with tuned parameters
edx %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth(method = "loess", span = params_date[1], method.args = list(degree = params_date[2])) +
  xlab("Date (rounded by week)") +
  ylab("Mean rating per week") +
  geom_hline(yintercept = mean(edx$rating))

# RMSE for this model
model_4_rmse <- min(rmses)

# Incorporation of this result in the results table
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Regularized Movie + User + Date Effects Model (CV)",  
                                 RMSE = model_4_rmse))
rmse_results %>% knitr::kable() %>%
  kable_styling(bootstrap_options = "bordered", full_width = FALSE, position = "center") %>%
  kable_styling(bootstrap_options = "striped" , full_width = FALSE, position = "center") %>%
  kable_styling(latex_options = "hold_position") %>%
  row_spec(0, bold = TRUE)


###################################
# Addition of a release year effect
###################################

# Extraction of the movie release year
edx <- edx %>%
  mutate(year = str_extract(title, "\\s\\(\\d{4}\\)$"),
         year = as.numeric(str_extract(year, "\\d{4}"))) %>%
  select(-title) # remove title column

# Visualization of mean rating versus release year
edx %>%
  group_by(year) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(year, rating)) +
  geom_point() +
  xlab("Release year") +
  ylab("Mean rating") +
  geom_hline(yintercept = mean(edx$rating))

# Visualization of rating count versus release year
edx %>%
  group_by(year) %>%
  summarize(n = n()) %>%
  ggplot(aes(year, n)) +
  geom_bar(stat = "identity") +
  xlab("Release year") +
  ylab("Count")

# Creation of 5 folds including release year
set.seed(1994)
folds_index <- createFolds(y = edx$rating, k = 5)

fold1_train <-edx[-folds_index[[1]],]
fold1_test <- edx[folds_index[[1]],]
fold1_test <- fold1_test %>% 
  semi_join(fold1_train, by = "movieId") %>%
  semi_join(fold1_train, by = "userId") %>%
  semi_join(fold1_train, by = "date") %>%
  semi_join(fold1_train, by = "year")

fold2_train <-edx[-folds_index[[2]],]
fold2_test <- edx[folds_index[[2]],]
fold2_test <- fold2_test  %>% 
  semi_join(fold2_train, by = "movieId") %>%
  semi_join(fold2_train, by = "userId") %>%
  semi_join(fold2_train, by = "date") %>%
  semi_join(fold2_train, by = "year")

fold3_train <-edx[-folds_index[[3]],]
fold3_test <- edx[folds_index[[3]],]
fold3_test <- fold3_test  %>% 
  semi_join(fold3_train, by = "movieId") %>%
  semi_join(fold3_train, by = "userId") %>%
  semi_join(fold3_train, by = "date") %>%
  semi_join(fold3_train, by = "year")

fold4_train <-edx[-folds_index[[4]],]
fold4_test <- edx[folds_index[[4]],]
fold4_test <- fold4_test  %>% 
  semi_join(fold4_train, by = "movieId") %>%
  semi_join(fold4_train, by = "userId") %>%
  semi_join(fold4_train, by = "date") %>%
  semi_join(fold4_train, by = "year")

fold5_train <-edx[-folds_index[[5]],]
fold5_test <- edx[folds_index[[5]],]
fold5_test <- fold5_test  %>% 
  semi_join(fold5_train, by = "movieId") %>%
  semi_join(fold5_train, by = "userId") %>%
  semi_join(fold5_train, by = "date") %>%
  semi_join(fold5_train, by = "year")

# Loess smoothing of rating by year; tuning of span and degree parameters
set.seed(1994)
l <- bestlambda # optimal lambda
spans <- c(seq(0.1,0.9,0.1))
degs <- 1:2
params <- expand.grid(s = spans, d = degs)  # grid of parameters

rmses <- apply(params,1,function(param){
  s <- param[1]
  d <- param[2]
  #fold1
  mu <- mean(fold1_train$rating)
  b_d <- fold1_train %>%
    group_by(date) %>%
    summarize(rating = mean(rating - mu)) %>%
    mutate(b_d = predict(loess(rating ~ as.numeric(date), data = ., span = params_date[1], degree = params_date[2]))) %>%
    select(-rating)
  b_y <- fold1_train %>% 
    left_join(b_d, by = "date") %>% 
    group_by(year) %>%
    summarize(rating = mean(rating - mu - b_d)) %>%
    mutate(b_y = predict(loess(rating ~ year, data = ., span = s, degree = d))) %>%
    select(-rating)
  b_i <- fold1_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - b_d - b_y)/(n()+l))
  b_u <- fold1_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by= "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_d - b_y - b_i)/(n()+l))
  predicted_ratings <- fold1_test %>% 
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId")  %>%
    mutate(pred = mu + b_i + b_u + b_d + b_y) %>%
    .$pred
  rmse1 <- RMSE(predicted_ratings, fold1_test$rating)
  #fold2
  mu <- mean(fold2_train$rating)
  b_d <- fold2_train %>%
    group_by(date) %>%
    summarize(rating = mean(rating - mu)) %>%
    mutate(b_d = predict(loess(rating ~ as.numeric(date), data = ., span = params_date[1], degree = params_date[2]))) %>%
    select(-rating)
  b_y <- fold2_train %>% 
    left_join(b_d, by = "date") %>% 
    group_by(year) %>%
    summarize(rating = mean(rating - mu - b_d)) %>%
    mutate(b_y = predict(loess(rating ~ year, data = ., span = s, degree = d))) %>%
    select(-rating)
  b_i <- fold2_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - b_d - b_y)/(n()+l))
  b_u <- fold2_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by= "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_d - b_y - b_i)/(n()+l))
  predicted_ratings <- fold2_test %>% 
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId")  %>%
    mutate(pred = mu + b_i + b_u + b_d + b_y) %>%
    .$pred
  rmse2 <- RMSE(predicted_ratings, fold2_test$rating)
  #fold3
  mu <- mean(fold3_train$rating)
  b_d <- fold3_train %>%
    group_by(date) %>%
    summarize(rating = mean(rating - mu)) %>%
    mutate(b_d = predict(loess(rating ~ as.numeric(date), data = ., span = params_date[1], degree = params_date[2]))) %>%
    select(-rating)
  b_y <- fold3_train %>% 
    left_join(b_d, by = "date") %>% 
    group_by(year) %>%
    summarize(rating = mean(rating - mu - b_d)) %>%
    mutate(b_y = predict(loess(rating ~ year, data = ., span = s, degree = d))) %>%
    select(-rating)
  b_i <- fold3_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - b_d - b_y)/(n()+l))
  b_u <- fold3_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by= "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_d - b_y - b_i)/(n()+l))
  predicted_ratings <- fold3_test %>% 
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId")  %>%
    mutate(pred = mu + b_i + b_u + b_d + b_y) %>%
    .$pred
  rmse3 <- RMSE(predicted_ratings, fold3_test$rating)
  #fold4
  mu <- mean(fold4_train$rating)
  b_d <- fold4_train %>%
    group_by(date) %>%
    summarize(rating = mean(rating - mu)) %>%
    mutate(b_d = predict(loess(rating ~ as.numeric(date), data = ., span = params_date[1], degree = params_date[2]))) %>%
    select(-rating)
  b_y <- fold4_train %>% 
    left_join(b_d, by = "date") %>% 
    group_by(year) %>%
    summarize(rating = mean(rating - mu - b_d)) %>%
    mutate(b_y = predict(loess(rating ~ year, data = ., span = s, degree = d))) %>%
    select(-rating)
  b_i <- fold4_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - b_d - b_y)/(n()+l))
  b_u <- fold4_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by= "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_d - b_y - b_i)/(n()+l))
  predicted_ratings <- fold4_test %>% 
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId")  %>%
    mutate(pred = mu + b_i + b_u + b_d + b_y) %>%
    .$pred
  rmse4 <- RMSE(predicted_ratings, fold4_test$rating)
  #fold5
  mu <- mean(fold5_train$rating)
  b_d <- fold5_train %>%
    group_by(date) %>%
    summarize(rating = mean(rating - mu)) %>%
    mutate(b_d = predict(loess(rating ~ as.numeric(date), data = ., span = params_date[1], degree = params_date[2]))) %>%
    select(-rating)
  b_y <- fold5_train %>% 
    left_join(b_d, by = "date") %>% 
    group_by(year) %>%
    summarize(rating = mean(rating - mu - b_d)) %>%
    mutate(b_y = predict(loess(rating ~ year, data = ., span = s, degree = d))) %>%
    select(-rating)
  b_i <- fold5_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - b_d - b_y)/(n()+l))
  b_u <- fold5_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by= "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_d - b_y - b_i)/(n()+l))
  predicted_ratings <- fold5_test %>% 
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId")  %>%
    mutate(pred = mu + b_i + b_u + b_d + b_y) %>%
    .$pred
  rmse5 <- RMSE(predicted_ratings, fold5_test$rating)
  #combination
  return((rmse1+rmse2+rmse3+rmse4+rmse5)/5)
})

# Plot of RMSE versus span 
qplot(params$s, rmses, col = as.factor(params$d), xlab = "Span", ylab = "RMSE", geom = c("point", "line")) + labs(colour = "Degree")

# Optimal loess parameters
params_year <- params[which.min(rmses),]
params_year

# RMSE for this model
model_5_rmse <- min(rmses)

# Visualization of the loess fit with tuned parameters
edx %>%
  group_by(year) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(year, rating)) +
  geom_point() +
  geom_smooth(method = "loess", span = params_year[1], method.args = list(degree = params_year[2])) +
  xlab("Release year") +
  ylab("Mean rating") +
  geom_hline(yintercept = mean(edx$rating))

# Incorporation of this result in the results table
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Regularized Movie + User + Date + Year Effects Model (CV)",  
                                 RMSE = model_5_rmse))
rmse_results %>% knitr::kable() %>%
  kable_styling(bootstrap_options = "bordered", full_width = FALSE, position = "center") %>%
  kable_styling(bootstrap_options = "striped" , full_width = FALSE, position = "center") %>%
  kable_styling(latex_options = "hold_position") %>%
  row_spec(0, bold = TRUE)


######################################################################
# Final regularization of the movie + user + date + year effects model
######################################################################

set.seed(1994)
lambdas <- seq(4,6,0.1)
rmses <- sapply(lambdas, function(l){
  #fold1
  mu <- mean(fold1_train$rating)
  b_d <- fold1_train %>% 
    group_by(date) %>%
    summarize(rating = mean(rating - mu)) %>%
    mutate(b_d = predict(loess(rating ~ as.numeric(date), data = ., span = params_date[1], degree = params_date[2]))) %>%
    select(-rating)
  b_y <- fold1_train %>%
    left_join(b_d, by = "date") %>%
    group_by(year) %>%
    summarize(rating = mean(rating - mu - b_d)) %>%
    mutate(b_y = predict(loess(rating ~ year, data = ., span = params_year[1], degree = params_year[2]))) %>%
    select(-rating)
  b_i <- fold1_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - b_d - b_y)/(n()+l))
  b_u <- fold1_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by= "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_d - b_y - b_i)/(n()+l))
  predicted_ratings <- fold1_test %>% 
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId")  %>%
    mutate(pred = mu + b_d + b_y + b_i + b_u) %>%
    .$pred
  rmse1 <- RMSE(predicted_ratings, fold1_test$rating)
  #fold2
  mu <- mean(fold2_train$rating)
  b_d <- fold2_train %>% 
    group_by(date) %>%
    summarize(rating = mean(rating - mu)) %>%
    mutate(b_d = predict(loess(rating ~ as.numeric(date), data = ., span = params_date[1], degree = params_date[2]))) %>%
    select(-rating)
  b_y <- fold2_train %>%
    left_join(b_d, by = "date") %>%
    group_by(year) %>%
    summarize(rating = mean(rating - mu - b_d)) %>%
    mutate(b_y = predict(loess(rating ~ year, data = ., span = params_year[1], degree = params_year[2]))) %>%
    select(-rating)
  b_i <- fold2_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - b_d - b_y)/(n()+l))
  b_u <- fold2_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by= "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_d - b_y - b_i)/(n()+l))
  predicted_ratings <- fold2_test %>% 
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId")  %>%
    mutate(pred = mu + b_d + b_y + b_i + b_u) %>%
    .$pred
  rmse2 <- RMSE(predicted_ratings, fold2_test$rating)
  #fold3
  mu <- mean(fold3_train$rating)
  b_d <- fold3_train %>% 
    group_by(date) %>%
    summarize(rating = mean(rating - mu)) %>%
    mutate(b_d = predict(loess(rating ~ as.numeric(date), data = ., span = params_date[1], degree = params_date[2]))) %>%
    select(-rating)
  b_y <- fold3_train %>%
    left_join(b_d, by = "date") %>%
    group_by(year) %>%
    summarize(rating = mean(rating - mu - b_d)) %>%
    mutate(b_y = predict(loess(rating ~ year, data = ., span = params_year[1], degree = params_year[2]))) %>%
    select(-rating)
  b_i <- fold3_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - b_d - b_y)/(n()+l))
  b_u <- fold3_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by= "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_d - b_y - b_i)/(n()+l))
  predicted_ratings <- fold3_test %>% 
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId")  %>%
    mutate(pred = mu + b_d + b_y + b_i + b_u) %>%
    .$pred
  rmse3 <- RMSE(predicted_ratings, fold3_test$rating)
  #fold4
  mu <- mean(fold4_train$rating)
  b_d <- fold4_train %>% 
    group_by(date) %>%
    summarize(rating = mean(rating - mu)) %>%
    mutate(b_d = predict(loess(rating ~ as.numeric(date), data = ., span = params_date[1], degree = params_date[2]))) %>%
    select(-rating)
  b_y <- fold4_train %>%
    left_join(b_d, by = "date") %>%
    group_by(year) %>%
    summarize(rating = mean(rating - mu - b_d)) %>%
    mutate(b_y = predict(loess(rating ~ year, data = ., span = params_year[1], degree = params_year[2]))) %>%
    select(-rating)
  b_i <- fold4_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - b_d - b_y)/(n()+l))
  b_u <- fold4_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by= "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_d - b_y - b_i)/(n()+l))
  predicted_ratings <- fold4_test %>% 
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId")  %>%
    mutate(pred = mu + b_d + b_y + b_i + b_u) %>%
    .$pred
  rmse4 <- RMSE(predicted_ratings, fold4_test$rating)
  #fold5
  mu <- mean(fold5_train$rating)
  b_d <- fold5_train %>% 
    group_by(date) %>%
    summarize(rating = mean(rating - mu)) %>%
    mutate(b_d = predict(loess(rating ~ as.numeric(date), data = ., span = params_date[1], degree = params_date[2]))) %>%
    select(-rating)
  b_y <- fold5_train %>%
    left_join(b_d, by = "date") %>%
    group_by(year) %>%
    summarize(rating = mean(rating - mu - b_d)) %>%
    mutate(b_y = predict(loess(rating ~ year, data = ., span = params_year[1], degree = params_year[2]))) %>%
    select(-rating)
  b_i <- fold5_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu - b_d - b_y)/(n()+l))
  b_u <- fold5_train %>%
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by= "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_d - b_y - b_i)/(n()+l))
  predicted_ratings <- fold5_test %>% 
    left_join(b_d, by = "date") %>%
    left_join(b_y, by = "year") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId")  %>%
    mutate(pred = mu + b_d + b_y + b_i + b_u) %>%
    .$pred
  rmse5 <- RMSE(predicted_ratings, fold5_test$rating)
  #combination
  return((rmse1+rmse2+rmse3+rmse4+rmse5)/5)
})

# Plot of RMSE versus lambda
qplot(lambdas, rmses, xlab = "Lambda", ylab = "RMSE", geom = c("point", "line"))

# Optimal lambda of this final model
final_bestlambda <- lambdas[which.min(rmses)]
final_bestlambda

# RMSE for this final model
model_6_rmse <- min(rmses)

# Incorporation of this result in the results table
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Final Regularized Movie + User + Date + Year Effects Model (CV)",  
                                 RMSE = model_6_rmse))
rmse_results %>% knitr::kable() %>%
  kable_styling(bootstrap_options = "bordered", full_width = FALSE, position = "center") %>%
  kable_styling(bootstrap_options = "striped" , full_width = FALSE, position = "center") %>%
  kable_styling(latex_options = "hold_position") %>%
  row_spec(0, bold = TRUE)


##################################################
# Application of the final model on validation set
##################################################

# Transformation of validation set
validation <- validation %>%
  mutate(date = ymd(round_date(as_datetime(timestamp), unit = "week")),
         year = str_extract(title, "\\s\\(\\d{4}\\)$"),
         year = as.numeric(str_extract(year, "\\d{4}"))) %>%
  select(-timestamp, -title) %>%
  semi_join(edx, by = "date") %>%
  semi_join(edx, by = "year")

# Final model on validation set
mu <- mean(edx$rating)
b_d <- edx %>% 
  group_by(date) %>%
  summarize(rating = mean(rating - mu)) %>%
  mutate(b_d = predict(loess(rating ~ as.numeric(date), data = ., span = params_date[1], degree = params_date[2]))) %>%
  select(-rating)
b_y <- edx %>%
  left_join(b_d, by = "date") %>%
  group_by(year) %>%
  summarize(rating = mean(rating - mu - b_d)) %>%
  mutate(b_y = predict(loess(rating ~ year, data = ., span = params_year[1], degree = params_year[2]))) %>%
  select(-rating)
b_i <- edx %>%
  left_join(b_d, by = "date") %>%
  left_join(b_y, by = "year") %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu - b_d - b_y)/(n()+final_bestlambda))
b_u <- edx %>%
  left_join(b_d, by = "date") %>%
  left_join(b_y, by = "year") %>%
  left_join(b_i, by= "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_d - b_y - b_i)/(n()+final_bestlambda))
predicted_ratings <- validation %>% 
  left_join(b_d, by = "date") %>%
  left_join(b_y, by = "year") %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId")  %>%
  mutate(pred = mu + b_d + b_y + b_i + b_u) %>%
  .$pred

# RMSE of the final model on the validation set
final_rmse <- RMSE(predicted_ratings, validation$rating)

# Incorporation of the final result in the results table
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Final Model on Validation Set",  
                                 RMSE = final_rmse))
rmse_results %>% knitr::kable() %>%
  kable_styling(bootstrap_options = "bordered", full_width = FALSE, position = "center") %>%
  kable_styling(bootstrap_options = "striped" , full_width = FALSE, position = "center") %>%
  kable_styling(latex_options = "hold_position") %>%
  row_spec(0, bold = TRUE) %>%
  row_spec(8, bold = TRUE)