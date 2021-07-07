#######################################################################################
# Title: HarvardX PH125.9x Data Science: Capstone - MovieLens Project
# Author: Robert Walscheid	
# Date: 07/07/2021
#######################################################################################

#######################################################################################
# 1.) PREREQUISITES	###################################################################
#######################################################################################

# Download and install the necessary packages for this project.	
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")	
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")	
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")	
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")	
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")	
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")	
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")	
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")	
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")	
	
# Load the necessary libraries for this project.	
library(tidyverse)	
library(caret)	
library(data.table)	
library(scales)	
library(lubridate)	
library(dplyr)	
library(gridExtra)	
library(kableExtra)	
library(recosystem)	

# Download the MovieLens 10M dataset to the computer's temporary directory.	
dl <- tempfile()	
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)	
	
# Create the data frame "ratings" and fill it with the 4 columns of data	
# that were delimited by "::" in the ratings.dat file, labeling them	
# userId, moveiId, rating, and timestamp, respectively.	
#	
# Note: this process could take a couple of minutes.	
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),	
                 col.names = c("userId", "movieId", "rating", "timestamp"))	

# Summary information of the "ratings" dataset	
summary(ratings)	

# Structure of the "ratings" dataset	
str(ratings)	

# Create the 'movies' data frame and fill it with the 3 columns of data	
# that were delimited by '::' in the movies.dat file, labeling them moveiId, 	
# title, and genres, respectively:	
#	
# Note: this process could take a couple of minutes.	
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)	
colnames(movies) <- c("movieId", "title", "genres")	
	
# The movies matrix array needs to be converted to a usable dataframe, 	
# assigning the proper class types for this project to each column.	
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),	
                                           title = as.character(title),	
                                           genres = as.character(genres))	

# Summary information of the "movies" dataset	
summary(movies)	

# Structure of the "movies" dataset	
str(movies)	

# Create the 'movielens' data frame by joining the 'ratings' and 'movies' 	
# datasets by 'movieId'.	
movielens <- left_join(ratings, movies, by = "movieId")	

# Summary information of the "movielens" dataset	
summary(movielens)	

# Structure of the "movielens" dataset	
str(movielens)	

# The training dataset, "edx", and validation test dataset, "temp", will be created 	
# from a 90%/10% split of the MovieLens data, respectively.	
set.seed(1, sample.kind="Rounding")	
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)	
edx <- movielens[-test_index,]	
temp <- movielens[test_index,]	

# Because the data is split into two subsets, one that will validate the other, it 
# is important to ensure that the unique identifiers (`movieId` and `userId` in this
# case) in the final hold-out validation data exists in the `edx` training data, 
# otherwise the calculated RMSE could be skewed.  	
# 	
# This final hold-out test set (which will aptly be be named `validation`), used to 
# simulate the `edx` data, will be created by joining together the data from the 
# corresponding `movieId` and `userId` records from the `edx` and `temp` data frames.  
# The join results in the `validation` data frame consisting of 999,999 rows of data. 	
#
# Create validation dataset, ensuring the userId and movieId combinations are 	
# also in the edx set.	
validation <- temp %>% 	
  semi_join(edx, by = "movieId") %>%	
  semi_join(edx, by = "userId")	

# Summary information for the "validation" dataset.	
summary(validation)	

# Once the `validation` dataset is created, any rows that were removed from 
# `validation` need to be added back to `edx` to ensure 100% of the original 
# `movielens` data (10,000,054 rows) is accounted for between the training and 
# validation data frames.  Looking at the difference in length in the summary 
# output of `temp` (1,000,007) and `validation` (999,999), it can be observed 
# that 8 rows will be added back to the `edx` data frame (totaling 9,000,055 rows).	
# 	
# Add rows removed from validation set back into edx set	
removed <- anti_join(temp, validation)	
edx <- rbind(edx, removed)	

# Remove all unnecessary variables	
rm(dl, ratings, movies, test_index, temp, movielens, removed)	

# As a requirement for this project, the final hold-out test dataset (`validation`) 
# is to only be used for evaluating and testing the RMSE of the final algorithm, and
# not during model development.  Since 10% of the entire `movielens` dataset has 
# already been allocated to the final hold-out (`validation`), the `edx` dataset 
# itself will be split into a 90% training data subset (`edx_train`) and 10% test 
# subset (`edx_test`), with 8,100,065 and 899,990 rows, respectively, to be used 
# to build and test algorithms in the Modeling section of this report.  This will 
# bring the overall training/testing data split to approximately 80% (training)/20% (testing) 
# for the project after both splits: 90% `edx` (with `edx` split into another 90% 
# `edx_train`/10% `edx_test`)/10% `validation` (final hold-out), which is a 
# commonly-used split percentage for larger datasets.	
# 	
# The analysis and charting in the next section (Section 4.3) will use the larger 
# `edx` dataset, while the modeling section (Section 5) will use the smaller 
# `edx_train`/`edx_test` datasets for model development.	
# 	
# The training dataset, "edx_train", and validation test dataset, "edx_temp", will be 	
# created from a 90%/10% split of the "edx" dataset, respectively.	
set.seed(1, sample.kind="Rounding")	
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)	
edx_train <- edx[-edx_test_index,]	
edx_temp <- edx[edx_test_index,]	

# Create edx_test dataset, ensuring the userId and movieId combinations are also	
# in edx_train set.	
edx_test <- edx_temp %>% 	
  semi_join(edx_train, by = "movieId") %>%	
  semi_join(edx_train, by = "userId")	

# Summary information for the "validation" dataset.	
summary(edx_test)	

# Once the `edx_test` dataset is created, any rows that were removed from `edx_test` need 
# to be added back to `edx_train` to ensure 100% of the original `edx` data (9,000,055 rows)
# is accounted for.  Looking at the difference in length in the summary output of `edx_temp`
# (8,100,048) and `edx_test` (899,990), it can be observed that 17 rows will be added back
# to the `edx_train` data frame (totaling 8,100,065 rows).	
#
# Add rows removed from edx_test set back into the edx_train set	
edx_removed <- anti_join(edx_temp, edx_test)	
edx_train <- rbind(edx_train, edx_removed)	
 
# Remove all unnecessary variables	
rm(edx_test_index, edx_temp, edx_removed)	


#######################################################################################
# 2.) DATASET ANALYSIS ################################################################
#######################################################################################
# 	
# Reviewing the datasets in raw format, along with creating correlative visual aids 
# is necessary when comparing and contrasting the data.  While simple movie and user 
# data analysis may be made with the dataset as-is, other variables will need to be 
# created to dig deeper.  To get a general overview of the `edx` data and its 
# structure, the below output can be studied. In the below `edx` Object Summary, 
# the average (mean) rating for all movies is shown as 3.512.

# Summary information for the "edx" dataset.	
summary(edx)	

# Get a quick glimpse of the data in edx.	
glimpse(edx, width=80)	

# Number of unique users in the edx userId column:	
n_distinct(edx$userId)  #69,878	
	
# Number of unique movies in the edx movieId column:	
n_distinct(edx$movieId)  #10,677	
	
# Number of unique titles in the edx title column.	
# (There might be movies with different movieIds that have the same title)	
n_distinct(edx$title)  #10,676	

# Seeing the results of the `movieId` count (10,677) as being greater than the `title` 
# count (10,676) can be explained by there being two different `movieId`s that share 
# the same title.	
# 	
# Given that there are 69,878 unique user IDs and 10,677 unique movie IDs, simple math 
# shows: 69,878 users * 10,677 movies = a 746,087,406 total ratings potential if all 
# users rated every movie.  The summary output of the `edx` dataset above shows that
# there are only 9,000,055 ratings, which is approximately 12.1% of the total possible.  
# The unknown variations of which users rated which movies provides insight not only
# into how sparse the dataset is, but also how challenging the creation of an accurate
# recommendation system will be.	

# When comparing user ratings, a simple 100-user sample can be used to create a sparse 
# matrix to show how random the data is.  Each orange square shows a movie/user 
# combination that has a rating in the training dataset:	
#   	
# -- Creating a 100x100 Sparse Matrix --	
# Take a sample of 100 unique userIds and assign it to vector uuid.	
set.seed(1, sample.kind = "Rounding")	
uuid <- sample(unique(edx$userId), 100)	
	
edx %>% filter(userId %in% uuid) %>%  #Filter out non-uuid edx data	
  select(userId, movieId, rating) %>% #Pick only 3 columns of data	
  mutate(rating = 1) %>%              #Set rating as a boolean value (check box)	
  spread(movieId, rating) %>%         #Select the data/value options to chart	
  select(sample(ncol(.), 100)) %>%    #Random sample of 100 ratings	
  as.matrix() %>% t(.) %>%            #Plot as a matrix and transpose the data	
  image(1:100, 1:100,. , xlab = "Movies", ylab = "Users") + #Build chart	
  abline(h= 0:100 + 0.5, v = 0:100 + 0.5, col = "grey")  #Color the 100x100 chart	

# Calculate mean movie rating in edx dataset	
mu <- mean(edx$rating)	

# The simple distribution of movie ratings can be seen in Figure 4.3.1.  The movie 
# ratings ranged between 0.5 to 5, at 0.5 increments.  Most of the ratings were 4's
# and 3's, respectively, and whole number ratings were given more often by users than 
# half-ratings (*X*.5 ratings):	
#
# Ratings Distribution Bar Graph	
edx %>% ggplot(aes(x=rating)) +	
  geom_bar(color="orange", fill="dodgerblue4") +	
  scale_x_continuous(breaks=seq(0, 5, by=0.5)) +	
  scale_y_continuous(breaks=seq(0, 4000000, by=250000), labels=comma) +	
  labs(x = "Movie Rating", 	
       y = "Number of Ratings", 	
       title = "Ratings Distribution",	
       subtitle="(by rating)",	
       caption = "Source Data: edx\nFigure 4.3.1") 	

# Viewing the ratings distribution by both `moveId` and `userId` in Figure 4.3.2  
# (using `scale_x_log10`) shows that some movies were rated more than others and some 
# users rated more movies than others.	
#
# Bar chart of number of ratings by movieId	
chart_movieId_ratings_count <- edx %>% count(movieId) %>% 	
  ggplot(aes(n)) + 	
  geom_histogram(bins=50, color="orange", fill="dodgerblue4") +	
  scale_x_log10() + 	
  labs(x="movieId", 	
       y="Number of Ratings", 	
       title = "Ratings Distribution",	
       subtitle="(by movieId)",	
       caption = " \n ") + 	
  theme(panel.border = element_rect(color="black", fill=NA)) 	
	
# Bar chart of number of ratings by userId	
chart_userId_ratings_count <- edx %>% count(userId) %>% 	
  ggplot(aes(n)) + 	
  geom_histogram(bins=50, color="orange", fill="dodgerblue4") +	
  scale_x_log10() + 	
  labs(x="userId", 	
       y="Number of ratings",	
       title = "Ratings Distribution",	
       subtitle="(by userId)",	
       caption = "Source Data: edx\nFigure 4.3.2") + 	
  theme(panel.border = element_rect(color="black", fill=NA)) 	
	
# Set the two bar charts next to each other.	
grid.arrange(chart_movieId_ratings_count, chart_userId_ratings_count, nrow = 1)	

# Figure 4.3.3 shows the top 20 movies by the quantity of user ratings.  While 17 out
# of 20 (85%) of the top 20 movies were released in the 1990's, the chart does not 
# show whether these movies were actually rated around the year they were released, 
# or any other relevant details.	
# 	
# Bar chart of top 20 movies  	
  edx %>% group_by(title) %>%	
  summarize(n=n()) %>%	
  top_n(20,n) %>%	
  arrange(desc(n)) %>%	
  ggplot(aes(x=reorder(title, n), y=n)) +	
  geom_bar(stat="identity", color="orange", fill="dodgerblue4") + 	
  coord_flip(y=c(0, 40000)) +	
  labs(x="",	
       y="Number of ratings",	
       title="Top 20 Movies",	
       subtitle="(by number of ratings)",	
       caption="Source Data: edx\nFigure 4.3.3") +	
  geom_text(aes(label=n), hjust=-0.2, size=3.5)	

# Taking the difference between the first and last rating's timestamp shows that the
# ratings were collected over a 14 year period:	
# 	
# Build a table showing the first rating date, last rating date, and the rating period in years.	
kable(tibble(date(as_datetime(min(edx$timestamp))),	
             date(as_datetime(max(edx$timestamp)))) %>% 	
      mutate("Rating Period" = duration(max(edx$timestamp)-min(edx$timestamp))),	
      col.names = c("First Rating Date", "Last Rating Date", "Rating Period")) %>%	
  row_spec(0,background="#104E8B", color="white") %>% 	
  column_spec(3, bold=TRUE) %>% 	
  kable_styling(bootstrap_options="bordered", 	
                full_width=FALSE, 	
                position="center",	
                latex_options="HOLD_position")	

# Viewing how the ratings were distributed each year over that 14 year period 
# (Figure 4.3.4) does not provide much useful information by itself:	
#
# Create a histogram of ratings by year.	
edx %>% mutate(year_rated = year(as_datetime(timestamp))) %>%	
  group_by(year_rated) %>%	
  summarize(n=n()) %>%	
  ggplot(aes(x=year_rated, y=n)) +	
  geom_bar(stat="identity", color="orange", fill="dodgerblue4") + 	
  scale_x_continuous(breaks=seq(1995, 2010, by=1)) +	
  scale_y_continuous(breaks=seq(0, 2000000, by=250000), labels=comma, limits = c(0,1250000)) +	
  coord_flip(clip = "off") +	
  labs(x="Data Collection Year",	
       y="Number of ratings",	
       title="Rating Distribution",	
       subtitle="(by year in data collection period)",	
       caption="Source Data: edx\nFigure 4.3.4") +	
  geom_text(aes(label=n), hjust=-0.1, size=3.5) +	
  theme(panel.border = element_rect(color="black", fill=NA)) 	

# The first (1995) and last (2008) years during the ratings collection period contain very 
# few ratings, and will need to be taken into account when reviewing the average movie
# ratings given over that 14 year period in **Figure 4.3.5** below.  Since there were 
# only 2 ratings in 1995, it is not surprising that the average rating for that year is
# skewing the smooth line.  Changing from year to day shows a much flatter smooth line, 
# which means the date the movie is rated only has a small impact on the rating itself.	
# 	
# Chart the mean rating distribution by year rated	
chart_rating_by_year <- edx %>% mutate(year_rated = year(as_datetime(timestamp))) %>%	
  group_by(year_rated) %>%	
  summarize(rating = mean(rating)) %>%	
  ggplot(aes(year_rated,rating)) +	
  geom_point(alpha=0.25) + 	
  geom_smooth(method=loess, color="dodgerblue4") +	
  geom_hline(aes(yintercept=mean(rating)), color="orange", linetype='dashed', size=1) +	
  labs(x="Year Rated" , 	
       y="Average Movie Rating", 	
       title="Rating Distribution",	
       subtitle="(average rating by year rated)",	
       caption = " \n ") +	
  theme(panel.border = element_rect(color="black", fill=NA)) 	
	
# Chart the mean rating distribution by day rated	
chart_rating_by_day <- edx %>% mutate(day_rated = day(as_datetime(timestamp))) %>%	
  group_by(day_rated) %>%	
  summarize(rating = mean(rating)) %>%	
  ggplot(aes(day_rated,rating)) +	
  geom_point(alpha=0.25) + 	
  geom_smooth(method=loess, color="dodgerblue4") +	
  geom_hline(aes(yintercept=mean(rating)), color="orange", linetype='dashed', size=1) +	
  labs(x="Day Rated" , 	
       y="Average Movie Rating", 	
       title="Rating Distribution",	
       subtitle="(average rating by day rated)",	
       caption = "Source Data: edx\nFigure 4.3.5") +	
  theme(panel.border = element_rect(color="black", fill=NA)) 	
	
# Set the two bar charts next to each other.	
grid.arrange(chart_rating_by_year, chart_rating_by_day, nrow = 1)	

# Comparing the average movie ratings for the top 20 most-rated movies to the overall 
# `edx` dataset mean shows that the most-rated movies were also rated higher than 
# average, which is not surprising that good movies are rated higher more often.
# 	
# Create table of the Top-20 movies, quantity of ratings, their ratings mean, and how much	
# greater or less than the mean ratings for that particular movie were as compared to the 	
# overall ratings mean for all movies.	
kable(tibble(edx %>% group_by(title) %>%	
               summarize(Number_of_Ratings=n(), Rating_Mean=mean(rating)) %>%	
               mutate(plus_minus_rating_ave = Rating_Mean - mean(edx$rating)) %>%	
               top_n(20,Number_of_Ratings) %>%	
               arrange(desc(Number_of_Ratings))),	
      col.names = c("Movie Title", "# of Ratings", "Average Rating", "+/- edx Mean")) %>%	
  row_spec(0,background="#104E8B", color="white") %>% 	
  column_spec(4, bold=TRUE) %>% 	
  kable_styling(bootstrap_options="bordered", 	
                font_size=10,	
                full_width=FALSE, 	
                position="center",	
                latex_options="HOLD_position")	

# For further age-based comparisons in this section, it would be more efficient from a 
# computing-resource perspective to extract the following variables into a new data frame 
# called `analysis_age_data` (a copy of the `edx` dataset), using the default `timestamp` 
# column and splitting the name of the movie and the release year from the default `title` 
# column: 	
# 	
# 1. `age_at_rating` = The age of each movie when they were rated. (year rated - year released)	
# 2. `ratings_per_age` = The number of ratings for each movie of a particular age.  	
# 3. `movies_per_age` = The number of movies that are a particular age.  	
# 4. `avg_ratings_per_age` = The average rating of all movies of a given age when rated.	
# 5. `ratings_per_movie` = The number of ratings given for each movie.  	
#' 	
# Build time/age-based analysis dataset with year the movies were rated, movie title 	
# without the release year, the movie release year, age of movie at time of rating, 	
# number of ratings by movie age, number of movies by age, average ratings of a movie 	
# based on age, and number of ratings per movie.	
analysis_age_data <- edx %>% mutate(year_rated = year(as_datetime(timestamp)),	
                            title_without_year = str_replace(title,"^(.+)\\s\\((\\d{4})\\)$","\\1__\\2" )) %>% 	
  separate(title_without_year,c("title_without_year","year_released"),"__") %>%	
  select(-timestamp) %>% 	
  mutate(age_at_rating = as.numeric(year_rated)-as.numeric(year_released)) %>%	
  group_by(age_at_rating) %>% 	
  summarize(ratings_per_age=n(),	
            movies_per_age=n_distinct(movieId),	
            avg_ratings_per_age=mean(rating),	
            ratings_per_movie=n()/n_distinct(movieId))	

# Looking at the total number movie ratings by movie age when rated, Figure 4.3.6 shows
# that newer movies were rated more frequently within the first 3 years of their release, 
# with a sharp decline over the following 20 years.  Graphing the number of ratings given 
# only to movies within the movie's age group (Figure 4.3.7) continues to show that newer
# movies were still more frequently rated, but by normalizing by the number of movies
# greatly slows the rate of decline.	
# 	
# Bar chart of the number of movie ratings by the movie's age.	
chart_total_ratings_by_movie_age <- analysis_age_data %>% ggplot(aes(age_at_rating, ratings_per_age)) +	
  geom_bar(stat="identity", size=0.5, color="dodgerblue4", fill="orange") +	
  scale_y_continuous(labels = comma) + 	
  labs(x="Age of Movies at Rating", 	
       y="Number of Movie Ratings\n(total)",	
       title="Ratings Distribution",	
       subtitle="(total number of ratings by\nmovie age at rating)", 	
       caption="Source Data: edx\nFigure 4.3.6")	
	
# Bar chart of the per-movie rating distribution by the movie's age.	
chart_indiv_ratings_by_movie_age <- analysis_age_data %>% ggplot(aes(age_at_rating, ratings_per_movie)) +	
  geom_bar(stat="identity", size=0.5, color="dodgerblue4", fill="orange") +	
  scale_y_continuous(labels = comma) + 	
  labs(x="Age of Movies at Rating", 	
       y="Number of Movie Ratings\n(per movie)",	
       title="Ratings Distribution",	
       subtitle="(per-movie ratings by\nmovie age at rating)", 	
       caption="Source Data: edx\nFigure 4.3.7")	
	
# Set the two bar charts next to each other.	
grid.arrange(chart_total_ratings_by_movie_age, chart_indiv_ratings_by_movie_age, nrow = 1)	


# Bar chart of the number of movies by age category.	
analysis_age_data %>% ggplot(aes(age_at_rating, movies_per_age)) +	
  geom_bar(stat="identity", size=0.5, color="orange", fill="dodgerblue4") +	
  scale_y_continuous(labels = comma) + 	
  labs(x="Age of Movies at Rating", 	
       y="Number of Movies",	
       title="Movie Distribution",	
       subtitle="(by age at rating)", 	
       caption="Source Data: edx\nFigure 4.3.8")	

# When comparing both Figure 4.3.6 (total ratings by movie age) and Figure 4.3.8 (number 
# of movies by age) above, it appears that most ratings were for movies between 0-25 
# years old when they were rated.  Plotting the average movie rating by movie age when
# rated (Figure 4.3.9) shows that users gave a better rating to older movies vs. newer 
# movies:	
#
# Plot the average movie rating by movie age.	
analysis_age_data %>% ggplot(aes(age_at_rating,avg_ratings_per_age)) +	
  geom_point(alpha=0.25, color="dodgerblue4") + 	
  geom_smooth(method=loess, color="dodgerblue4") +	
  geom_hline(aes(yintercept=mean(edx$rating)), color="orange", linetype="dashed", size=1) +	
  labs(x="Movie Age at Rating" , 	
       y="Average Movie Rating", 	
       title="Rating Distribution",	
       subtitle="(average rating by movie age)",	
       caption = "Source Data: edx\nFigure 4.3.9") +	
  theme(panel.border = element_rect(color="black", fill=NA)) 	

# Plotting the average movie rating by movie release year (Figure 4.3.10 below) shows a 
# similar trend in user preference for older movies vs. newer movies.	
# 	
# Plot the average movie rating by movie release year.	
edx %>% mutate(title_without_year = str_replace(title,"^(.+)\\s\\((\\d{4})\\)$","\\1__\\2" )) %>% 	
  separate(title_without_year,c("title_without_year","year_released"),"__") %>%	
  group_by(year_released) %>%	
  summarize(rating = mean(rating)) %>%	
  ggplot(aes(as.numeric(year_released),rating)) +	
  geom_point(alpha=0.25) + 	
  geom_smooth(method = "loess", color="dodgerblue4") +	
  scale_x_continuous(breaks=seq(1910, 2010, by=10)) +	
  geom_hline(aes(yintercept=mean(rating)), color="orange", linetype='dashed', size=1) +	
  labs(x="Movie Release Year" , 	
       y="Average Movie Rating", 	
       title="Rating Distribution",	
       subtitle="(average rating by movie release year)",	
       caption = "Source Data: edx\nFigure 4.3.10") +	
  theme(panel.border = element_rect(color="black", fill=NA)) 	

# The significantly lower number of ratings given to older movies needs to be taken into
# account when viewing both Figure 4.3.9 (average ratings based on movie age at rating) 
# and Figure 4.3.10 (average ratings based on the year the movie was released).	
#
# Gathering the movie release years shows 93-year period between the first and last movie 
# in this dataset.  Since the data collection period was over a 14 year period (starting 
# in 1995), there is the possibility that not all users were able to rate all of the newer 
# movies.	
# 	
# Gather a list of all movie release years	
release_years <- edx %>% mutate(title_without_year = str_replace(title,"^(.+)\\s\\((\\d{4})\\)$","\\1__\\2" )) %>% 	
  separate(title_without_year,c("title_without_year","year_released"),"__") %>%	
  group_by(year_released) %>% 	
  summarize(n=n())	
# Create a table with the earliest and latest movie release year	
kable(tibble(min(release_years$year_released), 	
             max(release_years$year_released)),	
      col.names=c("First Movie Release Year", "Last Movie Release Year"), 	
      align="c") %>%	
  row_spec(0,background="#104E8B", color="white") %>% 	
  kable_styling(bootstrap_options="bordered", 	
                full_width=FALSE, 	
                position="center",	
                latex_options="HOLD_position")	

# When looking at the frequency in which each movie was rated over time, Figure 4.3.11 
# shows that movies rated more often each year were also often rated higher.  This can
# easily be explained since more people watched popular movies.	
# 	
# Plot the average rating based on rating frequency.	
edx %>% mutate(year = year(as_datetime(timestamp))) %>%	
  group_by(movieId) %>%	
  summarize(n = n(), years = max(as.numeric(release_years$year_released)) - min(as.numeric(release_years$year_released)),	
            title = title[1],	
            rating = mean(rating)) %>%	
  mutate(rate = n/years) %>%	
  ggplot(aes(rate, rating)) +	
  geom_point(alpha=0.25) +	
  geom_smooth(method = "loess", color="dodgerblue4") +	
  geom_hline(aes(yintercept=mean(rating)), color="orange", linetype='dashed', size=1) +	
  labs(x="Rating Frequency (per Year)" , 	
       y="Average Movie Rating", 	
       title="Rating Distribution",	
       subtitle="(average movie rating by rating frequency)",	
       caption = "Source Data: edx\nFigure 4.3.11") +	
  theme(panel.border = element_rect(color="black", fill=NA)) 	

# Now, to begin analyzing more genre-related data, the `movielens` data structure output 
# in section 4.1 showed that a movie being rated had the possibility of being assigned 
# one or more genres.  To begin charting genre-based comparisons, the `genres` data for
# each movie will be separated from their pipe (`|`) delimiter and stored in a column.
# The following variables will additionally need to be extracted into a new data frame
# (copied from `edx`) called `analysis_genre_data`, which will be referenced for 
# genre-based data analysis later in this section.	
# 	
# 1. `ratings_per_genre` = The number of ratings given to a particular genre.  	
# 2. `movies_per_genre` = The number of movies that are assigned to a particular genre.  	
# 3. `users_per_genre` = The number of users that rated a movie of a particular genre.  	
# 4. `avg_ratings_per_genre` = The average rating of movies assigned a particular genre.  	
# 5. `ratings_per_movie` = The number of ratings given for each movie.  	
#
# **Note: Preliminary testing shows the presence of a "(no genres listed)" entry, which 
# this code will ignore (further details provided later in this section).
#
# Build Genre-based analysis dataset with number of ratings/genre, number of 	
# movies/genre, number of times users rated a particular genre, average rating 	
# of each genre, and the number of ratings/movie.	
analysis_genre_data <- edx %>% separate_rows(genres,sep = "\\|") %>% 	
  mutate(value=1) %>% 	
  group_by(genres) %>% 	
  summarize(ratings_per_genre=n(),	
            movies_per_genre=n_distinct(movieId),	
            users_per_genre=n_distinct(userId),	
            avg_ratings_per_genre=mean(rating),	
            ratings_per_movie=n()/n_distinct(movieId)) %>% 	
  filter(genres != "(no genres listed)") %>%	
  arrange(desc(ratings_per_genre))	

# When separating the genres, a total of 20 different categories are found.  One such 
# genre category is labeled "(no genres listed)", which accounts for 7 movie ratings
# (see Figure 4.3.12), and is for movies that are not assigned to a specific genre.	
# 	
# Bar Graph of Ratings by Genre	
edx %>% separate_rows(genres,sep = "\\|") %>% 	
  group_by(genres) %>% 	
  summarize(n=n()) %>% 	
  ggplot(aes(x=reorder(genres, n), y=n)) +	
  geom_bar(stat="identity", color="orange", fill="dodgerblue4") + 	
  coord_flip(y=c(0, 4500000)) +	
  scale_y_continuous(breaks=seq(0, 4500000, by=1000000), labels=comma) +	
  labs(x="", 	
       y="Number of ratings",	
       title="Ratings Distribution\n(by genre)", 	
       caption="Source Data: edx\nFigure 4.3.12") +	
  geom_text(aes(label=n), hjust=-0.1, size=3)	

# Extracting the 7 ratings that were for movies in the `(no genres listed)` category 
# yields only a single movie, "Pull My Daisy (1958)", with `moveId` 8606.  	
#
# Movies rated that were not assigned a genre.	
kable(tibble(edx %>% 	
               filter(genres=="(no genres listed)") %>%	
               select(movieId, title, genres)), 	
      col.names=c("Movie ID", "Movie Title", "Genre")) %>%	
  row_spec(0,background="#104E8B", color="white") %>% 	
  kable_styling(bootstrap_options="bordered", 	
                full_width=FALSE, 	
                position="center",	
                latex_options="HOLD_position")	

# Given that there is only a single movie with the `(no genre listed)` label, any model 
# considering genres as a data point in this project will ignore this category, leaving 
# 19 total genre categories:	
# 	
# Bar chart of number of movies by genre	
analysis_genre_data %>% ggplot(aes(reorder(genres, movies_per_genre), movies_per_genre)) +	
  geom_bar(stat="identity", color="orange", fill="dodgerblue4") + 	
  coord_flip(y=c(0, 5750)) +	
  scale_y_continuous(breaks=seq(0, 5750, by=1000), labels=comma) +	
  labs(x="Movie Genres", 	
       y="Number of Movies",	
       title="Ratings Distribution",	
       subtitle="(number of movies by genre)", 	
       caption="Source Data: edx\nFigure 4.3.13") +	
  geom_text(aes(label=movies_per_genre), hjust=-0.1, size=3)	

# Looking at the distribution of movies by genre in **Figure 4.3.13** above, 49.98% of 
# all movies were in the Drama genre (5,336 Drama/10,677 Total), and 34.68% were in the 
# Comedy genre (3,703 Comedy/10,677 Total).  Since each movie may be in more than one 
# genre category, these top-two genres most likely share several movies with the same movieId.	
# 	
# While there was a significant number of movies that were in the Drama and Comedy genre
# category, Figure 4.3.14 shows that movies in either of those categories were among the
# fewest to be rated on average.	
# 	
# Bar chart of average number of times each movie was rated according to its genre	
analysis_genre_data %>% ggplot(aes(reorder(genres, ratings_per_movie), ratings_per_movie)) +	
  geom_bar(stat="identity", color="orange", fill="dodgerblue4") + 	
  coord_flip(y=c(0, 2000)) +	
  scale_y_continuous(breaks=seq(0, 2000, by=500), labels=comma) +	
  labs(x="Movie Genres", 	
       y="Average Number of Times Each Movie Was Rated\n(based on its genre)",	
       title="Ratings Distribution",	
       subtitle="(average number of ratings per movie by genre)", 	
       caption="Source Data: edx\nFigure 4.3.14") +	
  geom_text(aes(label=round(ratings_per_movie), hjust=-0.2))	

# When comparing the average ratings of movies by their genre (Figure 4.3.15), the
# highest-rated (on average) genres were those that contained the fewest movies (see 
# Figure 4.3.13) as well as were rated the fewest times on average (see Figure 4.3.14).
# While Figure 4.3.15 definitely shows that genre has an effect on the rating, the amount
# of impact appears to be slight. 	
#
# Bar chart of average movie rating by genre	
analysis_genre_data %>% ggplot(aes(reorder(genres, avg_ratings_per_genre), avg_ratings_per_genre)) +	
  geom_bar(stat="identity", color="orange", fill="dodgerblue4") + 	
  coord_flip(y=c(0, 5)) +	
  scale_y_continuous(breaks=seq(0, 5, by=1)) +	
  geom_hline(yintercept=mu,col="orange",linetype="dashed", size=1) +	
  labs(x="Movie Genres", 	
       y="Average Movie Rating",	
       title="Ratings Distribution\n(average movie rating by genre)", 	
       caption="Source Data: edx\nFigure 4.3.15") +	
  geom_text(aes(label=round(avg_ratings_per_genre, digits = 4), hjust=-0.1))	


#######################################################################################
# 3.) MODELING ########################################################################
#######################################################################################

# Throughout the data analysis performed above, some data point comparisons show the 
# possibility of higher correlations while others appear they would have little impact 
# to the overall RMSE value, and thus the recommendation model.  To determine the impact
# of each individual bias, various models will be tested, starting from the most basic 
# baseline model through the addition of individual biases, bias combinations, 
# regularization, and matrix factorization.  Each resultant RMSE will be summarized 
# in a list with previous results as to make it easy to compare all results and
# determine which model performs the best (having the lowest RMSE).	
# 	
#######################################################################################
# 3a.) MODEL 1: Predict by Overall Average Movie Rating Only (Naive Model)	
#######################################################################################

# The first model is the simplest, as it assumes the same rating for all movies and all 
# users, with all the differences explained by random variation.  All variations in this 
# model are viewed as equally weighted errors.	
#
# Model 1: mean (mu) calculation
mu_edx_train <- mean(edx_train$rating) 
mu_edx_train

# Test the model by calculating the RMSE using the edx_test dataset.
model1_rmse <- RMSE(edx_test$rating, mu_edx_train)  
model1_rmse

# Create a table containing the Model 1 data results.	
model_table_titles <- "Model 1: Average Rating Only (Naive Model)"	
model_table_rmses <- model1_rmse	
kable(tibble(model_table_titles, model_table_rmses),	
      col.names = c("Model", "RMSE")) %>%	
  row_spec(0,background="#104E8B", color="white") %>% 	
  column_spec(2, bold=TRUE) %>% 	
  kable_styling(bootstrap_options="bordered", 	
                full_width=FALSE, 	
                position="center",	
                latex_options="HOLD_position")	

#######################################################################################
# 3b.) MODEL 2: Account For The Individual Movie's Average Rating (Movie Effect)	
#######################################################################################
#
# In order to improve upon Model 1, other variables need to be considered.  As seen 
# earlier in section 2, the average rating for each movie varies, adding a bias that
# we can factor into a model:	
# 	
# Mean (mu) calculation
mu_edx_train <- mean(edx_train$rating) 

# Each individual movie's mean rating calculation
movie_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_edx_train))

# Predict the ratings against the edx_test dataset.
predicted_ratings_m2 <- mu_edx_train + edx_test %>% 
  left_join(movie_avgs, by = "movieId") %>%
  pull(b_i)

# Test the "Movie Effect" model by calculating the RMSE using the edx_test dataset.
model2_rmse <- RMSE(predicted_ratings_m2, edx_test$rating)
model2_rmse

# Create RMSE Results Table for Models 1-2
model_table_titles <- c(model_table_titles, "Model 2: Movie Effect")	
model_table_rmses <- c(model_table_rmses, model2_rmse)	
kable(tibble(model_table_titles, model_table_rmses),	
      col.names = c("Model", "RMSE")) %>%	
  row_spec(0,background="#104E8B", color="white") %>% 	
  column_spec(2, bold=TRUE) %>% 	
  kable_styling(bootstrap_options="bordered", 	
                full_width=FALSE, 	
                position="center",	
                latex_options="HOLD_position")	

# Adding the "Movie Effect" bias improved the prediction, as shown by the lower calculated 
# RMSE.  Plotting the bias (Figure 5.2.1) shows the slightly negative effect the movie's 
# individual rating had on the ratings distribution.  This takes into account the difference 
# of the individual movie's average rating and the overall average rating of all movies, 
# allowing for the prediction adjustment of mu by b_i.	
#   	
# Graph the effect of the "Movie Effect" bias on the ratings distribution	
chart_movie_bias <- movie_avgs %>% ggplot(aes(b_i)) +	
  geom_histogram(bins=30, color="orange", fill="dodgerblue4") +	
  scale_y_continuous(labels = comma) + 	
  labs(x="Movie Effect Bias (b_i)",	
       y="Rating",	
       title="Bias on Ratings Distribution",	
       subtitle = "(Movie Effect)",	
       caption="Source Data: edx_train\nFigure 5.2.1") +	
  theme(panel.border = element_rect(color="black", fill=NA)) 	
chart_movie_bias	

#######################################################################################
# 3c.) MODEL 3: Account For The User's Rating (Movie + User Effect)	
#######################################################################################
# 
# Model 2 takes into account the ratings for each individual movie, but that model can 
# be improved upon by additionally taking into consideration the rating bias of the user
# (b_u) toward the movie.  Some users rate box-office hit movies (statistically high-rated
# movies on average by a majority of the rating users) poorly, while others may rate all 
# movies high (or low).  This bias can have a great impact to those movies that have fewer 
# ratings. 	
# 
# Mean (mu) calculation
mu_edx_train <- mean(edx_train$rating)

# Each individual movie's mean rating calculation
movie_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_edx_train))

# User's mean rating calculation, using the movie_avgs and b_i vectors from Model 1.
user_avgs <- edx_train %>% 
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_edx_train - b_i))

# Predict the ratings against the edx_test dataset.
predicted_ratings_m3 <- edx_test %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(predict_m3 = mu_edx_train + b_i + b_u) %>%
  pull(predict_m3)

# Test the "Movie + User Effect" model by calculating the RMSE using the edx_test 
# dataset.
model3_rmse <- RMSE(predicted_ratings_m3, edx_test$rating)
model3_rmse

# Create RMSE Results Table for Models 1-3
model_table_titles <- c(model_table_titles, "Model 3: Movie + User Effect")
model_table_rmses <- c(model_table_rmses, model3_rmse)
kable(tibble(model_table_titles, model_table_rmses),
      col.names = c("Model", "RMSE")) %>%
  row_spec(0,background="#104E8B", color="white") %>% 
  column_spec(2, bold=TRUE) %>% 
  kable_styling(bootstrap_options="bordered", 
                full_width=FALSE, 
                position="center",
                latex_options="HOLD_position")

# Adding the "User Effect" bias with the "Movie Effect" bias improved the prediction,
# lowering the calculated RMSE.  While this RMSE already meets the requirement of this
# project, additional modeling will be done to see what the impact of other biases, 
# regularization, and matrix factorization will have in lowering the RMSE further for 
# the recommendation system.  Plotting the bias (Figure 5.3.1) shows the slightly
# positive effect the user's average rating has on the ratings distribution.	
#     	
# Graph the effect of the "User Effect" bias on the ratings distribution
chart_user_bias <- user_avgs %>% ggplot(aes(b_u)) +
  geom_histogram(bins=30, color="orange", fill="dodgerblue4") +
  scale_y_continuous(labels = comma) + 
  labs(x="User Effect Bias (b_u)",
       y="Rating",
       title="Bias on Ratings Distribution",
       subtitle = "(User Effect)",
       caption="Source Data: edx_train\nFigure 5.3.1") +
  theme(panel.border = element_rect(color="black", fill=NA)) 
chart_user_bias

#######################################################################################
# 3d.) MODEL 4: Account For The Movie's Age When Rated (Movie + User + Age Effect)
#######################################################################################
# 
# An additional bias to consider would be the age of the movie when it was rated.  In 
# the initial data analysis of the `edx` data set in section 2, it was found that 
# newer movies had more ratings than older movies, but older movies received higher 
# average ratings.  Several factors or biases can contribute to a curve like this,
# such as the user rating the movie right after seeing it online or in the theater,
# or some users may prefer older, more classic movies, while others may only want to
# watch the latest movies.  This bias (b_a) will be added to the "Movie + User Effect"
# model to account for an additional dimension of data.
# 	
# Mean (mu) calculation
mu_edx_train <- mean(edx_train$rating)

# Each individual movie's mean rating calculation
movie_age_avgs <- edx_train %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(year_rated = year(as_datetime(timestamp)),
         title_without_year = str_replace(title,"^(.+)\\s\\((\\d{4})\\)$","\\1__\\2" )) %>% 
  separate(title_without_year,c("title_without_year","year_released"),"__") %>%
  mutate(age_at_rating = as.numeric(year_rated) - as.numeric(year_released)) %>%
  group_by(age_at_rating) %>% 
  summarize(b_a = mean(rating - mu_edx_train - b_i - b_u))

# Predict the ratings against the edx_test dataset.
predicted_ratings_m4 <- edx_test %>% 
  mutate(year_rated = year(as_datetime(timestamp)),
         title_without_year = str_replace(title,"^(.+)\\s\\((\\d{4})\\)$","\\1__\\2" )) %>% 
  separate(title_without_year,c("title_without_year","year_released"),"__") %>%
  mutate(age_at_rating = as.numeric(year_rated) - as.numeric(year_released)) %>%
  group_by(age_at_rating) %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(movie_age_avgs, by = "age_at_rating") %>%
  mutate(predict_m4 = mu_edx_train + b_i + b_u + b_a) %>%
  pull(predict_m4)

# Test the "Movie + User + Age Effect" model by calculating the RMSE using the 
# edx_test dataset.
model4_rmse <- RMSE(predicted_ratings_m4, edx_test$rating)
model4_rmse

# Create RMSE Results Table for Models 1-4
model_table_titles <- c(model_table_titles, "Model 4: Movie + User + Age Effect")
model_table_rmses <- c(model_table_rmses, model4_rmse)
kable(tibble(model_table_titles, model_table_rmses),
      col.names = c("Model", "RMSE")) %>%
  row_spec(0,background="#104E8B", color="white") %>% 
  column_spec(2, bold=TRUE) %>% 
  kable_styling(bootstrap_options="bordered", 
                full_width=FALSE, 
                position="center",
                latex_options="HOLD_position")

# Adding the "Age Effect" bias with the "Movie + User Effect" model only very slightly 
# improved the prediction, lowering the calculated RMSE.  Plotting the bias (Figure 5.4.1)
# shows the slightly positive effect the user's average rating has on the ratings distribution.	
# 	
# Graph the effect of the "Age Effect" bias on the ratings distribution
chart_age_bias <- movie_age_avgs %>% ggplot(aes(b_a)) +
  geom_histogram(bins=30, color="orange", fill="dodgerblue4") +
  scale_y_continuous(labels = comma) + 
  labs(x="Age Effect Bias (b_a)",
       y="Rating",
       title="Bias on Ratings Distribution",
       subtitle = "(Age Effect)",
       caption="Source Data: edx_train\nFigure 5.4.1") +
  theme(panel.border = element_rect(color="black", fill=NA)) 
chart_age_bias

#######################################################################################
# 3e.) MODEL 5: Account For The Movie's Genre (Movie + User + Genre Effect)
#######################################################################################
#
# Toward the end of section 4.3, the analysis of whether genre had an impact on the movie's
# ratings showed a slight, but definitive effect when the genres were split to individual
# categories.  Adding this bias will show whether the imbalance of the number of ratings 
# for each genre to the above-average ratings for fewest-rated genres (especially when 
# each movie can belong to more than one), will have a positive or negative effect on the 
# RMSE value.	
# 	
# Split the genres for each movie into their individual form for both the edx_train 
# and edx_test datasets, copying all data into a new data frame for each.
edx_train_split_genres <- edx_train %>% separate_rows(genres, sep = "\\|")
edx_test_split_genres <- edx_test %>% separate_rows(genres, sep = "\\|")

# Mean (mu) calculation
mu_edx_train <- mean(edx_train_split_genres$rating) 

# Calculate each movie's mean rating
movie_avgs <- edx_train_split_genres %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_edx_train))

# Calculate each user's mean rating for each movie
user_avgs <- edx_train_split_genres %>% 
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_edx_train - b_i))

# Calculate the mean for each movie rated by each user that was in a 
# particular genre category.
genre_avgs <- edx_train_split_genres %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu_edx_train - b_i - b_u))

# Predict the ratings against the edx_test_split_genres dataset.
predicted_ratings_m5 <- edx_test_split_genres %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(genre_avgs, by = "genres") %>%
  mutate(predict_m5 = mu_edx_train + b_i + b_u + b_g) %>%
  pull(predict_m5)

# Test the "Movie + User + Genre Effect" model by calculating the RMSE using the 
# edx_test dataset.
model5_rmse <- RMSE(predicted_ratings_m5, edx_test_split_genres$rating)
model5_rmse

# Create RMSE Results Table for Models 1-5
model_table_titles <- c(model_table_titles, "Model 5: Movie + User + Genre Effect")
model_table_rmses <- c(model_table_rmses, model5_rmse)
kable(tibble(model_table_titles, model_table_rmses),
      col.names = c("Model", "RMSE")) %>%
  row_spec(0,background="#104E8B", color="white") %>% 
  column_spec(2, bold=TRUE) %>% 
  kable_styling(bootstrap_options="bordered", 
                full_width=FALSE, 
                position="center",
                latex_options="HOLD_position")

# Adding the "Genre Effect" bias with the "Movie + User Effect" model only very 
# slightly improved the prediction, lowering the calculated RMSE.  Plotting the bias 
# (Figure 5.5.1) shows the slightly positive effect the user's average rating has on 
# the ratings distribution.	
#
# Graph the effect of the "Genre Effect" bias on the ratings distribution
chart_genre_bias <- genre_avgs %>% ggplot(aes(b_g)) +
  geom_histogram(bins=30, color="orange", fill="dodgerblue4") +
  scale_y_continuous(labels = comma) + 
  labs(x="Genre Effect Bias (b_g)",
       y="Rating",
       title="Bias on Ratings Distribution",
       subtitle = "(Genre Effect)",
       caption="Source Data: edx_train\nFigure 5.5.1") +
  theme(panel.border = element_rect(color="black", fill=NA)) 
chart_genre_bias

#######################################################################################
# 3f.) MODEL 6: Regularization of The Movie + User Effect
#######################################################################################
#
# Remembering Figure 4.3.2 (Rating Distribution by `movieId` and `userId`), some movies 
# were rated a greater number of times more than others, and some users rated a greater 
# number of movies than others.  Regularization is a method that can limit the effects 
# of either or both of these two variables by penalizing large estimates that come from
# small sample sizes in order to improve the RMSE results.  It assumes smaller weights 
# generate simpler models, which helps avoid overfitting.  This model will be regularizing
# both movies and users.
# 	
# Set the lambda tuning parameter sequence to be 1-10 at 0.25 intervals.
lambdas <- seq(0, 10, 0.25)

# Calculate the individual rmse values using each lambda parameter interval.
rmses_m6 <- sapply(lambdas, function(l){
  mu_edx_train <- mean(edx_train$rating)
  b_i <- edx_train %>%
    group_by(movieId) %>%
    summarize(b_i=sum(rating - mu_edx_train)/(n() + l))
  b_u <- edx_train %>% 
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u=sum(rating - mu_edx_train -b_i)/(n() + l))
  predicted_ratings_m6 <- edx_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(predict_m6 = mu_edx_train + b_i + b_u) %>%
    pull(predict_m6)
  return(RMSE(predicted_ratings_m6, edx_test$rating))
})

# Find the optimal lambda value that had the best RMSE
model6_opt_lambda <- lambdas[which.min(rmses_m6)]
model6_opt_lambda
 	
# Add the lambda and RMSE values to a data frame to be plotted
rmses_m6_plot <- data.frame(lambdas, rmses_m6)
# Plot the Cross-Validation results
rmses_m6_plot %>% ggplot(aes(lambdas, rmses_m6)) +
  geom_point(alpha=0.6, color="dodgerblue4", size = 3) +
  geom_hline(aes(yintercept=min(rmses_m6)), color="orange", linetype='dashed', size=1) +
  geom_vline(aes(xintercept=model6_opt_lambda), color="orange", linetype='dashed', size=1) +
  labs(x=expression(paste(lambda, ' Values')), 
       y="Calculated RMSE", 
       title="Cross-Validation",
       subtitle="(RMSE by lambda value)",
       caption = "Source Data: edx_train\nFigure 5.6.1") +
  theme(panel.border = element_rect(color="black", fill=NA)) 

# The cross-validation plot in Figure 5.6.1 above shows the intersection points 
# between the lowest RMSE and the lambda value used when calculating it.  The remaining 
# portion of the regularization algorithm can now be run with the optimal lambda value. 	
#
# Use the optimal lambda value to calculate the regularized movie bias
b_i <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_i=sum(rating - mu_edx_train)/(n() + model6_opt_lambda))
# Use the optimal lambda value to calculate the regularized user bias
b_u <- edx_train %>% 
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u=sum(rating - mu_edx_train - b_i)/(n() + model6_opt_lambda))
# Predict the ratings against the edx_test dataset.
predicted_ratings_m6 <- edx_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(predict_m6 = mu_edx_train + b_i + b_u) %>%
  pull(predict_m6)

# Test the Regularized "Movie + User Effect" model by calculating the RMSE using 
# the edx_test dataset.
model6_rmse <- RMSE(predicted_ratings_m6, edx_test$rating)
model6_rmse

# Create RMSE Results Table for Models 1-6
model_table_titles <- c(model_table_titles, "Model 6: Regularized Movie + User Effect")
model_table_rmses <- c(model_table_rmses, model6_rmse)
kable(tibble(model_table_titles, model_table_rmses),
      col.names = c("Model", "RMSE")) %>%
  row_spec(0,background="#104E8B", color="white") %>% 
  column_spec(2, bold=TRUE) %>% 
  kable_styling(bootstrap_options="bordered", 
                full_width=FALSE, 
                position="center",
                latex_options="HOLD_position")

# Adding Regularization to the "Movie + User Effect" model improved the prediction a 
# little, lowering the originally calculated RMSE.  Plotting the bias (Figure 5.6.2)
# shows the slightly positive effect the regularized user's average rating has on the
# ratings distribution.	
#
# Graph the effect of the "Regularized Movie + User Effect" bias on the ratings distribution
b_u %>% ggplot(aes(b_u)) +
  geom_histogram(bins=30, color="orange", fill="dodgerblue4") +
  scale_y_continuous(labels = comma) + 
  labs(x="Regularized User Effect Bias (b_u)",
       y="Number of Ratings",
       title="Bias on Ratings Distribution",
       subtitle = "(Regularized User Effect)",
       caption="Source Data: edx_train\nFigure 5.6.2") +
  theme(panel.border = element_rect(color="black", fill=NA)) 

#######################################################################################
# 3g.) MODEL 7: Regularization of The Movie + User + Age + Genre Effect
#######################################################################################
#
# Since the regularized "Movie + User Effect" model was able to lower the RMSE, this model
# will show what would happen if more biases were added to the equation.  The 
# non-regularized biases were plotted for all four factors (movie, user, age, genre) 
# individually, which showed varying impacts to the ratings.  This model will place all 
# four together and regularization will be used (using the same lambda value for each) 
# to penalize large estimates.
# 	
# Convert the timestamp to a date/time format and extract the year, separate
# the movie release year from the movie's title, store the difference between
# the two as the movie's age when rated, then split the genres for each movie
# into their individual form for both the edx_train and edx_test datasets, 
# copying all data into a new data frame for each.
edx_train_m7 <- edx_train %>% mutate(year_rated = year(as_datetime(timestamp)),
                                     title_without_year = str_replace(title,"^(.+)\\s\\((\\d{4})\\)$","\\1__\\2" )) %>% 
  separate(title_without_year,c("title_without_year","year_released"),"__") %>%
  mutate(age_at_rating = as.numeric(year_rated) - as.numeric(year_released)) %>%
  separate_rows(genres, sep = "\\|")

edx_test_m7 <- edx_test %>% mutate(year_rated = year(as_datetime(timestamp)),
                                   title_without_year = str_replace(title,"^(.+)\\s\\((\\d{4})\\)$","\\1__\\2" )) %>% 
  separate(title_without_year,c("title_without_year","year_released"),"__") %>%
  mutate(age_at_rating = as.numeric(year_rated) - as.numeric(year_released)) %>%
  separate_rows(genres, sep = "\\|")

# Set the lambda tuning parameter sequence to be 1-20 at 0.25 intervals.
lambdas <- seq(0, 20, 0.25)

# Calculate the individual rmse values using each lambda parameter interval.
rmses_m7 <- sapply(lambdas, function(l){
  mu_edx_train <- mean(edx_train_m7$rating)
  b_i <- edx_train_m7 %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_edx_train)/(n() + l))
  b_u <- edx_train_m7 %>% 
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu_edx_train - b_i)/(n() + l))
  b_a <- edx_train_m7 %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(age_at_rating) %>% 
    summarize(b_a = sum(rating - mu_edx_train - b_i - b_u)/(n() + l))
  b_g <- edx_train_m7 %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_a, by = "age_at_rating") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu_edx_train - b_i - b_u - b_a)/(n() + l))
  predicted_ratings_m7 <- edx_test_m7 %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_a, by = "age_at_rating") %>%
    left_join(b_g, by = "genres") %>%
    mutate(predict_m7 = mu_edx_train + b_i + b_u + b_a + b_g) %>%
    pull(predict_m7)
  return(RMSE(predicted_ratings_m7, edx_test_m7$rating))
})

# Find the optimal lambda value that had the best (lowest) RMSE
model7_opt_lambda <- lambdas[which.min(rmses_m7)]
model7_opt_lambda

# Add the lambda and RMSE values to a data frame to be plotted
rmses_m7_plot <- data.frame(lambdas, rmses_m7)
# Plot the Cross-Validation results
rmses_m7_plot %>% ggplot(aes(lambdas, rmses_m7)) +
  geom_point(alpha=0.6, color="dodgerblue4", size = 3) +
  geom_hline(aes(yintercept=min(rmses_m7)), color="orange", linetype='dashed', size=1) +
  geom_vline(aes(xintercept=model7_opt_lambda), color="orange", linetype='dashed', size=1) +
  labs(x=expression(paste(lambda, ' Values')), 
       y="Calculated RMSE", 
       title="Cross-Validation",
       subtitle="(RMSE by lambda value)",
       caption = "Source Data: edx_train\nFigure 5.7.1") +
  theme(panel.border = element_rect(color="black", fill=NA)) 

# The cross-validation plot in Figure 5.7.1 shows the intersection points between the 
# lowest RMSE and the lambda value used when calculating it.  The remaining portion 
# of the regularization algorithm can now be run with the chosen lambda value. 	
#
# Use the optimal lambda value to calculate the "Movie + User + Age + Genre
# Effect" predicted ratings.

# Use the optimal lambda value to calculate the regularized movie bias
b_i <- edx_train_m7 %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_edx_train)/(n() + model7_opt_lambda))
# Use the optimal lambda value to calculate the regularized user bias
b_u <- edx_train_m7 %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_edx_train - b_i)/(n() + model7_opt_lambda))
# Use the optimal lambda value to calculate the regularized movie age at rating bias
b_a <- edx_train_m7 %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(age_at_rating) %>%
  summarize(b_a = sum(rating - mu_edx_train - b_i - b_u)/(n() + model7_opt_lambda))
# Use the optimal lambda value to calculate the regularized movie genre bias
b_g <- edx_train_m7 %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_a, by = "age_at_rating") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu_edx_train - b_i - b_u - b_a)/(n() + model7_opt_lambda))
# Predict the ratings against the edx_test_m7 dataset.
predicted_ratings_m7 <- edx_test_m7 %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_a, by = "age_at_rating") %>%
  left_join(b_g, by = "genres") %>%
  mutate(predict_m7 = mu_edx_train + b_i + b_u + b_a + b_g) %>%
  pull(predict_m7)

# Test the Regularized "Movie + User + Age + Genre Effect" model by calculating   
# the RMSE using the edx_test_m7 dataset.
model7_rmse <- RMSE(predicted_ratings_m7, edx_test_m7$rating)
model7_rmse

# Create RMSE Results Table for Models 1-7
model_table_titles <- c(model_table_titles, "Model 7: Regularized Movie + User + Age + Genre Effect")
model_table_rmses <- c(model_table_rmses, model7_rmse)
kable(tibble(model_table_titles, model_table_rmses),
      col.names = c("Model", "RMSE")) %>%
  row_spec(0,background="#104E8B", color="white") %>% 
  column_spec(2, bold=TRUE) %>% 
  kable_styling(bootstrap_options="bordered", 
                full_width=FALSE, 
                position="center",
                latex_options="HOLD_position")

# Taking this approach of regularizing all of the tested biases together and lumping them
# into one model only has a small incremental decrease in the calculated RMSE for each bias 
# added.  Continually adding to the complexity of the model can potentially introduce the 
# issue of over-fitting, though these results could also have been anticipated based on 
# Figures 5.2.1, 5.3.1, 5.4.1, and 5.5.1 (grouped together) that all chart the effect 
# their respective bias has on the ratings distribution.  Adding both the movie's age 
# effect and genre effect to the originally regularized "Movie + User Effect" only 
# lowered the RMSE a small amount.	
#
# Set the four bias bar charts in a grid pattern.
grid.arrange(chart_movie_bias, chart_user_bias, chart_age_bias, chart_genre_bias, nrow = 2)

#######################################################################################
# 3h.) MODEL 8: Using Matrix Factorization
#######################################################################################
#
# One popular technique to use for recommender systems is matrix factorization.  The idea 
# is to approximate the whole rating matrix by the product of two matrices-- in this case,
# each user gets a row and each movie gets a column:	
# 	
# The previous models used did not take into account any similar ratings patterns of the 
# users or movies.  Matrix factorization observes these patterns by factorizing the 
# residuals (r) into vectors p (unrelated user effects) and q (principal components).  
# 	
# This model will use the "recosystem" package (Recommender System using Matrix 
# Factorization), which is simply an R wrapper of the LIBMF library that uses a fast 
# parallel stochastic gradient method for matrix factorization.
# 	
# The use of the `recosystem` package in this model consists of the following steps:	
# 1. Create a `recosystem`-formatted copy of both the `edx` and `validation` datasets, passing in only the 	
#    `userId`, `movieId` and `rating` columns.	
# 2. Create a model object (a Reference Class object in R) by calling `Reco()`.  	
# 3. Call the `$tune()` method to select best tuning parameters along a set of candidate values.	
#    Note: If tune() is not specified, the default parameters will be used, and the calculation	
#          period will be exponentially longer.  Cross validation will be used to tune these parameters,	
#          and will be chosen by minimizing the RMSE as a loss function.	
# 4. Train the model by calling the `$train()` method, calling in a number of parameters that will 	
#    come from the result of calling `$tune()`.	
# 5. Use the `$predict()` method against the validation dataset to compute the predicted values.	
# 6. Search the resultant data frame for any predictions that are higher than 5, and lower than 0.5 and	
#    set those rows to 5 and 0.5, respectively.	
# 7. Calculate the RMSE value.	
# 	
# Create a Recosystem-formatted training dataset from the in-memory edx_train dataset.
edx_train_mf <- with(edx_train, data_memory(user_index = userId,
                                            item_index = movieId, 
                                            rating = rating))

# Create a Recosystem-formatted testing dataset from the in-memory edx_test dataset.
edx_test_mf <- with(edx_test, data_memory(user_index = userId, 
                                          item_index = movieId,
                                          rating = rating))
# Create the model object
r <- Reco()

# Cross validation (k=5 by default) of Recosystem-formatted edx_train training dataset  
# to compute the residuals from a set of tuning parameters.  If the "tune_opts" list is  
# left blank, all defaults will be used, which can take 8 or more hours to run.  The 
# first time this was run, all defaults were used and the below values were determined
# to be optimal parameters.
tune_opts <- r$tune(edx_train_mf,
                    opts = list(dim = 30, # Number of latent factors
                                lrate = 0.1, # Learning rate, or step size in gradient descent
                                costp_l1 = 0, # L1 regularization parameter for user factors
                                costp_l2 = 0.01, # L2 regularization parameter for user factors
                                costq_l1 = 0, # L1 regularization parameter for item factors
                                costq_l2 = 0.1, # L2 regularization parameter for item factors
                                nthread  = 6, # Number of parallel computing threads
                                niter = 20)) # Number of iterations

# Train the algorithm with the recosystem-formatted edx_train dataset, using the optimal
# tuning parameters.
r$train(edx_train_mf, opts = c(tune_opts$min, nthread = 6, niter = 20))

# Run the recosystem predict function against the edx_test dataset.
y_hat_ratings <-  r$predict(edx_test_mf, out_memory())

# Find which rows have rating predictions over the max value of 5 (the highest rating).
over5 <- which(y_hat_ratings > 5) 
# Set all discovered rows with ratings above 5 to max rating value of 5.
y_hat_ratings[over5] <- 5

# Find which rows have rating predictions under the min value of 0.5 (the lowest rating).
under0_5 <- which(y_hat_ratings < 0.5) 
# Set all discovered rows with ratings under 0.5 to minimum rating value of 0.5.
y_hat_ratings[under0_5] <- 0.5

# Test the Matrix Factorization Model's RMSE value
model8_rmse <- RMSE(edx_test$rating, y_hat_ratings)
model8_rmse

# To aid the model, knowing there cannot be any rating above 5 or below 0.5, any 
# predictions that fit that profile are adjusted to the highest possible value of 5 
# or lowest of 0.5, respectively.	
# 	
# Create RMSE Results Table for Models 1-8
model_table_titles <- c(model_table_titles, "Model 8: Matrix Factorization")
model_table_rmses <- c(model_table_rmses, model8_rmse)
kable(tibble(model_table_titles, model_table_rmses),
      col.names = c("Model", "RMSE")) %>%
  row_spec(0,background="#104E8B", color="white") %>% 
  column_spec(2, bold=TRUE) %>% 
  kable_styling(bootstrap_options="bordered", 
                full_width=FALSE, 
                position="center",
                latex_options="HOLD_position")

# Based on the substantial drop in the calculated RMSE, observing similarities in the 
# rating pattern of users to movies has a significant impact on the predictability of 
# a user's movie preference (by their rating).  	
#
# The `niter` property (number of training iterations on the `edx_train_mf` dataset) 
# in the recosystem tuning parameters was additionally tested with the value of 40 
# to see what would happen, and the overall RMSE lowered even further to 0.78429.  
# Though, simply adding additional iterations can quickly add the issue of "over-smoothing".	

#######################################################################################
# 4.) RESULTS #########################################################################
#######################################################################################
#
# Having done all of the initial data analysis in section 4 and pinpointing which 
# factors may contribute to (or at least warrant a model to test) an effective movie 
# recommendation system, stepping through eight (8) different models yielded the best 
# results with the use of matrix factorization.  While testing the various performance 
# tuning options of the recosystem package did result in a few extended compute-intensive 
# timeframes (up to 12 hours on an 8-core hyperthreaded processor with 64GB RAM), the
# final tests with the optimal parameters used in this report only took around 3 minutes
# to complete from start to finish.  In contrast, Model 7 (Regularization of The Movie + 
# User + Age + Genre Effect) consistently takes between 30-45 minutes to complete from 
# start to finish because the number of computations compound for each additional effect.
#
# Create Final RMSE Results Table for all models
kable(tibble(model_table_titles, model_table_rmses),
      col.names = c("Model", "RMSE")) %>%
  row_spec(0,background="#104E8B", color="white") %>% 
  column_spec(2, bold=TRUE) %>% 
  kable_styling(bootstrap_options="bordered", 
                full_width=FALSE, 
                position="center",
                latex_options="HOLD_position")

# The final hold-out testing of this model using the initial `edx` and `validation` 
# datasets yielded the following final RMSE:	
# 	
# Create a Recosystem-formatted training dataset from the in-memory edx dataset.
edx_mf <- with(edx, data_memory(user_index = userId,
                                item_index = movieId, 
                                rating = rating))

# Create a Recosystem-formatted testing dataset from the in-memory validation dataset.
validation_mf <- with(validation, data_memory(user_index = userId,
                                              item_index = movieId,
                                              rating = rating))
# Create the model object
r <- Reco()

# Cross validation (k=5 by default) of Recosystem-formatted edx training dataset to 
# compute the residuals from a set of tuning parameters.  If the "tune_opts" list is  
# left blank, all defaults will be used, which can take 8 or more hours to run.  The 
# first time this was run, all defaults were used and the below values were determined
# to be optimal parameters.
tune_opts <- r$tune(edx_mf,
                    opts = list(dim = 30, # Number of latent factors
                                lrate = 0.1, # Learning rate, or step size in gradient descent
                                costp_l1 = 0, # L1 regularization parameter for user factors
                                costp_l2 = 0.01, # L2 regularization parameter for user factors
                                costq_l1 = 0, # L1 regularization parameter for item factors
                                costq_l2 = 0.1, # L2 regularization parameter for item factors
                                nthread  = 6, # Number of parallel computing threads
                                niter = 20)) # Number of iterations

# Train the algorithm with the recosystem-formatted edx dataset, using the optimal
# tuning parameters.
r$train(edx_mf, opts = c(tune_opts$min, nthread = 6, niter = 20))

# Run the recosystem predict function against the validation test dataset.
y_hat_ratings <-  r$predict(validation_mf, out_memory())

# Find which rows have rating predictions over the max value of 5 (the highest rating).
over5 <- which(y_hat_ratings > 5) 
# Set all discovered rows with ratings above 5 to max rating value of 5.
y_hat_ratings[over5] <- 5

# Find which rows have rating predictions under the min value of 0.5 (the lowest rating).
under0_5 <- which(y_hat_ratings < 0.5) 
# Set all discovered rows with ratings under 0.5 to minimum rating value of 0.5.
y_hat_ratings[under0_5] <- 0.5

# Calculate the Matrix Factorization Model's RMSE value
final_model_rmse <- RMSE(validation$rating, y_hat_ratings)
final_model_rmse

# Create Final Model RMSE Results Table
kable(tibble("FINAL MODEL: Matrix Factorization", final_model_rmse),
      col.names = c("Model", "RMSE")) %>%
  row_spec(0,background="#104E8B", color="white") %>% 
  column_spec(1, bold=TRUE) %>% 
  column_spec(2, color="red", bold=TRUE) %>% 
  kable_styling(bootstrap_options="bordered", 
                full_width=FALSE, 
                position="center",
                latex_options="HOLD_position")

#######################################################################################
# 5.) CONCLUSION ######################################################################
#######################################################################################
#
# The purpose of this project was to create a movie recommendation system using the 
# MovieLens 10M dataset, with the goal of obtaining a Root Mean Square Error (RMSE) value
# that is less than 0.86490.  	
# 	
# Interestingly, the model with the lowest RMSE value only used the `movieId` and `userId`
# features.  While adding considerations for movie, user, movie age at rating, and movie 
# genre biases did improve the ratings prediction (having achieved the target RMSE in 
# earlier models), matrix factorization was most effective in discovering the patterns
# that exist between the users and movies in the sparse dataset.  The large MovieLens 
# dataset used in this project aided in preventing lesser-known/rated movies from skewing
# the RMSE results, which is one of the reasons that regularization did not provide a 
# significant improvement in the Regularized Movie + User model.
#
# After testing 8 different predictive models, using their respective RMSE values as the
# success indicator for each, the Matrix Factorization model (using the `recosystem` 
# package) yielded the lowest value.
