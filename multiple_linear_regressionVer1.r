# Multiple Linear Regression
# 1. Clean Environment & Load Libraries
library(caret)
library(caTools)
library(dplyr)

# 2. Source, Collect and Load Data
## 2.1 Source and Collect Data

#The data is obatained from where?

## 2.2 Load Data
# Importing the dataset

dataset = read.csv("~/IIT Semester One/Applied Statistics/Project final/50_Startups.csv")


plot(dataset)
# 3. Process and Clean Data
# Explore missing value patterns
# Check if there are NA values
NANumbers <- sum(is.na(dataset)) 
paste("**** The number of NA values in this data set =", NANumbers)
# Plotting percentage of missing values per feature
library(naniar)
gg_miss_var(dataset, show_pct = TRUE)
# We don?t have missing values but we drop na values just to check

## 3.3 Data parsing (Set dummy variables)
# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))
dataset$State=as.integer(dataset$State)
# Display new data frame
head(dataset)


## 3.4 Data histograms
par(mfrow=c(1,1))
hist(dataset$R.D.Spend, col = "wheat",main = "Research and Development Spending",xlab = "R&D Spending")
hist(dataset$Administration, col = "skyblue",main = "Administration",xlab = "Administration Cost")
hist(dataset$Marketing.Spend, col = "green", main = "Marketing Spend",xlab = "Marketing Spending")
hist(dataset$State, col = "coral2",main = "State",xlab = "State")

## 3.5 Checking & Treating Outliers
boxplot(dataset$R.D.Spend,
        ylab = "R.D.Spend")
boxplot(dataset$Administration,
        ylab = "Administration Cost")
boxplot.stats(dataset$Administration)$out
boxplot(dataset$Marketing.Spend,
        ylab = "Marketing Spending")
boxplot(dataset$State,
        ylab = "State")

#It can be noted that from the boxplots there are no outliers from the dataset.

## 3.6 Data Correlation Analysis
library(corrplot)
corrplot(cor(dataset), addCoef.col = 'black', method="color")

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')

#4.0 Data Analysis Stage

## 4.1 Training/Testing Set Split

set.seed(123)
training.samples=dataset$Profit%>%
  createDataPartition(p=.8, list =FALSE)
training_set=dataset[training.samples,]
test_set=dataset[-training.samples,]



# training_set = scale(training_set)
# test_set = scale(test_set)

# 4.2 Model Fitting Multiple Linear Regression to the Training set
regressor = lm(formula = Profit ~ .,
               data = training_set)
summary(regressor)
predy = predict(regressor, newdata = test_set)
mse <- mse(training_set$Profit,predy)
cat("MSE" , mse, "\n")
coef(regressor)
confint(regressor)

## 4.3 Preparing X and Y vectors for lasso regression

x_train <- model.matrix(training_set$Profit~., data = training_set)[, -1]
y_train <- training_set$Profit

## 4.4 Displaying how the coefficients vary with lambda

library(glmnet)
lasso_model <- glmnet(x_train, y_train, alpha = 1)
plot(lasso_model, "lambda")
plot(lasso_model, "norm")
set.seed(4)
grid <- 10^seq(2, -2, length = 100)
lasso_cv_model <- cv.glmnet(x_train, y_train, alpha = 1, lambda = grid)
plot(lasso_cv_model)

best_lambda <- lasso_cv_model$lambda.min
cat("Best Lambda: ", best_lambda, "\n")
coef(lasso_cv_model)

## 4.5 Performance on test data


library(Metrics)
x_test <- model.matrix(test_set$Profit~. , data = test_set)[, -1]
lasso_pred <- predict(lasso_cv_model, s = best_lambda, newx = x_test)
test_mse <- mse(test_set$Profit, lasso_pred)
cat("Test MSE" , test_mse, "\n")

# The resulting model dhave coefficients for all predictor variables.All the coefficients for other predictors got smaller with the except Intercept. This was expected as lasso performs feature selection through shrinking irrelevant coefficients to 0

# 5.4 Model Accuracy
## 5.4.1 Make predictions
predictions <- regressor %>% predict(test_set)

## 5.4.2 Model performance

# (a) Compute the prediction error, RMSE

RMSE(predictions, test_set$Profit)

# (b) Compute R-square

R2(predictions, test_set$Profit)

## 5.4.3  Conclusion

#From the output above, the R2 is 0.98, meaning that the observed and the predicted outcome values are highly correlated, which is very good.

Error_rate= 9534.568/mean(test_set$Profit)
#The prediction error RMSE is 9534.568, representing an error rate of 9534.568/mean(test_set$Profit)  = 8.3%, which is good.
