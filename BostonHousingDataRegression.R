#Boston Housing Data (regression) - mlbench


# Boston Housing Data
data(BostonHousing)
dim(BostonHousing)
head(BostonHousing)

# target variable is medv

library(caret)

anyNA(BostonHousing)

inTrain <- createDataPartition(BostonHousing$medv, p = 0.8, list = FALSE)
train <- BostonHousing[inTrain, ]
test <- BostonHousing[-inTrain, ]

summary(train)

# regression 
ctrl<-trainControl(method = 'cv',number = 10)
lr <- train(medv ~., data = train, preProcess = c( "center", "scale"), method = "lm" , trControl = ctrl)
lr_pred <- predict(lr, test)
modelvalues<-data.frame(obs = test$medv, pred=lr_pred)
defaultSummary(modelvalues)
plot(test$medv, lr_pred)

# svmLinear
svm_Linear <- train(medv ~., data = train, method = "svmLinear",
                    tuneLength = 10)
svm_pred <- predict(svm_Linear, newdata = test)
modelvalues<-data.frame(obs = test$medv, pred=svm_pred)
defaultSummary(modelvalues)
plot(test$medv, svm_pred)

# svmRadial
svm_r <- train(medv ~., data = train, method = "svmRadial",
                    tuneLength = 10)
svm_pred <- predict(svm_r, newdata = test)
modelvalues<-data.frame(obs = test$medv, pred=svm_pred)
defaultSummary(modelvalues)
plot(test$medv, svm_pred)

