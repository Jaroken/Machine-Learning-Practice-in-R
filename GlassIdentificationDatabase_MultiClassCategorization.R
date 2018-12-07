
# Glass Identification Database
data(Glass)
dim(Glass)
levels(Glass$Type)
head(Glass)

library(caret)

inTrain <- createDataPartition(Glass$Type, p = 0.7, list = FALSE)
train <- Glass[inTrain, ]
test <- Glass[-inTrain, ]


### svm Radial 
ctrl<-trainControl(method = 'cv',number = 10)
svmR <- train(Type ~., data = train, method = "svmRadial", trControl = ctrl,
      tuneLength = 10)
svmR
pred_test <- predict(svmR, test)

confusionMatrix(data = pred_test, reference = test$Type)

### svm linear
svmR <- train(Type ~., data = train, method = "svmLinear",
              tuneLength = 10)
svmR
pred_test <- predict(svmR, test)

confusionMatrix(data = pred_test, reference = test$Type)

### nieve bayes
nbR <- train(Type ~., data = train, method = "nb",
              tuneLength = 10)
nbR
pred_test <- predict(nbR, test)

confusionMatrix(data = pred_test, reference = test$Type)

### Latent Drichlet Allocation
ldaR <- train(Type ~., data = train, method = "lda",
             tuneLength = 10)
ldaR
pred_test <- predict(ldaR, test)

confusionMatrix(data = pred_test, reference = test$Type)


### rf
rfR <- train(Type ~., data = train, method = "rf",
              tuneLength = 10)
rfR
pred_test <- predict(rfR, test)

confusionMatrix(data = pred_test, reference = test$Type)


### nnet
nnetR <- train(Type ~., data = train, method = "nnet",
             tuneLength = 10)
nnetR
pred_test <- predict(nnetR, test)

confusionMatrix(data = pred_test, reference = test$Type)
