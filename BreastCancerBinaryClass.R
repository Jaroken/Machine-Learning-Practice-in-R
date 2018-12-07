# Wisconsin Breast Cancer Database (binary classification) - mlbench
library(mlbench)

'
Description: Predict whether a cancer is malignant or benign from biopsy details.
Type: Binary Classification
Dimensions: 699 instances, 11 attributes
Inputs: Integer (Nominal)
Output: Categorical, 2 class labels
'

# Wisconsin Breast Cancer Database
data(BreastCancer)
dim(BreastCancer)
levels(BreastCancer$Class)
head(BreastCancer)

library(caret)

inTrain <- createDataPartition(BreastCancer$Class, p = 0.7, list = FALSE)
train <- BreastCancer[inTrain,]
test <- BreastCancer[-inTrain, ]

str(train)
table(train$Class)
train$Id <- NULL

anyNA(train)
sum(is.na(train$Bare.nuclei))
train <- train[complete.cases(train),]
test$Id <- NULL
test <- test[complete.cases(test),]
set.seed(3233)

# svmLinear
start_t <- Sys.time()
svm_Linear <- train(Class ~., data = train, method = "svmLinear",
                    tuneLength = 10)
test_pred <- predict(svm_Linear, newdata = test)
end_t <- Sys.time()

svm_r <- list(cmatrix = confusionMatrix(test_pred, test$Class), time = end_t- start_t)

# random forest
start_t <- Sys.time()
rf <- train(Class ~., data = train, method = 'rf',
                  tuneLength = 10)
test_pred <- predict(rf, newdata = test)
end_t <- Sys.time()

rf_r <- list(cmatrix = confusionMatrix(test_pred, test$Class), time = end_t- start_t)

# Linear Discriminant Analysis.
start_t <- Sys.time()
ldam <- train(Class ~., data = train, method = 'lda',
            tuneLength = 10)
test_pred <- predict(ldam, newdata = test)
end_t <- Sys.time()
lda_r <- list(cmatrix = confusionMatrix(test_pred, test$Class), time = end_t- start_t)

# logit boost
start_t <- Sys.time()
logitm <- train(Class ~., data = train, method = 'LogitBoost',
              tuneLength = 10)
test_pred <- predict(logitm, newdata = test)
end_t <- Sys.time()
logit_r <- list(cmatrix = confusionMatrix(test_pred, test$Class), time = end_t- start_t)

## Logistic regression
start_t <- Sys.time()
lreg<-train(Class ~., data = train,method="glm",family=binomial())
test_pred <- predict(lreg, newdata = test)
end_t <- Sys.time()
logr_r <- list(cmatrix = confusionMatrix(test_pred, test$Class), time = end_t- start_t)

### boosted Tree
start_t <- Sys.time()
boostedt <- train(Class ~., data = train, method = 'bstTree')
test_pred <- predict(boostedt, newdata = test)
end_t <- Sys.time()
boostt_r <- list(cmatrix = confusionMatrix(test_pred, test$Class), time = end_t- start_t)


### nnet
start_t <- Sys.time()
nnetm <- train(Class ~ ., data = train, 
      method='nnet', 
      trace = FALSE)
test_pred <- predict(nnetm, newdata = test)
end_t <- Sys.time()
nnet_r <- list(cmatrix = confusionMatrix(test_pred, test$Class), time = end_t- start_t)

### naive bayes
start_t <- Sys.time()
nbr <- train(Class ~ ., data=spam,  method = "nb")
test_pred <- predict(nbr, newdata = test)
end_t <- Sys.time()
nb_r <- list(cmatrix = confusionMatrix(test_pred, test$Class), time = end_t- start_t)

### compare
svm_r$overall[2]
rf_r$overall[2]
lda_r$overall[2]
logit_r$overall[2] 
boostt_r$overall[2] 
nnet_r$overall[2]
nb_r$overall[2]

svm_r$time
rf_r$time
lda_r$time
logit_r$time 
boostt_r$time 
nnet_r$time
nb_r$time
