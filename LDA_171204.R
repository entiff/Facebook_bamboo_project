#LDA
library(lda)
library(topicmodels)
ldaform <- dtm2ldaformat(td, omit_empty=TRUE)

result.lda <- lda.collapsed.gibbs.sampler(documents = ldaform$documents,
                                          K = 30, vocab = ldaform$vocab,
                                          num.iterations = 5000, burnin = 1000,
                                          alpha = 0.01, eta = 0.01)
#num.iterations posteria number of update
#burnin = burning of first value
#alpha = probability of topics in document / 1 = uniform
#eta = parameter setting = probability of words in one topic

attributes(result.lda)
dim(result.lda$topics)
result.lda$topics
top.topic.words(result.lda$topics)
result.lda$topic_sums #how many words 

#######################Visualization############################

theta <- t(apply(result.lda$topic_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(result.lda$topics) + eta, 2, function(x) x/sum(x)))

MovieReviews <- list(phi = phi,
                     theta = theta,
                     doc.length = 85432,
                     vocab = ldaform$vocab,
                     term.frequency = TermFreq)

options(encoding = 'UTF-8') #한글로 결과 보기

library(LDAvis)

# create the JSON object to feed the visualization:
json <- createJSON(phi = MovieReviews$phi, 
                   theta = MovieReviews$theta, 
                   doc.length = MovieReviews$doc.length, 
                   vocab = MovieReviews$vocab, 
                   term.frequency = MovieReviews$term.frequency, encoding='UTF-8')
#install.packages("servr")
library(servr)

serVis(json, out.dir = 'vis', open.browser = TRUE)

###################################################################

save.image(file = "LSALDA_171204.RData")

###################################################################