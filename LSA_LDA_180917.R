#=========================================================#
# Load Package
#=========================================================#

packages = c("Rfacebook", "tm", "lsa", "wordcloud","ggplot2","KoNLP",
             "GPArotation","cluster","RWeka","ROAuth","fpc","stringr","ape","devtools")

for (i in packages){
  if(!require( i , character.only = TRUE))
  {install.packages(i, dependencies = TRUE)}
}

#devtools::install_github('haven-jeon/KoSpacing')
library(KoSpacing) #if you interested in this package, visit here (https://github.com/haven-jeon/KoSpacing)

spacing("제발이것좀띄어쓰기해주세요왜이렇게띄어쓰기를제대로안하는사람이많아요보아즈여러분들은아닐거라고믿어요글쓸때는띄어쓰기꼭합시다")

# if you have trouble with installing, check the version of java and reinstall

#=========================================================#
# Setting
#=========================================================#

setwd("D://Facebook_bamboo_project/")

pdf.options(family="Korea1deb") #not to tear down the letters
options(java.parameters=c("-Xmx8g","-Dfile.encoding=UTF-8")) #to increse heap size of rjava
pal <- brewer.pal(9,"Set1")
options(mc.cores=1)

useNIADic()

#=========================================================#
# Crawling Data
#=========================================================#

Posts <- read.csv("Bamboo_posts_17v.csv",header = TRUE)
Posts <- data.frame(Posts$Sepped.v2)
Posts <- as.data.frame(Posts[1:1000,])

#=========================================================#
# Preprocessing
#=========================================================#

# 1. remove pucntuation: punctuation raise error

posts_v1 <- as.character(Posts$`Posts[1:1000, ]`)
posts_v1 <- gsub("[[:punct:]]","",posts_v1)

# [:punct:] => [][!"#$%&'()*+,./:;<=>?@\^_`{|}~-]
# I recommend you should play with regular expression if you interested in NLP -> https://regexr.com/

# 2. remove redundant whitespace

posts_v1 <- gsub("\\s+"," ", posts_v1)

# 3. spacing

posts_v1 <- spacing(posts_v1) #more than 200 charaters, it doesn't working

# 4. extract noun

ExtractWord <- function(doc){
  
  doc <- as.character(doc)
  
  doc2 <- paste(SimplePos22(doc))
  
  doc3 <- str_match(doc2, "([가-힣]+)/NC")
  
  if(dim(doc3)[2] == 2){
    doc4 <- doc3[,2]
    doc4 <- doc4[!is.na(doc4)]
    return(doc4)
  }
}

#To know about the simplepos22 -> https://brunch.co.kr/@mapthecity/9 

nouns = sapply(posts_v1, words, USE.NAMES = F)
txt_noun1 <- nouns

# 4. remove stopwords

words <- read.csv("2nd_stopwords.txt",header=FALSE)
words <- as.character(words$V1)

#=========================================================#
# Word Embedding
#=========================================================#

corpus <- Corpus(VectorSource(nouns)) # assign vector value
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, words)

#=========================================================#
# LSA
#=========================================================#

# 1. Document-Term Matrix

uniTokenizer <- function(x) unlist(strsplit(as.character(x), "[[:space:]]+"))

control = list(tokenize = uniTokenizer,
               wordLengths=c(2,20),
               stopwords = c("\\n","\n","것","c"),
               weighting = function (x) {weightTfIdf(x, TRUE)})

dtm <- DocumentTermMatrix(corpus, control=control) #invert to dtm using tokenizer
#dtm <- DocumentTermMatrix(corpus,control=list(tokenize=extractNoun))

object.size(dtm)

Encoding(dtm$dimnames$Terms) = "UTF-8" #fucking encoding...ha...
#Encoding(rownames(dtm)) <- "UTF-8"

findFreqTerms(dtm,lowfreq = 5)
# 2. Decrease Sparsity

td <- removeSparseTerms(dtm,0.9999) # decrease matirx's sparsity using frequency
as.numeric(object.size(dt)/object.size(dtm)) * 100 # ratio of reduced/ original

td$dimnames$Terms[1:10]

# 3. Visualize Words (Word Cloud) 

library(slam) # for apply function on DocumentTermMatrix

check <- which(row_sums(as.matrix(td))>=1) # Once we make reduced matrix, maybe there is empty row. so we need to get rid of them
td<-(td[check,])
colTotal<-apply(td,1,sum)
which(colTotal<=0) # there is no empty row, you need to identify '0'
findFreqTerms(td, lowfreq=5)

TermFreq<-colSums(as.matrix(td))
TermFreq2 <-subset(TermFreq,TermFreq>4)
gframe<-data.frame(term=names(TermFreq2),freq=TermFreq2)
ggplot(data=gframe)+aes(x=term,y=freq)+geom_bar(stat="identity")+coord_flip()
wordcloud(names(TermFreq2),TermFreq2,random.color=TRUE,colors=pal) # WOW ~

# 4. Model(LSA)

LSA <-lsa(td,dim=5) # dimensions=5
st <-LSA$tk
wd <- LSA$dk
strength <- LSA$sk

rot <- GPForth(wd, Tmat=diag(ncol(wd)), normalize=FALSE, eps=1e-5,
               maxit=10000, method="varimax",methodArgs=NULL) #After lsa, we get 3 matrixes, and we use 'dk' matrix for varimax rotation

cord <- st %*% diag(strength) %*% rot$Th # after rotation, merge

signs <- sign(colSums(rot$loadings)) # we can get how much each word explain each dimension

cord<-cord %*% diag(signs)

text_lsa<-data.frame(cord=cord,file=file1[check])

# 5. Visualize Result(LSA)

showmedim <- function(dimen){
  t<-rot$loadings[,dimen]
  tt<-abs(t)
  terms<-names(tt)
  wordcloud(terms,tt,scale=c(4,1),rot.per=0,max.words=50, colors=pal)
}

showmedim(1)
showmedim(2)
showmedim(3)
showmedim(4)
showmedim(5)

#=========================================================#
# LDA
#=========================================================#

# 1. Model(LDA)

#install.packages("lda")
#install.packages("topicmodels")

library(lda)
library(topicmodels)
ldaform <- dtm2ldaformat(td, omit_empty=TRUE)

result.lda <- lda.collapsed.gibbs.sampler(documents = ldaform$documents,
                                          K = 30, vocab = ldaform$vocab,
                                          num.iterations = 5000, burnin = 1000,
                                          alpha = 0.01, eta = 0.01)

#=====================================================================#
# Cf. parameters & options                                            #
#                                                                     #
#     1. num.iterations => posteria number of update                  #
#     2. burnin = burning of first value                              #
#     3. alpha = probability of topics in document / 1 = uniform      #
#     4. eta = parameter setting = probability of words in one topic  #
#                                                                     #
#=====================================================================#

attributes(result.lda)
dim(result.lda$topics)
result.lda$topics
top.topic.words(result.lda$topics)
result.lda$topic_sums #how many words 

# 2. Visualization

alpha = 0.01
eta = 0.01

theta <- t(apply(result.lda$topic_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(result.lda$topics) + eta, 2, function(x) x/sum(x)))

Bamboo_topics <- list(phi = phi,
                     theta = theta,
                     doc.length = 85432,
                     vocab = ldaform$vocab,
                     term.frequency = TermFreq)

options(encoding = 'UTF-8') #fucking encoding ... ha...

#install.packages("LDAvis")
library(LDAvis)

# create the JSON object to feed the visualization

json <- createJSON(phi = MovieReviews$phi, 
                   theta = MovieReviews$theta, 
                   doc.length = MovieReviews$doc.length, 
                   vocab = MovieReviews$vocab, 
                   term.frequency = MovieReviews$term.frequency, encoding='UTF-8')

#install.packages("servr")
library(servr)

serVis(json, out.dir = 'vis', open.browser = TRUE) # WOW!

############################################################
#################### End!!! DO SAVE !!! ####################
############################################################

# save.image(file = "LSA_LDA_180917.RData")

############################################################