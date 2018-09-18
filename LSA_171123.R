#=========================================================#
# Load Package
#=========================================================#

packages = c("Rfacebook", "tm", "lsa", "wordcloud","ggplot2","KoNLP",
             "GPArotation","cluster","RWeka","ROAuth","fpc","stringr","ape","devtools")

for (i in packages){
  if(!require( i , character.only = TRUE))
  {install.packages(i, dependencies = TRUE)}
}

#=========================================================#
# Setting
#=========================================================#

setwd("D://Facebook_bamboo_project")

pdf.options(family="Korea1deb") #not to tear down the letters
options(java.parameters=c("-Xmx8g","-Dfile.encoding=UTF-8")) #to increse heap size of rjava
pal <- brewer.pal(9,"Set1")
options(mc.cores=1)

useNIADic()

#=========================================================#
# Crawling Data
#=========================================================#

load("image1029.RData")

#=========================================================#
# Preprocessing
#=========================================================#

# 1. remove pucntuation: punctuation raise error

file1 <- unlist(file1)
file1 <- gsub("[[:punct:]]"," ",file1)

# 2. remove redundant whitespace

file1 <- gsub("\\s+"," ", file1)

# 3. extract noun

words <- function(doc){
  
  doc <- as.character(doc)
  
  doc2 <- paste(SimplePos22(doc))
  
  doc3 <- str_match(doc2, "([가-??+)/NC")
  
    if( dim(doc3)[2] == 2){
      doc4 <- doc3[,2]
      doc4 <- doc4[!is.na(doc4)]
      return(doc4)
      }
 }
 
nouns = sapply(file1,words, USE.NAMES = F)
 
txt_noun1 <- nouns

# 4. remove stopwords

words<-read.csv("2nd_stopwords.txt",header=FALSE)
words <- as.character(words$V1)

#=========================================================#
# Word Embedding
#=========================================================#

corpus <- Corpus(VectorSource(txt_noun1)) #명사??벡터???부??
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, words)

#=========================================================#
# LSA
#=========================================================#

# 1. Document-Term Matrix

uniTokenizer <- function(x) unlist(strsplit(as.character(x), "[[:space:]]+"))
control = list(tokenize = uniTokenizer,
               removeNumbers = TRUE,
               wordLengths=c(2,20),
               removePunctuation = TRUE,
               stopwords = c("\\n","\n","것","c"),
               weighting = function (x) {weightTfIdf(x, TRUE)})

dtm <- DocumentTermMatrix(corpus, control=control) #invert to dtm using tokenizer
object.size(dtm)

Encoding(dtm$dimnames$Terms) ='UTF-8' #fucking encoding...ha...

findFreqTerms(dtm,lowfreq = 100)

# 2. Decrease Sparsity

dt <- removeSparseTerms(dtm,0.999) # decrease matirx's sparsity using frequency
as.numeric(object.size(dt)/object.size(dtm)) * 100 # ratio of reduced/ original

dt$dimnames$Terms[1:10]

# 3. Visualize Words (Word Cloud) 

library(slam) # for apply function on DocumentTermMatrix

check<-which(row_sums(td>1)) # Once we make reduced matrix, maybe there is empty row. so we need to get rid of them
dt<-(dt[check,])
colTotal<-apply(dt,1,sum)
which(colTotal<=0) # there is no empty row, you need to identify '0'
findFreqTerms(dt, lowfreq=5)

TermFreq<-colSums(as.matrix(dt))
TermFreq2 <-subset(TermFreq,TermFreq>4)
gframe<-data.frame(term=names(TermFreq2),freq=TermFreq2)
ggplot(data=gframe)+aes(x=term,y=freq)+geom_bar(stat="identity")+coord_flip()
wordcloud(names(TermFreq2),TermFreq2,random.color=TRUE,colors=pal) # WOW ~

# 4. Model(LSA)

LSA <-lsa(dt,dim=5) # dimensions=5
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

# 6. Model(LDA)

#install_github("cpsievert/LDAvisData")
data(reviews, package = "LDAvisData")

#install.packages("lda")
#install.packages("LDAvis")
library(lda)
library(LDAvis)

doc.list <- txt_noun1[1:1000]

term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)

del <- names(term.table) %in% words | term.table < 5
term.table <- term.table[!del]
vocab <- names(term.table)

get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}

documents <- lapply(doc.list, get.terms)

D <- length(documents)  # number of documents (2,000)
W <- length(vocab)  # number of terms in the vocab (14,568)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [312, 288, 170, 436, 291, ...]
N <- sum(doc.length)  # total number of tokens in the data (546,827)
term.frequency <- as.integer(term.table)  # frequencies of terms in the corpus [8939, 5544, 2411, 2410, 2143, ...]

K <- 5
G <- 5000
alpha <- 0.02
eta <- 0.02

library(lda)

set.seed(357)
t1<-Sys.time()

# LDA
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab=vocab,
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
t2-t1

theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

Bamboo_posts <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)

options(encoding = 'UTF-8') #fucking encoding....ha....

library(LDAvis)

# 7. Visualize Result(LDA)
# create the JSON object to feed the visualization:

json <- createJSON(phi = Bamboo_posts$phi, 
                   theta = Bamboo_posts$theta, 
                   doc.length = Bamboo_posts$doc.length, 
                   vocab = Bamboo_posts$vocab, 
                   term.frequency = Bamboo_posts$term.frequency, encoding='UTF-8')

#install.packages("servr")
library(servr)

serVis(json, out.dir = 'vis', open.browser = TRUE)

###################################################################
# End!!! DO SAVE !!!

save.image(file = "LSA_LDA_180917.RData")

###################################################################
