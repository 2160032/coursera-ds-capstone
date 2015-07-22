## text-predictor, transformation/prediction functions
##
## (c) Patrick Charles; see http://pchuck.net
## redistribute or modify under the GPL; see http://www.gnu.org/licenses
##
## developed for the Coursera/Johns Hopkins Data Science specialization
## capstone project in collaboration with SwiftKey; see http://swiftkey.com/en/
##

### transformation functions

## fix contractions 
expandContractions <- function(doc) {
    doc <- gsub("won't", "will not", doc) # a special case of "n't"
    doc <- gsub("can't", "can not", doc) # another special case of "n't"
    doc <- gsub("n't", " not", doc)
    doc <- gsub("'ll", " will", doc)
    doc <- gsub("'re", " are", doc)
    doc <- gsub("'ve", " have", doc)
    doc <- gsub("'m", " am", doc)
    doc <- gsub("it's", "it is", doc) # a special case of 's
    ##doc <- gsub("'s", "", doc) # otherwise, possessive w/ no expansion
    return(doc)
}

## custom transformation - specified texts to spaces
toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x))

## custom transformation - UTF-8 to ASCII (remove special characters)
removeSpecial <- content_transformer(function(x)
    iconv(x, "ASCII", "UTF-8", sub=""))

## given a set of texts, apply cleaning transformations
## and return a tm corpus containing the documents
##
createCleanCorpus <- function(texts, remove.punct=TRUE, remove.profanity=FALSE, profanity=NULL) {
    texts <- expandContractions(texts)
    filtered <- VCorpus(VectorSource(texts))
  
    # remove digits
    filtered <- tm_map(filtered, removeNumbers)

    # substitute slashes, @'s and pipes to spaces
    filtered <- tm_map(filtered, toSpace, "/|@|\\|")

    # remove special characters
    filtered <- tm_map(filtered, removeSpecial)

    # convert to lower case
    filtered <- tm_map(filtered, content_transformer(tolower))

    # conditionally remove punctuation
    if(remove.punct) {
        filtered <- tm_map(filtered, removePunctuation,
                           preserve_intra_word_dashes=TRUE)
    }

    # remove profanity
    if(remove.profanity) {
        filtered <- tm_map(filtered, removeWords, profanity)
    }
    
    # strip excess whitespace
    filtered <- tm_map(filtered, stripWhitespace)
}


### plotting - ngrams

## ngram plotting function  
plotGram <- function(threshold, freq, wf, type) {
    ggplot(subset(wf, freq >= wf$freq[threshold]),
           aes(reorder(word, freq), freq)) + 
        geom_bar(stat="identity") + 
        theme(axis.text.x=element_text(angle=45, hjust=1)) +
        ggtitle(paste("Most Common ", type, "s", sep="")) +
        xlab(type) + ylab("Frequency")
}


### prediction

## count the number of words in the character string provided
##
wordCount <- function(text) {
    length(unlist(strsplit(text, " ")))
}

## return a string containing the last 'n' words of text
##   text - a string of characters containing words
##   n - the number of words to extract
##
## returns a string of characters containing the last 'n' words
##
lastWords <- function(text, n) {
    paste(tail(unlist(strsplit(text, " ")), n), collapse=" ")
}
  
## return, ordered by frequency, all the n-grams starting with 'words'
##   words - a string of characters containing words to search for
##   nf - a dataframe of n-gram frequencies to search
##
## returns a vector containing up to count suggested next words
##
findBestMatches <- function(words, nf, count) {
    # determine the size of the ngrams provided
    nf.size <- length(unlist(strsplit(as.character(nf$word[1]), " ")))
    # drop leading words longer than the ngrams
    words.pre <- lastWords(words, nf.size - 1)
    # matching ngrams that start with the provided words
    f <- head(nf[grep(paste("^", words.pre, " ", sep=""), nf$word), ], count)
    # strip away the search words from all the results
    r <- gsub(paste("^", words.pre, " ", sep=""), "", as.character(f$word))
    # filter incomplete word suggestions and filtering artifacts
    r[!r %in% c("s", "<", ">", ":", "-", "o", "j", "c", "m")]
}
  
## given an input text, return the predicted next word
##   text - a character string containing words
##   nfl - n-gram frequency dataframes list
##
## returns a character string containing the predicted next word
##
predictNextWord <- function(text, nfl, count=1) {
    text.wc <- wordCount(text)

    prediction <- NULL

    if(text.wc > 3) prediction <- findBestMatches(text, nfl$f5, count)
    if(length(prediction)) return(prediction)

    if(text.wc > 2) prediction <- findBestMatches(text, nfl$f4, count)
    if(length(prediction)) return(prediction)

    if(text.wc > 1) prediction <- findBestMatches(text, nfl$f3, count)
    if(length(prediction)) return(prediction)

    prediction <- findBestMatches(text, nfl$f2, count)
    if(length(prediction)) return(prediction)

    ## text not found in any length n-grams?? randomly select from
    ## highest frequency words
    as.character(sample(head(nfl$f1$word, nfl$r), count))
}

## clean the input text and perform prediction
cleanPredictNextWord <- function(text, nfl, count=1) {
    text <- as.character(createCleanCorpus(text)[[1]], remove.punct=TRUE)
    predictNextWord(text, nfl, count)
}

## given an input text, predict the current word
##   text - a character string containing a portion of a word
##   nfl - n-gram frequency dataframes list
##
## returns a character string containing the predicted current word
##
predictCurrentWord <- function(text, nfl, count=1) {
    current <- as.character(createCleanCorpus(lastWords(text, 1))[[1]])
    nf <- nfl$f1
    # matching ngrams that start with the provided letters
    f <- head(nf[grep(paste("^", current, sep=""), nf$word), ], count)
    as.character(head(f$word, count))
}


### testing

## test the timing and accuracy of predictions on a set of test strings
##   corpus.test - a tm corpus of test strings
##   nfl - n-gram frequency dataframes list
## assumes that corpus.test has had createCleanCorpus() applied to it
## returns the accuracy as the fraction correctly predicted and the proc.time
## 
testTimeAccuracy <- function(corpus.test, nfl, count=1) {
  # extract a random substring of the provided text
  #   text - a string of characters containing words
  # returns both a substring and the actual next word, for prediction testing
  randomSubstring <- function(text) {
    # convert characters to a vector
    wv <- unlist(strsplit(text, " "))
    wv.start <- as.integer(runif(1, 1, length(wv) - 1))
    wv.length <- as.integer(runif(1, 1, length(wv) - wv.start + 1))
    wv.sub <- paste(wv[wv.start:(wv.start + wv.length - 1)], collapse=" ")
    wv.next <- paste(wv[(wv.start + wv.length):(wv.start + wv.length)],
                     collapse=" ")
    list("sub"=wv.sub, "nxt"=wv.next)
  }

  ptm <- proc.time()
  success <- 0
  invalid <- 0
  for(i in 1:length(corpus.test)) {
    testText <- corpus.test[[i]]$content

    # exclude testing texts with only a single word (e.g. nothing to predict!)
    if(wordCount(testText) > 1) {
      ts <- randomSubstring(testText)
      if(ts$nxt %in% cleanPredictNextWord(ts$sub, nfl, count))
          success = success + 1
    }
    else {
      invalid <- invalid + 1 # count of invalid tests
    }
  }
  accuracy <- success / (length(corpus.test) - invalid)
  time <- proc.time() - ptm

  list("accuracy"=accuracy, "time"=time)
}

