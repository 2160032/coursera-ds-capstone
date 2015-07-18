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
createCleanCorpus <- function(texts, remove.punct=TRUE) {
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
        filtered <- tm_map(filtered, removePunctuation)
    }

    # strip excess whitespace
    filtered <- tm_map(filtered, stripWhitespace)
}


### plotting - ngrams

## ngram plotting function  
plotGram <- function(threshold, freq, wf, type) {
    ggplot(subset(wf, freq > wf$freq[threshold]),
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
    r[!r %in% c("s", "<", ">", ":", "-", "o")]
}
  
## given an input text, return the predicted next word
##   text - a character string containing words
##   nfl - n-gram frequency dataframes list
##
## returns a character string containing the predicted next word
##
predictNext <- function(text, nfl, count=1) {
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
cleanPredictNext <- function(text, nfl, count=1) {
    text <- as.character(createCleanCorpus(text)[[1]], remove.punct=TRUE)
    predictNext(text, nfl, count)
}

