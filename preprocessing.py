import re

#deal with nearly all apostrophe use, except possesives
def expandApostrophes(text):
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"there's", "there is", text)
    return text

#stop punctuation sticking to words
def spacePunctuation(text):
    text = re.sub(r"\.(\Z| )", " . ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    #normalize quotes but only those outside words
    text = re.sub(r"(\A| )['’]", " ' ", text)
    text = re.sub(r"['’](\Z| )", " ' ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r";", " ; ", text)
    #same treatment for hashtags
    text = re.sub(r"#", " ", text) #BERT no like <hashtag> tokens until read otherwise
    return text

#get text ready for further processing
def preprocess(text):
    #print(text)
    textClean = text
    #remove special-token characters (not found in urls)
    textClean = re.sub(r"[<>]", " ", textClean)
    #remove URLs (regex from my CS918 coursework)
    textClean = re.sub(r"((https?|ftp)://)?[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+\.[a-zA-Z]{2,}[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*"," ", textClean) #BERT no like <url> tokens until read otherwise
    #standardize silly characters
    textClean = re.sub(r"&amp;", " and ", textClean)
    textClean = re.sub(r"x27;", "\'", textClean)
    textClean = re.sub(r"–", "-", textClean)
    textClean = re.sub(r"([`\"]|'')", "'", textClean)
    textClean = re.sub(r"[éë]", "e", textClean)
    textClean = re.sub(r"( )+", " ", textClean)
    #remove undesirable characters (keep the ones below)
    textClean = re.sub(r"[^<>A-Za-z0-9\-_@#!?.,'’:; ]", " ", textClean)
    #remove mentions (todo: make this cleaner, sometimes they're important)
    #textClean = re.sub(r"[@][A-Za-z0-9\-_.]+"," ", textClean)
    #de-mid-comma numbers
    textClean = re.sub(r"([0-9]),([0-9])", r"\1\2", textClean)
    #stop punctuation sticking to words
    textClean = spacePunctuation(textClean)
    #remove extra spaces
    textClean = re.sub(r" +"," ", textClean)
    #tokenize
    #textClean = textClean.split()
    return textClean

#get text ready for further processing
def preprocessReaction(text):
    #print(text)
    textClean = text
    #remove special-token characters (not found in urls)
    textClean = re.sub(r"[<>]", " ", textClean)
    #remove URLs (regex from my CS918 coursework)
    textClean = re.sub(r"((https?|ftp)://)?[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+\.[a-zA-Z]{2,}[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*"," ", textClean) #BERT no like <url> tokens until read otherwise
    #standardize silly characters
    textClean = re.sub(r"&amp;", " and ", textClean)
    textClean = re.sub(r"x27;", "\'", textClean)
    textClean = re.sub(r"–", "-", textClean)
    textClean = re.sub(r"([`\"]|'')", "'", textClean)
    textClean = re.sub(r"[éë]", "e", textClean)
    textClean = re.sub(r"( )+", " ", textClean)
    #remove undesirable characters (keep the ones below)
    textClean = re.sub(r"[^<>A-Za-z0-9\-_@#!?.,'’:; ]", " ", textClean)
    #remove mentions (todo: make this cleaner, sometimes they're important)
    textClean = re.sub(r"[@][A-Za-z0-9\-_.]+","hey", textClean)
    #de-mid-comma numbers
    textClean = re.sub(r"([0-9]),([0-9])", r"\1\2", textClean)
    #stop punctuation sticking to words
    textClean = spacePunctuation(textClean)
    #remove extra spaces
    textClean = re.sub(r" +"," ", textClean)
    #tokenize
    #textClean = textClean.split()
    return textClean