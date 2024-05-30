import re
import wordsegment as ws
ws.load()
from wordsegment import load, segment
load()
import stanfordnlp
import subprocess
import random

nlp = stanfordnlp.Pipeline()
interesting_relations = ("obl:npmod", "compound", "advcl", "nummod", "acl:relcl", "nsubj:pass", "acl", "amod", "aux:pass")

#words = []
#with open("10k-words.txt") as f:
#    for word in f.readlines():
#        words.append(word[:-1]) #no /n

class Tweet:
    def __init__(self):
        self.text = None
        #self.textPOS = None
        #self.namedEntities = None
        #self.verbs = None
        self.urls = None
        self.tags = None
        self.label = None
        self.date = None

#deal with nearly all apostrophe use, except possesives
def expandApostrophes(tweet):
    tweet = re.sub(r"i'm", "i am", tweet)
    tweet = re.sub(r"'d", " would", tweet)
    tweet = re.sub(r"it's", "it is", tweet)
    tweet = re.sub(r"'ve", " have", tweet)
    tweet = re.sub(r"'ll", " will", tweet)
    tweet = re.sub(r"can't", "can not", tweet)
    tweet = re.sub(r"won't", "will not", tweet)
    tweet = re.sub(r"n't", " not", tweet)
    tweet = re.sub(r"'re", " are", tweet)
    tweet = re.sub(r"that's", "that is", tweet)
    tweet = re.sub(r"he's", "he is", tweet)
    tweet = re.sub(r"what's", "what is", tweet)
    tweet = re.sub(r"there's", "there is", tweet)
    return tweet

#stop puncauation sticking to words
def spacePunctuation(tweet):
    tweet = re.sub(r"\.(\Z| )", " . ", tweet)
    tweet = re.sub(r",", " , ", tweet)
    tweet = re.sub(r"!", " ! ", tweet)
    #normalize quotes but only those outside words
    tweet = re.sub(r"(\A| )'", " ' ", tweet)
    tweet = re.sub(r"'(\Z| )", " ' ", tweet)
    tweet = re.sub(r":", " : ", tweet)
    tweet = re.sub(r"\?", " ? ", tweet)
    tweet = re.sub(r";", " ; ", tweet)
    #same treatment for hashtags
    tweet = re.sub(r"#", " <hashtag> ", tweet)
    return tweet

#get all the tags
def getHashtags(tweet):
    hashtags = []
    for i in range(len(tweet)-1):
        if tweet[i] == "<hashtag>":
            hashtags.append(tweet[i+1])
    return hashtags

def segmentHashtags(tweet):
    segtags = []
    for v, i in enumerate(tweet):
        if i == "<hashtag>":
            if v < len(tweet)-1 and not tweet[v+1] == "<hashtag>": #sanity check
                segtags.append(segment(tweet[v+1]))
    for i in segtags:
        pointer = tweet.index("<hashtag>")
        tweet = tweet[:pointer] + i + tweet[pointer+2:]
    return (tweet)

#strip ending hastags and users
end_conjunctions = ["in", "on", "by", "for", "via", "from", "and", "the"]
def stripEnding(tweet):
    if len(tweet) >= 3:
        while tweet[-2] == "<hashtag>" and tweet[-3] not in end_conjunctions:
            tweet = tweet[:-2]
            if len(tweet) < 3:
                return tweet
        while (tweet[-1][0] == "@" and tweet[-2] not in end_conjunctions) or (tweet[-2][0] == "@" and tweet[-3] not in end_conjunctions):
            if tweet[-2][0] == "@":
                tweet = tweet[:-2]
            else:
                tweet = tweet[:-1]
            if len(tweet) < 3:
                return tweet
    return tweet

#get text ready for further processing
def preprocess(text):
    #print(text)
    textClean = text
    t = Tweet()
    #remove special-token characters (not found in urls)
    textClean = re.sub(r"[<>]", " ", textClean)
    #remove URLs (regex from my CS918 coursework)
    t.urls = (re.findall(r"((?:(?:https?|ftp)://)?[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+\.[a-zA-Z]{2,}[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*)", textClean))
    textClean = re.sub(r"((https?|ftp)://)?[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+\.[a-zA-Z]{2,}[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*"," ", textClean)  #remove URLs entirely
    #remove user mentions (but leave the token) (actually not anymore)
    #textClean = re.sub(r"@[A-Za-z0-9_]+", " user ", textClean)
    #standardize silly characters
    textClean = re.sub(r"&amp;", " and ", textClean)
    textClean = re.sub(r"–", "-", textClean)
    textClean = re.sub(r"([`\"]|'')", "'", textClean)
    textClean = re.sub(r"[éë]", "e", textClean)
    #remove undesirable characters (keep the ones below)
    textClean = re.sub(r"[^<>A-Za-z0-9\-_@#!?.,':; ]", " ", textClean)
    #de-mid-comma numbers
    textClean = re.sub(r"([0-9]),([0-9])", r"\1\2", textClean)
    #stop punctuation sticking to words
    textClean = spacePunctuation(textClean)
    #tokenize
    textClean = textClean.split()
    #get hashtags
    t.tags = getHashtags(textClean)
    #strip ending hashtags and mantions
    textClean = stripEnding(textClean)
    #remove @s from remaining mentions
    for v, i in enumerate(textClean):
        if i[0] == "@":
            textClean[v] = i[1:]
    #segment any remaining hashtags
    textClean = segmentHashtags(textClean)
    #delete other hashtag tokens (update: there shouldn't be any now)
    textClean = [i for i in textClean if i != "<hashtag>"]
    t.text = textClean
    return t

def prepareNormal(tweet):
    #print("Text -- ", tweet.text)
    output = " ".join(tweet.text)

    tags = []
    for v, j in enumerate(tweet.tags):
        if v == 2:
            break
        j = " ".join(segment(j))
        tags.append(j)
    tag_prefix = "(" + " | ".join(tags) + ")"
    if tag_prefix != "()":
        output = tag_prefix + " " + output

    output = "before:"+tweet.date + " " + output

    return output

def prepareRandom(tweet):
    output = random.choices(words,k=3)
    output = " ".join(output)
    #output = "before:"+tweet.date + " " + output
    print(output)
    return output

def prepareClausIE(tweet):
    with open("temp.txt", "w") as f:
        f.write(" ".join(tweet.text) + "\n")

    while True:
        p = subprocess.Popen(["java", "-jar", "clausie/clausie.jar", "-ftemp.txt", "-otemp2.txt"],stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if len(stderr) > 200:  #hack for error checking
            print("Did not work for this line")
            output = " ".join(tweet.text)
            return "before:" + tweet.date + " " + output
        else:
            break

    # read output of clausIE
    stuff = []
    with open("temp2.txt", "r") as f:
        lines = f.read().splitlines()
        for i in lines:
            i = i.replace("\"", " ")
            i = i.replace("\t", " ")
            i = i.split(" ")
            i = i[1:]
            stuff.append(i)

    # keep only things in clauses
    text = tweet.text
    keeplist = [0] * len(text)
    for j in stuff:
        text_clone = text
        pointer = 0
        for k in j:
            if k in text_clone:
                temp_pointer = text_clone.index(k)
                keeplist[pointer + temp_pointer] = 1
                pointer += temp_pointer + 1
                text_clone = text_clone[temp_pointer + 1:]

    text = [t for v, t in enumerate(text) if keeplist[v] == 1]
    output = " ".join(text)

    # if invalid or empty, just use all
    if len(output) < 3:
        output = " ".join(text)

    tags = []
    for v, j in enumerate(tweet.tags):
        if v == 2:
            break
        j = " ".join(segment(j))
        tags.append(j)
    tag_prefix = "(" + " | ".join(tags) + ")"
    if tag_prefix != "()":
        output = tag_prefix + " " + output
    output = "before:"+tweet.date + " " + output
    print(tweet.text)
    print(output)
    return output

def prepareStanfordNLP(tweet):
    text = " ".join(tweet.text)
    text = nlp(text)
    ordered_sentences = []
    for s in text.sentences:
        ordered_words = {}
        for w in s.words:
            if w.dependency_relation in interesting_relations:
                ind_word = int(w.index) - 1
                ind_governor = w.governor - 1
                ordered_words[ind_word] = s.words[ind_word].text
                ordered_words[ind_governor] = s.words[ind_governor].text
                # print(w.text, w.dependency_relation, s.words[w.governor - 1].text if w.governor > 0 else "root")
        temp = []
        for j in ordered_words:
            temp.append((j, ordered_words[j]))
        temp = sorted(temp)
        ordered_sentences.append(temp)
    output = []
    for j in ordered_sentences:
        for k in j:
            if k[1] != "user":
                output.append(k[1])
    output = " ".join(output)
    tags = []
    for v, j in enumerate(tweet.tags):
        if v == 2:
            break
        j = " ".join(segment(j))
        tags.append(j)
    tag_prefix = "(" + " | ".join(tags) + ")"
    if tag_prefix != "()":
        output = tag_prefix + " " + output
    if (len(output.split(" "))) <= 2:
        output = " ".join(tweet.text)
    output = "before:"+tweet.date + " " + output
    print(output)
    return output

def fixDate(raw_data):
    data = raw_data.split(" ")
    month = data[1]
    if month[:3] == "Jan":
        month = "01"
    elif month[:3] == "Feb":
        month = "02"
    elif month[:3] == "Mar":
        month = "03"
    elif month[:3] == "Apr":
        month = "04"
    elif month[:3] == "May":
        month = "05"
    elif month[:3] == "Jun":
        month = "06"
    elif month[:3] == "Jul":
        month = "07"
    elif month[:3] == "Aug":
        month = "08"
    elif month[:3] == "Sep":
        month = "09"
    elif month[:3] == "Oct":
        month = "10"
    elif month[:3] == "Nov":
        month = "11"
    elif month[:3] == "Dec":
        month = "12"
    day = data[2]
    year = data[5]
    return year+"-"+month+"-"+day