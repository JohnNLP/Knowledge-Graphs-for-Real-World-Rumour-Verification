import pickle
from operator import itemgetter
from rank_bm25 import BM25Okapi
from transformers import DPRReader, DPRReaderTokenizer

# DPR approach
'''
tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base', return_dict=True)
'''

with open("2_disambiguated_data.pkl", 'rb') as f:
    data = pickle.load(f)

print("Sentence finding")
for counter, i in enumerate(data):
    #print(counter, "/", len(data))
    #if counter == 1435: #rip 1435
    #    continue
    #old approach
    '''
    keywords = clausie(data[i]["rumour"]).split(" ")
    '''
    #sequential approach
    r_keywords = data[i]["rumour"].split(" ")
    r_keywords = [i.lower() for i in r_keywords]
    r_keywords = [i[:5] if len(i) > 5 else i for i in r_keywords]
    r_keywords = set(r_keywords)

    #used for multiple approaches
    sentences = []
    titles = []
    for article in data[i]["articles"]:
        try:
            for para in data[i]["articles"][article]["paras"]:
                para = para.split(" ")
                sentence_buffer = []
                for word in para:
                    sentence_buffer.append(word)
                    if word == ".":
                        if len(sentence_buffer) >= 2 and sentence_buffer[-2].lower() == "st": #special case for st. abbreviation
                            continue
                        sentences.append(sentence_buffer)
                        titles.append(data[i]["articles"][article]["title"])
                        sentence_buffer = []
                if len(sentence_buffer) > 0:
                    sentences.append(sentence_buffer)
                    titles.append(data[i]["articles"][article]["title"])
        except KeyError: #title or paras are missing due to being empty
            continue
    sentences_spare = sentences.copy()
    if len(sentences) == 0:
        data[i]["evidence"] = []
        continue
    matches = {}

    if len(sentences) > 200:
        sentences = sentences[:200]
    if len(titles) > 200:
        titles = titles[:200]
    for v, s in enumerate(sentences):
        if len(s) > 130:
            s = s[:130]
            sentences[v] = s
    for v, s in enumerate(titles):
        if len(s) > 130:
            s = s[:130]
            titles[v] = s
    print(len(sentences), max([len(s) for s in sentences]), max([len(s) for s in titles]))

    #DPR approach
    '''
    questions = [data[i]["rumour"]] * len(sentences)
    titles = titles #for readability
    texts = [" ".join(s) for s in sentences]
    encoded_inputs = tokenizer(questions=questions,titles=titles,texts=texts,return_tensors='pt',padding=True)
    outputs = model(**encoded_inputs)
    relevance_logits = outputs.relevance_logits.tolist()
    for v, score in enumerate(relevance_logits):
        matches[v] = score
    '''

    #bm25 approach
    '''
    for v, s in enumerate(sentences):
        s = [j for j in s if len(j) >= 3]
        s = [j[:5] if len(j) > 5 else j for j in s]
        s = [w.lower() for w in s]
        s = [j for j in s if len(j) > 0]
        sentences[v] = s
    bm25 = BM25Okapi(sentences)
    '''


    for v, s in enumerate(sentences):
        #sequential approach
        s = [j for j in s if len(j) >= 3]
        s = [j[:5] if len(j) > 5 else j for j in s]
        s = [w.lower() for w in s]
        s = [j for j in s if len(j) > 0]
        s_set = set(s)
        intersection = r_keywords.intersection(s_set)
        score = 0
        streak = 0
        for w, j in enumerate(s):
            if j in intersection and j != "news":
                if not (w==0 or s[w-1] in intersection):
                    streak = 0
                streak += 1
                if len(j) >= 4:
                    score += streak
            else:
                streak = 0
        score /= max(1,len(s)) #penalise long matches that match just because they're long
        if len(s) <= 10:
            score = 0
        matches[v] = score


    #BM25 approach
    '''
    r_keywords = data[i]["rumour"].split(" ")
    r_keywords = [i.lower() for i in r_keywords]
    r_keywords = [i[:5] if len(i) > 5 else i for i in r_keywords]
    #print(r_keywords)
    stuff = bm25.get_scores(r_keywords)
    #print(bm25.get_top_n(r_keywords,sentences,n=1))
    #exit()
    for v, score in enumerate(stuff):
        matches[v] = score
    '''

    sentence_indexes = (sorted(matches.items(), key=itemgetter(1), reverse=True)[:5])
    temp_sentences = []
    for index in sentence_indexes:
        temp_sentences.append(sentences_spare[index[0]])
    data[i]["evidence"] = temp_sentences
    print("====")
    print(data[i]["rumour"])
    print("")
    for k in data[i]["evidence"]:
        print(k)
print("> done")



pickle.dump(data, open("3_sentences.pkl","wb"))