import pickle
from openie import StanfordOpenIE
import nltk

# https://stanfordnlp.github.io/CoreNLP/openie.html#api
# Default value of openie.affinity_probability_cap was 1/3.
properties = {'openie.affinity_probability_cap': 2/3}

def get_clauses(tweet):
    stuff = []
    with StanfordOpenIE(properties=properties) as client:
        for triple in client.annotate(" ".join(tweet)):
            stuff.append([[triple["subject"]],[triple["relation"]],[triple["object"]]])
    return stuff

def get_triple_nouns(tweet):
    nouns = set()
    stuff = nltk.pos_tag(tweet.split(" "))
    for i in stuff:
        if i[1][:2] == "NN" or i[1][:2] == "VB": #truncating POS tags, not words themselves!
            nouns.add(i[0])
    stuff = get_clauses(tweet.split(" "))
    all_words = set()
    for triple in stuff:
        for chunk in triple:
            for word in chunk:
                all_words.add(word)
    return all_words.intersection(nouns), stuff

###########################################

with open("3_sentences.pkl", 'rb') as f:
    all_data = pickle.load(f)

for v, i in enumerate(all_data):
    #if all_data[i]["event"] != "germanwings-crash-all-rnr-threads":
    #    continue
    evidence = all_data[i]["evidence"]
    evidence_chunks = []
    for j in evidence:
        if len(j) > 500:
            print("<<too long>>")
            continue
        print(v, len(all_data), j)
        x = get_clauses(j)
        evidence_chunks.append(x)
    all_data[i]["evidence_chunks"] = evidence_chunks

#get nouns
nouns = {}
for v, i in enumerate(all_data):
    new_nouns, rumour_triples = get_triple_nouns(all_data[i]["rumour"])
    all_data[i]["rumour_triples"] = rumour_triples
    new_nouns = [n.lower() for n in new_nouns]
    print(v,"/",len(all_data),new_nouns)
    if all_data[i]["event"] not in nouns:
        nouns[all_data[i]["event"]] = {}
    for j in new_nouns:
        if j in nouns[all_data[i]["event"]]:
            nouns[all_data[i]["event"]][j] += 1
        else:
            nouns[all_data[i]["event"]][j] = 1
#keep only the most common nouns
for event in nouns:
    to_pop = set()
    for noun in nouns[event]:
        if nouns[event][noun] < 4 or len(noun) <= 3:
            to_pop.add(noun)
    for noun in to_pop:
        nouns[event].pop(noun)

with open('4_nouns.pkl', 'wb') as f:
    pickle.dump(nouns, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('4_data_and_triples.pkl', 'wb') as f:
    pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)

for i in nouns:
    print(i, nouns[i])