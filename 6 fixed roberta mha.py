import pickle
import itertools
import torch
import torch.nn as nn
import random
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from transformers import AlbertModel, AlbertTokenizer, AlbertConfig
from transformers import BertModel, BertTokenizer, BertConfig
from pytorch_transformers import AdamW
from scipy.special import softmax
import numpy as np
import json
import re
import math

EPOCHS = 30
MAX_LEN_1 = 40
MAX_LEN_2 = 130
model_name = "roberta-base" #"albert-base-v2"  #1/4
#model_name = "albert-base-v2"
config = RobertaConfig.from_pretrained(model_name)  #2/4
#config = AlbertConfig.from_pretrained(model_name)
##config.update({"add_pooling_layer": True})
tokenizer = RobertaTokenizer.from_pretrained(model_name)  #3/4
#tokenizer = AlbertTokenizer.from_pretrained(model_name)
NUM_CLASSES = 3
ORIGINAL_LABELS_SYN_EVIDENCE = True
if ORIGINAL_LABELS_SYN_EVIDENCE:
    print("Labels: Original")
    print("Evidence: Replaced")
else:
    print("Labels: New")
    print("Evidence: Untouched")

#load data
with open("4_data_and_triples.pkl", 'rb') as f: #why not 5!?
    all_data = pickle.load(f)
with open("5_kg.pkl", 'rb') as f:
    graphs = pickle.load(f)
with open("5_triple_origins.pkl", "rb") as f:
    triple_origins = pickle.load(f)

#index events
events = []
for i in all_data:
    events.append(all_data[i]["event"])
events = set(events)
events = list(events)
events = sorted(events)
temp = events[4]
events.pop(4)
events = [temp] + events
print(events)

#fix labels
with open("relabel_final.json", "rb") as f:
    relabel = json.load(f)
if ORIGINAL_LABELS_SYN_EVIDENCE == False:
    pop = []
    for i in relabel:
        if i not in all_data:
            print("something very silly happened")
            exit()
        all_data[i]["annotation"] = relabel[i]
        if relabel[i] == "remove":
            pop.append(i)
    for i in pop:
        all_data.pop(i)

#bad = []
#for i in all_data:
#    if all_data[i]["annotation"] == "unverified":
#        bad.append(i)
#for i in bad:
#    all_data.pop(i)

def fixPunctuation(text):
    if len(text) == 0:
        return ""
    while re.match("[ -]", text[0]): #strip bad chars from start of text
        text = text[1:]
        if len(text) == 0:
            return ""
    while re.match("[ -]", text[-1]): #strip bad chars from end of text
        text = text[:-1]
    text = re.sub(r" +", " ", text)
    text = re.sub(r" ,", ",", text)
    text = re.sub(r" \.", ".", text)
    text = re.sub(r" :", ":", text)
    text = re.sub(r" ;", ";", text)
    return text

def printMatrix(labels, pred):
    x = confusion_matrix(labels, pred)
    temp = f1_score(labels, pred, average=None)
    if len(temp) == 3:
        print("      Predicted:")
        print("          F  T U")
        print("Actual: F", x[0][0], x[0][1], x[0][2])
        print("        T", x[1][0], x[1][1], x[1][2])
        print("        U", x[2][0], x[2][1], x[2][2])
        print("Macro F1:", round(f1_score(labels, pred, average="macro"), 4))
        a, b, c = f1_score(labels, pred, average=None)
        print("False F1:", round(a, 4))
        print("True F1:", round(b, 4))
        print("Unverified F1:", round(c, 4))
    else:
        print(labels)
        print(pred)
        print(x)
        try:
            a, b = f1_score(labels, pred, average=None)
            print("False F1:", round(a, 4))
            print("True F1:", round(b, 4))
        except:
            print("all 1 class found")

def numberiseAnnotation(annotation):
    if annotation == "false" or annotation == "FALSE" or annotation == "comment":
        return 0
    elif annotation == "true" or annotation == "TRUE" or annotation == "deny":
        return 1
    elif annotation == "unverified" or annotation == "UNVERIFIED" or annotation == "query":
        return 2
    elif annotation == "support":
        return 3
    else:
        print("Unexpected annotation found")
        exit()

def stringifyAnnotation(annotation):
    if annotation == 0:
        return "TRUE"
    elif annotation == 1:
        return "FALSE"
    elif annotation == 2:
        return "UNVERIFIED"
    else:
        print("Unexpected annotation found")
        exit()

#get paths from node-with-start-text to node-with-end-text
def getPaths(graph, start, end):
    output = []
    node = graph[start]
    #length 1
    paths = {} #made of edges
    out_edges = node.getOutEdges()
    for edge_name in out_edges:
        new_edge = out_edges[edge_name]
        if new_edge.getTarget() == new_edge.getBase(): #no self-edges
            continue
        paths[new_edge] = {}
        if new_edge.getTarget().getText() == end: #add to output
            output.append([new_edge])
    #length 2
    for edge in paths:
        target = edge.getTarget()
        out_edges = target.getOutEdges()
        for edge_name in out_edges:
            new_edge = out_edges[edge_name]
            if new_edge.getTarget() == new_edge.getBase(): #no self-edges
                continue
            if edge == new_edge: #no duplicates (impossible here anyways)
                continue
            paths[edge][new_edge] = {}
            if new_edge.getTarget().getText() == end: #add to output
                output.append([edge, new_edge])
    #length 3
    #'''
    for edge in paths:
        for edge2 in paths[edge]:
            target = edge2.getTarget()
            out_edges = target.getOutEdges()
            for edge_name in out_edges:
                new_edge = out_edges[edge_name]
                if new_edge.getTarget() == new_edge.getBase(): #no self-edges
                    continue
                if edge == new_edge or edge2 == new_edge: #no duplicates
                    continue
                paths[edge][edge2][new_edge] = {}
                if new_edge.getTarget().getText() == end: #add to output
                    output.append([edge, edge2, new_edge])
    #'''
    return output

'''
event = "charliehebdo-all-rnr-threads"
graph = graphs[event]
x = getPaths(graph, "people", "attack")
for i in x:
    print(i)
'''

#given a tweet, link its contents to graph nodes
for rumour in all_data:
    event = all_data[rumour]["event"]
    graph = graphs[event]
    overlap = []
    text = " "+all_data[rumour]["rumour"].lower()+" " #no partial words
    text = text.split(" ")
    #text = ["dead" if i == "died" else i for i in text]
    text = [i[:5] if len(i) > 5 else i for i in text]
    for node in graph:
        #if len(node) > 3 and " "+node+" " in text: #no partial words
        if len(node) <= 3:
            continue
        temp = node.split(" ")
        #temp = ["dead" if i == "died" else i for i in temp]
        temp = [i[:5] if len(i) > 5 else i for i in temp]
        all_contained = True
        for i in temp:
            if i not in text:
                all_contained = False
        if all_contained:
            overlap.append(node)
    all_data[rumour]["overlap"] = overlap

#frequency of graph nodes matching rumours
frequencies = {}
for event in events:
    frequencies[event] = {}
    graph = graphs[event]
    for node in graph:
        frequencies[event][node] = 0
for rumour in all_data:
    event = all_data[rumour]["event"]
    text = " "+all_data[rumour]["rumour"].lower()+" " #no partial words
    text = text.split(" ")
    #text = ["dead" if i == "died" else i for i in text]
    text = [i[:5] if len(i) > 5 else i for i in text]
    for thing in frequencies[event]:
        if len(thing) <= 3:
            continue
        temp = thing.split(" ")
        #temp = ["dead" if i == "died" else i for i in temp]
        temp = [i[:5] if len(i) > 5 else i for i in temp]
        all_contained = True
        for i in temp:
            if i not in text:
                all_contained = False
        if all_contained:
            frequencies[event][thing] += 1

#for i in frequencies:
#    for j in frequencies[i]:
#        print(frequencies[i][j], j)

#add frequencies to matches
for rumour in all_data:
    event = all_data[rumour]["event"]
    overlap = all_data[rumour]["overlap"]
    for v, thing in enumerate(overlap):
        overlap[v] = [frequencies[event][thing], thing]
    all_data[rumour]["overlap"] = overlap

#remove extra matches, taking into account frequencies
MATCHES_TO_KEEP_INITIALLY = 2
og_overlap = {}
for rumour in all_data:
    event = all_data[rumour]["event"]
    overlap = all_data[rumour]["overlap"]
    overlap = sorted(overlap) #keep rarest finds
    og_overlap[rumour] = overlap.copy()
    final_words = []
    for i in overlap:
        if len(final_words) < MATCHES_TO_KEEP_INITIALLY:
            final_words.append(i[1])
    all_data[rumour]["graph_words"] = final_words

#find graph paths, adding the next least common word each time
for rumour in all_data:
    event = all_data[rumour]["event"]
    words = all_data[rumour]["graph_words"]
    spare_words = [i[1] for i in og_overlap[rumour] if i[1] not in words]
    all_words = words + spare_words
    graph = graphs[event]
    paths = []
    if len(words) < 2:
        pass #need >=2 nodes to have a path
    else:
        while paths == []:
            perms = itertools.permutations(words, 2)
            for i in perms:
                paths += getPaths(graph, i[0], i[1])
            if len(spare_words) == 0:
                break
            else:
                x = spare_words.pop(0)
                words.append(x)
    all_data[rumour]["paths"] = paths
    #now do it again but find ALL paths not just with rarest words
    all_paths = []
    if len(all_words) < 2:
        pass
    else:
        perms = itertools.permutations(all_words, 2)
        for i in perms:
            all_paths += getPaths(graph, i[0], i[1])
    all_data[rumour]["all_paths"] = all_paths

#score and order paths
classes = [] #for sanity checking later
for rumour in all_data:
    text = all_data[rumour]["rumour"].lower()
    text = text.split(" ")
    #text = ["dead" if i == "died" else i for i in text]
    text = [i[:3] if len(i) >= 3 else i for i in text]
    text_not_set = text.copy()
    text = set(text)
    paths = all_data[rumour]["all_paths"]
    paths_not_normalized = [None]*len(paths) #something sus happens if this is init. the same as paths above, maybe a compiler tries to be clever
    for v, path in enumerate(paths):
        pathtext = []
        pathtext.append(path[0].getBase().getText())
        for i in range(len(path)):
            pathtext.append(path[i].getText())
            pathtext.append(path[i].getTarget().getText())
        pathtext = " ".join(pathtext)
        pathtext = pathtext.split(" ")
        #pathtext = ["dead" if i == "died" else i for i in pathtext]
        pathtext = [i[:3] if len(i) >= 3 else i for i in pathtext]
        pathtext_list = pathtext.copy()
        pathtext = set(pathtext)
        intersection = text.intersection(pathtext)
        score = 0
        streak = 0
        for w, i in enumerate(text_not_set):
            if i in intersection and i != "news"[:3]:
                if not (w==0 or text_not_set[w-1] in intersection):
                    streak = 0
                streak += 1
                if len(i) >= 3:
                    score += streak
            else:
                streak = 0
        paths_not_normalized[v] = [score, path]
        score /= len(pathtext) #penalise long matches that match just because they're long
        paths[v] = [score, path]
    paths = sorted(paths, reverse=True)
    paths_not_normalized = sorted(paths_not_normalized)
    all_data[rumour]["pathscore"] = 0
    classes.append(all_data[rumour]["annotation"])
    classes.append(all_data[rumour]["event"])
    if len(paths) > 0:
        all_data[rumour]["all_paths"] = paths
        all_data[rumour]["ev_rare_path"] = None
        all_data[rumour]["ev_score_path"] = None
        all_data[rumour]["ev_bad_path"] = [None, None, None]
        #####all_data[rumour]["ev_original"] = None #####
        #bad_target = random.randint(math.floor(len(paths)-max(1,len(paths)/4)), len(paths)-1) #bottom quarter of paths
        for v, i in enumerate(paths_not_normalized): #the worst not_normalized matches
            if v >= 3: #only want the worst 3
                break
            path = paths_not_normalized[v][1]
            pathtext = []
            pathtext.append(path[0].getBase().getText())
            for i in range(len(path)):
                pathtext.append(path[i].getText())
                pathtext.append(path[i].getTarget().getText())
            sources = ""
            already_added = []
            for node in path:
                triple_flattened = node.getBase().getText() + " " + node.getText() + " " + node.getTarget().getText()
                new = triple_origins[all_data[rumour]["event"]][triple_flattened] + " "
                if new not in already_added:
                    sources += new
                already_added.append(new)
                if v == 2: #only length 1 allowed for original evidence which is always a single sentence
                    break
            all_data[rumour]["ev_bad_path"][v] = fixPunctuation(sources)
        for v, i in enumerate(paths):
            path = paths[v][1]
            pathtext = []
            pathtext.append(path[0].getBase().getText())
            for i in range(len(path)):
                pathtext.append(path[i].getText())
                pathtext.append(path[i].getTarget().getText())
            sources = ""
            already_added = []
            for node in path:
                triple_flattened = node.getBase().getText() + " " + node.getText() + " " + node.getTarget().getText()
                new = triple_origins[all_data[rumour]["event"]][triple_flattened] + " "
                if new not in already_added:
                    sources += new
                already_added.append(new)
            if v == 0:
                all_data[rumour]["ev_score_path"] = fixPunctuation(sources)
                all_data[rumour]["pathscore"] = paths[v][0]
            if v == 1:
                all_data[rumour]["ev_rare_path"] = fixPunctuation(sources)
                print("SUBSTITUTING ev_rare_path WITH ev_score_path #2")
            #if path in all_data[rumour]["paths"] and all_data[rumour]["ev_rare_path"] is None: #highest scoring one in both paths and all_paths
            #    all_data[rumour]["ev_rare_path"] = fixPunctuation(sources)
            #    print("Using Rare as usual...")
        if all_data[rumour]["ev_score_path"] is None or all_data[rumour]["ev_rare_path"] is None:
            print("Something very silly happened around line 400")
            #exit()
    else:
        all_data[rumour]["paths"] = None
        all_data[rumour]["ev_rare_path"] = None
        all_data[rumour]["ev_score_path"] = None
        all_data[rumour]["ev_bad_path"] = [None, None, None]

#retrieve original evidence too
for rumour in all_data:
    original_evidence = all_data[rumour]["evidence"]
    if original_evidence is None or len(original_evidence) == 0:
        all_data[rumour]["ev_original"] = None
    else:
        all_data[rumour]["ev_original"] = fixPunctuation(" ".join(original_evidence[0]))

#evaluate graph path stuff
for rumour in all_data:
    #if all_data[rumour]["annotation"] != "unverified":
    #    continue
    print("===")
    print(rumour)
    print(all_data[rumour]["rumour"])
    print(all_data[rumour]["graph_words"])
    x = all_data[rumour]["paths"]
    if x is not None:
        for v, i in enumerate(x):
            #print(i)
            if v >= 2:
                break
    print(all_data[rumour]["ev_rare_path"])
    print(all_data[rumour]["ev_score_path"])
    print(all_data[rumour]["ev_original"])
    print(all_data[rumour]["annotation"])
    print("bad --", all_data[rumour]["ev_bad_path"])


print("f",classes.count("false"))
print("t",classes.count("true"))
print("u",classes.count("unverified"))
for i in events:
    print(i, classes.count(i))

'''
#consider de-duplicating paths
pred_actual_all = []
test_y_all = []
for r in all_data:
    entry = all_data[r]
    label = entry["annotation"]
    if entry["final_evidence"] is None:
        pred_actual_all.append(numberiseAnnotation("false"))
        test_y_all.append(numberiseAnnotation(label))
    elif entry["paths"][0][0] < 1.3:
        pred_actual_all.append(numberiseAnnotation("unverified")) #notably these scores are lower in f, reconsider later if some other difference is spotted there
        test_y_all.append(numberiseAnnotation(label))
    else:
        pred_actual_all.append(numberiseAnnotation("true"))
        test_y_all.append(numberiseAnnotation(label))
printMatrix(test_y_all, pred_actual_all)
exit()
'''

#'''
for rumour in all_data:
    if random.randint(0,100) == 1:
        print("")
        print("")
        print("")
        print(all_data[rumour]["rumour"])
        print(all_data[rumour]["annotation"])
        print(all_data[rumour]["ev_original"])
        print(all_data[rumour]["ev_score_path"])
        print(all_data[rumour]["ev_rare_path"])
exit()
#'''


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare(text, maxlen):
    out = tokenizer.encode(text,max_length=maxlen, truncation=True) #with new models, make sure [BOS] and [EOS] tokens are being added
    if len(out) < maxlen:
        pad = tokenizer.encode("<pad>", add_special_tokens=False)
        if len(pad) != 1:
            print("Wrong padding token used!")
            print(pad)
            exit()
        for i in range(maxlen-len(out)):
            out += pad
    return out

def prepareJoint(text, text2):
    #print(text, "  |  ", text2)
    out = tokenizer.encode(text,max_length=MAX_LEN_1, truncation=True)
    out2 = tokenizer.encode(text2,max_length=MAX_LEN_2+1, truncation=True)
    out2 = out2[1:] #remove bos token (is now the correct length also)
    out += out2
    pad = tokenizer.encode("<pad>", add_special_tokens=False)
    if len(pad) != 1:
        print("Wrong padding token used!")
        print(pad)
        exit()
    if len(out) < MAX_LEN_1+MAX_LEN_2:
        for i in range(MAX_LEN_1+MAX_LEN_2-len(out)):
            out += pad
    return out

def makeTrainable(dict):
    top_x = []
    top_y = []
    bottom = []
    bottom_rumour = []
    bottom_y = []
    extra_x = []
    ids = []

    for r in dict:

        if NUM_CLASSES == 2:
            if dict[r]["annotation"] == "unverified":
                continue
        #if dict[r]["event"][0] == "f": ###
        #    continue ###

        temp_top_x = []
        temp_top_y = []
        temp_bottom = []
        temp_rumour = []
        temp_bottom_y = None
        temp_extra_x = None

        dict[r]["rumour"] = fixPunctuation(dict[r]["rumour"]).lower()
        rumour = fixPunctuation(dict[r]["rumour"]).lower()

        #reactions
        '''
        for i in dict[r]["reactions"]:
            reaction = dict[r]["reactions"][i]
            if i in reaction_labels:
                temp_top_x.append(fixPunctuation(reaction).lower())
                temp_top_y.append(numberiseAnnotation(reaction_labels[i]))
        if len(temp_top_x) > 5:
            temp_top_x = temp_top_x[:5]
            temp_top_y = temp_top_y[:5]
        while len(temp_top_x) < 5:
            temp_top_x.append("")
            temp_top_y.append(999) #todo: figure this out
        for v, i in enumerate(temp_top_x):
            temp_top_x[v] = prepareJoint(rumour, temp_top_x[v])
        '''

        #todo: rumour + reaction per thing, lower, bertify

        #rumour
        #temp_bottom_r_x.append(rumour)
        #temp_bottom_r_x = [prepare(i,50) for i in temp_bottom_r_x]
        '''
        #evidence -- graph path
        if dict[r]["paths"] is None:
            evidence = "there is no evidence for this claim."
        else:
            path = dict[r]["paths"][0][1]
            pathtext = []
            for i in range(len(path)):
                temp = []
                temp.append(path[i].getBase().getText())
                temp.append(path[i].getText())
                temp.append(path[i].getTarget().getText())
                pathtext.append(" ".join(temp) + ".")
            evidence = " ".join(pathtext)
        temp_bottom.append(evidence)
        '''
        #evidence -- some relevant sentence -- found how?
        evidence = dict[r]["ev_score_path"] #ev_rare_path, ev_score_path, ev_original
        #print("Using ev_score_path")
        if ORIGINAL_LABELS_SYN_EVIDENCE == True:
            if dict[r]["annotation"] == "unverified":
                evidence = dict[r]["ev_bad_path"][2]
        if evidence is not None:
            temp_bottom.append(evidence.lower())
        else:
            temp_bottom.append("there is no evidence for this claim.")

        evidence = dict[r]["ev_rare_path"] #ev_rare_path, ev_score_path, ev_original
        #print("Using ev_rare_path")
        if ORIGINAL_LABELS_SYN_EVIDENCE == True:
            if dict[r]["annotation"] == "unverified":
                evidence = dict[r]["ev_bad_path"][2]
        if evidence is not None:
            temp_bottom.append(evidence.lower())
        else:
            temp_bottom.append("there is no evidence for this claim.")
        #'''
        evidence = dict[r]["ev_original"] #ev_rare_path, ev_score_path, ev_original
        #print("Using ev_original")
        if ORIGINAL_LABELS_SYN_EVIDENCE == True:
            if dict[r]["annotation"] == "unverified":
                evidence = dict[r]["ev_bad_path"][2]
        if evidence is not None:
            temp_bottom.append(evidence.lower())
        else:
            temp_bottom.append("there is no evidence for this claim.")
        #'''
        '''
        for ev in dict[r]["evidence"]: #add google (or is it dataset?) evidence -- todo: may duplicate evidence from the previous bit
            temp_bottom.append(fixPunctuation(" ".join(ev)).lower())
        '''

        MAX_EVIDENCE = 3
        if len(temp_bottom) > MAX_EVIDENCE:
            temp_bottom = temp_bottom[:MAX_EVIDENCE]
        if len(temp_bottom) < MAX_EVIDENCE: #indeed it's nearly always 5, if not just continue (ignore the 0.5% this is not true for) -- todo: maybe give them F/U label
            print("Less than MAX_EVIDENCE thingies")
            continue
        #while len(temp_bottom) < MAX_EVIDENCE:
        #    temp_bottom.append("")

        temp_bottom = [prepare(i, MAX_LEN_2) for i in temp_bottom]
        temp_rumour = [prepare(rumour, MAX_LEN_1)]
        temp_bottom_y = numberiseAnnotation(dict[r]["annotation"])


        #extra
        #reaction_count = min(1, dict[r]["reaction_count"]/15.0)
        #reaction_length = min(1, dict[r]["reaction_length"]/120.0)
        #user_verified = 0
        #if dict[r]["user_verified"] == True:
        #    user_verified = 1
        #evidence_score = dict[r]["pathscore"]
        temp_extra_x = [] #[reaction_count, reaction_length, user_verified, evidence_score]
        #print(temp_extra_x)

        top_x.append(temp_top_x)
        top_y.append(temp_top_y)
        bottom.append(temp_bottom)
        bottom_rumour.append(temp_rumour)
        bottom_y.append(temp_bottom_y)
        extra_x.append(temp_extra_x)
        ids.append(r)

        #print("====")
        #print(temp_top_x)
        #print(temp_top_y)
        #print(temp_bottom_r_x)
        #print(temp_bottom_e_x)
        #print(temp_bottom_y)
        #print(temp_extra_x)

    return top_x, top_y, bottom, bottom_rumour, bottom_y, extra_x, ids

def printData(labels, pred):
    for v, i in enumerate(pred):
        #print("\nr:", rumours[v])
        #print("s:", sentences[v])
        print("p:", stringifyAnnotation(i), "a:", stringifyAnnotation(labels[v]))

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.bottom_r_encoder = RobertaModel(config)  #4/4
        self.bottom_e_encoder = RobertaModel(config)  #4/4
        #self.bottom_r_encoder = AlbertModel(config)
        #self.bottom_e_encoder = AlbertModel(config)
        #self.pooler = ContextPooler(config) #as in the official SeqClassif code

        self.mha = nn.MultiheadAttention(embed_dim=768, num_heads=8, dropout=0.15)
        if NUM_CLASSES == 2:
            self.linear = nn.Linear(768,2)
        else:
            self.linear = nn.Linear(768,3)

        self.drop = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, top_x, bottom_r_x, bottom_e_x, extra_x, bottom_e_x_mask, bottom_r_x_mask): #todo: unswap
        bottom_r_x = self.bottom_r_encoder(bottom_r_x, attention_mask=bottom_r_x_mask)["pooler_output"] #todo: batches, epochs, and stuff
        bottom_e_x = self.bottom_e_encoder(bottom_e_x, attention_mask=bottom_e_x_mask)["pooler_output"]
        #print(bottom_r_x.shape)
        #print(bottom_e_x.shape)
        bottom, atn_weights = self.mha(key=bottom_e_x, query=bottom_r_x, value=bottom_e_x)
        #print(bottom.shape)
        ##bottom = torch.flatten(bottom) #for rer
        #print(bottom.shape)
        #print(atn_weights.shape)
        #print(bottom.shape)
        ###extra_x = torch.unsqueeze(extra_x, 0)
        #print(extra_x.shape)
        ###bottom = torch.cat((bottom,extra_x),1)
        #print(bottom.shape)
        out = self.linear(bottom)
        ##out = torch.unsqueeze(out, 0) #for rer
        #print(out.shape)
        return out

test_y_all = []
pred_actual_all = []
ids_preds = {}

for event in events:
    #if event[:2] != "ot":
    #    print("skipping", event)
    #    continue

    print("\nNEXT:",event)
    test_data = {}
    train_data = {}

    for v, i in enumerate(all_data):
        if all_data[i]["event"] == event:
            test_data[i] = all_data[i]
        else:
            train_data[i] = all_data[i]

    '''
    z = []
    for i in test_data:
        a = test_data[i]["annotation"]
        z.append(a)
    print(z.count("true"), z.count("false"), z.count("unverified"))
    continue
    '''

    MASK_TOKEN = 1
    train_top_x, train_top_y, train_bottom, train_bottom_r, train_bottom_y, train_extra_x, _ = makeTrainable(train_data) #todo: consider masking first and second thingies separately, and wrt this mask
    train_bottom_r_x_mask = [[[int(k != MASK_TOKEN) for k in j] for j in i] for i in train_bottom]
    #train_bottom_e_x_mask = [[[int(k != 0) for k in j] for j in i] for i in train_bottom_e_x]
    test_top_x, test_top_y, test_bottom, test_bottom_r, test_bottom_y, test_extra_x, test_ids = makeTrainable(test_data)
    test_bottom_r_x_mask = [[[int(k != MASK_TOKEN) for k in j] for j in i] for i in test_bottom]

    train_bottom_other_mask = [[[int(k != MASK_TOKEN) for k in j] for j in i] for i in train_bottom_r]
    test_bottom_other_mask = [[[int(k != MASK_TOKEN) for k in j] for j in i] for i in test_bottom_r]

    if train_bottom_r_x_mask[0][0][0] != 1:
        print(train_bottom[0][0])
        print(train_bottom_r_x_mask[0][0])
        print("Attention masking is broken, does not start with 1")
        exit()
    #test_bottom_e_x_mask = [[[int(k != 0) for k in j] for j in i] for i in test_bottom_e_x]


    #data dump
    #'''
    kgat_train = []
    kgat_test = []

    print("======")
    for i in train_data:
        print(i, train_data[i])
        special_train = {"id": None, "evidence": None, "claim": None, "label": None}
        id = i
        label = train_data[i]["annotation"]
        if label == "true":
            label = "SUPPORTS"
        elif label == "false":
            label = "REFUTES"
        elif label == "unverified":
            label = "NOT ENOUGH INFO"
        else:
            print("Unexpected label")
            exit()
        claim = train_data[i]["rumour"]
        evidence = [train_data[i]["ev_score_path"], train_data[i]["ev_rare_path"], train_data[i]["ev_original"]]
        evidence = [j for j in evidence if j is not None]
        evidence_flags = [1, 1, 1] #1 in train 0 in test
        evidence_titles = [train_data[i]["event"][:-16]]*3 #sentences have their own context and no clear title, so no need to give title for later concat
        evidence_ids = [0, 1, 2]
        special_train["id"] = int(id)
        special_train["evidence"] = []
        if len(evidence) > 0:
            special_train["evidence"] = [[evidence_titles[0], evidence_ids[0], evidence[0], evidence_flags[0]]]
        if len(evidence) > 1:
            special_train["evidence"] += [[evidence_titles[1], evidence_ids[1], evidence[1], evidence_flags[1]]]
        if len(evidence) > 2:
            special_train["evidence"] += [[evidence_titles[2], evidence_ids[2], evidence[2], evidence_flags[2]]]
        special_train["claim"] = claim
        special_train["label"] = label
        kgat_train.append(special_train)
        print(special_train)
    with open("data_"+event+".json", 'w') as f:
        for entry in kgat_train:
            json.dump(entry, f)
            f.write('\n')

    for i in test_data:
        print(i, test_data[i])
        special_test = {"id": None, "evidence": None, "claim": None, "label": None}
        id = i
        label = test_data[i]["annotation"]
        if label == "true":
            label = "SUPPORTS"
        elif label == "false":
            label = "REFUTES"
        elif label == "unverified":
            label = "NOT ENOUGH INFO"
        else:
            print("Unexpected label")
            exit()
        claim = test_data[i]["rumour"]
        evidence = [test_data[i]["ev_score_path"], test_data[i]["ev_rare_path"], test_data[i]["ev_original"]]
        evidence = [j for j in evidence if j is not None]
        evidence_flags = [0, 0, 0] #1 in train 0 in test
        evidence_titles = [test_data[i]["event"][:-16]]*3 #sentences have their own context and no clear title, so no need to give title for later concat
        evidence_ids = [0, 1, 2]
        special_test["id"] = int(id)
        special_test["evidence"] = []
        if len(evidence) > 0:
            special_test["evidence"] = [[evidence_titles[0], evidence_ids[0], evidence[0], evidence_flags[0]]]
        if len(evidence) > 1:
            special_test["evidence"] += [[evidence_titles[1], evidence_ids[1], evidence[1], evidence_flags[1]]]
        if len(evidence) > 2:
            special_test["evidence"] += [[evidence_titles[2], evidence_ids[2], evidence[2], evidence_flags[2]]]
        special_test["claim"] = claim
        special_test["label"] = label
        kgat_test.append(special_test)
        print(special_test)
    with open("data_test_"+event+".json", 'w') as f:
        for entry in kgat_test:
            json.dump(entry, f)
            f.write('\n')
    continue
    #'''

    #'''
    #balance classes
    a = train_bottom_y.count(0)
    b = train_bottom_y.count(1)
    c = train_bottom_y.count(2)
    temp_f = []
    temp_t = []
    temp_u = []
    for v, i in enumerate(train_bottom_y):
        if i == 0:
            temp_f.append([train_top_x[v], train_top_y[v], train_bottom[v], train_bottom_y[v], train_extra_x[v], train_bottom_r_x_mask[v], train_bottom_r[v], train_bottom_other_mask[v]])
        elif i == 1:
            temp_t.append([train_top_x[v], train_top_y[v], train_bottom[v], train_bottom_y[v], train_extra_x[v], train_bottom_r_x_mask[v], train_bottom_r[v], train_bottom_other_mask[v]])
        elif i == 2:
            temp_u.append([train_top_x[v], train_top_y[v], train_bottom[v], train_bottom_y[v], train_extra_x[v], train_bottom_r_x_mask[v], train_bottom_r[v], train_bottom_other_mask[v]])
    random.shuffle(temp_f)
    random.shuffle(temp_t)
    random.shuffle(temp_u)
    longest = max(len(temp_f), len(temp_t), len(temp_u)) #extend
    while len(temp_f) < longest:
        temp_f += temp_f
        if len(temp_f) > longest:
            temp_f = temp_f[:-(len(temp_f)-longest)]
    while len(temp_t) < longest:
        temp_t += temp_t
        if len(temp_t) > longest:
            temp_t = temp_t[:-(len(temp_t)-longest)]
    while len(temp_u) < longest:
        if NUM_CLASSES == 2:
            break
        temp_u += temp_u
        if len(temp_u) > longest:
            temp_u = temp_u[:-(len(temp_u)-longest)]
    train_top_x = [i[0] for i in temp_f] + [i[0] for i in temp_t] + [i[0] for i in temp_u] #regroup
    train_top_y = [i[1] for i in temp_f] + [i[1] for i in temp_t] + [i[1] for i in temp_u]
    train_bottom = [i[2] for i in temp_f] + [i[2] for i in temp_t] + [i[2] for i in temp_u]
    train_bottom_y = [i[3] for i in temp_f] + [i[3] for i in temp_t] + [i[3] for i in temp_u]
    train_extra_x = [i[4] for i in temp_f] + [i[4] for i in temp_t] + [i[4] for i in temp_u]
    train_bottom_r_x_mask = [i[5] for i in temp_f] + [i[5] for i in temp_t] + [i[5] for i in temp_u]
    train_bottom_r = [i[6] for i in temp_f] + [i[6] for i in temp_t] + [i[6] for i in temp_u]
    train_bottom_other_mask = [i[7] for i in temp_f] + [i[7] for i in temp_t] + [i[7] for i in temp_u]
    #'''

    #print(len(train_x), len(train_x2), len(train_y))
    #print(train_y)

    #shuffle
    indices = np.arange(len(train_bottom_y))
    np.random.shuffle(indices)
    train_top_x = np.array(train_top_x)[indices].tolist()
    train_top_y = np.array(train_top_y)[indices].tolist()
    train_bottom = np.array(train_bottom)[indices].tolist()
    train_bottom_r = np.array(train_bottom_r)[indices].tolist()
    train_bottom_y = np.array(train_bottom_y)[indices].tolist()
    train_extra_x = np.array(train_extra_x)[indices].tolist()
    train_bottom_r_x_mask = np.array(train_bottom_r_x_mask)[indices].tolist()
    train_bottom_other_mask = np.array(train_bottom_other_mask)[indices].tolist()

    #account for class imbalance
    a = train_bottom_y.count(0)
    b = train_bottom_y.count(1)
    c = train_bottom_y.count(2)



    #print(a,b,c)

    if NUM_CLASSES == 2:
        weights = torch.tensor([1/a, 1/b], dtype=torch.float).to(device)
    else:
        weights = torch.tensor([1/a, 1/b, 1/c], dtype=torch.float).to(device)

    print(weights)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = CustomModel().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-7, correct_bias=True) #vs 0.48/9 with 1.0

    #make tensors and move them to gpu
    train_top_x = torch.tensor(train_top_x).to(device)
    train_top_y = torch.tensor(train_top_y).to(device)
    train_bottom = torch.tensor(train_bottom).to(device)
    train_bottom_r = torch.tensor(train_bottom_r).to(device)
    train_bottom_y = torch.tensor(train_bottom_y).to(device)
    train_extra_x = torch.tensor(train_extra_x).to(device)
    train_bottom_r_x_mask = torch.tensor(train_bottom_r_x_mask).to(device)
    train_bottom_other_mask = torch.tensor(train_bottom_other_mask).to(device)

    test_top_x = torch.tensor(test_top_x).to(device)
    test_bottom = torch.tensor(test_bottom).to(device)
    test_bottom_r = torch.tensor(test_bottom_r).to(device)
    test_extra_x = torch.tensor(test_extra_x).to(device)
    test_bottom_r_x_mask = torch.tensor(test_bottom_r_x_mask).to(device)
    test_bottom_other_mask = torch.tensor(test_bottom_other_mask).to(device)

    #data is already batched per-rumour
    #not anymore it's not!
    train_top_x = torch.tensor(train_top_x).to(device)
    train_top_y = torch.tensor(train_top_y).to(device)
    train_bottom_r_x = torch.tensor(train_bottom_r).to(device)
    train_bottom_e_x = torch.tensor(train_bottom).to(device)
    train_bottom_y = torch.tensor(train_bottom_y).to(device)
    train_extra_x = torch.tensor(train_extra_x).to(device)
    train_bottom_r_x_mask = torch.tensor(train_bottom_r_x_mask).to(device)
    train_bottom_e_x_mask = torch.tensor(train_bottom_other_mask).to(device)

    test_top_x = torch.tensor(test_top_x).to(device)
    test_bottom_r_x = torch.tensor(test_bottom_r).to(device)
    test_bottom_e_x = torch.tensor(test_bottom).to(device)
    test_extra_x = torch.tensor(test_extra_x).to(device)
    test_bottom_r_x_mask = torch.tensor(test_bottom_r_x_mask).to(device)
    test_bottom_e_x_mask = torch.tensor(test_bottom_other_mask).to(device)

    #15 1 140

    BATCH_SIZE = 16
    for epoch in range(EPOCHS):
        model.train()
        loss = 0
        loss_readout = 0
        for v, i in enumerate(train_bottom_r_x):
            y_pred = model(train_top_x[v], train_bottom_r_x[v], train_bottom_e_x[v], train_extra_x[v], train_bottom_r_x_mask[v], train_bottom_e_x_mask[v]) #batch size 1 for now
            loss += criterion(y_pred, torch.unsqueeze(train_bottom_y[v], 0)) #unsqueeze to make it into batch of size 1 []
            if v>0 and (v%BATCH_SIZE == 0 or v == len(train_bottom_r_x)-1):
                model.zero_grad()
                loss.backward()
                optimizer.step()
                #optimizer.zero_grad()
                loss_readout += loss
                loss = 0
            if v%100 == 0:
                print(v, len(train_bottom_r_x))
        print("EPOCH", epoch + 1, "loss:", loss_readout)


        #get predictions
        model.eval()
        pred = []
        for v, i in enumerate(test_bottom):
            with torch.no_grad():
                output = model(None, test_bottom_r[v], test_bottom[v], test_extra_x[v], test_bottom_r_x_mask[v], test_bottom_other_mask[v])
            predicted = output.cpu().numpy()
            pred += [j for j in predicted]

        pred = np.array([softmax(i) for i in pred])
        pred_actual = []
        for v, i in enumerate(pred):
            if NUM_CLASSES == 2:
                if i[0] >= i[1]:##### and i[0] >= i[2]:
                    prediction = 0
                elif i[1] >= i[0]:# and i[1] >= i[2]:
                    prediction = 1
                #else:
                #    prediction = 2
            else:
                if i[0] >= i[1] and i[0] >= i[2]:
                    prediction = 0
                elif i[1] >= i[0] and i[1] >= i[2]:
                    prediction = 1
                else:
                    prediction = 2
            pred_actual.append(prediction)
            r = test_ids[v]
            ids_preds[r] = {"annotation":test_bottom_y[v], "predicted":prediction, "rumour":all_data[r]["rumour"], "evidence":all_data[r]["ev_rare_path"]}

        #printData(test_y, pred_actual)
        print("\n"+event)
        printMatrix(test_bottom_y, pred_actual)

        if epoch == EPOCHS-1:
            test_y_all += test_bottom_y
            pred_actual_all += pred_actual

#final evaluation
print("\nOVERALL RESULTS")
printMatrix(test_y_all, pred_actual_all)

with open('data.json', 'w') as f:
    json.dump(ids_preds, f)