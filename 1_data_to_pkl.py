import os
import json
import math
from functools import reduce
import numpy as np
import hashlib
from preprocessing import preprocess, preprocessReaction
import torch
from operator import itemgetter
import pickle

###LOAD DATA###
rootdir = "./Data-Normal/"
print("Loading rumours from", rootdir)
#code by Andrew Clark from http://code.activestate.com/recipes/577879-create-a-nested-dictionary-from-oswalk/
dir = {} #soon-to-be tree of directories
rootdir = rootdir.rstrip(os.sep)
start = rootdir.rfind(os.sep) + 1
for path, dirs, files in os.walk(rootdir):
    folders = path[start:].split(os.sep)
    subdir = dict.fromkeys(files)
    parent = reduce(dict.get, folders[:-1], dir)
    parent[folders[-1]] = subdir
#end
print("> done")
data_dir = dir["./Data-Normal/all-rnr-annotated-threads"]
data = {}
for i in data_dir:
    for k in data_dir[i]["rumours"]:
        if k not in data:
            data[k] = {}
            data[k]["articles"] = {}
            data[k]["query"] = None
        for y in data_dir[i]["rumours"][k]:
            with open("./Data-Normal/all-rnr-annotated-threads/" + i + "/rumours/" + k + "/annotation.json") as f:
                info = json.load(f)
                temp_true = int(info["true"])
                temp_misinformation = int(info["misinformation"])
                if temp_true == 1 and temp_misinformation == 0:
                    data[k]["annotation"] = "true"
                elif temp_true == 0 and temp_misinformation == 1:
                    data[k]["annotation"] = "false"
                elif temp_true == 0 and temp_misinformation == 0:
                    data[k]["annotation"] = "unverified"
                else:
                    print("strange truth value found", temp_true, temp_misinformation)
                    continue
                data[k]["event"] = i #could be earlier but meh
            #if l == "google":
                #for y in data_dir[i]["rumours"][k]:
            if y == "google":
                for article in data_dir[i]["rumours"][k][y]:
                    if article[-4:] != "json": #ensure is not a txt file with URL
                        continue
                    with open("./Data-Normal/all-rnr-annotated-threads/" + i + "/rumours/" + k + "/google/" + article) as f:
                        data[k]["articles"][article] = {}
                        file = json.load(f)
                        if len(file["title"]) == 0 or len(file["paras"]) == 0:
                            continue
                        data[k]["articles"][article]["title"] = file["title"]
                        data[k]["articles"][article]["paras"] = file["paras"]
                        #print("T",file["title"])
                        #print("P",file["paras"])
                        data[k]["query"] = file["query"] #only 1 per rumour ofc, unless just representative URL
            elif y == "source-tweets":
                with open("./Data-Normal/all-rnr-annotated-threads/" + i + "/rumours/" + k + "/source-tweets/" + k + ".json") as f:
                    file = json.load(f)
                    data[k]["rumour"] = file["text"]
                    #print("R",file["text"])
                    data[k]["user_verified"] = file["user"]["verified"]
            elif y == "reactions":
                data[k]["reactions"] = {}
                data[k]["indirect_reactions"] = {}
                reaction_avg_length = 0
                reaction_count = 0
                for reaction in data_dir[i]["rumours"][k][y]:
                    with open("./Data-Normal/all-rnr-annotated-threads/" + i + "/rumours/" + k + "/reactions/" + reaction) as f:
                        file = json.load(f)
                        in_reply_to = file["in_reply_to_status_id_str"]
                        if in_reply_to != k: #direct replies only
                            data[k]["indirect_reactions"][reaction[:-5]] = file["text"]
                            continue
                        data[k]["reactions"][reaction[:-5]] = file["text"]
                        reaction_avg_length += len(file["text"])
                        reaction_count += 1
                data[k]["reaction_count"] = reaction_count
                if reaction_count > 0:
                    data[k]["reaction_length"] = reaction_avg_length/reaction_count
                else:
                    data[k]["reaction_length"] = 0


###PREPROCESSING###
print("Preprocessing")
missing_rumours = []
for i in data:
    try:
        data[i]["rumour"] = preprocess(data[i]["rumour"])
    except KeyError:
        print("There is no rumour for "+i+", removing it")  #todo: check why these files are not being picked up (17 of them)
        missing_rumours.append(i)
        continue
    for reaction in data[i]["reactions"]:
        r = data[i]["reactions"][reaction]
        data[i]["reactions"][reaction] = preprocessReaction(r)
    for reaction in data[i]["indirect_reactions"]:
        r = data[i]["indirect_reactions"][reaction]
        data[i]["indirect_reactions"][reaction] = preprocessReaction(r)
    for article in data[i]["articles"]:
        try:
            data[i]["articles"][article]["title"] = preprocess(data[i]["articles"][article]["title"])
        except KeyError:
            pass
            #print("There is no title")
        try:
            temp_paras = []
            for para in data[i]["articles"][article]["paras"]:
                temp_paras.append(preprocess(para))
            data[i]["articles"][article]["paras"] = temp_paras
        except KeyError:
            pass
            #print("There are no paras")
for i in missing_rumours:
    data.pop(i)
print("> done")

pickle.dump(data, open("1_original_data.pkl","wb"))
