import os
import json
import math
from functools import reduce
import numpy as np
import hashlib
import multiprocessing
from googleapiclient.discovery import build
from _webcleaner import getContent
from _preprocessing import preprocess
from _preprocessing import fixDate
from _preprocessing import prepareNormal
import time

#WARNING: USES SOME SORT OF OLD PREPROCESSING
#although there is preprocessing in the legacy code, it might never be used in its current state -- didn't check properly

#
import signal
#

#base data directory
rootdir = "./Data_PHEME9/"
print("Loading rumours from", rootdir)

#build a tree of the data directories
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

def get_rumours_twitter(files, basepath):
    rumour_dict = {}
    for rumour in files: #iterate through the rumour+response files
        rumour_dict[rumour] = {"source-tweets": {}, "reactions": {}}
        with open(basepath+"/"+rumour+"/source-tweets/"+rumour+".json", encoding="utf-8") as f: #the base rumour
            data = json.load(f)
            all = preprocess(data["text"])
            if data["lang"] != "en":
                continue
            all.date = fixDate(data["created_at"])
            try:
                with open(basepath + "/" + rumour + "/annotation.json", encoding="utf-8") as g: # response tree
                    info = json.load(g)
                    temp_true = int(info["true"])
                    temp_misinformation = int(info["misinformation"])
                    if temp_true == 1 and temp_misinformation == 0:
                        all.label = "true"
                    elif temp_true == 0 and temp_misinformation == 1:
                        all.label = "false"
                    elif temp_true == 0 and temp_misinformation == 0:
                        all.label = "unverified"
                    else:
                        print("strange truth value found", temp_true, temp_misinformation)
            except: #not all rumours have a class provided
                #####rumour_dict.pop(rumour)
                print("Error reading truth value")
                #####continue
            rumour_dict[rumour]["source-tweets"][rumour] = all
        with open (basepath+"/"+rumour+"/structure.json", encoding="utf-8") as g: #response tree
            for r in files[rumour]["reactions"]:
                with open(basepath+"/"+rumour+"/reactions/"+r) as f: #responses to base rumour
                    data = json.load(f)
                    if data["lang"] != "en":
                        continue
                    tag = r[:-5] #remove the ".json" to get response id
                    response = preprocess(data["text"])
                    rumour_dict[rumour]["reactions"][tag] = response
    return rumour_dict

print("Setting up data...")

data_dir = dir["./Data_PHEME9/all-rnr-annotated-threads"]
events = []
for i in data_dir:
    events.append((i, data_dir[i]["rumours"])) #rumours only

all_tweets = {}
for i in events:
    print("> processing", i[0][:-16])
    temp = get_rumours_twitter(i[1], "./Data_PHEME9/all-rnr-annotated-threads/"+i[0]+"/rumours")
    all_tweets[i[0]] = temp  # stratify by event

print("Gathering search queries...")
paths_queries = {}
for i in all_tweets:
    for j in all_tweets[i]:
        for l in all_tweets[i][j]["source-tweets"]:
            if "google" not in dir["./Data_PHEME9/all-rnr-annotated-threads"][i]["rumours"][j]:
                continue
            path = "./Data_PHEME9/all-rnr-annotated-threads/"+i+"/rumours/"+j+"/google/"
            for file in dir["./Data_PHEME9/all-rnr-annotated-threads"][i]["rumours"][j]["google"]:
                if file[-4:] == "json":
                    continue
                with open(path+file, "r") as f:
                    url = f.readline()
                    paths_queries[path+file] = url

got = 0
done = 0
had = 0
failed = 0
for path in paths_queries:
    print(done, "/", len(paths_queries))
    done += 1
    url = paths_queries[path]
    truncated_path = path[:-36]
    hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    if os.path.isfile(truncated_path + hash + ".json"):  # already downloaded
        print("Had")
        had += 1
        continue
    try:
        content = getContent(url)
        content["query"] = paths_queries[path]
        with open(truncated_path + hash + ".json", "w") as f:
            json.dump(content, f)
        got += 1
        print("Got")
    except Exception as e:
        failed += 1
        print(e)
        print(url)
        #signal.alarm(0)


print(done, got, had, failed)














