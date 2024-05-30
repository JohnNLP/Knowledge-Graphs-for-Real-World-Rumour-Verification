import os
import json
import math
from functools import reduce
import numpy as np
import hashlib
import multiprocessing
from googleapiclient.discovery import build
from _preprocessing import preprocess
from _preprocessing import prepareRandom
from _preprocessing import prepareNormal
from _preprocessing import prepareStanfordNLP
from _preprocessing import prepareClausIE
from _preprocessing import fixDate
from _webcleaner import getContent
import time

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
data_dir = dir["./Data_PHEME9/all-rnr-annotated-threads"]
events = []
for i in data_dir:
    events.append((i, data_dir[i]["rumours"])) #rumours only

def get_rumours_twitter(files, basepath):
    got = 0
    error = 0
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
                    got += 1
            except: #not all rumours have a class provided
                #####rumour_dict.pop(rumour)
                print("Error reading truth value")
                error += 1
                #####continue
            rumour_dict[rumour]["source-tweets"][rumour] = all
        with open(basepath+"/"+rumour+"/structure.json", encoding="utf-8") as g: #response tree
            for r in files[rumour]["reactions"]:
                with open(basepath+"/"+rumour+"/reactions/"+r) as f: #responses to base rumour
                    #print(rumour, r, f)
                    data = json.load(f)
                    if data["lang"] != "en":
                        continue
                    tag = r[:-5] #remove the ".json" to get response id
                    response = preprocess(data["text"])
                    rumour_dict[rumour]["reactions"][tag] = response
    print("no errors reading truth value for", got, "/", got+error)
    return rumour_dict

print("Setting up data...")
all_tweets = {}
for i in events:
    print("> processing", i[0][:-16])
    temp = get_rumours_twitter(i[1], "./Data_PHEME9/all-rnr-annotated-threads/"+i[0]+"/rumours")
    all_tweets[i[0]] = temp  # stratify by event

'''
print("Sampling available tweets...")
removed = 0
kept = 0
for event in list(all_tweets): #force copy of keys
    for number in list(all_tweets[event]):
        if int(number[-12:-6])%11 != 0:
            removed += 1
            all_tweets[event].pop(number)
        else:
            kept += 1
print("> kept " + str(kept) + " of " + str(kept+removed) + " rumours")
'''

'''
print("Gathering urls...")
urls_paths = {}
for i in all_tweets:
    for j in all_tweets[i]:
        for k in all_tweets[i][j]:
            for l in all_tweets[i][j][k]:
                for url in all_tweets[i][j][k][l].urls:
                    path = "./Data_PHEME9/all-rnr-annotated-threads/"+i+"/rumours/"+j+"/web/"
                    urls_paths[url] = path

count = 0
total = 0
print("Downloading linked articles...")
#get valid URLs and download their data
for url in urls_paths:
    total += 1
    try:
        hash = hashlib.md5(url.encode("utf-8")).hexdigest()
        path = urls_paths[url]
        if os.path.isfile(path+hash+".json"): #already downloaded
            count += 1
            continue
        print(url)
        if not os.path.exists(path): #make path if not exists
            os.mkdir(path)
        content = getContent(url)
        with open(path+hash+".json", "w") as f:
            json.dump(content, f)
    except Exception as e:
        print("Error -", e)
print(count)
print(total)

'''

print("Gathering search queries...")
paths_queries = {}
for i in all_tweets:
    for j in all_tweets[i]:
        for l in all_tweets[i][j]["source-tweets"]:
            path = "./Data_PHEME9/all-rnr-annotated-threads/"+i+"/rumours/"+j+"/google/"
            paths_queries[path] = prepareNormal(all_tweets[i][j]["source-tweets"][l])

#print(len(paths_queries))
#exit()

print("Downloading articles using web search...")

api_key = "AIzaSyBEjyJVekBC6lVwrtV48MjhrtNKnQX4I54"
cse_id = "63aace4a671524f7f"

got = 0
done = 0
failed = 0

def google_query(query, api_key, cse_id):
    time.sleep(3) #avoid minute rate limit
    try:
        print(query)
        query_service = build("customsearch", "v1", developerKey=api_key)
        query_results = query_service.cse().list(q=query, cx=cse_id, num=10).execute()
        print(query_results)
        if query_results["searchInformation"]["totalResults"] != "0":
            return query_results["items"]
        else:
            print("No results")
            return []
    except Exception as e:
        print("Error -", e)
        return []

#
def handler(signum, frame):
    raise Exception("Too slow")
#

print("Saving to files...")
for path in paths_queries:
    print(done)
    done += 1
    results = google_query(paths_queries[path],api_key,cse_id)
    if results == []:
        failed += 10
    for result in results:
        try:
            url = result["link"]
            hash = hashlib.md5(url.encode("utf-8")).hexdigest()
            #if os.path.isfile(path + hash + ".json"):  # already downloaded
            #    continue
            if not os.path.exists(path):  # make path if not exists
                os.mkdir(path)
            #
            #signal.signal(signal.SIGALRM, handler)
            #signal.alarm(2)
            #
            #time.sleep(1)
            try:
                #uncomment this block for direct webpage retrieval
                '''
                content = getContent(url)
                content["query"] = paths_queries[path]
                with open(path + hash + ".json", "w") as f:
                    json.dump(content, f)
                got += 1
                print("Got")
                '''
                #delete this block for direct webpage retrieval
                with open(path + hash + ".txt", "w", encoding="utf-8") as f:
                    f.write(url)
                    f.close()
                #signal.alarm(0)
            except Exception as e:
                failed += 1
                print(e)
                #signal.alarm(0)
        except Exception as e:
            failed += 1
            print("Error -", e)

print(got)
print(failed)
