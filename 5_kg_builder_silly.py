import pickle
from kg import *

with open("4_data_and_triples.pkl", 'rb') as f:
    all_data = pickle.load(f)
with open("4_nouns.pkl", 'rb') as f:
    nouns2 = pickle.load(f)

events = []
for i in all_data:
    events.append(all_data[i]["event"])
events = set(events)
events = list(events)

#matching triples with their original evidence sentence
triple_origins = {}
for i in events:
    triple_origins[i] = {}
for i in all_data:
    raw_evidence = all_data[i]["evidence"]
    triples = all_data[i]["evidence_chunks"]
    event = all_data[i]["event"]
    for v, j in enumerate(triples):
        article = " ".join(raw_evidence[v])
        article_triples = triples[v]
        for triple in article_triples:
            temp = []
            for spo in triple:
                temp += spo
            temp = " ".join(temp)
            triple_origins[event][temp.lower()] = article

with open('5_triple_origins.pkl', 'wb') as f:
    pickle.dump(triple_origins, f, protocol=pickle.HIGHEST_PROTOCOL)

#gather nouns
nouns = {}
nouns_shortened = {}
for i in events:
    nouns[i] = {}
    nouns_shortened[i] = {}
for event in nouns2:
    for i in nouns2[event]:
        if i.lower() == "breaking":
            continue
        nouns[event][i.lower()] = nouns2[event][i]
        nouns_shortened[event][i.lower()[:-2]] = nouns2[event][i]

for rumour in all_data:
    all_data[rumour]["rumour"] = all_data[rumour]["rumour"].split(" ")
with open('5_data_and_triples.pkl', 'wb') as f:
    pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)

no_evidence_v2 = [] #rumours for which no evidence was found

for rumour in all_data:
    #overlap = event-relevant nouns in this rumour
    rumour_set = set([i[:-2].lower() for i in all_data[rumour]["rumour"]])
    event_set = set(i[:-2].lower() for i in nouns[all_data[rumour]["event"]])
    overlap = rumour_set.intersection(event_set)
    if "" in overlap:
        overlap.remove("")

    #find evidence triples containing words in overlap [why not all rumour related ones!?]
    new_evidence = []
    new_evidence_rarities = []
    matched_word = ""
    for evidence in all_data[rumour]["evidence_chunks"]:
        for triple in evidence:
            if len(triple) != 3: #this can rarely happen
                continue
            good = False
            for part in triple:
                for word in part:
                    if word[:-2].lower() in overlap: #todo: an interesting approach to share evidence between happenings (use others of that word?)
                        matched_word = word.lower()[:-2]
                        good = True
            if good or True:
                #print(triple)
                new_evidence.append([triple[0],triple[1],triple[2]])
                new_evidence_rarities.append(1) #frequency of the last matched word in the event as a whole

    #up to 20 triples per event
    print(len(new_evidence))
    if len(new_evidence) >= 20:
        temp = []
        for v, i in enumerate(new_evidence):
            temp.append((new_evidence_rarities[v], i))
        temp = sorted(temp, reverse=True)
        new_evidence = []
        for i in range(20):
            new_evidence.append(temp[i][1])

    #if no matches ignore the rumour
    if len(new_evidence) == 0:
        no_evidence_v2.append(rumour)
        continue

    #update evidence with the triples, ready for graphing
    all_data[rumour]["evidence"] = []
    for v, i in enumerate(new_evidence):
        all_data[rumour]["evidence"].append(i)

#todo: for now we pop the non-matchers
for i in no_evidence_v2:
    all_data.pop(i)

#build the graph
nodes = {}
for rumour in all_data:
    #print(all_data[rumour]["evidence"])
    event = all_data[rumour]["event"]
    if event not in nodes:
        nodes[event] = {}
    for i in all_data[rumour]["evidence"]:
        subj = i[0][0].lower()
        obj = i[2][0].lower()
        if subj not in nodes[event]:
            nodes[event][subj] = Node(subj)
        if obj not in nodes[event]:
            nodes[event][obj] = Node(obj)
        p = Edge(nodes[event][subj], i[1][0].lower(), nodes[event][obj])
        nodes[event][subj].addOutEdge(p)
        nodes[event][obj].addInEdge(p)
        #print("I added",p,"to",subj,"and",obj)
        #nodes[event][obj].printInEdges()
        #if obj == "supermarket" and subj == "several hostages":
        #    exit()

'''
#remove edges without nodes
no_edges = []
for i in nodes:
    if nodes[i].getEdgeCount == 0:
        no_edges.append(i)
for i in no_edges:
    nodes.pop(i)
'''
#print the remainder
for rumour in nodes:
    print(rumour+"\n\n")
    for node in nodes[rumour]:
        print("=======")
        nodes[rumour][node].printOutEdges()
        nodes[rumour][node].printInEdges()

for rumour in all_data:
    print(all_data[rumour]["rumour"])
    good = []
    words = all_data[rumour]["rumour"]
    event = all_data[rumour]["event"]
    words = [i.lower() for i in words]
    for node in nodes[event]:
        actual_node = nodes[event][node]
        for fact in actual_node.getEdgesAsText():
            overlap = 0
            for word in fact.split(" "):
                if word.lower() in words:
                    overlap += 1
            if overlap > 0:
                good.append([overlap, fact])
    good = sorted(good, reverse=True)
    best = good[:10]
    all_data[rumour]["evidence"] = [i[1] for i in best]
    for i in best:
        print(i)
    print(len(good))

#event defining nouns are good for facts, but not matching facts to rumours
for i in nouns:
    print(i, nouns[i])

import sys
sys.setrecursionlimit(10000)
with open('5_kg.pkl', 'wb') as f:
    pickle.dump(nodes, f, protocol=pickle.HIGHEST_PROTOCOL)

print("===")

#get connected components
all_connected_components = []
graph = nodes[events[0]].copy()
locked = []
opened = []
while True:
    #print("===")
    this_component = []
    found_start = False
    for word in graph: #open the first available node
        node = graph[word]
        if node.isLocked():
            continue
        else:
            node.setOpen()
            opened.append(node.getText())
            found_start = True
            break
    if not found_start: #all nodes are locked (done)
        break
    #print(opened)
    while len(opened) > 0:
        new_opens = []
        for word in opened:
            node = graph[word]
            out_edges = node.getOutEdges()
            in_edges = node.getInEdges()
            for i in out_edges:
                other_node = out_edges[i].getTarget()
                if not other_node.isLocked() and not other_node.isOpen():
                    new_opens.append(other_node.getText())
                    other_node.setOpen()
            for i in in_edges:
                other_node = in_edges[i].getBase()
                if not other_node.isLocked() and not other_node.isOpen():
                    new_opens.append(other_node.getText())
                    other_node.setOpen()
            this_component.append(word)
            locked.append(word)
            node.setLocked()
        opened = [i for i in set(new_opens)]
        #print(opened)
    all_connected_components.append(this_component)

print("found",len(all_connected_components),"connected components")
for i in all_connected_components:
    print(len(i), i)

#'''
from pyvis.network import Network
graph = nodes[events[3]].copy()

net = Network()
nodes = [i+":" for i in graph]
net.add_nodes(nodes)

for i in graph:
    out = graph[i].getOutEdges()
    for j in out:
        net.add_edge(graph[i].getText()+":",out[j].getTarget().getText()+":")

net.show('mygraph.html')
#'''



