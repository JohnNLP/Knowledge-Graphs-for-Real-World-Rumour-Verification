class Node:
    def __init__(self, text):
        self.text = text
        self.out_edges = {}
        self.in_edges = {}
        self.type = ""
        self.locked = False
        self.open = False

    def setLocked(self):
        self.locked = True

    def setOpen(self):
        self.open = True

    def isLocked(self):
        return self.locked

    def isOpen(self):
        return self.open

    def addOutEdge(self, edge): #if 2 edges are identical, overwriting will happen! [very rare, and ok]
        self.out_edges[edge.base.text + " " + edge.text + " " + edge.target.text] = edge

    def addInEdge(self, edge):
        self.in_edges[edge.base.text + " " + edge.text + " " + edge.target.text] = edge

    def setType(self, type):
        self.type = type

    def getText(self):
        return self.text

    def printOutEdges(self):
        print("> Outgoing Edges for",self.text)
        for i in self.out_edges:
            e = self.out_edges[i]
            if e.getUncertain() and not e.getNegated():
                print(e.getBase().getText(), "(maybe)", e.getText(), "|", e.getTarget().getText())
            elif e.getNegated() and not e.getUncertain():
                print(e.getBase().getText(), "NOT", e.getText(), "|", e.getTarget().getText())
            elif e.getNegated() and e.getUncertain():
                print(e.getBase().getText(), "(maybe) NOT", e.getText(), "|", e.getTarget().getText())
            else:
                print(e.getBase().getText(), "|", e.getText(), "|", e.getTarget().getText())

    def printInEdges(self):
        print("> Incoming Edges for",self.text)
        for i in self.in_edges:
            e = self.in_edges[i]
            if e.getUncertain() and not e.getNegated():
                print(e.getBase().getText(), "(maybe)", e.getText(), "|", e.getTarget().getText())
            elif e.getNegated() and not e.getUncertain():
                print(e.getBase().getText(), "NOT", e.getText(), "|", e.getTarget().getText())
            elif e.getNegated() and e.getUncertain():
                print(e.getBase().getText(), "(maybe) NOT", e.getText(), "|", e.getTarget().getText())
            else:
                print(e.getBase().getText(), "|", e.getText(), "|", e.getTarget().getText())

    def getOutEdges(self):
        return self.out_edges

    def getInEdges(self):
        return self.in_edges

    def getEdgeCount(self):
        return len(self.in_edges) + len(self.out_edges)

    def getEdgesAsText(self):
        out = []
        for i in self.out_edges:
            e = self.out_edges[i]
            out.append(e.getBase().getText() + " " + e.getText() + " " + e.getTarget().getText())
        for i in self.in_edges:
            e = self.in_edges[i]
            out.append(e.getBase().getText() + " " + e.getText() + " " + e.getTarget().getText())
        return out

    def __repr__(self):
        return self.text

class Edge:
    def __init__(self, base, text, target):
        self.base = base
        self.text = text
        self.target = target
        self.negated = False
        self.uncertain = False

    def setNegated(self):
        self.negated = True

    def setUncertain(self):
        self.uncertain = True

    def getNegated(self):
        return self.negated

    def getUncertain(self):
        return self.uncertain

    def getText(self):
        return self.text

    def getBase(self):
        return self.base

    def getTarget(self):
        return self.target

    def __repr__(self):
        return self.base.getText() + " | " + self.text + " | " + self.target.getText()

    def __lt__(self, other):
        return len(self.text) < len(other.text)
