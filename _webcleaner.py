import urllib.request, urllib.parse
from urllib.request import Request
import bs4

def getContent(url):
    request = Request(url, None, {'User-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.35 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.35'}) #counters anti-robot filters
    response = urllib.request.urlopen(request, timeout=10)
    webContent = response.read()
    stuff = bs4.BeautifulSoup(webContent,"html.parser")
    title = stuff.find('title').get_text()
    if url[0:12] == "https://t.co" or url[0:11] == "http://t.co": #bypass twitter URL shortening
        request = Request(title, None, {'User-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.35 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.35'})  # counters anti-robot filters
        response = urllib.request.urlopen(request, timeout=10)
        webContent = response.read()
        stuff = bs4.BeautifulSoup(webContent, "html.parser")
        title = stuff.find('title').get_text()

    paras = stuff.findAll('p',limit=10)
    dict = {"title" : title, "paras" : []}

    remove_start = True
    for p in paras:
        if isinstance(p, bs4.element.Tag):
            if len(p.getText()) < 80:
                if remove_start == True:
                    continue
            else:
                remove_start = False
            if len(p.getText()) > 0:
                dict["paras"].append(p.getText())

    return dict
