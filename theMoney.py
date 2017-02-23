import plotly
plotly.tools.set_credentials_file(username='vworri', api_key='xxxxxxxxxxxxxxxxxxxxx')
import sys
import re
from plotly import tools
import plotly.plotly as py
from plotly.tools import FigureFactory as FF
import plotly.graph_objs as go

import pandas as pd
import math as m
import numpy as np
import pprint
y = []
l = []
f = sys.argv[1]
K = int(sys.argv[2])
txt = open(f)
txt = txt.read()
txt = re.sub(r'[^\w\s]','',txt)
body = txt.lower().split()
primaryWord = input("Put in Primary term: ")
secondaryWord= input("Put in Secondary term: ")

def find_cooc(A,B,k=K):
    count = body.count(A)
    cooccurrence =0
    index = 0
    while index < len(body):
        item = body[index]
        if (item == A and  (B in body[index+1:k+index+1] or B in body[index- k-1:index-1])):
            cooccurrence += 1
        index +=1
        index+=0
    return cooccurrence/(1 + count)

def wordCount():
    wordCountDict = {}
    for word in body:
        if word not in wordCountDict:
            wordCountDict[word] = body.count(word)
    return wordCountDict

def wordCountlog(Wordcount):
    wordCountDictLog = {}
    for key, val in Wordcount.items():
        wordCountDictLog[key] = 1 + m.log(float(val))
    return wordCountDictLog.values()

def invDocFreq(Wordcount):
    N = 1
    num_docs_per_word = 1
    wordCountDictLog = {}
    for key, val in Wordcount.items():
        wordCountDictLog[key] =m.log(1 + N/num_docs_per_word)
    return wordCountDictLog.values()


def Augfreq(Wordcount,maxfreq):
    augmentedFrequency = {}
    for key, val in Wordcount.items():
        augmentedFrequency[key] =  0.5 + 0.5*(val/maxfreq)
    return augmentedFrequency
def createDataFrame(Wordcount,Wordcountlog,WordcountAug,WordcountInv):
    TextDataframe = pd.DataFrame(index=Wordcount.keys())
    TextDataframe['Raw Frequency']= (Wordcount.values())
    TextDataframe['Log Frequency']= (Wordcountlog)
    TextDataframe['Augmented Frequency']= (WordcountAug)
    TextDataframe['Inverse Document Term Frequency']= WordcountInv
    TextDataframe['TF-IDF'] = TextDataframe['Raw Frequency'] * TextDataframe['Inverse Document Term Frequency']
    return TextDataframe
def createScatterFreqs (Wordcount,Wordcountlog):
    trace1 = go.Scatter(
        x = list(Wordcount.keys()),
        y = list(Wordcount.values()),
        mode = 'markers',
        name = "Raw Frequency",
        marker = dict(color = 'rgb(174, 53, 214)')
    )
    trace2 = go.Scatter(
        x = list(Wordcount.keys()),
        y = list(Wordcountlog),
        mode = 'markers',
        name = "Log Frequency",
        marker = dict(color = 'rgb(65, 77, 244)')
    )
    layout = dict(
        title = "Log and Raw Frequencies",
        xaxis = dict(title= 'Unique Words'),
        yaxis = dict(title= 'Frequency')
    )

    data = [trace1, trace2]
    plotly.offline.plot({
        "data": data,
        "layout": layout})
def creatHeatMap(Wordcount,Wordcountlog):
    heatMap = [
        go.Contour(
            z = list(Wordcountlog),
            x = list(Wordcount.keys()),
            y = list(Wordcount.values()),
            colorscale = 'magma'
        )
    ]
    py.iplot(heatMap)

def createWordMatrix(wordcount,k=K):
    matrix1 = []
    for k1, v1 in wordcount.items():
        matrix2 = []
        for k2, v1 in wordcount.items():
            matrix2.append(find_cooc(k1,k2,k))
        matrix1.append(matrix2)
    return matrix1


def makeKgraphs(Wordcount, Wordcountkey):
    for k in np.arange(1,6):
        matrix = createWordMatrix(Wordcount, k)
        trace = [go.Contour(
            z = matrix,
            x = list(Wordcountkey),
            y = list(Wordcountkey),
        )]
        py.iplot(trace, filename=('Catinthehat K= '+k))
########################################################################################################################
#get counts of all unique words in txt
#w_c = wordCount()
########################################################################################################################
#get value of most frequent word
#maxfreq = max(list(w_c.values()))
########################################################################################################################
#get counts of all unique words in txt on a log scale
#w_c_log = wordCountlog(w_c)
########################################################################################################################
#get augmented frequency of each unique word (take into account how many texts you are working with)
#augmentedFrequency = Augfreq(w_c)
########################################################################################################################
#try and find the interesting values by looking for rare words
#inverseDocumentFreq = invDocFreq(w_c)
########################################################################################################################
#create pandas DataFrame

def TextDataFrame():
    w_c = wordCount()
    maxfreq = max(list(w_c.values()))
    w_c_log = wordCountlog(w_c)
    augmentedFrequency = Augfreq(w_c,maxfreq)
    inverseDocumentFreq = invDocFreq(w_c)
    TextInfo = createDataFrame(w_c,w_c_log,augmentedFrequency,inverseDocumentFreq)
    return TextInfo
print(TextDataFrame().head())
