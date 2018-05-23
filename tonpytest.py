import pandas as pd
import csv
import re 
import enchant
import os.path
import numpy as np
from pdfminer.pdfparser import PDFParser,PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal,LAParams
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
from tensorflow.contrib import learn
def changePdfToText(filePath):
    file = open(path, 'rb') 
    praser = PDFParser(file)
    doc = PDFDocument()
    praser.set_document(doc)
    doc.set_parser(praser)
    doc.initialize()
    if not doc.is_extractable:
        raise PDFTextExtractionNotAllowed
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    pdfStr = ''
    for page in doc.get_pages(): 
        interpreter.process_page(page)
        layout = device.get_result()
        for x in layout:
            if (isinstance(x, LTTextBoxHorizontal)):
                pdfStr = pdfStr + x.get_text() #+ '\n'
    pat = '[a-zA-Z]+'  
    text = pdfStr
    cleantext = re.findall(pat,text)
    #d = enchant.Dict("en_US")
    #clean_text = ' '.join(c for c in cleantext if d.check(c) is True)
    oldcount = len(cleantext)
    if oldcount <= 10000:
        return cleantext,oldcount
    else:
        onectext = cleantext[0:10000]
        count = len(onectext)
        return onectext,count
    
path = '/home/yuri/Convolutional Neural Networks for Sentence Classification.pdf'
result = changePdfToText(path)

#input glove
glove = pd.read_table('/home/yuri/glove.6B.100d.txt',sep=" ",index_col=0,header=None,quoting=csv.QUOTE_NONE,encoding='ISO-8859-1')
glove_index = list(glove.index)
glove_dict = {}
for i in range (len(glove_index)):
    glove_dict.update({glove_index[i]:i})
keys = list(glove_dict.keys())
lenkeys = len(keys)
oldlist = result[0]
listnum = result[1]
print (listnum)

left=[]
for i  in range(0,listnum):
    if oldlist[i] not in keys:
        left.append(oldlist[i])
max_document_length = max([len(x.split(" ")) for x in left])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(left)))
print (len(left))
print (len(x))

checknum = listnum
s = []
for i in range(0,checknum):
    if oldlist[i] in keys:
        num = keys.index(oldlist[i])
        s.append(num)
    else:
        j = left.index(oldlist[i])
        loc = x[j] + lenkeys
        y = list(loc)
        z = loc[0]
        s.append(z)
print (s)
