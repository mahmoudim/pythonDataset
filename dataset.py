import io
import nltk
import regex as regex
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem.porter import PorterStemmer
import re
import string
from multiprocessing import Pool
import sys, os

global newcorpus
global finalcorpus
global corpus
global vocab
global f
global g
global t


def count():
    global corpus,newcorpus,vocab
    for doc in corpus:
        wordcount={}
        for word in doc:
            if word not in wordcount:
                wordcount[word] = 1
            else:
                wordcount[word] += 1
        newcorpus.append(wordcount)
        vocab+=list(wordcount)


def writeDS():
    global finalcorpus
    with open ("finalDS.txt","w") as totalfile:
        index=0
        for wordcount in finalcorpus:
            index+=1
            totalfile.write('%s\t'%(wordcount.__len__()))
            for u,v in wordcount.items():
                totalfile.write('%s:%s '%(u,v))
            totalfile.write('\n')


def convert():
    global finalcorpus,newcorpus
    temp={}
    sorted_vocab=list(sorted(set(vocab)))
    sorted_vocab_dict={}
    num=0
    for v in sorted_vocab:
        sorted_vocab_dict[v]=str(num)
        num+=1
    # voc=open("vocab.txt",'w')
    voc=io.open("vocab.txt","w",encoding="utf-8")
    for l in sorted_vocab:
        voc.write(l+'\n')
    voc.close()
    for wordcount in newcorpus:
        for key,value in wordcount.items():
            temp[sorted_vocab_dict[key]]=wordcount[key]
        finalcorpus.append(temp)
        temp={}

def sent_tok(summery):
    return sent_tokenize(summery,'english') #.decode('utf-8')


def wrd_tok(s_tok):
    temp=[]
    for s in s_tok:
        temp=temp+word_tokenize(s,'english')
    w_t=[]
    
    for token in temp:
        if token.strip().__len__() >=3:
            w_t+=[regex.sub("[^\P{P}-]+", " ", token)]
    return w_t


def tag_tok(temp):
    return nltk.pos_tag(temp)

def steming(doc):
    porter = PorterStemmer()
    final_doc = []
    for word in doc:
        final_doc+=[porter.stem(word.lower())]
    return final_doc


def rem_stopwords(temp):
    en_stop=temp[1]
    rem_sw=[]
    for i in temp:
        if i.strip(string.whitespace) not in en_stop:
            if i.strip(string.whitespace).__len__()>=3:
                rem_sw+=[i]
    return rem_sw


if __name__ == '__main__':
    
    pathname = os.path.dirname(sys.argv[0])        
    
    if(len(pathname)<1):
        pathname="."

    N_cores=12
    corpus=[]
    newcorpus=[]
    finalcorpus=[]
    vocab=[]
    en_stop=dict()
    f=io.open("dataset.txt","r",encoding="utf-8")
    g=open( pathname+"/exeptation.txt",'w')
    t=open( pathname+"/stopwords.txt")
    for i in t:
        en_stop[i.strip()]=1
    i=0
    pool=Pool(N_cores)
    s_tok=pool.map(sent_tok,f)
    print("scentence tokenization done")
    w_tok=pool.map(wrd_tok,s_tok)
    print("word tokenization done")
    rem_swords=pool.map(rem_stopwords,w_tok)
    print("stopwords removed")
    stem_c=pool.map(steming,rem_swords)
    print("stemming done")
    
    pool.close()
    pool.join()
    corpus=stem_c
    count()
    convert()
    writeDS()
    g.close()
    f.close()
