
import codecs
import io
import nltk
import regex as regex
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem.porter import PorterStemmer
import re
import os,sys
import string
from multiprocessing import Pool
import unicodedata

def count(core_data):
    newcorpus=[]
    vocab=[]
    for doc in core_data:
        wordcount={}
        for word in doc:
            if word not in wordcount:
                wordcount[word] = 1
            else:
                wordcount[word] += 1
        newcorpus.append(wordcount)
        vocab+=list(wordcount)
    return [list(set(vocab)),newcorpus]

def writeDS(corpus):
    DS_file =open("final_DS.txt", "w")
    for C in corpus:
        for doc in C:
            DS_file.write(doc.strip()+"\n")
    DS_file.close()


def convert(core_data):
    curpus=core_data[0]
    vocab=core_data[1]
    finalcorpus=[]
    for doc in curpus:
        temp=""
        for word,count in doc.items():
            temp+=str(vocab[word])+":"+str(count)+" "
        finalcorpus.append(temp)
    return finalcorpus

def sent_tok(core_data):
    s_tok=[]
    for summery in core_data:
        temp=sent_tokenize(summery.strip(),'english')
        s_tok.append(temp)
    return s_tok


def wrd_tok(s_tok):
    res=[]
    for doc in s_tok:
        temp = []
        for s in doc:
            temp+=word_tokenize(s.strip(),'english')
        w_t=[]
        # remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
        for token in temp:
            if token.strip().__len__() >=3:
                w_t.append(regex.sub("[^\P{P}-]+", " ", token))
        res.append(w_t)
    return res

def exep(temp):
    dashexep=[]
    for l in temp:
        if l.__contains__("\u2014"):
            word=l.replace("\u2014","-")
            for token in word.split("-"):
                if token.__len__() >2:
                    dashexep+=token
            temp.remove(l)
            # g.write(l.replace("\u2014","-")+"\n")
    for token in dashexep:
        temp+=token
    return temp

def steming(C):
    porter = PorterStemmer()
    curpus=[]
    exeptions=[]
    for doc in C:
        stem_doc = []
        for word in doc:
            try:
                stem_doc.append(porter.stem(word))
            except:
                stem_doc.append(word)
                exeptions.append(word)
        curpus.append(stem_doc)
    return [curpus,exeptions]



def rem_stopwords(core_data):
    en_stop=core_data[1]
    C=[]
    for doc in core_data[0]:
        doc_rem=[]
        for word in doc:
            if not word.isdigit():
                if word.strip() not in en_stop:
                    if word.strip().__len__()>=3:
                        doc_rem.append(word)
        C.append(doc_rem)
    return C



def writetemp(s_tok,name):
    d=io.open(name+".txt","w",encoding="utf-8")
    for i in s_tok:
        for w in i:
            d.write(w+" ")
        d.write(u"\n")
    d.close()


def merge_vocab(vocab):
    V=[]
    for v in vocab:
        V+=v
    new_vocab=sorted(list(set(V)))
    new_vocab_dict=dict()
    # new_vocab = list(set(V))
    voc = codecs.open("vocab.txt", "w", "utf-8")
    j=0
    for v in new_vocab:
        voc.write(v+"\n")
        new_vocab_dict[v]=j
        j+=1
    voc.close()
    return new_vocab,new_vocab_dict

def lenn(data):
    t=0
    for item in data:
        t+=len(item)
    return t



if __name__ == '__main__':
    pathname = os.path.dirname(sys.argv[0])

    if (len(pathname) < 1):
        pathname = "."

    N_cores = 12
    corpus = []
    newcorpus = []
    finalcorpus = []
    vocab = []
    en_stop = dict()
    f = io.open("dataset.txt", "r", encoding="utf-8")
    data=[]
    file_size=0
    for line in f:
        file_size+=1
        data.append(line.strip())
    f.close()
    print(file_size)
    lim=int(file_size/N_cores)
    begg=0
    endd=lim
    core_data=[]
    print (endd - begg)
    for i_ in range(N_cores-1):
        core_data.append(data[begg:endd])
        begg=endd
        endd+=lim
        print (endd - begg)
    core_data.append(data[begg:])
    print ("orig",lenn(core_data))

    t=codecs.open(pathname+"/stopwords.txt","r","utf-8")
    for i in t:
        en_stop[i.strip()]=1
    t.close()
    removed_nouns=[]
    pool=Pool(N_cores)
    s_tok=pool.map(sent_tok,core_data)
    print("scentence tokenization done",lenn(s_tok))
    w_tok=pool.map(wrd_tok,s_tok)
    print("word tokenization done",lenn(w_tok))
    pool.close()
    pool.join()
    duck=[]
    for tr in w_tok:
        duck += [[tr] + [en_stop]]
    pool = Pool(N_cores)
    rem_swords = pool.map(rem_stopwords, duck)
    print("stopwords removed", lenn(rem_swords))
    result = pool.map(steming, rem_swords)
    pool.close()
    pool.join()
    stem_c = []
    exeptions = []
    for item in result:
        stem_c.append(item[0])
        if len(item[1]) > 0:
            for exep in item[1]:
                exeptions.append(exep)
    f = io.open("exeptions.txt", "w", encoding='utf-8')
    for exep in list(set(exeptions)):
        f.write(exep + "\n")
    f.close()
    pool = Pool(N_cores)
    print("stemming done", lenn(stem_c))
    result = pool.map(count, stem_c)
    pool.close()
    pool.join()
    vocab = []
    corpus = []
    for res in result:
        vocab.append(res[0])
        corpus.append(res[1])
    new_Vocab,new_vocab_dict = merge_vocab(vocab)
    print( "vocab file has been writed")
    core_data = []
    for res in corpus:
        core_data.append([res, new_vocab_dict])
    pool = Pool(N_cores)
    new_corpus = pool.map(convert, core_data)
    pool.close()
    pool.join()
    print("convertion Done!", lenn(new_corpus))
    writeDS(new_corpus)
