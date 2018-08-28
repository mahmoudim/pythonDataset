import io
import nltk
import regex as regex
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem.porter import PorterStemmer
import re
import string
from multiprocessing import Pool

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
            for u,v in wordcount.iteritems():
                totalfile.write('%s:%s '%(u,v))
            totalfile.write('\n')


def convert():
    global finalcorpus,newcorpus
    temp={}
    sorted_vocab=list(sorted(set(vocab)))
    # voc=open("vocab.txt",'w')
    voc=io.open("vocab.txt","w",encoding="utf-8")
    for l in sorted_vocab:
        voc.write(l+'\n') #.decode('ascii').replace("\u2014","-")
    voc.close()
    for wordcount in newcorpus:
        for key,value in wordcount.iteritems():
            temp[str(sorted_vocab.index(key))]=wordcount[key]
        finalcorpus.append(temp)
        temp={}

def sent_tok(summery):
    return sent_tokenize(summery,'english') #.decode('utf-8')


def wrd_tok(s_tok):
    temp=[]
    for s in s_tok:
        temp=temp+word_tokenize(s,'english')
    w_t=[]
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    for token in temp:
        if token.strip().__len__() >=3:
            w_t+=[regex.sub("[^\P{P}-]+", " ", token)]#token.translate(remove_punctuation_map)]
    return w_t


def tag_tok(temp):
    return nltk.pos_tag(temp)


def remove_nouns(tagged_tokens):
    temp=[]
    for s in tagged_tokens: #   remove nouns
          if s[1]!="NNP" and s[1]!="CD" and s[1]!="JJ":
              temp=temp+[s[0].strip(string.whitespace)]
          # else:
          #     removed_nouns.append(s)
    return temp


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

def steming(doc):
    porter = PorterStemmer()
    final_doc = []
    for word in doc:
        final_doc+=[porter.stem(word.lower())]
    return final_doc


def rem_stopwords(temp):
    en_stop=temp[1]
    rem_sw=[]
    a=temp[0]
    for i in a:
        if not en_stop.__contains__(i.strip(string.whitespace)):
            if i.strip(string.whitespace).__len__()>=3:
                rem_sw+=[i]
    return rem_sw



def writetemp(s_tok,name):
    d=io.open(name+".txt","w",encoding="utf-8")
    for i in s_tok:
        for w in i:
            d.write(w+" ")
        d.write(u"\n")
    d.close()

if __name__ == '__main__':
    N_cores=4
    corpus=[]
    newcorpus=[]
    finalcorpus=[]
    vocab=[]
    en_stop=[]
    f=io.open("dataset","r",encoding="utf-8")
    g=open("exeptation.txt",'w')
    t=open("stopwords.txt")
    for i in t:
        en_stop.append(i.strip())
    i=0
    removed_nouns=[]
    pool=Pool(N_cores)
    print("scentence tokenization start")
    s_tok=pool.map(sent_tok,f)
    print("scentence tokenization done")
    w_tok=pool.map(wrd_tok,s_tok)
    print("word tokenization done")
    tagged_tokens=pool.map(tag_tok,w_tok)
    print("tagging done")
    rem_n=pool.map(remove_nouns,tagged_tokens)
    print("nouns removed")
    # ex_w=pool.map(exep,rem_n)
    # print("exeption done")
    duck=[]
    for tr in rem_n:
        duck+=[[tr]+[en_stop]]
    rem_swords=pool.map(rem_stopwords,duck)
    print("stopwords removed")
    stem_c=pool.map(steming,rem_swords)
    print("stemming done")
    # p=open("removed_nouns.txt","w")
    # for e in removed_nouns:
    #     p.write(e.decode("ascii")+"\n")
    # p.close()
    pool.close()
    pool.join()
    corpus=stem_c
    count()
    convert()
    writeDS()
    writetemp(s_tok,"s-tok")
    writetemp(w_tok,"w-tok")
    writetemp([[s[0]] for i in tagged_tokens for s in i],"tag-tok")
    writetemp(rem_n,"nonrem-tok")
    writetemp(rem_swords,"remswords")
    writetemp(stem_c,"stem-tok")
    g.close()
    f.close()
