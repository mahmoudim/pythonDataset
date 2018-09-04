import numpy as np
import math
import sys,os

def stat(DS,new_DS,num_vocab,treshold):
    g = open("statictics.txt", "w")
    g.write("number of documents: " + str(num_doc) + "\n")
    g.write("size of vocabulary: " + str(num_vocab) + "\n")
    avg_len = 0.0
    max_len = 0
    min_len = 10000
    zero_doc=0
    for doc in DS:
        l = len(doc)
        avg_len += l
        if l > max_len:
            max_len = l
        if l < min_len:
            min_len = l
    g.write("average document lenght: " + str(float(avg_len / num_doc)) + "\n")
    g.write("maximum document lenght: " + str(max_len) + "\n")
    g.write("minimum document lenght: " + str(min_len) + "\n")
    g.write("size of removed words: " + str(num_vocab - treshold) + "\n")
    g.write("\t*****\tafter remove words\t*****\n ")
    avg_len = 0.0
    max_len = 0
    min_len = 10000
    under5=0
    under10=0
    for doc in new_DS:
        l = len(doc)
        avg_len += l
        if l > max_len:
            max_len = l
        if l < min_len:
            min_len = l
        if l < 5:
            under5+=1
        if l < 10:
            under10+=1
    g.write("average document lenght: " + str(float(avg_len / num_doc)) + "\n")
    g.write("maximum document lenght: " + str(max_len) + "\n")
    g.write("minimum document lenght: " + str(min_len) + "\n")
    g.write("document lenght under 5: " + str(under5) + "\n")
    g.write("document lenght under 10: " + str(under10) + "\n")
    g.close()


if __name__ == '__main__':
    treshold = int(sys.argv[1])
    removeDoctreshold=5
    f=open("../final_DS.txt","r")
    DS=[]
    # DS_word=[]
    for line in f:
        WC={}
        # W=[]
        for item in line.strip().split(" "):
            if not item.strip()=="":
                WC[int(item.strip().split(":")[0])]=int(item.strip().split(":")[1])
            # W.append(int(item.strip().split(":")[0]))
        # DS_word.append()
        DS.append(WC)
    f.close()
    f=open("../vocab.txt","r")
    vocab=[]
    for line in f:
        vocab.append(line.strip())
    f.close()
    num_vocab=len(vocab)
    num_doc=len(DS)
    tf=np.zeros(num_vocab).tolist()
    df=np.zeros(num_vocab).tolist()
    tf_idf = np.zeros(num_vocab).tolist()
    for doc in DS:
        for word,count in doc.items():
            tf[word]+=count
            df[word]+=1
    for i in range(num_vocab):
        tf_idf[i]= math.log(tf[i]+1)*math.log(float(num_doc/df[i])+1)
    new_vocab=[]
    vocab_new_id=(np.zeros(num_vocab)-1).tolist()
    index=-1
    for id in np.argsort(tf_idf).tolist()[::-1][:treshold]:
        index+=1
        new_vocab.append(vocab[id])
        vocab_new_id[id]=index
    new_DS=[]
    new_DS_text=[]
    ids=open("../ids")
    idfiles=ids.read()
    idlines=idfiles.split('\n')
    newiddocs=[]
    docCount=0
    for doc in DS:
        WC={}
        valid=0
        t=""
        for word,count in doc.items():
            if not vocab_new_id[word]==-1:
                n=vocab_new_id[word]
                WC[n]=count
                t+=str(n)+":"+str(count)+" "
                valid+=1
        new_DS.append(WC)
        if valid>removeDoctreshold :
            newiddocs.append(idlines[docCount])
            new_DS_text.append(str(valid)+" "+t.strip())
        docCount+=1
    g=open("new_DS.txt","w")
    for doc in new_DS_text:
        g.write(doc+"\n")
    g.close()
    g=open("new_ids.txt","w")
    for doc in newiddocs:
        g.write(doc+"\n")
    g.close()
    g=open("new_vocab.txt","w")
    for v in new_vocab:
        g.write(v.strip()+"\n")
    g.close()
    stat(DS,new_DS,num_vocab,treshold)