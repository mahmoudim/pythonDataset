import numpy as np
import math
import sys
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
    for doc in new_DS:
        l = len(doc)
        avg_len += l
        if l > max_len:
            max_len = l
        if l < min_len:
            min_len = l
    g.write("average document lenght: " + str(float(avg_len / num_doc)) + "\n")
    g.write("maximum document lenght: " + str(max_len) + "\n")
    g.write("minimum document lenght: " + str(min_len) + "\n")
    g.close()


if __name__ == '__main__':
    treshold=int(sys.argv[1])
    f=open("finalDS.txt","r")
    DS=[]
    # DS_word=[]
    for line in f:
        WC={}
        # W=[]
        line=line.strip().split('\t')[1]
        l=line.strip().split(" ")
        for item in l:
            d=item.strip()
            if not d=="":
                a=d.split(":")
                if(len(a)>=2):
                    WC[int(a[0])]=int(a[1])
            # W.append(int(item.strip().split(":")[0]))
        # DS_word.append()
        DS.append(WC)
    f.close()
    f=open("vocab.txt","r")
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
        tf_idf[i]= math.log(1+tf[i])*math.log(float(num_doc/df[i]))
    
    new_vocab=[]
    vocab_new_id=(np.zeros(num_vocab)-1).tolist()
    index=-1
    for id in np.argsort(tf_idf).tolist()[::-1][:treshold]:
        index+=1
        new_vocab.append(vocab[id])
        vocab_new_id[id]=index
    new_DS=[]
    new_DS_text=[]
    for doc in DS:
        WC={}
        t=""
        for word,count in doc.items():
            if not vocab_new_id[word]==-1:
                n=vocab_new_id[word]
                WC[n]=count
                t+=str(n)+":"+str(count)+" "
        new_DS.append(WC)
        new_DS_text.append(t.strip())
    g=open("new_DS.txt","w")
    for doc in new_DS_text:
        g.write(doc+"\n")
    g.close()
    g=open("new_vocab.txt","w")
    for v in new_vocab:
        g.write(v.strip()+"\n")
    g.close()
    stat(DS,new_DS,num_vocab,treshold)





