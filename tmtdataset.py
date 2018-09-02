import csv
f=open("new_DS.txt","r")
tmt_ds=[]
DS=[]
ln=-1
for line in f:
    ln+=1
    # print ln
    WC=[]
    t=""
    if len(line.strip().split(" "))==0:
        print("Oooops")
        exit()
    for item in line.strip().split(" "):
        if not item.strip()=="" and len(item.split(":"))>1:
            v = item.strip().split(":")[0]
            for i_ in range(int(item.strip().split(":")[1])):
                t+=v+" "
    tmt_ds.append(t.strip())
f.close()
doc_id=[]
f=open("new_ids.txt","r")
for line in f:
    doc_id.append(line.strip())
f.close()
csvfile=open("dataset.csv", 'w')
writer=csv.writer(csvfile,delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
print(len(doc_id),len(tmt_ds))
for i in range(len(doc_id)):
    writer.writerow([doc_id[i]]+[tmt_ds[i]])
