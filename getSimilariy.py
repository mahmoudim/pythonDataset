import multiprocessing
import numpy as np
import csv
import sklearn.metrics.pairwise as pws
import sys


def read_theta(file_name):
    doc_id=[]
    theta=[]
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            R=[]
            doc_id.append(row[0])
            for item in row[1:]:
                # print item
                R.append(float(item))
            theta.append(R)
    return theta,doc_id


def normalize(theta):
    th=[]
    for row in theta:
        t=[]
        for item in row:
            if item<0:
                t.append(0.0)
            else:
                t.append(item)
        s=sum(t)
        temp=[]
        for item in t:
            temp.append(item/s)
        th.append(temp)
    return th


def getsimHelinger(a, b):
    return (1.0/np.sqrt(2))*np.sqrt(np.power(np.sqrt(a) - np.sqrt(b), 2).sum())


if __name__ == '__main__':
    file_name = sys.argv[1]
    theta, doc_id = read_theta(file_name)
    norm_theta = np.asarray(normalize(theta))

    res1 = pws.pairwise_distances(norm_theta, metric=getsimHelinger, n_jobs=12)
    for i in range(0,len(res1[0])):
        res1[i][i] = 1
    res1 = 1 - res1
    res1 = ((res1 - np.min(res1)) / (np.max(res1) - np.min(res1)))
    np.save("simHelinger", res1)

    with open("new_ids.txt", 'w') as resultFile:
        for str in doc_id:
            resultFile.write(str+"\n")


