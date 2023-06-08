from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pandas as pd
import math
import numpy as np

stop_words = set(stopwords.words('english'))

stop_words.remove('in')
stop_words.remove('to')
stop_words.remove('where')



def getInputFiles(filelist):#the funcation that reed all paths in the file
    with open(filelist) as f:
        return [a for a in f.read().split()]

def preprocess(data): #the funcation that remove punc
    for p in "!.,:@#$%^&?<>*()[}{]-=;/\"\\\t\n":
        if p in '\n;?:!.,.':
            data = data.replace(p,' ')
        else: data = data.replace(p,'')
    return data.lower()

def computeTF(wordDict):
    tfDict = {}
    for word, count in wordDict.items():
        if count>0:
            tfDict[word] =1+math.log10(float(count))
        else:
            tfDict[word]=0
    return tfDict

def computeIDF(docList):#the funcation that compute idf
    N =10#the number of doc
    for word, val in docList.items():
        if N==val:  #check if num of doc =number of tf
            docList[word]=1
        else:
             docList[word] =float(math.log10(N / float(val)))

    return docList



def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        #j=float(idfs[word])
       # tfBow[word] = int(val)
        tfidf[word] = int(val)*idfs[word]
    return tfidf






print('-----------------------------------------------------part1-----------------------------------------------')
files = getInputFiles("input.txt")  #get list of all paths
filenumber = 1  #intialize the file number=1
index = {} #the dectionary than
for i in range(len(files)):
    with open(files[i]) as f:
        l=f.read()#read all things in the doc
        doc = [a for a in preprocess(l).split()]#remove the punc and split the doc and put the token in list
        f1 = [w for w in doc if not w.lower() in stop_words]#change all item to lower and remove the stopwords
        for idx, word in enumerate(f1):#use to get the postion of the all items in list
            if word in index:#check if the token is in the index
                if filenumber in index[word][0]:#check if the file number in the index
                    index[word][0][filenumber].append(idx+1)#insert the postion in the file number of token

                else:
                    index[word][0][filenumber] = [idx+1]#insert the file numer and the first postion

            else:
                index[word] = [] # initialize the list.
                index[word].append({})# initialize the dic that contain The postings list is initially empty.
                index[word][0][filenumber] = [idx+1]# Add doc number and postion .
        filenumber += 1 # Increment the file number
print("the token after remove stop words")
for l in index.keys():
 print(l)


print("-----------------------------------------------------part2-----------------------------------------------")
diction={}#the dic that contain every token and the number doc that contain the token
print("The Positional Index :-  (term :{ DocNum , pos } , ... ) \n")
for l,k in sorted(index.items()):#using to print every token and his postion
     s=len(k[0])

     print(l,":",k)
     diction[l]=int(s)#add token and the number doc that contain the 

print ("\n")
print ("The df of all token :- \n",diction)
print ("\n")


x=input("enter query phase :- ")
doc = [a for a in preprocess(x).split()]#remove the punc and split the doc and put the token in list
query = [w for w in doc if not w.lower() in stop_words]#change all item to lower and remove the stopwords
print("The query after processing    :- ",query)

h=[]
for l,k in sorted(index.items()):
    for i in range (len(sorted(query))):
        if l==query[i]:
            o=[]
            for j,n in k[0].items():
              o.append(j)

            h.append(o)
h.reverse()
print("The Docs That Match The query :- ",h,"\n")


p=list(set.intersection(*map(set,h)))
print("the matched files is :- ",p)
for k in p:
    print("And it's path is  :- ",files[k-1])
print("\n")



print("-----------------------------------------------------part3-----------------------------------------------")
print ("\n")





idf=computeIDF(diction)
print("The idf of all token :- \n",idf)
print("\n")





dic={}
for l,k in sorted(index.items()):
    dic[l]="|"
dic1={}
for l,k in sorted(index.items()):
    dic1[l]=0
    for j, n in k[0].items():
        if j == 1:
           dic1[l]=len(n)
print("1- dic1 ",dic1)
dic2={}
for l,k in sorted(index.items()):
    dic2[l]=0
    for j, n in k[0].items():
        if j == 2:
           dic2[l]=len(n)
print("2- dic2 ",dic2)
dic3={}
for l,k in sorted(index.items()):
    dic3[l]=0
    for j, n in k[0].items():
        if j == 3:
           dic3[l]=len(n)
print("3- dic3 ",dic3)
dic4={}
for l,k in sorted(index.items()):
    dic4[l]=0
    for j, n in k[0].items():
        if j == 4:
           dic4[l]=len(n)
print("4- dic4 ",dic4)
dic5={}
for l,k in sorted(index.items()):
    dic5[l]=0
    for j, n in k[0].items():
        if j == 5:
           dic5[l]=len(n)
print("5- dic5 ",dic5)
dic6={}
for l,k in sorted(index.items()):
    dic6[l]=0
    for j, n in k[0].items():
        if j == 6:
           dic6[l]=len(n)
print("6- dic6 ",dic6)
dic7={}
for l,k in sorted(index.items()):
    dic7[l]=0
    for j, n in k[0].items():
        if j == 7:
           dic7[l]=len(n)
print("7- dic7 ",dic7)
dic8={}
for l,k in sorted(index.items()):
    dic8[l]=0
    for j, n in k[0].items():
        if j == 8:
           dic8[l]=len(n)
print("8- dic8 ",dic8)
dic9={}
for l,k in sorted(index.items()):
    dic9[l]=0
    for j, n in k[0].items():
        if j == 9:
           dic9[l]=len(n)
print("9- dic9 ",dic9)
dic10={}
for l,k in sorted(index.items()):
    dic10[l]=0
    for j, n in k[0].items():
        if j == 10:
           dic10[l]=len(n)
print("1 -dic10",dic10)
print("\n\n")


print("The term Frequency for each term in each document                        \n")
df1=pd.DataFrame.from_dict([dic1,dic2,dic3,dic4,dic5,dic6,dic7,dic8,dic9,dic10]).T
df1.columns = ["doc1","doc2","doc3","doc4","doc5","doc6","doc7","doc8","doc9","doc10"]
print(df1,"\n --------------------------------------")

print ("\n\n")


# computeTF
dictfw1=computeTF(dic1)
dictfw2=computeTF(dic2)
dictfw3=computeTF(dic3)
dictfw4=computeTF(dic4)
dictfw5=computeTF(dic5)
dictfw6=computeTF(dic6)
dictfw7=computeTF(dic7)
dictfw8=computeTF(dic8)
dictfw9=computeTF(dic9)
dictfw10=computeTF(dic10)



print("\nThe term Frequency weight for each term in each document                        \n")
df2=pd.DataFrame.from_dict([dictfw1,dictfw2,dictfw3,dictfw4,dictfw5,dictfw6,dictfw7,dictfw8,dictfw9,dictfw10]).T
df2.columns = ["doc1","doc2","doc3","doc4","doc5","doc6","doc7","doc8","doc9","doc10"]
print(df2,"\n --------------------------------------\n")


# computeTFIDF and  Print TFIDf
print("\nThe TF.IDF matrix.")
tf_idf1=computeTFIDF(dictfw1,idf)
tf_idf2=computeTFIDF(dictfw2,idf)
tf_idf3=computeTFIDF(dictfw3,idf)
tf_idf4=computeTFIDF(dictfw4,idf)
tf_idf5=computeTFIDF(dictfw5,idf)
tf_idf6=computeTFIDF(dictfw6,idf)
tf_idf7=computeTFIDF(dictfw7,idf)
tf_idf8=computeTFIDF(dictfw8,idf)
tf_idf9=computeTFIDF(dictfw9,idf)
tf_idf10=computeTFIDF(dictfw10,idf)

idf_tf=pd.DataFrame.from_dict([tf_idf1,tf_idf2,tf_idf3,tf_idf4,tf_idf5,tf_idf6,tf_idf7,tf_idf8,tf_idf9,tf_idf10]).T
idf_tf.columns = ["doc1","doc2","doc3","doc4","doc5","doc6","doc7","doc8","doc9","doc10"]
print(idf_tf,"\n --------------------------------------\n")

print ("\n\n\n")

# compute The Length Of Each Docs
document_length=pd.DataFrame()
def get_docs_length(col):
    return np.sqrt(idf_tf[col].apply(lambda x: x**2).sum())
for column in idf_tf.columns :
    document_length.loc[0,column+'_len'] = get_docs_length(column)


length_of_docs = get_docs_length(idf_tf.columns)
print("The Length Of Each Docs :-\n\n",length_of_docs)
print ("\n\n")



#get_normalized(idf_tf.columns,x)
normalized_term_freq_idf = pd.DataFrame()

def get_normalized(col, x):
    try:
        return x / document_length[col+'_len'].values[0]
    except:
        return 0

for column in idf_tf.columns:
    normalized_term_freq_idf[column] = idf_tf[column].apply(lambda x : get_normalized(column, x))
print("The normalized tf_idf  :-\n")
print(normalized_term_freq_idf)
print("\n\n")
#normalized_tf_idf= get_normalized(col, x)
#print("The normalized   :-\n\n",normalized_tf_idf)
#print ("\n\n")


######################## The query processing ####################################
query_tf = {}
for l, k in sorted(index.items()):#using to put all token =0
    query_tf[l] = 0
for o in query:#using to make for loop in the query
   for l, k in sorted(query_tf.items()):#using for loop in the query_tf
        if o == l:#check if the item in the query=item in query_tf
            query_tf[l] =k+1#add 1 to the value in the query_tf
        else:
            query_tf[l] = k
print("the tf of query\n\n", query_tf)
print ("\n\n")
query_tfw =computeTF(query_tf)
print("the tfw of query\n\n", query_tfw)
print ("\n\n")
tf_idf_query=computeTFIDF(query_tfw,idf)
print("the tf.idf of query\n\n", tf_idf_query)
print ("\n\n")
##################################################################################
vector = TfidfVectorizer()
y=vector.fit_transform(index)
y=y.T.toarray()
df=pd.DataFrame(y, index=vector.get_feature_names())
q=[x]
q_vector=vector.transform(q).toarray().reshape(df.shape[0])
similarity={}
for i in range(10):
    similarity[i]= np.dot(df.loc[:,i].values,q_vector)/np.linalg.norm(df.loc[:,i])*np.linalg.norm(df.loc[:,i])

similarity_sorted=sorted(similarity.items(),key=lambda y:y[1])


for document ,score in similarity_sorted:
    if score >0.5:
        print("similarity score = ", score)
        print("doc is: ",document+1 )

