# DS_Final_project
![img](/data-science.png)

## Customer Churm prediction - Image Pre-Processing - NLP Text Pre-processing - Recommendation system-
### The repo Consists of 4 parts - 
  * Customer Churm Prediction - E-commerce Website Customer Churn Rate
  * Image Pre-processing -I used Pillow as my primary library for Image processing
  * NLP Text Pre-processing - It consists of 10 Pre-processing steps for the NLP model
  * Recommendation System - I Build a recommendation system using the Spotify dataset
---
Problem statement Customer Churm Prediction :
  * We do have an E-commerce dataset, we need to identify or predict who are all coming as visitors and converted as our customers. Using the given data, there is a column called “has_converted” as the target variable. Classify to find whether the user will convert as a customer or not.

STEP - 1
Import Necessary package 
```
import pandas as pd
from pycaret.classification import *
import plotly_express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from sklearn.ensemble import GradientBoostingClassifier
```
Import Dataset and Clean the data - 
```
#Importing dataset
df=pd.read_csv("Dataset.csv")v
#No of unique Obversation in each columns
df.nunique()
#Finding Null values
df.isnull().sum().sort_values(ascending=False)
#Finding Duplicate values
df.drop_duplicates(inplace=True))
#Find Info
df1.info()
```
```
Int64Index: 9207 entries, 0 to 99934
Data columns (total 23 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   count_session           9207 non-null   int64  
 1   count_hit               9207 non-null   int64  
 2   totals_newVisits        9207 non-null   int64  
 3   historic_session        9207 non-null   int64 
 ...
```
 EDA Process 
   * Plot the feature and find the distribution of the data
Building Model :
  * I use random forest to find the feature importance of the dataset
```
from sklearn.ensemble import RandomForestClassifier

x=df1.drop("has_converted",axis=1) # DROPPED THE TARGET VARIABLE TEST DATASET
y=df1["has_converted"] # ADD THE TARGET TO MAKE THE TRAIN DATASET
rf=RandomForestClassifier(n_estimators=400)
rf.fit(x,y)

	    columns              score
1 transactionRevenue	     0.508024
2 sessionQualityDim	       0.080653
3 time_on_site	           0.076759
4 num_interactions	       0.042958
5 avg_session_time_page	   0.039708
6 historic_session_page    0.039371
```
Cross-valuation to find the optimal algorithm
```
#CROSS VALIDATION OF 3 ALGORITHM
gat=GradientBoostingClassifier()
rf=RandomForestClassifier()
lg=LogisticRegression()

model=[]
cros_valu=[]

for i in (gat,rf,lg):
    model.append(i.__class__.__name__)
    cros_valu.append(cross_validate(i,X_train,y_train,scoring=("accuracy","recall","precision")))
.....

  Model	                        Accuracy	Precision	Recall
0	GradientBoostingClassifier	  0.934513	0.924061	0.917506
1	RandomForestClassifier	      0.936375	0.925073	0.921258
2	LogisticRegression	          0.785846	0.853569	0.582680

```
Based on the result I selected the Gradient Boosting as the final algorithm
```
#GradientBoostingClassifier 
gat=GradientBoostingClassifier()
gat.fit(X_train,y_train)

#Prediction
x_test_gat=scaler.transform(X_test)
gat_pred=gat.predict(x_test_gat)

#Performance Metric
print(classification_report(y_test,gat_pred))

#Confusion metric
fig,ig=plt.subplots()
sns.heatmap(confusion_matrix(y_test,gat_pred,normalize="true"),annot=True,ax=ig)
ig.set_title("Confusion Metric")
ig.set_ylabel("Real Value")
ig.set_xlabel("Prediction")
plt.show()
```
Once I make it final I pickle the algorithm
```
import pickle
with open("modelgrad.pkl","wb") as f:
    pickle.dump(gat,f)
```
Final - step once pickle made to run in streamlit application 
---
Step - 2 - Image Pre-processing
Importing necessary packages 
```
import easyocr
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt  
from PIL import Image,ImageEnhance,ImageFilter,ImageChops,ImageColor,ImageOps
```
Image prep-processing using pillow library 

```
car=Image.open("D:\DTM9\Final Project\\01.jpg")
print("Format of the Image : ",car.format,"Size of the Image :",car.size,"Mode of the Image :",car.mode)
#croping
cropped_img=car.crop((300,150,700,1000))#Tuple represent the (left,upper,right,bottom)
crop=plt.imshow(cropped_img)

#Resizing & Reduce The image 
resize=car.resize((800,900))
reduce=car.reduce(15)

#Imge Rotation 
flip_lr=car.transpose(Image.FLIP_TOP_BOTTOM)

#Convert the image modes 
#RBGA,RGB,CMYK,Binary,Grayscale
#getbands()-Return the bands in image
#convert() - Convert the image to desire format
gi=car.getbands()
gray=car.convert("L")
cmyk1=car.convert("CMYK")
rgba=car.convert("RGBA")
b=car.convert("1")

#Image equilizer 
eq=ImageOps.solarize(car)

#Image Enhancer 
en=ImageEnhance.Brightness(car)
en1=en.enhance(1.0)
```
Step - 3 Text NLP pre-processing 
Import necessary package 
```
import nltk
from nltk import word_tokenize,WordNetLemmatizer,pos_tag,PorterStemmer
from nltk.corpus import stopwords
from contractions import contractions_dict
import contractions
import re
import string
from textblob import TextBlob
from collections import Counter
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import seaborn as sns
```
Steps using NLPTK  for pre-processing 
```
#NLTK PRE_Processing step
# remove punctuation 
re_pun=text.translate(str.maketrans("","",string.punctuation))

#lowercase
lc=text.lower()

#word_tokenize
wt=word_tokenize(text)

#part of speech
ps=pos_tag(wt)

#for loop pre-processing
#stop word
word=set(stopwords.words("english"))
r_sw=[i for i in wt if i not in word]

#stem
ste=PorterStemmer()
stem=[ste.stem(i) for i in wt]

#leme
le=WordNetLemmatizer()
leme=[le.lemmatize(i) for i in wt]

#spell check 
check=str(TextBlob(text).correct())

#part of speech
part=pos_tag(token)
```
Step 4 - Building The Recommendation system
Import package 
```
#importing the important package for the model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk import WordNetLemmatizer,word_tokenize,PorterStemmer
from nltk.corpus import stopwords
import sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
```
Import Dataset and clean the data -
```
#Importing dataset
df=pd.read_csv("Dataset.csv")v
#No of unique Obversation in each columns
df.nunique()
#Finding Null values
df.isnull().sum().sort_values(ascending=False)
#Finding Duplicate values
df.drop_duplicates(inplace=True))
#Find Info
df1.info()
```
View the data info
```
RangeIndex: 57650 entries, 0 to 57649
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   artist  57650 non-null  object
 1   song    57650 non-null  object
 2   link    57650 non-null  object
 3   text    57650 non-null  object
dtypes: object(4)
```
Word tokenization as it is a content filtering 
```
stem=PorterStemmer()

def token(text):
    token=word_tokenize(text)
    stemword=[stem.stem(i) for i in token]
    return " ".join(stemword)

# use lambda funcation to apply this funcation in all the text dataset
spotiy['text']=spotiy['text'].apply(lambda x: token(x))
```
Once made the tokenizatin import  vector to make it vectorization 
```
vector=TfidfVectorizer(analyzer="word",stop_words="english")
t2n=vector.fit_transform(spotiy['text'])
```
Now build the recommendation system function 
```
def recommander(song_name):
    ids=spotiy[spotiy['song']==song_name].index[0]
    dis=sorted(list(enumerate(songsim[ids])),reverse=True,key=lambda x:x[1])  
    song=[]
    art=[]
    for i in dis[1:6]:
        s=spotiy.iloc[i[0]].song
        a=spotiy.iloc[i[0]].artist
        song.append(s)
        art.append(spotiy.iloc[i[0]].artist)
    return song,art
```
Let's see the output - 

```
recommender("Dragontown")
'Come Let Go',
'Chi',
'Come Back Baby',
'I Like It Like That',
'It Comes Back To You'],
'Xavier Rudd', 'Korn', 'Elton John', 'Kenny Loggins', 'Imagine Dragons'
```
Final step- 

* Build the Streamlit application for deployment of the project
