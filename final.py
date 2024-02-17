#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#:Important package to be import:
import streamlit as st  
import pandas as pd
from PIL import Image,ImageColor,ImageEnhance,ImageFilter,ImageOps
from streamlit_option_menu import option_menu 
import plotly_express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import easyocr
import string
from nltk import word_tokenize,WordNetLemmatizer,pos_tag,PorterStemmer
from nltk.corpus import stopwords
import re
import string
from textblob import TextBlob
from collections import Counter
import spacy
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
import spotipy
from spotipy import SpotifyClientCredentials
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#:PAGE STEPUP:
st.set_page_config(page_title="DS_FINAL PROJECT",
                   layout="wide",
                   page_icon="🧊",
                   initial_sidebar_state="collapsed"
                   )
#Hide the streamlit hambuger Icon,footer note and header
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

image1=Image.open("D:\DTM9\Final Project\data-science (1).png")
st.image(image1)
reader=easyocr.Reader(['en'])

if st.session_state.get('switch_button', False):
    st.session_state['menu_option'] = (st.session_state.get('menu_option',0) + 1) % 5
    manual_select = st.session_state['menu_option']
else:
    manual_select = None

with st.sidebar:    
    selected = option_menu("Main menu", ["Customer Conversion", "Image processing","NLP Pre-processing","Recommendation",'About'], 
        icons=['check2-all', 'images','body-text','award-fill','person-lines-fill'],
        manual_select=manual_select, key='menu_4')
st.button(f":red[Switch Tab] {st.session_state.get('menu_option',1)}", key='switch_button')
selected
df=pd.read_csv("D:\DTM9\Final Project\classification_data1.csv")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#:customer churn prediction:
if selected == "Customer Conversion":
    with open("modelgrad.pkl","rb") as f:
        gat_model= pickle.load(f)
    st.subheader(":gray[Customer Churn Prediction]")
    st.write("###### Select The Parameter")
    cl1,cl2=st.columns(2)
    with cl1:
        revenue=st.number_input("Revenue",min_value=0,max_value=15464117626,step=10)
        session=st.slider("Session Quality Dimension",0,150,10)
        num_inter=st.slider("Number of Interaction",0,197630,1500)
    with cl2:
        history=st.slider("Historic Session Page",0,99896,1200)
        time=st.slider("Time on Site",0,1250267,1600)
        average=st.slider("Average Session on page",0.0,5441.0,50.2)

    data={"Revenue":revenue,"Session_Quality":session,"Num_interaction":num_inter,"Historic_Session":history,
          "Time_on_site":time,"Average_session":average}
    data1=pd.DataFrame([data])
    st.write("###### Selected Parameter")
    st.dataframe(data1)
    cl1,cl2=st.columns(2)
    with cl1:
        d1=[[revenue,session,num_inter,history,time,average]]
        on=st.button("Predict")
        if on:
            pred=gat_model.predict(np.array(d1))
            if pred:
                st.write("#### :green[Converted as Customer]")
            else:
                st.write("#### :red[Not Converted as Customer]") 
    st.write("---")
    st.markdown("<h5 style='text-align: center; color: #949494;'>Evaluation Metric</h5>", unsafe_allow_html=True)
    cl3,cl4=st.columns(2)
    with cl3:
        #Finding the distribution of top 6 feature importance
        on=st.toggle("Distribution of Features")
        if on:
            x=df[['transactionRevenue',"sessionQualityDim","time_on_site","num_interactions","historic_session_page","avg_session_time_page"]]
            for i in x.columns:
                plt.figure(figsize=(10, 5))
                sns.histplot(data=x[i], bins=30, kde=True)
                plt.title('Distribution of Feature Importance Variable')
                st.pyplot(plt)
    with cl4:
        on=st.toggle("Preformance comparation")
        if on:
            X=df[['transactionRevenue',"sessionQualityDim","time_on_site","num_interactions","historic_session_page","avg_session_time_page"]]
            y=df['has_converted']
            X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.5,random_state=45)

            gat=GradientBoostingClassifier()
            rf=RandomForestClassifier()
            lg=LogisticRegression()

            model=[]
            cros_valu=[]

            for i in (gat,rf,lg):
                model.append(i.__class__.__name__)
                cros_valu.append(cross_validate(i,X_train,y_train,scoring=("accuracy","recall","precision")))
                
            recall=[]
            precision=[]
            accuracy=[]
            for i in range(len(cros_valu)):
                accuracy.append(cros_valu[i]['test_accuracy'].mean())
                precision.append(cros_valu[i]['test_precision'].mean())
                recall.append(cros_valu[i]['test_recall'].mean())

            data={'Model':model,"Accuracy":accuracy,"Precision":precision,"recall":recall}
            performance=pd.DataFrame(data)
            st.dataframe(performance)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#:Image preprocessing:
if selected == "Image processing":
    st.subheader("Image Pre-Processing")
    upload_image=st.file_uploader("Upload the image ",type=["jpg","png","jpeg"])
    if upload_image is not None:
        image1=Image.open(upload_image)
        st.image(image1,width=250)
        st.write("Format of Image :",image1.format,"--","Mode of the Image :",image1.mode)
        st.write("---")
        st.markdown("<h4 style='text-align:center;color:#949494;'>Basic Resizing and Cropping</h4>",unsafe_allow_html=True)
        cl1,cl2,cl3=st.columns([1,1,1])
        with cl1:
            tog=st.toggle("Cropping")
            if tog:
                col1,col2,col3=st.columns([1,1,1])
                with col1:
                    sd=image1.crop((0,0,800,600))
                    sd1=st.button("Standard Photo size 4:3")
                    if sd1:
                        st.image(sd,width=250)
                with col2:
                    sd=image1.crop((0,0,1920,1080))
                    sd1=st.button("HD Photo size 16:9")
                    if sd1:
                        st.image(sd,width=250)
                with col1:
                    sd=image1.crop((0,0,1080,1080))
                    sd1=st.button("Instagram Photo size 4:3")
                    if sd1:
                        st.image(sd,width=250)

        with cl2:
            on=st.toggle("Resize")
            if on:
                height=st.number_input("Height",min_value=1,max_value=2580,step=100)
                width=st.number_input("Width",min_value=1,max_value=2580,step=100)
                resize=image1.resize((height,width))
                st.image(resize)
        with cl3:
            on=st.toggle("Reduce")
            if on:
                times=st.number_input("Reduce in Times",min_value=1,max_value=250,step=10)
                reduce=image1.reduce(times)
                st.image(reduce)
        st.write("---")
        st.markdown("<h4 style='text-align:center;color:#949494;'>Basic Manipulation</h4>",unsafe_allow_html=True)
        cl1,cl2,cl3=st.columns([1,1,1])
        with cl1:
            on=st.toggle("Rotation")
            if on:
                flip_lr=st.button("Flip left right")
                if flip_lr:
                    flip_lr1=image1.transpose(Image.FLIP_LEFT_RIGHT)
                    st.image(flip_lr1)
                flip_TB=st.button("Flip Top Bottom")
                if flip_TB:
                    flip_lr1=image1.transpose(Image.FLIP_TOP_BOTTOM)
                    st.image(flip_lr1)
                flip_TP=st.button("Transpose")
                if flip_TP:
                    flip_lr1=image1.transpose(Image.TRANSPOSE)
                    st.image(flip_lr1)
                flip_TV=st.button("Transverse")
                if flip_TV:
                    flip_lr1=image1.transpose(Image.TRANSVERSE)
                    st.image(flip_lr1)
        with cl2:
            on=st.toggle("Change Image Format")
            if on:
                g=st.button("Grayscale")
                if g:
                    gray=image1.convert("L")
                    st.write("Image Formart : ",image1.format)
                    st.image(gray)
                r=st.button("Monochrome")
                if r:
                    rgba=image1.convert("1")
                    st.write("Image Format :",image1.format)
                    st.image(rgba)
                c=st.button("CMYK")
                if c:
                    cmyk=image1.convert("CMYK")
                    st.write("Image Format : ",image1.format)
                    st.image(cmyk)
        with cl3:
            on=st.toggle("Color convertion and Equalizer")
            if on:
                eq=st.button("Equalizer")
                if eq:
                    e=ImageOps.equalize(image1)
                    st.image(e)
                iv=st.button("Invert")
                if iv:
                    i=ImageOps.invert(image1)
                    st.image(i)
                mi=st.button("Mirror")
                if mi:
                    mir=ImageOps.mirror(image1)
                    st.image(mir)
        st.write("---")
        st.markdown("<h4 style='text-align:center;color:#949494;'>Basic Enhancement and Filter</h4>",unsafe_allow_html=True)
        cl01,cl02=st.columns([1,1])
        with cl01:
            on=st.toggle("Image Enhancement")
            if on:
                bi=st.button("Enhance Brightness")
                #fl=st.number_input("Enter the value",1.0,20.0,step=1.0)
                if bi:
                    bi=ImageEnhance.Brightness(image1)
                    en1=bi.enhance(2.0)
                    st.image(en1)
            if on:
                bi=st.button("Enhance Color")
                #fl=st.number_input("Enter the value",1.0,20.0,step=1.0)
                if bi:
                    bi=ImageEnhance.Color(image1)
                    en1=bi.enhance(2.0)
                    st.image(en1)
            if on:
                bi=st.button("Enhance Contrast")
                #fl=st.number_input("Enter the value",1.0,20.0,step=1.0)
                if bi:
                    bi=ImageEnhance.Contrast(image1)
                    en1=bi.enhance(2.0)
                    st.image(en1)
        with cl02:
            on=st.toggle("Image Filter")
            if on:
                bi=st.button("Image blur")
                if bi:
                    dt=image1.filter(ImageFilter.BLUR)
                    st.image(dt)
                bi=st.button("Image Boxblur")
                if bi:
                    dt=image1.filter(ImageFilter.BoxBlur(2.0))
                    st.image(dt)
                bi=st.button("Image Edge Enhance")
                if bi:
                    dt=image1.filter(ImageFilter.EDGE_ENHANCE)
                    st.image(dt)
                bi=st.button("Image Details")
                if bi:
                    dt=image1.filter(ImageFilter.DETAIL)
                    st.image(dt)
                bi=st.button("Image Find Edges")
                if bi:
                    dt=image1.filter(ImageFilter.FIND_EDGES)
                    st.image(dt)
                bi=st.button("Image Smooth")
                if bi:
                    dt=image1.filter(ImageFilter.SMOOTH)
                    st.image(dt)
        st.write("---")
        st.markdown("<h4 style='text-align:center;color:#949494;'>Text Detection on Image</h4>",unsafe_allow_html=True)
        on=st.toggle("Image to text Conversion")
        if on:
            reader=easyocr.Reader(["en"])
            result=reader.readtext(image1,detail=0)
            st.text(result)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#:Text preprocessing:
if selected=="NLP Pre-processing":
    st.subheader("Text Pre-Processing")
    textbox=st.text_input("Enter the Text for Pre-processing")
    st.markdown("<h4 style='text-align:center;color:#949494;'>Pre-processing steps</h4>",unsafe_allow_html=True)
    cl1,cl2,cl3,cl4=st.columns([1,1,1,1])
    with cl1:
        button=st.button("Remove Punctuation ")
        if button:
            re_pun=textbox.translate(str.maketrans("","",string.punctuation))
            st.write(re_pun)
    with cl2:
        button=st.button("Lower case")
        if button:
            #lowercase
            lc=textbox.lower()
            st.write(lc)
    with cl3:
        button=st.button("Word Tokenization")
        if button:#word_tokenize
            wt=word_tokenize(textbox)
            st.text(wt)
    with cl4:
        button=st.button("Part of Speech")
        if button:
            #part of speech
            wt=word_tokenize(textbox)
            ps=pos_tag(wt)
            st.text(ps)
    st.write("---")
    cl1,cl2,cl3=st.columns([1,1,1])
    with cl1:
        button=st.button("Spell check")
        if button:
            #spell check
            sp=str(TextBlob(textbox).correct())
            st.write(sp)
    with cl2:
        button=st.button("Remove Numbers")
        if button:
            #remove number 
            re_num=re.sub(r'\d+',"",textbox)
            st.write(re_num)
    with cl3:
        button=st.button("Remove Special character")
        if button:        
            #remove specil character
            re_sp=re.sub(r'[^A-Za-z0-9\s]','',textbox)
            st.write(re_sp)
    st.write("---")
    cl1,cl2,cl3=st.columns([1,1,1])
    with cl1:
        button=st.button("Remove stop word")
        if button:
            wt=word_tokenize(textbox)
            word=set(stopwords.words("english"))
            r_sw=[i for i in wt if i not in word]
            st.dataframe(pd.DataFrame({"Remove stop words":r_sw}))
    with cl2:
        button=st.button("Stemming")
        if button:
            wt=word_tokenize(textbox)
            #stem
            ste=PorterStemmer()
            stem=[ste.stem(i) for i in wt]
            st.dataframe(pd.DataFrame({"Stemming":stem}))
    with cl3:
        button=st.button("Lemmatization")
        if button:
            wt=word_tokenize(textbox)
            #leme
            le=WordNetLemmatizer()
            leme=[le.lemmatize(i) for i in wt]
            st.dataframe(pd.DataFrame({"Lemmatized":leme}))
    st.write("---")
    st.markdown("<h4 style='text-align:center;color:#949494;'>High and low frequent words</h4>",unsafe_allow_html=True)
    cl1,cl2=st.columns([1,1])
    with cl1:
        button=st.toggle("High Frequent words")
        if button:
            wt=word_tokenize(textbox)
            word=set(stopwords.words("english"))
            r_sw=[i for i in wt if i not in word]        
            coun_word=Counter(r_sw)
            n=st.number_input("Enter frequent time",min_value=2,max_value=100,step=1)
            ther=n
            hw_word=[i for i,count in coun_word.items()if count>ther]
            remove_hw=[i for i in wt if i not in hw_word]
            df=pd.DataFrame({"High Frequen":hw_word})
            st.dataframe(df.head(8))
            df=pd.DataFrame({"Filter from Frequent WORDS":remove_hw})
            st.dataframe(df.head(8))
    with cl2:
        button=st.toggle("RARE words")
        if button:
            wt=word_tokenize(textbox)
            word=set(stopwords.words("english"))
            r_sw=[i for i in wt if i not in word]        
            coun_word=Counter(r_sw)
            n=st.number_input("Rare times",min_value=1,max_value=100,step=1)
            ther=n
            hw_word=[i for i,count in coun_word.items()if count<ther]
            remove_hw=[i for i in wt if i not in hw_word]
            df=pd.DataFrame({"RARE WORDS":hw_word})
            st.dataframe(df.head(8))
            df=pd.DataFrame({"Filter from RARE WORDS":remove_hw})
            st.dataframe(df.head(8))
    st.write("---")
    st.markdown("<h4 style='text-align:center;color:#949494;'>Key word extraction and word cloud</h4>",unsafe_allow_html=True)
    cl1,cl2=st.columns([1,1])
    with cl1:
        on=st.toggle("Keyword extraction")
        if on:
            nlp=spacy.load("en_core_web_sm")
            def key_word(text):
                r=[]
                pos=['PROPN','NOUN','ADJ']
                doc=nlp(text.lower())# load the desired text and convert into in spacy
                for i in doc:
                    if(i.text in nlp.Defaults.stop_words or i.text in string.punctuation):
                        continue
                    if(i.pos_ in pos):
                        r.append(i.text)
                return r
            output=set(key_word(textbox))
            out=key_word(textbox)
            df=pd.DataFrame({"keyword":out})
            st.dataframe(df.head(8))
    with cl2:
        on=st.toggle("Wordcloud")
        if on:
            nlp=spacy.load("en_core_web_sm")
            def key_word(text):
                    r=[]
                    pos=['PROPN','NOUN','ADJ']
                    doc=nlp(text.lower())# load the desired text and convert into in spacy
                    for i in doc:
                        if(i.text in nlp.Defaults.stop_words or i.text in string.punctuation):
                            continue
                        if(i.pos_ in pos):
                            r.append(i.text)
                    return r
            output=key_word(textbox)
            wc=" ".join(output)
            wordcl=WordCloud(width=200,height=150,background_color="white").generate(wc)
            plt.figure(figsize=(10,15))
            plt.imshow(wordcl)
            plt.axis("off")
            st.pyplot(plt)
    st.write("---")
    st.markdown("<h4 style='text-align:center;color:#949494;'>Sentiment Analysis</h4>",unsafe_allow_html=True)
    on=st.toggle("Sentiment Analysis")
    if on:
        se=SentimentIntensityAnalyzer()
        sen=se.polarity_scores(textbox)
        df=pd.DataFrame([sen])
        st.dataframe(df)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#:Recommendation System:
if selected=="Recommendation":
    st.header("Music-Recommendation-System")
    st.caption("Using Cosine Similarity - Content filtering")
    #df=pickle.load(open("df_music.pkl","rb"))
    df=pd.read_csv("D:\DTM9\Final Project\spotiy.csv")
    rec=pickle.load(open("music_recommendation.pkl","rb"))


    client_id="7a6fecb0006b49f1928339407e6bc240"
    secrect_id="1230af281002470f96b9b80ada86fe77"
    
    client=SpotifyClientCredentials(client_id=client_id,client_secret=secrect_id)
    spotify=spotipy.Spotify(client_credentials_manager=client)

    def poster(songname,artist_n):
        query=f"track:{songname} artist:{artist_n}"
        results=spotify.search(q=query,type="track")

        if results and results['tracks']['items']:
            track=results['tracks']['items'][0]
            album_url=track['album']['images'][0]['url']
            print(album_url)
            return album_url
        else:
            return"https://i.postimg.cc/0QNxYz4V/social.png"

    def recommender(song_name):
        id=df[df['song']==song_name].index[0]
        result1=sorted(list(enumerate(rec[id])),reverse=True,key=lambda x:x[1])
        song=[]
        posters=[]
        for i in result1[1:6]:
            art=df.iloc[i[0]].artist
            print(art)
            s=df.iloc[i[0]].song
            print(s)
            posters.append(poster(s,art))
            song.append(s)
        return song,posters
    
    music_list=df['song'].values
    select=st.selectbox("select the song",music_list)

    if st.toggle('Show Recommendation'):
        recommended_music_names,recommended_music_posters = recommender(select)
        col1, col2, col3, col4, col5= st.columns(5)
        with col1:
            st.text(recommended_music_names[0])
            st.image(recommended_music_posters[0])
        with col2:
            st.text(recommended_music_names[1])
            st.image(recommended_music_posters[1])

        with col3:
            st.text(recommended_music_names[2])
            st.image(recommended_music_posters[2])
        with col4:
            st.text(recommended_music_names[3])
            st.image(recommended_music_posters[3])
        with col5:
            st.text(recommended_music_names[4])
            st.image(recommended_music_posters[4])
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#:About:
if selected=="About":
    st.subheader(":gray[My Contact]")
    st.image(Image.open("D:\\DTM9\\CS-3\\flyer.png"))
    st.subheader(":black[Project - Airbnb Data Anlysis]")
    st.link_button(":blue[LinkedIn]","https://www.linkedin.com/in/narayana-ram-sekar-b689a9201/")
    st.link_button(":black[GitHub]","https://github.com/Narennrs1")
