#!/usr/bin/env python
# coding: utf-8

# <h1> Import Library

# <h3> Import pustaka yang diperlukan

# In[1]:


import librosa # untuk menganalisa musik atau audio
import soundfile # untu membaca dan membuat file audio
import os, glob, pickle 
# untuk membuat / menghapus file (os), untuk mengambil file / menentukan path yang cocok (glob)
# converting object to bytek (pickle)
import numpy as np # untuk array multidimensi dan untuk proses array
from sklearn.metrics import accuracy_score # untuk menghitung tingkat akurasi
# from sklearn.pipeline import make_pipeline


# <h1> Exploratory Data Analysis

# <h3> Mengambil sample untuk masing-masing tipe suara untuk tujuan eksplorasi data</h3>
# <p> Directory perlu disesuaikan dengan penyimapanan pribadi</p>

# In[5]:


sample_calm_path = "C:\\MLCourse\\MLCourse\\Ilmudata\\speech_data\\Actor_10\\03-01-02-01-01-01-10.wav"
sample_happy_path = "C:\\MLCourse\\MLCourse\\Ilmudata\\speech_data\\Actor_10\\03-01-03-01-01-01-10.wav"
sample_sad_path = "C:\\MLCourse\\MLCourse\\Ilmudata\\speech_data\\Actor_10\\03-01-04-01-01-01-10.wav"
sample_angry_path = "C:\\MLCourse\\MLCourse\\Ilmudata\\speech_data\\Actor_10\\03-01-05-01-01-01-10.wav"
sample_surprised_path = "C:\\MLCourse\\MLCourse\\Ilmudata\\speech_data\\Actor_10\\03-01-08-01-01-01-10.wav"
samples = [sample_calm_path, sample_happy_path, sample_sad_path, sample_angry_path, sample_surprised_path]


# In[6]:


sample_calm_path2 = "C:\\MLCourse\\MLCourse\\Ilmudata\\speech_data\\Actor_07\\03-01-02-01-01-01-07.wav"
sample_happy_path2 = "C:\\MLCourse\\MLCourse\\Ilmudata\\speech_data\\Actor_07\\03-01-03-01-01-01-07.wav"
sample_sad_path2 = "C:\\MLCourse\\MLCourse\\Ilmudata\\speech_data\\Actor_07\\03-01-04-01-01-01-07.wav"
sample_angry_path2 = "C:\\MLCourse\\MLCourse\\Ilmudata\\speech_data\\Actor_07\\03-01-05-01-01-01-07.wav"
sample_surprised_path2 = "C:\\MLCourse\\MLCourse\\Ilmudata\\speech_data\\Actor_07\\03-01-08-01-01-01-07.wav"
samples_2 = [sample_calm_path2, sample_happy_path2, sample_sad_path2, sample_angry_path2, sample_surprised_path2]


# In[7]:


# mengimport class IPython.display untuk menampilkan suara
import IPython.display as ipd
ipd.Audio(sample_calm_path)


# In[9]:


# membaca dan menampilkan sample rate
# SoundFile => audio library
with soundfile.SoundFile(sample_calm_path) as sound_file:
    X = sound_file.read(dtype="float32")
    sample_rate=sound_file.samplerate
print(f"Sample rate is {sample_rate} Hz")


# In[10]:


# visualisasi file suara
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import librosa.display
import seaborn as sns
sns.set()

x , sr = librosa.load(sample_calm_path)
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)


# Grafik diatas adalah gelombang suara dari salah satu sampel yang diambil

# In[11]:


# Visualisasi suara berdasarkan frekuensi dengan menggunakan Transformasi Fourier Diskrit
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()


# Gelombang diatas adalah grafik Frekuensi vs Time dari sampel yang sama

# In[12]:


librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()


# Gelombang diatas adalah grafik log Frekuensi vs Time dari sampel yang sama

# In[13]:


# MFCCs => Mel-frequency cepstral coefficients 
mfccs = librosa.feature.mfcc(x, sr=sample_rate)
print(mfccs.shape)


# In[14]:


#The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope. It models the characteristics of the human voice.
#Displaying  the MFCCs:
plt.figure(figsize=(15, 7))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')


# In[15]:


chromagram = librosa.feature.chroma_stft(x, sr=sample_rate)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')


# Grafik diatas adalah Chromagram dari sampel di atas. Chromagram adalah pitch yang dimiliki oleh suara yang terekam. Chromagram biasanya digunakan untuk klasifikasi genre musik sehingga Pitch memiliki jarak dari nada C ke nada B

# Kemudian dibawah kita akan mencoba mencari pola dari Chromagram untuk setiap sampel emosi. Kita memiliki dua sampel untuk setiap emosi sehingga kita mencoba dua kali.

# In[17]:


# Mengabungkan hasil semua visualisasi
observed_emotions=['calm', 'happy', 'sad', 'angry', 'surprised']
num_of_emotions = len(samples)
chroma_list = []

for emotion in samples:
    x = librosa.load(emotion)[0]
    chroma_list.append(librosa.feature.chroma_stft(x, sr=sample_rate))

fig, axs = plt.subplots(num_of_emotions, figsize=(15, 25))
fig.suptitle('chromagram for multiple emotions')

for i in range(num_of_emotions):
    librosa.display.specshow(chroma_list[i], x_axis='time', y_axis='chroma', cmap='coolwarm', ax=axs[i])
    axs[i].set_title(observed_emotions[i])


# In[18]:


observed_emotions=['calm', 'happy', 'sad', 'angry', 'surprised']
num_of_emotions = len(samples_2)
chroma_list = []

for emotion in samples_2:
    x = librosa.load(emotion)[0]
    chroma_list.append(librosa.feature.chroma_stft(x, sr=sample_rate))

fig, axs = plt.subplots(num_of_emotions, figsize=(15, 25))
fig.suptitle('chromagram for multiple emotions')

for i in range(num_of_emotions):
    librosa.display.specshow(chroma_list[i], x_axis='time', y_axis='chroma', cmap='coolwarm', ax=axs[i])
    axs[i].set_title(observed_emotions[i])


# Dari sini dapat disimpulkan bahwa dengan penglihatan saja kita tidak dapat menemukan pola.

# <h1> Feature Engineering

# Dengan informasi dari website terpercaya, kita melakukan Feature Engineering dengan cara mengambil 3 fitur penting yang bisa didapatkan dari file audio yaitu MFCC(Mel Frequency Cepstrum Coefficient), Chroma, dan MEL

# Mel adalah transformasi logaritmik dari frekuensi gelombang <br>
# Chroma adalah pitch dari gelombang <br>
# MFCC adalah gelombang yang diproses dengan Discrete Fourier Transform, Log Amplitude Spectrum kemudian Mel Scaling

# In[20]:


# https://www.youtube.com/watch?v=4_SH2nfbQZ8
#https://towardsdatascience.com/how-i-understood-what-features-to-consider-while-training-audio-files-eedfb6e9002b
# mengambil fitur penting dari file suara
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result


# In[21]:


def extract_individual_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
#             result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
#             result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
#             result=np.hstack((result, mel))
    return mfcc, chroma, mel


# Berikut adalah label emosi yang diberikan dan kita hanya memilih untuk memprediksi 5 emosi

# In[22]:


# membuat dictionary "emotions" yang berisikan tipe emosi
# misalnya 01 untuk neutral dan 02 untuk calm
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

observed_emotions=['calm', 'happy', 'sad', 'angry', 'surprised']


# In[23]:


from sklearn.model_selection import train_test_split
def load_data():
    x,y=[],[]
    for file in glob.glob("C:\\MLCourse\\MLCourse\\Ilmudata\\speech_data\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return np.array(x), y


# In[24]:


#memisahkan data mfcc, chroma, dna mel dalam list yang berbeda
# file y untuk jenis emosi
def load_individual_data():
    mfcc, chroma, mel ,y = [],[],[],[]
    for file in glob.glob("C:\\MLCourse\\MLCourse\\Ilmudata\\speech_data\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        a,b,c=extract_individual_feature(file, mfcc=True, chroma=True, mel=True)
        mfcc.append(a)
        chroma.append(b)
        mel.append(c)
        y.append(emotion)
    return mfcc,chroma,mel, y


# <h1> Create Dataframe</h1> 
# <h4> -Untuk visualisasi lebih baik lagi mengenai data yang dimiliki</h4>

# In[25]:


import pandas as pd # mengimport library pandas sebagai pd
# membuat list dengan isi "mfcc", "chroma", "mel", "emotions"
column_names = ["mfcc", "chroma", "mel", "emotions"] 
df = pd.DataFrame(columns = column_names)# Membuat data frame dengan label column names


# In[26]:


mfcc,chroma,mel,emo = load_individual_data()


# In[27]:


df["mfcc"] = mfcc # mengisi kolom "mfcc" dengan data mfcc
df["chroma"] = chroma # mengisi kolom "chroma" dengan data chorma 
df["mel"] = mel # mengisi kolom "mel" dengan data mel
df["emotions"] = emo # mengisi kolom "emo" dengan data emo


# In[28]:


df.head()# menampilkan 5 data teratas


# Menghitung berapa banyak data yang diberikan untuk masing-masing sampel emosi

# In[63]:


df['counts'] = df.emotions.map(df.emotions.value_counts())
df 


# In[70]:


emotions = ['calm', 'happy', 'sad', 'angry', 'surprised']
count = []
for i in range(df["emotions"].size):
    if df["emotions"].iloc[i] in emotions:
        emotions.remove(df["emotions"].iloc[i])
        count.append(df["counts"].iloc[i])
print(count)


# In[72]:


plt.bar(observed_emotions, count, align='center')


# Terlihat bahwa distribusi kelas adalah rata. Masing-masing kelas memiliki 192 sampel. Hal ini penting untuk menentukan performance measure yang sesuai.

# Nilai chroma dari sampel audio satu adalah

# In[29]:


df["chroma"].iloc[0]


# Nilai mel dari sampel audio satu adalah

# In[30]:


df["mel"].iloc[0]


# In[31]:


df["chroma"].iloc[1].size


# In[32]:


df["mel"].iloc[1].size


# In[33]:


index_list = df[(df['mfcc'] == False)].index.tolist()


# In[34]:


index_list


# <h1> Train_test_split

# In[37]:


X, y= load_data()


# In[38]:


# membagi data untuk test dan train
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[39]:


# menghitung jumlah data x_train dan x_test
shape = (x_train.shape[0], x_test.shape[0])
shape # menampilkan ukuran


# In[53]:


x_train[0]# nilai data x_train pada index ke-0


# Karena data terdistribusi dengan seimbang, maka performance measure yang digunakan adalah akurasi

# <h1> One Versus Rest

# In[55]:


from sklearn.svm import SVC # import SVC
from sklearn.multiclass import OneVsRestClassifier # mengimport OneVsRestClassifier
clf = SVC() # menyimpan SVC sebagai label clf
ovr = OneVsRestClassifier(clf)
model = ovr.fit(x_train, y_train) # membuat model


# In[56]:


y_pred_ovr = model.predict(x_test)# prediksi x_train dari model yang sudah dibuat


# In[57]:


# Menampilkan nilai keakurasi dari model yang sudah ada
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_ovr)
print("Accuracy: {:.2f}%".format(accuracy*100))


# In[58]:


# menampikan confusion matrix dari model yang sudah dibuat
from sklearn.metrics import confusion_matrix 
print(confusion_matrix(y_test,y_pred_ovr))


# <h1> One Versus One

# In[59]:


# membuat model dengan menggunakan SVC One versus one
from sklearn.multiclass import OneVsOneClassifier
clf2 = SVC()
ovo = OneVsOneClassifier(clf2)
model2 = ovo.fit(x_train, y_train) # Membuat model


# In[60]:


y_pred_ovo = model2.predict(x_test) 
#memprediksi x_test dengan model yang sudah dibuat (model2)


# In[61]:


# menampilkan tingkat keakurasi
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_ovo)
print("Accuracy: {:.2f}%".format(accuracy*100))


# In[73]:


print(confusion_matrix(y_test,y_pred_ovo))


# <h1> Unspecified

# In[74]:


clf3 = SVC()
model3 = clf3.fit(x_train, y_train)


# In[75]:


y_pred = model3.predict(x_test)


# In[76]:


accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))


# <h1> Tuning

# Menggunakan Grid Search untuk Tuning untuk mendapatkan parameter yang terbaik

# In[33]:


param_grid = {'estimator__C': [0.1,1, 10, 100], 'estimator__gamma': [1,0.1,0.01,0.001],'estimator__kernel': ['rbf', 'poly ', 'sigmoid']}

model_to_tune = OneVsRestClassifier(SVC())


# In[34]:


from sklearn.model_selection import GridSearchCV


# In[46]:


grid = GridSearchCV(model_to_tune ,param_grid ,refit=True, verbose=3)
grid.fit(X,Y)


# In[47]:


#svc c=1, gamma=0.001, kernel: rbf
print(grid.best_params_)


# In[48]:


# from pprint import pprint
# pprint(grid.cv_results_)


# In[77]:


best_clf = SVC(C=1, gamma=0.001, kernel='rbf')
best_model = best_clf.fit(x_train, y_train)


# In[78]:


y_pred2 = best_model.predict(x_test)


# In[79]:


accuracy = accuracy_score(y_true=y_test, y_pred=y_pred2)
print("Accuracy: {:.2f}%".format(accuracy*100))


# In[80]:


print(confusion_matrix(y_test,y_pred2))


# <h1> Ensembel Method = XGBoost

# In[81]:


from xgboost import XGBClassifier


# In[82]:


def labelEncoder(y):
    temp = []
    for i in range(len(y)):
        if y[i] == 'calm':
            temp.append(0)
        if y[i] == 'happy':
            temp.append(1)
        if y[i] == 'sad':
            temp.append(2)
        if y[i] == 'angry':
            temp.append(3)
        if y[i] == 'surprised':
            temp.append(4)
    return temp


# In[83]:


def labelDecoder(y):
    temp = []
    for i in range(len(y)):
        if y[i] == 0:
            temp.append('calm')
        if y[i] == 1:
            temp.append('happy')
        if y[i] == 2:
            temp.append('sad')
        if y[i] == 3:
            temp.append('angry')
        if y[i] == 4:
            temp.append('suprised')
    return temp


# In[84]:


y_train_encoded = labelEncoder(y_train)


# In[85]:


ensemble_model_encoding = XGBClassifier(use_label_encoder=False)
ensemble_model_encoding.fit(x_train, y_train_encoded)


# In[86]:


y_pred_ensemble_encoding = ensemble_model_encoding.predict(x_test)


# In[87]:


y_pred_ensemble_encoding = labelDecoder(y_pred_ensemble_encoding)


# In[88]:


accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_ensemble_encoding)
print("Accuracy: {:.2f}%".format(accuracy*100))


# In[89]:


print(confusion_matrix(y_test,y_pred_ensemble_encoding))


# In[90]:


ensemble_model = XGBClassifier()
ensemble_model.fit(x_train, y_train)


# In[91]:


y_pred_ensemble = ensemble_model.predict(x_test)


# In[92]:


accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_ensemble)
print("Accuracy: {:.2f}%".format(accuracy*100))


# In[93]:


print(confusion_matrix(y_test,y_pred_ensemble))


# In[ ]:


['calm', 'happy', 'sad', 'angry', 'surprised']


# In[99]:


wrongPred = ["Calm_Happy", "Calm_Sad" ,"Calm_Angry", "Calm_Surprised",
             "Happy_Calm", "Happy_Sad", "Happy_Agry","Happy_Surprised", 
             "Sad_Calm","Sad_Happy","Sad_Angry","Sad_Surprised",
             "Agry_Calm","Angry_Happy","Angry_Sad", "Angry_Surprised",
            "Surprised_Calm", "Surprised_Happy", "Surprised_Sad" ,"Surprised_Angry"]
count = [0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0]

tes = 0
for i in range(len(y_test)):
    if y_test[i] != y_pred_ensemble[i]:
        if y_pred_ensemble[i] == "calm":
            if y_test[i] == "happy":
                count[0] += 1
            if y_test[i] == "sad":
                count[1] += 1
            if y_test[i] == "angry":
                count[2] += 1
            if y_test[i] == "surprised":
                count[3] += 1
        if y_pred_ensemble[i] == "happy":
            if y_test[i] == "calm":
                count[4] += 1
            if y_test[i] == "sad":
                count[5] += 1
            if y_test[i] == "angry":
                count[6] += 1
            if y_test[i] == "surprised":
                count[7] += 1
        if y_pred_ensemble[i] == "sad":
            if y_test[i] == "happy":
                count[9] += 1
            if y_test[i] == "calm":
                count[8] += 1
            if y_test[i] == "angry":
                count[10] += 1
            if y_test[i] == "surprised":
                count[11] += 1
        if y_pred_ensemble[i] == "angry":
            if y_test[i] == "happy":
                count[13] += 1
            if y_test[i] == "sad":
                count[14] += 1
            if y_test[i] == "calm":
                count[12] += 1
            if y_test[i] == "surprised":
                count[15] += 1
        if y_pred_ensemble[i] == "surprised":
            if y_test[i] == "happy":
                count[17] += 1
            if y_test[i] == "sad":
                count[18] += 1
            if y_test[i] == "angry":
                count[19] += 1
            if y_test[i] == "calm":
                count[16] += 1
        print("Predicted {} but Actually {}".format(y_pred_ensemble[i], y_test[i]))
        tes += 1


# In[112]:


print(count)


# In[113]:


sum(count)


# In[114]:


print(tes)


# In[116]:


y_pos = range(len(wrongPred))
plt.bar(y_pos, count)
# Rotation of the bars names
plt.xticks(y_pos, wrongPred, rotation=90)


# <h1> Tuning Ensemble XGBoost

# In[84]:


parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}

estimators = XGBClassifier(eval_metric='mlogloss')


# In[86]:


import warnings
warnings.filterwarnings('ignore')
grid_ensemble = GridSearchCV(estimators,parameters, refit=True, verbose=3)
grid_ensemble.fit(X,Y)


# In[87]:


print(grid_ensemble.best_params_)


# In[88]:


# from pprint import pprint
# pprint(grid_ensemble.cv_results_)


# In[89]:


grid_model_best = XGBClassifier(learning_rate= 0.05, max_depth= 3, n_estimators= 180)
grid_model_best.fit(x_train, y_train)


# In[90]:


best_ensemble_pred = grid_model_best.predict(x_test)


# In[91]:


accuracy = accuracy_score(y_true=y_test, y_pred=best_ensemble_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))


# In[92]:


print(confusion_matrix(y_test,best_ensemble_pred))


# <h1> Another data for testing</h1>
# Mencoba untuk melakukan testing dengan dataset diluar data test dan predict 

# In[94]:


smellsgoodpath = "C:\\MLCourse\\MLCourse\\Ilmudata\\smellsgood.wav"
ipd.Audio(smellsgoodpath)


# In[104]:


x , sr = librosa.load(smellsgoodpath, mono=False)
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)


# In[105]:


cantbelieve = "C:\\MLCourse\\MLCourse\\Ilmudata\\cantbelieve_iletyou.wav"
ipd.Audio(cantbelieve)


# In[106]:


x , sr = librosa.load(cantbelieve, mono=False)
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)


# In[113]:


from librosa.core import istft

def extract_feature_modified(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        X = librosa.to_mono(X)
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result


# In[136]:


tes1 = extract_feature_modified(smellsgoodpath, mfcc=True, chroma=True, mel=True)
tes2 = extract_feature_modified(cantbelieve, mfcc=True, chroma=True, mel=True)


# In[137]:


tes_one = []
for i in range(180):
    tes_one.append(tes1)
    
tes_one = np.array(tes_one)


# In[138]:


ypred_tes1 = ensemble_model.predict(tes_one)


# In[139]:


ypred_tes1[0]


# In[140]:


tes_two = []
for i in range(180):
    tes_two.append(tes2)
    
tes_two = np.array(tes_two)


# In[141]:


ypred_tes2 = ensemble_model.predict(tes_two)


# In[142]:


ypred_tes2[0]


# <h1> KNN - K-nearest Neighbors </h1>

# Import Library yang diperlukan

# In[3]:


import librosa
import soundfile #read and write sound files
import os, glob, pickle # pckle => converting object to byte
import numpy as np
from sklearn.metrics import accuracy_score
# from sklearn.pipeline import make_pipeline


# In[4]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# <h3> Mengambil sample untuk masing-masing tipe suara untuk tujuan eksplorasi data</h3>
# <p> Directory perlu disesuaikan dengan penyimapanan pribadi</p>

# In[7]:


sample_calm_path = "C:\\Users\\My-Hp\\Semester 5\\Ilmu Data\\project\\speech_data\\Actor_10\\03-01-02-01-01-01-10.wav"
sample_happy_path = "C:\\Users\\My-Hp\\Semester 5\\Ilmu Data\\project\\speech_data\\Actor_10\\03-01-03-01-01-01-10.wav"
sample_sad_path = "C:\\Users\\My-Hp\\Semester 5\\Ilmu Data\\project\\speech_data\\Actor_10\\03-01-04-01-01-01-10.wav"
sample_angry_path = "C:\\Users\\My-Hp\\Semester 5\\Ilmu Data\\project\\speech_data\\Actor_10\\03-01-05-01-01-01-10.wav"
sample_surprised_path = "C:\\Users\\My-Hp\\Semester 5\\Ilmu Data\\project\\speech_data\\Actor_10\\03-01-08-01-01-01-10.wav"
samples = [sample_calm_path, sample_happy_path, sample_sad_path, sample_angry_path, sample_surprised_path]


# In[8]:


import IPython.display as ipd
ipd.Audio(sample_surprised_path)


# In[9]:


#https://towardsdatascience.com/how-i-understood-what-features-to-consider-while-training-audio-files-eedfb6e9002b
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma: #color
            stft=np.abs(librosa.stft(X))

        result=np.array([])
        if mfcc: #Mel-Frequency Cepstral Coefficients (untuk freq)
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel: # power spectrum
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result


# In[10]:


emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

observed_emotions=['calm', 'happy', 'sad', 'angry', 'surprised']


# In[11]:


from sklearn.model_selection import train_test_split
def load_data(test_size=0.2):
    
    x,y=[],[]
    for file in glob.glob("C:\\Users\\My-Hp\\Semester 5\\Ilmu Data\\project\\speech_data\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=1)


# In[12]:


x_train,x_test,y_train,y_test=load_data(test_size=0.2)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) #random state => pilih ngasal


# In[13]:


knn_clf=KNeighborsClassifier()
knn_clf.fit(x_train,y_train)
ypred=knn_clf.predict(x_test)


# In[16]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, ypred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, ypred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,ypred)
print("Accuracy: {:.2f}%".format(result2*100))


# In[15]:



leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(KNeighborsClassifier(),hyperparameters, cv=10, refit=True,verbose=2)
grid.fit(x_train,y_train)


# <p> Mencetak Parameter Terbaik</p>

# In[82]:


print('Best leaf_size:', grid.best_estimator_.get_params()['leaf_size'])
print('Best p:', grid.best_estimator_.get_params()['p'])
print('Best n_neighbors:', grid.best_estimator_.get_params()['n_neighbors'])


# In[81]:


ypred=grid.predict(x_test)


# In[19]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, ypred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, ypred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,ypred)
print("Accuracy: {:.2f}%".format(result2*100))

