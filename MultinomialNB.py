# Machine learning classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# For data manipulation
import pandas as pd
# To plot
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from sklearn.pipeline import Pipeline

loopsdb = pd.read_excel('/Disco2/ColabFold/HisKA1_PFAM/Secuencias_Loops.ods', engine='odf')
df = loopsdb[ (loopsdb['Model'] != "Broken")]

df['Loop_seq_split'] = df.apply(lambda row: " ".join(row["Loop_seq"]), axis=1)
df['Loop_seq_4_split'] = df.apply(lambda row: " ".join(row["Loop_seq_4"]), axis=1)

percentage_list=[]
chance=[]
chance4=[]
for percentage in range(1,99,5):
    print(percentage)
    for iterations in range(1,100,1):
        suma = 0
        suma4 = 0
        df_shuffle = df.sample(frac=1)
        df_train, df_test = np.split(df_shuffle, [int((percentage/100)*len(df))])

        text_clf = Pipeline([('vect', CountVectorizer(token_pattern='(?u)[a-zA-Z]+')),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])

        text_clf = text_clf.fit(df_train["Loop_seq_4_split"], df_train["Model"])
        predicted = text_clf.predict(df_test["Loop_seq_4_split"])
        avg = np.mean(predicted == df_test["Model"])
        suma4 += avg

        text_clf = text_clf.fit(df_train["Loop_seq_split"], df_train["Model"])
        predicted = text_clf.predict(df_test["Loop_seq_split"])
        avg = np.mean(predicted == df_test["Model"])
        suma += avg
    percentage_list.append(percentage)
    chance.append(suma)
    chance4.append(suma4)

print(chance)
print(chance4)

fig, ax = plt.subplots()
ax.plot(percentage_list, chance,label="Loop")
ax.plot(percentage_list, chance4,label="Loop+4aa")
ax.set(xlabel='Porcentaje usado en entrenamiento', ylabel='Precision del algoritmo',
       title='')
ax.legend()
ax.grid()
fig.savefig("test.png")
plt.show()