import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model
import pandas as pd
# To plot
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from keras.models import model_from_json

tf.random.set_seed(7)


# Function Definitions
def PreProcess(loopsdb,Variable_of_interest):
    df = loopsdb[(loopsdb['Model'] != "Broken")]

    df_shuffle = df.sample(frac=1)

    Model_int = []
    for row in df_shuffle['Model']:
        if row == "Cis":
            Model_int.append(0)
        else:
            Model_int.append(1)

    df_shuffle["Model_int"] = Model_int

    # Me quedo con 100 de cada una
    wantedLength=100
    cisRemain = len(df_shuffle[df_shuffle["Model"] == "Cis"])
    transRemain = len(df_shuffle[df_shuffle["Model"] == "Trans"])
    df_cis = df_shuffle[df_shuffle["Model"] == "Cis"].head(wantedLength)
    df_trans = df_shuffle[df_shuffle["Model"] == "Trans"].head(wantedLength)
    df_trans_unused = df_shuffle[df_shuffle["Model"] == "Trans"].tail(transRemain-wantedLength)
    df_cis_unused = df_shuffle[df_shuffle["Model"] == "Cis"].tail(cisRemain - wantedLength)

    df_cis = df_cis.sample(frac=1)
    df_trans = df_trans.sample(frac=1)

    # Separo en train y test
    percentage = 80
    cis_train_df, cis_test_df = np.split(df_cis, [int((percentage / 100) * len(df_cis))])
    trans_train_df, trans_test_df = np.split(df_trans, [int((percentage / 100) * len(df_trans))])

    train_df = pd.concat([trans_train_df, cis_train_df])
    test_df = pd.concat([trans_test_df, cis_test_df])

    train_texts = train_df[Variable_of_interest].values
    train_texts = [s.lower() for s in train_texts]
    test_texts = test_df[Variable_of_interest].values
    test_texts = [s.lower() for s in test_texts]

    return train_df, test_df, train_texts, test_texts, df_trans_unused,df_cis_unused

def Tokenize(train_texts, test_texts, padding_size):
    tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
    tk.fit_on_texts(train_texts)

    alphabet = "acdefghiklmnpqrstvwy"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1

    # Use char_dict to replace the tk.word_index
    tk.word_index = char_dict.copy()
    # Add 'UNK' to the vocabulary
    tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

    train_sequences = tk.texts_to_sequences(train_texts)
    test_texts = tk.texts_to_sequences(test_texts)

    train_data = pad_sequences(train_sequences, maxlen=padding_size, padding='post')
    test_data = pad_sequences(test_texts, maxlen=padding_size, padding='post')

    train_data = np.array(train_data, dtype='float32')
    test_data = np.array(test_data, dtype='float32')

    return tk, train_data, test_data

def testArray(test_array):
    test_sequences = tk.texts_to_sequences(test_array)
    test_d = pad_sequences(test_sequences, maxlen=padding_size, padding='post')
    test_d = np.array(test_d, dtype='float32')

    ynew = model.predict(test_d)

    cis_chance = []
    cis_chance_binary = []
    for item in ynew:
        cis_chance.append(item[1])
        if item[1] >= 0.5:
            cis_chance_binary.append(0) #0 significa cis
        else:
            cis_chance_binary.append(1) #0 significa trans

    # sumcis=sum(cis_chance)
    # avgcis=sumcis/len(ynew)
    # stdevcis=np.std(cis_chance)
    # avgtrans=1-avgcis
    # stdevtrans=stdevcis
    # print("Average for the set (continuous): Trans=%s +/-%s, Cis=%s +/-%s" % (round(avgtrans,3),round(stdevtrans,3),round(avgcis,3),round(stdevcis,3)))

    return cis_chance_binary

def analyzeResults(test_df,results):
    test_results_df = test_df
    test_results_df["Result"] = results
    test_results_df["Result"]
    hits = 0
    cis_hits = 0
    trans_hits = 0
    for index, row in test_results_df.iterrows():
        if row["Model_int"] == row["Result"]:
            hits += 1
            if row["Model"] == "Trans":
                trans_hits += 1
            else:
                cis_hits += 1
    if len(test_results_df[test_results_df["Model"] == "Cis"]) !=0:
        accuracyCis=cis_hits / len(test_results_df[test_results_df["Model"] == "Cis"])
    else:
        accuracyCis=0

    if len(test_results_df[test_results_df["Model"] == "Trans"]) != 0:
        accuracyTrans=trans_hits / len(test_results_df[test_results_df["Model"] == "Trans"])
    else:
        accuracyTrans=0
    print("Total accuracy:%s |   Trans accuracy:%s  | Cis accuracy:%s" % (hits / len(test_results_df),
                                                                          accuracyTrans, accuracyCis))
    return test_results_df

def testSequence(test):
    test = test.lower()
    test_array = []
    test_array.append(test)

    # tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
    # tk.fit_on_texts(test_array)
    # tk.word_index = char_dict.copy()
    # tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

    test_sequences = tk.texts_to_sequences(test_array)

    test_d = pad_sequences(test_sequences, maxlen=padding_size, padding='post')
    test_d = np.array(test_d, dtype='float32')

    ynew = model.predict(test_d)
    # show the inputs and predicted outputs
    cischance = ynew[0][1]
    transchance = ynew[0][0]
    print("X=%s, Trans Chance=%s, Cis Chance=%s" % (test, transchance, cischance))
    return

def trainModel(train_df, test_df, classVariable, epochs, patience):

    train_classes = train_df[classVariable].values
    train_class_list = [x - 1 for x in train_classes]
    test_classes = test_df[classVariable].values
    test_class_list = [x - 1 for x in test_classes]

    from keras.utils import to_categorical

    train_classes = to_categorical(train_class_list, num_classes=2)
    test_classes = to_categorical(test_class_list, num_classes=2)

    print(tk.word_index)
    vocab_size = len(tk.word_index)
    vocab_size

    embedding_weights = []
    embedding_weights.append(np.zeros(vocab_size))

    for char, i in tk.word_index.items():
        onehot = np.zeros(vocab_size)
        onehot[i - 1] = 1
        embedding_weights.append(onehot)
    embedding_weights = np.array(embedding_weights)

    # Parametros originales
    # conv_layers = [[256, 7, 3],
    #               [256, 7, 3],
    #               [256, 3, -1],
    #               [256, 3, -1],
    #               [256, 3, -1],
    #               [256, 3, 3]]

    # fully_connected_layers = [1024, 1024]

    # Parametros modificados
    conv_layers = [[64, 7, 3],
                   [64, 7, 3],
                   [64, 3, -1],
                   [64, 3, -1],
                   [64, 3, -1],
                   [64, 3, 3]]

    fully_connected_layers = [128, 128]

    num_of_classes = 2
    dropout_p = 0.5
    optimizer = 'adam'
    # loss = 'categorical_crossentropy'
    # loss="sparse_categorical_crossentropy"
    loss = "binary_crossentropy"
    embedding_size = 21
    input_size = padding_size
    embedding_layer = Embedding(vocab_size + 1,
                                embedding_size,
                                input_length=input_size,
                                weights=[embedding_weights])

    inputs = Input(shape=(input_size,), name='input', dtype='int64')

    x = embedding_layer(inputs)
    # Conv
    for filter_num, filter_size, pooling_size in conv_layers:
        x = Conv1D(filter_num, filter_size)(x)
        x = Activation('relu')(x)
        if pooling_size != -1:
            x = MaxPooling1D(pool_size=pooling_size)(x)  # Final shape=(None, 34, 256)
    x = Flatten()(x)  # (None, 8704)
    # Fully connected layers
    for dense_size in fully_connected_layers:
        x = Dense(dense_size, activation='relu')(x)  # dense_size == 1024
        x = Dropout(dropout_p)(x)
    # Output Layer
    predictions = Dense(num_of_classes, activation='sigmoid')(x)

    # Early Stopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)

    # Build model
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # Adam, categorical_crossentropy
    model.summary()

    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)

    x_train = train_data[indices]
    y_train = train_classes[indices]

    #x_train = train_data
    #y_train = train_classes

    # Training
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        batch_size=128,
                        epochs=epochs,
                        verbose=2,
                        callbacks=[callback])
    return model

def saveModel(model,filename,train_df,test_df):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("%s.h5" % (filename))
    test_df.to_csv("test-%s.csv"% (filename),index=False)
    train_df.to_csv("train-%s.csv"% (filename),index=False)
    print("Saved model and datasets to disk")

def loadModel(filename):
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s.h5" % (filename))
    print("Loaded model from disk")
    return loaded_model


loops_db = pd.read_excel('/Disco2/ColabFold/HisKA1_PFAM/Secuencias_Loops.ods', engine='odf')
Variable_of_interest = "Sequence"
train_df, test_df, train_texts, test_texts, df_trans_unused, df_cis_unused = PreProcess(loops_db,Variable_of_interest)
padding_size = 140
tk, train_data, test_data = Tokenize(train_texts, test_texts, padding_size)

model = trainModel(train_df, test_df, "Model_int", 80, 10)

filename= "80%"
# model = loadModel(filename)
# saveModel(model, filename)


#################Multi_Test#####################
df_to_test=df_trans_unused #normalmente es test_df, df_trans_unused o df_cis_unused
test_array = df_to_test[Variable_of_interest]
# test_array=test_df[test_df["Model"] =="Cis"]["Sequence"]
results = testArray(test_array)
test_results_df=analyzeResults(df_to_test,results)


#test_results_df.to_csv("results.csv",index=False)

#################Single_Test#####################

test = "QLKMMLAGVAHEVRNPIGGIALFSGILKEDLQAGAHADAGAHVERIQREVAYLQRIVEDFLAFAREQPL"
testSequence(test)
