import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
import warnings
warnings.filterwarnings("ignore")

def word2vec_train(lst, emb_dim = 150, seed = 42):
    """
    train a word2vec mode
    args: lst(list of string): sentences
          emb_dim(int): word2vec embedding dimensions
          seed(int): seed for word2vec
    return: word2vec model
    """
    tokenizer = RegexpTokenizer(r'\w+')
    sentences = []
    for i in lst:
        sentences.append([x.lower() for x in tokenizer.tokenize(str(i))])
    w2v = Word2Vec(sentences, size=emb_dim, min_count=1, seed=seed)
    return w2v

def get_sentence_emb(sentence, w2v):
    """
    get a sentence embedding vector
    *automatic initial random value to the new word
    args: sentence(string): sentence of log message
          w2v: word2vec model
    return: sen_emb(list of int): vector for the sentence
    """
    tokenizer = RegexpTokenizer(r'\w+')
    lst = []
    tokens = [x.lower() for x in tokenizer.tokenize(str(sentence))]
    if tokens == []:
        tokens.append('EmptyParametersTokens')
    for i in range(len(tokens)):
        words = list(w2v.wv.vocab.keys())
        if tokens[i] in words:
            lst.append(w2v[tokens[i]])
        else:
            w2v.build_vocab([[tokens[i]]], update = True)
            w2v.train([tokens[i]], epochs=1, total_examples=len([tokens[i]]))
            lst.append(w2v[tokens[i]])
    drop = 1
    if len(np.array(lst).shape) >= 2:
        sen_emb = np.mean(np.array(lst), axis=0)
        if len(np.array(lst)) >= 5:
            drop = 0
    else:
        sen_emb = np.array(lst)
    return list(sen_emb), drop

def word2emb(df_source, df_target, train_size_s, train_size_t, step_size, emb_dim):
    w2v = word2vec_train(np.concatenate((df_source.EventTemplate.values[:step_size*train_size_s], df_target.EventTemplate.values[:step_size*train_size_t])), emb_dim=emb_dim)
    print('Processing words in the source dataset')
    dic = {}
    lst_temp = list(set(df_source.EventTemplate.values))
    for i in tqdm(range(len(lst_temp))):
        (temp_val, drop) = get_sentence_emb([lst_temp[i]], w2v)
        dic[lst_temp[i]] = (temp_val, drop)
    lst_emb = []
    lst_drop = []
    for i in tqdm(range(len(df_source))):
        lst_emb.append(dic[df_source.EventTemplate.loc[i]][0])
        lst_drop.append(dic[df_source.EventTemplate.loc[i]][1])
    df_source['Embedding'] = lst_emb
    df_source['drop'] = lst_drop
    print('Processing words in the target dataset')
    dic = {}
    lst_temp = list(set(df_target.EventTemplate.values))
    for i in tqdm(range(len(lst_temp))):
        (temp_val, drop) = get_sentence_emb([lst_temp[i]], w2v)
        dic[lst_temp[i]] = (temp_val, drop)
    lst_emb = []
    lst_drop = []
    for i in tqdm(range(len(df_target))):
        lst_emb.append(dic[df_target.EventTemplate.loc[i]][0])
        lst_drop.append(dic[df_target.EventTemplate.loc[i]][1])
    df_target['Embedding'] = lst_emb
    df_target['drop'] = lst_drop

    df_source = df_source.loc[df_source['drop'] == 0]
    df_target = df_target.loc[df_target['drop'] == 0]

    print(f'Source length after drop none word logs: {len(df_source)}')
    print(f'Target length after drop none word logs: {len(df_target)}')

    return df_source, df_target

def sliding_window(df, window_size = 20, step_size = 4, target = 0, val_date = '2005.11.15'):
    df["Label"] = df["Label"].apply(lambda x: int(x != '-'))
    df = df[["Label", "Content", "Embedding", "Date"]]
    df['target'] = target
    df['val'] = 0
    log_size = df.shape[0]
    label_data = df.iloc[:, 0]
    logkey_data = df.iloc[:, 1]
    emb_data = df.iloc[:, 2]
    date_data = df.iloc[:, 3]
    new_data = []
    index = 0
    while index <= log_size-window_size:
        if date_data.iloc[index] == val_date:
            new_data.append([
                max(label_data[index:index+window_size]),
                logkey_data[index:index+window_size].values,
                emb_data[index:index+window_size].values,
                date_data.iloc[index],
                target,
                1
            ])
            index += step_size
        else:
            new_data.append([
                max(label_data[index:index+window_size]),
                logkey_data[index:index+window_size].values,
                emb_data[index:index+window_size].values,
                date_data.iloc[index],
                target,
                0
            ])
            index += step_size
    return pd.DataFrame(new_data, columns=df.columns)

def get_datasets(df_source, df_target, options, val_date="2005.11.15"):
    # Get source data preprocessed
    window_size = options["window_size"]
    step_size = options["step_size"]
    source = options["source_dataset_name"]
    target = options["target_dataset_name"]
    train_size_s = options["train_size_s"]
    train_size_t = options["train_size_t"]
    emb_dim = options["emb_dim"]
    times =  int(train_size_s/train_size_t) - 1

    df_source, df_target = word2emb(df_source, df_target, train_size_s, train_size_t, step_size, emb_dim)

    print(f'Start preprocessing for the source: {source} dataset')
    window_df = sliding_window(df_source, window_size, step_size, 0, val_date)
    r_s_val_df = window_df[window_df['val'] == 1]
    window_df = window_df[window_df['val'] == 0]

    # Training normal data
    df_normal = window_df[window_df["Label"] == 0]

    # shuffle normal data
    df_normal = df_normal.sample(frac=1, random_state=42).reset_index(drop=True)
    train_len = train_size_s

    train_normal_s = df_normal[:train_len]
    print("Source training size {}".format(len(train_normal_s)))

    # Test normal data
    test_normal_s = df_normal[train_len:]
    print("Source test normal size {}".format(len(test_normal_s)))

    # Testing abnormal data
    test_abnormal_s = window_df[window_df["Label"] == 1]
    print('Source test abnormal size {}'.format(len(test_abnormal_s)))

    print('------------------------------------------')
    print(f'Start preprocessing for the target: {target} dataset')
    # Get target data preprocessed
    window_df = sliding_window(df_target, window_size, step_size, 1, val_date)
    r_t_val_df = window_df[window_df['val'] == 1]
    window_df = window_df[window_df['val'] == 0]

    # Training normal data
    df_normal = window_df[window_df["Label"] == 0]
    # shuffle normal data
    df_normal = df_normal.sample(frac=1, random_state=42).reset_index(drop=True)
    train_len = train_size_t

    train_normal_t = df_normal[:train_len]
    print("Target training size {}".format(len(train_normal_t)))
    temp = train_normal_t[:]
    for _ in range(times):
        train_normal_t = pd.concat([train_normal_t, temp])

    # Testing normal data
    test_normal_t = df_normal[train_len:]
    print("Target test normal size {}".format(len(test_normal_t)))

    # Testing abnormal data
    test_abnormal_t = window_df[window_df["Label"] == 1]
    print('Target test abnormal size {}'.format(len(test_abnormal_t)))

    return train_normal_s, test_normal_s, test_abnormal_s, r_s_val_df, \
           train_normal_t, test_normal_t, test_abnormal_t, r_t_val_df