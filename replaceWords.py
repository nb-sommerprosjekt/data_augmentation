
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import random
from math import floor
from numpy import array_split, array_str
from time import sleep
import time
import gensim
import random
import os


def fix_corpus(folder, name_of_new_folder):

    if not os.path.exists(name_of_new_folder):
        os.makedirs(name_of_new_folder)


    arr_txt = [path for path in os.listdir(folder) if path.endswith(".txt")]
    for article_path in arr_txt:
        with open(folder+"/"+article_path,"r") as text_file:
            text = text_file.read().replace('\n',' ')
        edited_text = open(name_of_new_folder+'/'+article_path,'w')
        edited_text.write(text)
        edited_text.close()


def get_articles(folder):
    # Tar inn en textfil som er labelet på fasttext-format. Gir ut to arrays. Et med deweys og et med tekstene. [deweys],[texts]
    arr = os.listdir(folder)
    arr_txt = [path for path in os.listdir(folder) if path.endswith(".txt")]


    dewey_array = []
    docs = []
    for article_path in arr_txt:
            article = open(folder+'/'+article_path, "r")
            article = article.readlines()
            for article_content in article:
                dewey=article_content.partition(' ')[0].replace("__label__","")
                text  = article_content.replace("__label__"+dewey,"")
                docs.append(text)
                dewey_array.append(dewey[:3])
    text_names = [path.replace('.txt','') for path in arr_txt ]
    #print(len(dewey_array))
    #print(text_names)
    return text_names, dewey_array, docs

def find_synonyms(word):

    '''Takes a string as input and returns a array of synonyms in norwegian, if no synonyms exists, a empty array will be returned.'''
    class MyWordNet:
        def __init__(self, wn):
            self._wordnet = wn

        def synsets(self, word, pos=None, lang="nob"):
            return self._wordnet.synsets(word, pos=pos, lang = lang)

    wn = MyWordNet(wordnet)
    synonyms = []
    synonym_collection =[]
    for syn in wn.synsets(word, lang="nob"):
        for l in syn.lemmas(lang="nob"):
            #synonyms.append(l.name())
            if l.name() not in synonyms and l.name() != word:
                synonyms.append(l.name())
                synonym_collection.append(l.name())


    if len(synonyms) == 0:
        #print("bruker nynorsk")
        ##Checking if synonyms exists in NNO if there was none in NOB.
        wnn = MyWordNet(wordnet)
        for syn in wnn.synsets(word, lang="nno"):
            for l in syn.lemmas(lang="nno"):
                if l.name() not in synonyms and l.name() != word:
                    synonyms.append(l.name())
                    synonym_collection.append(l.name())

    #print(word + "---> " + str(synonym_collection))
    return synonyms
def replace_words_with_synonyms(tokenized_text,percentage):
    ''' Takes tokenized text as input, replaces words with synonyms or word2vec and outputs tokenized text as string with synonyms replaced'''
    num_words_to_replace = floor((percentage / 100) * len(tokenized_text))
    synonym_text = []

    words_replaced = 0
    synonyms_put_into_text = []
    # Replacing words with word from norsk synonymordbok
    for word in tokenized_text:
        synonym = find_synonyms(word.lower())
        if len(synonym) > 0 and words_replaced<(num_words_to_replace):
            synonym_text.append(synonym[0])
            #synonyms_put_into_text.append("!!!støy-->"+word+"-->" + synonym[0]+"<--støy!!!")
            synonyms_put_into_text.append(synonym[0])
            words_replaced = words_replaced + 1
            #print("Synonymordbok:"+ word + "---> " + str(synonym[0]))
        else:
            synonym_text.append(word)
    # Replacing remaining words with words from word2vec
    if (words_replaced < (num_words_to_replace)):
        words_left_to_replace = (num_words_to_replace-words_replaced)
        random_elements = random.sample(range(0,len(synonym_text)-1), len(synonym_text)-1)
        for index in random_elements:
            words_left_to_replace = (num_words_to_replace - words_replaced)
            if synonym_text[index] not in synonyms_put_into_text and words_left_to_replace>0:
                synonym_w2v = get_similar_words_from_word2vec(synonym_text[index], w2v_model)
                if len(synonym_w2v)>0:
                    temp_show = synonym_text[index]
                    # synonym_text[index] = "!!!støy--> "+temp_show+"-->"+synonym_w2v +"<--Støy!!!"
                    synonym_text[index] = synonym_w2v
                    words_replaced=words_replaced+1



    return synonym_text

def get_similar_words_from_word2vec(word, model):
    '''Function taking a word and a word2vec model as input, outputting the most similar word.'''
    try:
        similar_words = model.wv.similar_by_word(word, topn=1)
        most_similar_word = similar_words[0][0]
    except KeyError:
        most_similar_word = []

   # print(word + "---> " + str(most_similar_word))
    return most_similar_word
def remove_words_from_text(tokenized_text, percentage):
    ''' Chooses random words in the tokenized text and removes them. The amount is chosen by the percentage provided. It is not recommended to remove more than 10 percent. Returns reduced tokenized string.'''
    num_elements_to_remove = floor((percentage/100) * len(tokenized_text))
    words_to_remove = random.sample(tokenized_text, num_elements_to_remove)

    reduced_text = tokenized_text
    for words in words_to_remove:
        reduced_text.remove(words)
    return ' '.join(reduced_text)

def remove_parts_of_text(tokenized_text, percentage):
    '''Removes a random part of the input-tokenized text, size set by the percentage, and outputs a reduced tokenized text.'''
    number_of_splits = floor(100/percentage)
    array_of_text_parts = array_split(tokenized_text, number_of_splits)
    del array_of_text_parts[random.randint(0, len(array_of_text_parts) - 1)]
    reduced_array = [item for sublist in array_of_text_parts for item in sublist]

    return reduced_array

def add_words_to_text(tokenized_text, percentage, which_dewey_is_this_from):
    ''' Adds words randomly to text. Words are chosen either from word2vec or tfidf-matrix relevant to the articles dewey. This function returns the modified tokenized_text'''

def add_synonyms_to_parts_of_text(tokenized_text,number_of_splits,percentage):
    '''Module for adding noise to text. To add noise to the whole text, set number_of_splits=1. Number of documents output is set by number_of_splits.
    If number_of_splits=10, 10 documents will be returned where different parts of the documents have been induced with a percentage of noise set by percentage.'''
    array_of_text_parts = array_split(tokenized_text, number_of_splits)
    array_of_text_parts_with_synonyms = []
    #print(len(array_of_text_parts))
    #print(array_of_text_parts)
    # Making noise array
    for part_text in array_of_text_parts:
        array_of_text_parts_with_synonyms.append(replace_words_with_synonyms(part_text,percentage))
    #print(array_of_text_parts_with_synonyms)
     # Making new full tekst with the noise induced fractions included
    full_texts_with_noise_induced_on_parts_array = []

    for i in range(0,len(array_of_text_parts_with_synonyms)):

        temp = list(array_of_text_parts)

        temp[i] = array_of_text_parts_with_synonyms[i]
        flat_temp = []
        for sublist in temp:
            for item in sublist:
                 flat_temp.append(item)

        temp_text = ' '.join(flat_temp)

        full_texts_with_noise_induced_on_parts_array.append(temp_text)
        #print(full_texts_with_noise_induced_on_parts_array)

    return full_texts_with_noise_induced_on_parts_array

def create_fake_corpus(folder,artifical_folder, number_of_splits, noise_percentage):
    print("Henter tekst-data")
    text_names, dewey_array, docs = get_articles(folder)
    print("Tekstdata på plass")
    if not os.path.exists(artifical_folder):
        os.makedirs(artifical_folder)
    text_list = list(enumerate(text_names))
    total_number_of_texts = len(text_names)
    for index, text_name in text_list:
        start = time.time()
        print("tokenizer teksten")
        tokenized_text = word_tokenize(docs[index],language= 'norwegian')
        print("teksten er tokenized")
        print("Legger til støy og lager nye tekster")
        full_texts_with_noise = add_synonyms_to_parts_of_text(tokenized_text,number_of_splits, noise_percentage)
        print("Støy er lagt til")
        print("Starter utskrift av tekster")
        for noise_text_index, noise_text in enumerate(full_texts_with_noise):

            noise_text_w_label = "__label__" + str(dewey_array[index]) + ' ' + noise_text
            noise_split_file = open(artifical_folder + '/' + text_name.replace('_split','') + '_' + str(noise_text_index),'w')
            noise_split_file.write(noise_text_w_label)
            noise_split_file.close()
        end = time.time()

        print("Tekst nr " + str(index+1) + "/" + str(total_number_of_texts)+ "er ferdig prosessert")
        print("Tid brukt på denne runden: "+ str(end - start ))
if "__main__":

    ###GLOBAL VARIABLES
    w2v_model_path = "w2v_tgc/full.bin"
    w2v_model = gensim.models.Word2Vec.load(w2v_model_path)


    create_fake_corpus('corpus_training_split','artifical_corpus_10_20',10,20)
    # #print(find_synonyms(""))

    #tokenized_text = word_tokenize(tekst, language="norwegian")
    #
    # num_splits =1
    #lol = add_synonyms_to_parts_of_text(tokenized_text[:500],10,10)
    # #print(len(lol))
    # remove_parts_of_text(tokenized_text,10)
    # for i in range(0,len(lol)):
    #     split_file = open("noise_overall_new_w2v_induced_split_"+str(i)+".txt","w")
    #     split_file.write(lol[i])
    #     split_file.close()
    #
    # # print("original_tekst:")
    # print(tokenized_text[:50])



