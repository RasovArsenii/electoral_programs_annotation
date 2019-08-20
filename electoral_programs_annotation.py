#!/usr/bin/env python
# coding: utf-8

from pandas import DataFrame 
from re import split
from numpy import array, zeros, linalg 
from pickle import load
from gensim.utils import simple_preprocess
from os import walk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy import stats
from itertools import groupby

import ManifestoAPI
import re
import seaborn as sns
import warnings

class VSVM(object):
    data = []       # Raw sentences   
    __labels__ = [] # Labels for sentences
    vect_data = []  # Vectorized sentences
    
    x_test = []     # FOR
    y_test = []     # TESTING  
    x_test_text = []# Raw test sentences
    
    
    # Loading of pre-trained model and some necessary things
    def __init__(self):
        with open('TFIDFVectModel.pkl', 'rb') as input_file:
            self.__vect = load(input_file)         # Vectorizer model
            
        with open('TFIDFMatrix.pkl', 'rb') as input_file:
            self.__tf_idf = load(input_file)       # TF-IDF matrix
            
        with open('Dictionary.pkl', 'rb') as input_file:
            self.__dict = load(input_file)         # Dictionary of all words in matrix
            
    def __repr__(self):        
        return (f'VSvm,\n'
                f'{self.vect()},\n'
                f'{self.model()}.')
    
    #Returns fited vectorizer model    
    def vect(self):
        return(self.__vect)
        
    #Returns filled Tf-Idf matrix    
    def matr(self): 
        return(self.__tf_idf) 
    
    #Upload fited svm model
    def upload_model(self, path='GermanSVMModel2-3.pkl'):
        try:
            with open(path, 'rb') as input_file:
                self.__model = load(input_file)
        except Exception as ex:            
            raise ex                
        
    #Returns fited svm model (before it use upload_model)
    def model(self):
        try:
            return(self.__model)
        except Exception as ex:            
            raise ex

    #Builds TF-IDF matrix on given labeled data, also creates vectorizer model and dictionary
    #Takes two lists of strings(one with some sentences, another with their labels) and 
    #list with stop-words suitable for laguage of sentences
    def transform(self, sentences_list, labels, stop_words): 
        
        # Converts all labels to string type
        try:
            labels = [str(x) for x in labels]
        except TypeError as te:
            te.args = ['Type error exception occured.\nPlease check your \'labels\' variable. It should be list type, not numeral']
            raise te
            
        # Refreshes class field with new labels
        self.__labels__ = labels
        
        # Creates list with unique labels
        tags = [el for el, _ in groupby(sorted([tag for tag in labels]))] 
        
        # Creates dictionary and fills it with empty space
        sentences_by_tag = dict() 
        for tag in tags:                 
            sentences_by_tag[tag] = [] 
        try:    
            if len(sentences_list) != len(labels):
                warnings.warn('For correct results of algorithm variables \'sentences_list\' and \'labels\''
                              'should be the same length')
        except TypeError as te:
            te.args = ['Type error exception occured.\nPlease check your \'sentences_list\' variable. It should be list type, not numeral']
            raise te
        
        # Fills dictionary from 10 strings abow with all sentences for each label
        for i in zip(sentences_list, labels): 
            sentences_by_tag[i[1]].append(i[0])
               
        # Variable for all sentences, sorted by their label
        documents = [' '.join(x) for x in sentences_by_tag.values()]
        
        # Removes everything except letters and tabs and spaces from variable
        for i in range(len(documents)):
            documents[i] = re.sub(r'[^\w\s]+|[\d]+', "", documents[i], flags=re.UNICODE)
            
        # Variable for Tf-Idf vectorizer        
        self.__vect = TfidfVectorizer(lowercase=True, ngram_range = (2, 3), stop_words = stop_words) 
            
        # Creates and fills Tf-Idf matrix with all sentences (documents)   
        try:
            matrix = self.__vect.fit_transform(documents)
            
        except ValueError as ve:
            ve.args = ['Value error exception occured.\nPlease check your variables \'sentences_list\' and \'labels\'.'
                       'They Should be not empty.\nAlso they should be list type.\n'
                       '\'sentences_list\' should contain something that contains not only numerals']
            raise ve
            
        except TypeError as te:
            te.args = ['Type error exception occured.\nPlease make sure that your stop-words variable is a list']
            raise te
            
        except MemoryError as me:
            me.args = ['Memory error exception occured.\nPlease check your pagefile size and increase it']
            raise me
            
        # Dictionary - like variable that contains positions of each word in Tf-Idf matrix 
        positions = {} 
        j=0 
        for word in self.__vect.get_feature_names(): 
            positions[word] = j 
            j=j+1
            
        # Refreshes class field with new data 
        self.__dict = positions         
        self.__tf_idf = DataFrame(matrix.toarray(), columns=positions.keys(), index=sentences_by_tag.keys())       
        
    #Fits SVC model from sklearn using existing TF-IDF matrix
    #Takes two lists of strings (one (x) with some sentences, another (y) with their labels)
    def fit(self, x, y):
        
        # Refreshes class field with new data 
        self.data = x
        
        # Run function that transformates sentences to vectors with length of number of labels
        self.__create_vectors()
        
        # Variable for svm model
        svm = SVC(C = 10, kernel = 'linear', probability = True)
        
        # Fits model
        try:
            svm.fit(self.vect_data, y)
        except TypeError as te:
            te.args = ['Type error exception occured.\n'
                       'Probably your answers (y variable) contains objects of different types.\n'
                       'Please check it out and if the suspicions are confirmed - lead objects to the same type.\n'
                       'Also the error can be occured if \'y\' variable is empty'] 
            raise te
            
        # Refreshes class field with new data 
        self.__model = svm

    #Builds TF-IDF matrix on given labeled data, then fits SVC model using it
    #Takes two lists of strings(one with some sentences, another with their labels) and 
    #list with stop-words suitable for laguage of sentences
    def fit_transform(self, sentences_list, labels, stop_words):   
        
        # Refreshes class field with new data and runs transformtion function
        try:                        
            self.__labels__ = labels
            
            # Runs function that fits TF-IDF vectorizer and creates TF-IDF matrix from input data
            self.transform(sentences_list, self.__labels__, stop_words)            
        except Exception as ex:            
            raise ex
            
        # Splits data into test and train parts    
        try:
            x_train, x_test, y_train, y_test = train_test_split(sentences_list, self.__labels__, test_size=0.125 , random_state=228)
        except ValueError as ve:
            ve.args = [f'\'sentences_list\' and \'labels\' should be the same size\n'
                       f'Got {len(sentences_list)} and {len(self.__labels__)}']
            raise ve
        
        self.y_test = y_test
        self.data = x_test
        self.x_test_text = self.data.copy()
        self.x_test_text = self.x_test_text.reset_index(drop=True)
        
        # Runs function that transformates sentences to vectors with length of number of labels
        try:
            self.__create_vectors()
        except Exception as ex:            
            return(logging.error(ex, exc_info=True))
        
        self.x_test = self.vect_data
        
        # Runs function that fits svm model with vectorized data and labels for them
        try:       
            self.fit(x_train, y_train)
        except Exception as ex:            
            raise ex
        
    #Labels given data (list of sentences) based on the TF-IDF matrix
    #Takes one list of strings(with some sentences)
    def predict(self, sentences_list):
        self.data = sentences_list
        self.__create_vectors()
        return(DataFrame(self.data, columns=["sentence"]).join(DataFrame(self.__model.predict(self.vect_data), columns=["prediction"])))
    
    #Gives the list of vectors of probability in labels
    #(each vector is a set of numbers from 0 to 1)
    #Takes one list of strings(with some sentences)
    def predict_proba(self, sentences_list):
        self.data = sentences_list
        self.__create_vectors()
        return self.__model.predict_proba(self.vect_data)
    
    #Labels given data (list of sentences) based on the probability list
    #(use the same list of sentences as in prediction of the probability)
    #Takes a list of strings(with some sentences), a list of vectors (with probability of labels) and can take the probability threshold and the name of the trash label (tag)
    def interpretate_proba(self, sentences_list, proba, threshold=0.0969, trash_tag='000'):
        
        # Creates list of interpretated labels
        interp = []
        
        # Fills list with labels with the maximum probability and passing threshold
        try:
            i = 0
            for i in range(len(proba)):
                maximum = max(proba[i])
                if (maximum > threshold):
                    interp.append(self.__model.classes_[list(proba[i]).index(maximum)])
                else: interp.append(trash_tag)
        except LookupError as le:
            le.args = [f'Vectors in \'proba\' and model classes should be the same size\n'
                       f'Got {len(proba[i])} and {len(self.__model.classes_)}']
            raise le   
        
        interpretated = DataFrame(sentences_list)
        interpretated.columns = ['sentence']
        return  interpretated.join(DataFrame(interp, columns=['label']))
    
    #Prints information about fited model, such as accuracy, precision, recall and information about multiclass classification 
    def class_rep(self):      
        print(classification_report(self.y_test, self.model().predict(self.x_test), target_names=self._labels_))

    #Draw graphics that visualize correlation between human labeled text and machine labeled ones
    #Takes Pandas DataFrame  that contains some corpuses in Manifesto standart view(with columns:
    #document name(contains date, country, party) and code)
    def visualize_pearson(self, data):
        
        # Splits data into test and train parts (train part dropped)
        _, x_test, _, y_test = train_test_split(data['doc_name'], data['code'], test_size=0.125 , random_state=228)
        y_test.reset_index(drop=True, inplace=True)

        # Prepares data as annotated by human or computer
        hum_annot = DataFrame(y_test)
        hum_annot['doc_name'] = x_test.reset_index(drop=True)
        hum_annot.columns = ['code','doc_name']

        comp_annot = DataFrame(self.model().predict(self.x_test))
        comp_annot['doc_name'] = x_test.reset_index(drop=True)
        comp_annot.columns = ['code', 'doc_name']
        
        # Draws Pearson correlation
        self.__pears(data['doc_name','code'], comp_annot, hum_annot)
        
    #Downlaods some manifesto texts in specific format: csv file, header: manifesto_name, content, label
    #You can find them in C:\Users\User\ManifestoDetails\
    #Takes dictionary with some of that params:
    #     params = {     
    #     'language': 'german',
    #     'election_date': '2017'
    # }
    def get_manifesto_texts(self, params):   
        
        # Check current versions of cores and load them from their folder
        cores = self.__get_current_cores()
        
        # Check existence of meta data file 
        try:
            open(r'ManifestoDetails\meta.csv')
            
        except Exception as ex:
            ex.args = ['No meta file found! Please use function get_manifesto_meta or check ManifestoDetails folder']
            
        
        # If cores and meta were find downloads texts from Manifesto database to ManifestoDetails folder
        if cores != []:
            
            ManifestoAPI.get_texts(params, r'ManifestoDetails\meta.csv', r'ManifestoDetails\annot.csv', r'ManifestoDetails\not_annot.csv')
            
            print('Texts were successfully download. You can find them in your user folder, ManifestoDetails, annot.csv and not_annot.csv')
        else:
            print('No cores found! Please download some fo them using method \'get_manifesto_cores\'')
            
    #Downloads some manifesto meta data that uses to download some texts.
    #You can find it in C:\Users\User\ManifestoDetails\meta.csv
    #Takes dictionary with some of that params:
    #     params = {     
    #      "countryname": "Germany"
    # }
    def get_manifesto_meta(self, params):
        
        # Check current versions of cores and load them from their folder
        cores = self.__get_current_cores()
        
        # If cores were find downloads meta data from Manifesto database to ManifestoDetails folder
        if cores != []:
            
            ManifestoAPI.get_manifesto_metadata(params, cores, r'ManifestoDetails\meta.csv')
            
            print('Meta data was successfully download.'
                 'You can find them in your user folder, ManifestoDetails, meta\.csv')
        else:
            print('No cores found! Please download some fo them using method \'get_manifesto_cores\'')
        
    #Downloads some manifesto cores of specific versions
    #exmaple of versions: ['MPDS2018b', 'MPDSSA2018b']
    #You can get full list of current versions by using "get_core_versions"
    def get_manifesto_cores(self, versions):
        try:
            for i in versions:
                ManifestoAPI.api_get_core(i, kind = 'xlsx')
        except Exception as ex:
            ex.args = ['Please make sure that you gave correct versions of manifesto cores. To make sure use \'get_core_versions\' method']
            
    #Prints all current manifesto core versions
    def get_core_versions(self):
        print(ManifestoAPI.api_list_core_versions())
        
    #Check folder of cores and returns all that were find
    def __get_current_cores(self):
        
        # Directory for Manifesto cores
        pdir = r'ManifestoDetails\cores'
        
        # List for all files in folder
        contdir = []
        
        # "Walking" through content of directory and saving all file names
        for i in walk(pdir): 
            contdir.append(i)
        
        # Grabing useful information(core versions) from walking result
        cores = []
        for i in contdir[0][2]:
            cores.append(re.split('\.', i)[0])
        return(cores) 
    
    #Function that draws Pearson correlation chart for computer and human annotated texts 
    #Takes Pandas DataFrame that contains some corpuses in Manifesto standart view(with columns that contains such info as
    #document name and code) and
    #two other Pandas DataFrame that contains some pairs of document and his label(code), one of them contains algorithm predictions and
    #another - human labels
    def __pears(self, data, comp_annot, hum_annot):
        
        # Creates and fills lists with label frequency in document 
        comp = []
        hum = []
        for i in data['doc_name'].unique():
            for j in data['code'].unique():
                comp.append(comp_annot[(comp_annot.doc_name == i) & (comp_annot.code == j)].shape[0])
                hum.append(hum_annot[(hum_annot.doc_name == i) & (hum_annot.code == j)].shape[0])
        
        # Draws Pearson correlation
        pear = sns.jointplot(x=array(comp), y=array(hum), kind='reg')
        pear = pear.set_axis_labels("computer-annotated", "human-annotated")
        pear = pear.annotate(stats.pearsonr)
        print(pear)   
            
    #Changes the vector's length to 1
    #Takes list with n floats, where n is number of labels in Your data
    def __normalize(self,vector):
        normed = linalg.norm(vector)
        if normed == 0: 
            return vector
        return vector / normed
    
    #Converts sentence to an array of float numbers using the TF-IDF matrix  
    #Takes one sentence that must be string type.
    #Also you need to run transform method before using this one
    def __to_vector (self, sentence):
        
        # Variable for computing likelihood of entry into the label of sentence
        vector = zeros(len(self.__dict))
        
        # Check type of input data
        if isinstance(sentence, str):     
            
            # Splits sentence into words
            splitted_sentence = simple_preprocess(str(sentence), deacc=True)
            
            # Removes empty spaces
            splitted_sentence = list(filter(None, splitted_sentence))

            # If sentence have two or more words tries to find each phrase from two words in TF-IDF vocabulary
            # and increments appropriate position
            if len(splitted_sentence) > 1:
                for c in range (len(splitted_sentence)-1):
                    pair = splitted_sentence[c] + ' ' + splitted_sentence[c+1]
                    try:
                        position = self.__dict[pair]
                        vector[position] += 1
                    except KeyError:                    
                        continue
            # If sentence have three or more words tries to find each phrase from two words in TF-IDF vocabulary
            # and increments appropriate position            
            if len(splitted_sentence) > 2:
                for c in range (len(splitted_sentence)-2):
                    tripl = splitted_sentence[c] + ' ' + splitted_sentence[c+1] + splitted_sentence[c+2]
                    try:
                        position = self.__dict[tripl]
                        vector[position] += 1
                    except KeyError:                    
                        continue        

        return(self.__normalize(array(self.__tf_idf.dot(vector))))
    
    #Runs To_vector for a list of sentences (which is supposed to be in self.data)
    #Takes nothing, but You need to put some data in exemplar of that class or just run transform or fit_transform with correct args
    #before using this method
    def __create_vectors(self):
        
        # Variable for vectorized sentences
        vectors = []
        
        # Variable for active progress bar
        j=1
        
        # Variable for amount of sentences
        x = len(self.data)
        
        # Construction for active progress bar
        print("Vectorization in progress:")
        
        # Vectorizing each sentence
        for i in self.data:
            
            vectors.append(self.__to_vector(i))
            line = str(j) + '/' + str(x)            
            print(line,end="\r")
            j += 1
        # Refreshes class field with new data
        self.vect_data = vectors
        
        