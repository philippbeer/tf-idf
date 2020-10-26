from typing import List, Dict
import math
import itertools
import re


class TFIDF():
    def __init__(self, lowercase: bool,
                 bad_chars: List[str] = '[.,?:;]',
                 v_max: int = -1, l2: bool = False,
                 n_grams: int = 1) -> None:
        """params: lowercase (bool) - lowercase all words, bar_chars (string) -
            regex as string for chars to be removed from all docs in corpus,
            v_max (int) - max length of vocabulary assumed to be positive int"""
        self.vocabulary: List[str] = list() # keep word and id
        self.v_max: int = v_max
        self.lowercase: bool = lowercase
        self.idf_col: List[float] = list()
        self.bad_chars: str = bad_chars
        self.l2 = l2
        self.n_grams = n_grams

        
    def get_v_max(self) -> int:
        """returns length of vocabulary"""
        return len(self.vocabulary)
    
    def remove_bad_chars(self, corpus: List[str]) -> List[str]:
        """use regex to remove unwanted characters"""
        corpus_clean: List[str] = list()
        for doc in corpus:
            doc_tmp = ""
            doc_tmp = re.sub(self.bad_chars, "", doc)
            corpus_clean.append(doc_tmp)
        return corpus_clean
    
    def preprocess(self, collection: List[str]) -> List[List[str]]:
        """ make collection lower case if and tokenize"""
        # remove bad characters
        col_clean = self.remove_bad_chars(collection)
        # make lower case if applicable
        if self.lowercase:
            col_l = self.lower_case_corpus(col_clean)
        else:
            col_l = col_clean
        if self.n_grams == 1:
            return [s.split() for s in col_l]
        else:
            corpus_tokenized = [s.split() for s in col_l]
            col_n_grams = list()
            for doc in corpus_tokenized:
#                 print(f"doc is: {doc}")
                if self.n_grams > len(doc):
                    raise ValueError("n-grams value to large, choose\
                    smaller value")
                doc_n_grams = list()
                for i in range(len(doc)):
                    end_idx = self.n_grams + i
                    doc_tmp = ""
                    if end_idx <= len(doc):
#                         print(f"token added is: {' '.join(doc[i:end_idx])}")
                        doc_tmp = " ".join(doc[i:end_idx])
                        doc_n_grams.append(doc_tmp)
                    else:
                        break
                col_n_grams.append(doc_n_grams)
#             print(f"final: {col_n_grams}")
            return col_n_grams
                
                    
    
    def lower_case_corpus(self, collection: List[str]) -> List[str]:
        """params: collection (list of string)
            checks if each string element is lowercase
            returns: all string elements lowercase"""
        return [collection[i].lower() for i in range(len(collection))]
        
    
    def fit(self, corpus: List[str]) -> None:
        """build vocabulary and calculate idf"""
        # clear object variables
        self.idf_col = list()
        self.vocabulary = list()
        # build vocabulary
        # preprocess & tokenize data
        corpus_tokenized = self.preprocess(corpus)
        for doc in corpus_tokenized:
            for word in doc:
                if word not in self.vocabulary:
                    self.vocabulary.append(word)
        self.vocabulary.sort()
        # truncating to defined vocab length
        if (len(self.vocabulary) > self.v_max) and (self.v_max != -1):
            word_count = self.word_count(corpus_tokenized)
            words_allowed = [k for k in list(word_count.keys())[:self.v_max]]
            vocab_tmp = list()
            for word in self.vocabulary:
                if word in words_allowed:
                    vocab_tmp.append(word)
            self.vocabulary = vocab_tmp
        
        # calculate idf
        N = len(corpus_tokenized)
        for elm in self.vocabulary:
            occurrence_count = 0
            for doc in corpus_tokenized:
                if elm in doc:
                    occurrence_count += 1
            term = (1+N)/(occurrence_count+1)
            self.idf_col.append(round(math.log(term)+1, 3))
            
  

    def word_count(self, col: List[List[str]]) -> Dict[str, int]:
        """params: expects collection of space-tokenized string (list of
            list of str) -> executes a word count
            returns: dict with collection vocabulary sorted from most
            to least frequent"""
        vocab = dict()
        for document in col:
            for word in document:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        return dict(sorted(vocab.items(), key=lambda x: x[1],reverse=True))
 

    def tf_calc(self, col: List[List[str]]) -> List[List[float]]:
        """params: takes dictionary with count of for each word
            reduces dict to length of vocabulary
            returns: list of list with calculated tf-idf metric"""
        # calculate term frequency
        tf_col = list()
        for doc in col: # getting tokenized doc from collection
            tmp_tf = list()
            for elm in self.vocabulary: # build tf
                tmp_tf.append(doc.count(elm))
            tf_col.append(tmp_tf)
        return tf_col
                
    
    def transform(self, collection: List[List[str]]) -> List[List[float]]:
        """input list of documents returns list of lists (matrix)
            each sublist represents matrix in tf-idf scheme"""
        # tokenize collection
        col_tokenized = self.preprocess(collection)
        # calculate word count for potential vocabulary trunctation
        #word_count_col = self.word_count(col_tokenized)
        # calculate term frequency
        tf_col = self.tf_calc(col_tokenized)
        tf_idf = list()
        for tf in tf_col:
            if len(tf) == len(self.idf_col):
                tf_idf_tmp = [tf[i]*self.idf_col[i] for i in range(len(self.idf_col))]
            tf_idf.append(tf_idf_tmp)
        if self.l2:
            tf_idf_tmp = list()
            for vector in tf_idf:
                v_norm = math.sqrt(sum([elm ** 2 for elm in vector]))
                if v_norm == 0:
                    tf_idf_tmp.append(vector)
                    continue
                v_tmp = [round(elm/v_norm, 3) for elm in vector]
                tf_idf_tmp.append(v_tmp)
            tf_idf = tf_idf_tmp
                
        return tf_idf

class TFIDF_stop_word(TFIDF):
    def __init__(self, lowercase: bool,
                 bad_chars: List[str] = '[.,?:;]',
                 v_max: int = -1,
                stop_words: List[str] = None) -> None:
        super().__init__(lowercase=lowercase, v_max = v_max)
        self.stop_words = stop_words
        
    def preprocess(self, collection: List[str]) -> List[List[str]]:
        """params: collection (list of string)
        returns: checks if stop words are set and removes them,
            makes all words lowercase if set, tokenizes each document"""
        if self.stop_words is not None: 
            # if stop words are given - remove them
            col_tmp = list()
            for doc in collection:
                query = doc.split()
                tok_tmp = [word for word in query
                           if word.lower() not in self.stop_words]
                doc_tmp = " ". join(tok_tmp)
                col_tmp.append(doc_tmp)
            collection = col_tmp    
        # remove bad characters
        col_clean = self.remove_bad_chars(collection)
        # make lower case if applicable
        if self.lowercase:
            col_l = self.lower_case_corpus(col_clean)
        else:
            col_l = col_clean
        return [s.split() for s in col_l]
