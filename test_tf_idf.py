from tf_idf import TFIDF, TFIDF_stop_word

print("++++++++++++++++++++++++++++++++TEST Part I ++++++++++++++++++++++++++++++++")

# Testing
# class properties
corpus = TFIDF(lowercase=True, v_max=15, n_grams=2)
# corpus.fit(['a', 'b', 'c', 'd'])
# assert corpus.v_max == 4
# print(corpus.vocabulary)

# methods
s = ["This is a test: What do you want from me?",
     "nothing to clean here",
     ":?:mickey mouse!@.,#"]
assert corpus.remove_bad_chars(s) == ["This is a test What do you want from me",
                                      "nothing to clean here",
                                      "mickey mouse!@#"]

print("TEST 1:")
test_corpus = ['The hotel and the stay were great',
               'This was a great stay',
               'Great stay in a great destination',
               'Great destination']
corpus.fit(test_corpus)
print(f"vocabulary: {corpus.vocabulary}")

# assert corpus.transform(test_corpus) == [[1.916, 0.0, 1.0, 1.916, 0.0, 1.223, 3.833, 0.0, 0.0, 1.916],
#  [0.0, 0.0, 1.0, 0.0, 0.0, 1.223, 0.0, 1.916, 1.916, 0.0],
#  [0.0, 1.511, 2.0, 0.0, 1.916, 1.223, 0.0, 0.0, 0.0, 0.0],
#  [0.0, 1.511, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
print(corpus.transform(test_corpus))
print("TEST 2:")
#print(f"vocabulary: {corpus.vocabulary}")

corpus.transform(test_corpus)
test_corpus_2 = ['This was a wonderful stay',
                 'Dear customer thanks for your review',
                 ]
# assert corpus.transform(test_corpus_2) = [[0.0, 0.0, 0.0, 0.0, 0.0, 1.223, 0.0, 1.916 1.916, 0.0],
#  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
corpus.transform(test_corpus_2)

print("++++++++++++++++++++++++++++++++TEST Part II ++++++++++++++++++++++++++++++++")

bad_chars = [".", "?", ":", ";"]
stop_words = ["and", "a", "the", "it", "he",
              "she", "where", "was", "for"]

corpus = ['The hotel and the stay were great',
          'This was a great stay',
          'Great stay in a great destination',
          'Great destination']

print("################ TEST CASE 1 ###################")
test_case_1 = ['The hotel and the stay were great',
          'This was a great stay',
          'Great stay in a great destination',
          'Great destination']
tfidf = TFIDF_stop_word(lowercase=True)
tfidf.fit(corpus)
print("vocabulary: {}".format(tfidf.vocabulary))
full_vocab = tfidf.vocabulary
print(tfidf.transform(test_case_1))

print("################ TEST CASE 2 ###################")
test_case_1 = ['The hotel and the stay were great',
          'This was a great stay',
          'Great stay in a great destination',
          'Great destination']
tfidf = TFIDF_stop_word(lowercase=False)
tfidf.fit(corpus)
print("vocabulary: {}".format(tfidf.vocabulary))
print(tfidf.transform(test_case_1))

print("################ TEST CASE 3 ###################")
tfidf = TFIDF_stop_word(lowercase=True, stop_words=stop_words)
tfidf.fit(corpus)
print("vocabulary: {}".format(tfidf.vocabulary))
assert set(full_vocab) - set(stop_words) == set(tfidf.vocabulary)
print(tfidf.transform(test_case_1))

print("################ TEST CASE 4 ###################")
test_case_2 = ['This was a wonderful stay',
               'Dear customer thanks for your review',]
tfidf = TFIDF_stop_word(lowercase=False)
tfidf.fit(corpus)
print(tfidf.transform(test_case_2))

print("################ TEST CASE 5 ###################")
test_case_2 = ['This was a wonderful stay',
               'Dear customer thanks for your review',]
tfidf = TFIDF_stop_word(lowercase=True,
                       stop_words=stop_words)
tfidf.fit(corpus)
print(tfidf.transform(test_case_2))
