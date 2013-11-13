# coding=utf-8
import nltk
import random
import os
import re

from nltk.corpus import wordnet

from nltk import TaggerI, untag
from nltk.chunk import ChunkParserI, tree2conlltags
from nltk.chunk.util import conlltags2tree

from nltk.corpus import conll2000
from nltk.classify import MaxentClassifier

from nltk.corpus import CategorizedPlaintextCorpusReader as CPCR

'''
The chunk tagger class as it is used in the NLTK book and at:
http://streamhacker.com/2010/03/15/nltk-classifier-based-chunker-accuracy/
'''
class ConsecutiveNPChunkTagger(TaggerI):

    def __init__(self, feature_extractor, classifier):
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def tag(self, sentence):
        history = []

        for i, word in enumerate(sentence):
            featureset = self.feature_extractor(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)

        return zip(sentence, history)


    @classmethod
    def train(cls, train_sents, feature_extractor, classifier_cls, **kwargs):
        train_set = []

        for tagged_sent in train_sents:
            untagged_sent = untag(tagged_sent)
            history = []

            for i, (word, tag) in enumerate(tagged_sent):
                featureset = feature_extractor(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)

        classifier = classifier_cls.train(train_set, **kwargs)
        return cls(feature_extractor, classifier)

'''
The NPChunker as it is used in the NLTK book and at:
http://streamhacker.com/2010/03/15/nltk-classifier-based-chunker-accuracy/
'''
class ConsecutiveNPChunker(ChunkParserI):
    def __init__(self, train_sents, *args, **kwargs):
        tag_sents = [tree2conlltags(sent) for sent in train_sents]
        train_chunks = [[((w,t),c) for (w,t,c) in sent] for sent in tag_sents]
        self.tagger = ConsecutiveNPChunkTagger.train(train_chunks, *args, **kwargs)

    def parse(self, tagged_sent):
        if not tagged_sent: return None
        chunks = self.tagger.tag(tagged_sent)
        conlltags = [(w,t,c) for ((w,t),c) in chunks]
        return conlltags2tree(conlltags)

################################ FEATURES ##################################

'''
Features that we think are important:
-   Words have capitals except for stopwords
-   There are numbers included
-   Most words are nouns
-   There are brackets ()
-   Most words won’t be in the dictionary
-   Most words are names NP
-   The locations often include hall, school or house
'''

# Combine all the features into one dictionary
def combine_features(sent, i, history):
    features = dict(
    list(word(sent, i, history).items()) +
    list(pos(sent, i, history).items()) +
    list(pos_word(sent, i, history).items()) +
    list(prev_pos(sent, i, history).items()) +
    list(prev_pos_word(sent, i, history).items()) +
    list(next_pos(sent, i, history).items()) +
    list(next_pos_word(sent, i, history).items()) +
    list(prev_next_pos(sent, i, history).items()) +
    list(prev_next_pos_word(sent, i, history).items()) +
    list(is_capitalized(sent, i, history).items()) +
    list(is_all_capitalized(sent, i, history).items()) +
    list(is_numeric(sent, i, history).items()))

    return features;

#gets the word
def word(sent, i, history):
    word, pos = sent[i]
    return {'word': word}

#gets the part of speech of the word
def pos(sent, i, history):
    word, pos = sent[i]
    return {'pos': pos}
 
 #gets the part of speech  and the word
def pos_word(sent, i, history):
    word, pos = sent[i]
    return {'pos+word': "%s+%s" % (pos, word)}

 #gets the part of speech of the previous word, and the curent words
def prev_pos(sent, i, history):
    word, pos = sent[i]
 
    if i == 0:
        prevword, prevpos = '<START>', '<START>'
    else:
        prevword, prevpos = sent[i-1]
 
    return {'pos+prevpos': "%s+%s" % (pos, prevpos)}

 

 #gets the part of speech of the previous word, the current word, and the word
def prev_pos_word(sent, i, history):
    word, pos = sent[i]
 
    if i == 0:
        prevword, prevpos = '<START>', '<START>'
    else:
        prevword, prevpos = sent[i-1]
 
    return {'pos+prevpos+word': "%s+%s+%s" % (pos, prevpos, word)}

 
 #gets the part of speech of the next word, and the current word
def next_pos(sent, i, history):
    word, pos = sent[i]
 
    if i == len(sent) - 1:
        nextword, nextpos = '<END>', '<END>'
    else:
        nextword, nextpos = sent[i+1]
 
    return {'pos+nextpos': "%s+%s" % (pos, nextpos)}

 
#get the part of speech of the current and next words and the actual next word
def next_pos_word(sent, i, history):
    word, pos = sent[i]
 
    if i == len(sent) - 1:
        nextword, nextpos = '<END>', '<END>'
    else:
        nextword, nextpos = sent[i+1]
 
    return {'pos+nextpos+word': "%s+%s+%s" % (pos, nextpos, word)}
 
#get the part of speech of the current, previous and next words
def prev_next_pos(sent, i, history):
    word, pos = sent[i]
 
    if i == 0:
        prevword, prevpos = '<START>', '<START>'
    else:
        prevword, prevpos = sent[i-1]
 
    if i == len(sent) - 1:
        nextword, nextpos = '<END>', '<END>'
    else:
        nextword, nextpos = sent[i+1]
 
    return {'pos+nextpos+prevpos': "%s+%s+%s" % (pos, nextpos, prevpos)}
 

#get the part of speech of the current, previous and next words, and the current word
def prev_next_pos_word(sent, i, history):
    word, pos = sent[i]
 
    if i == 0:
        prevword, prevpos = '<START>', '<START>'
    else:
        prevword, prevpos = sent[i-1]
 
    if i == len(sent) - 1:
        nextword, nextpos = '<END>', '<END>'
    else:
        nextword, nextpos = sent[i+1]
 
    return {'pos+nextpos+word+prevpos': "%s+%s+%s+%s" % (pos, nextpos, word, prevpos)}

#return if the word is in the english dictionary
def in_dictionary(sent, i, history):
    word, pos = sent[i]
    return {'dictionary': wordnet.synsets(word)}

#return if the word is capitalized
def is_capitalized(sent, i, history):
    word, pos = sent[i]
    wordnet.synsets(word)
    return {'caps': word[0].isupper()}

#return if the word is in all caps 
def is_all_capitalized(sent, i, history):
    word, pos = sent[i]
    result = True
    for char in word:
        result = result = char.isupper()
    return {'allcaps': result}

#a helper fn that checks if a word is a number 
def number(s):
    try:
        float(s)
        return True
    except ValueError:
       return False

#return if the word is a number
def is_numeric(sent, i, history):
    word, pos = sent[i]
    return{'num': number(word)}

############################## END FEATURES ###############################

'''
Train the chunker with the training data and the NaiveBayesClassifier
'''
def train_chunker(filesDir):
    # Create chunked sentences in the CoNLL format.
    train_sents = conll2000.chunked_sents('train_locations.txt', chunk_types=['Loc'])

    # Train the chunker with the NaiveBayesClassifier
    chunker = ConsecutiveNPChunker(train_sents, combine_features, nltk.NaiveBayesClassifier)

    return chunker

'''
Test the chunker and print the accuracy, precision, recall, and F-Measure
'''
def test_chunker(filesDir, classifier):
    # Create chunked sentences in the CoNLL format.
    test_sents = conll2000.chunked_sents('test_locations.txt', chunk_types=['Loc'])
    print classifier.evaluate(test_sents);

    text = conll2000.raw('test_data_normal.txt')

    location_list = []
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    for sent in sent_tokenizer.tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'node'):
                if chunk.node == "GPE":
                    location = ' '.join(c[0] for c in chunk.leaves())
                    location_list.append(location)

    print location_list

'''A simple tagger '''
def tag_sentences(sentences):
    tagged_emails = []
    
    for e in sentences:
        tagged_emails.append(nltk.pos_tag(e))

    return tagged_emails

'''This method is used to tokenize the text into meaningfull tokens.'''
def tokenize_sentences(sentences):
    pattern = r'''(?x)              # Allow verbose regular expressions
      ([a-zA-Z]\.)+                 # Normal abbreviations
    | [A-Za-z]+(['-][A-Za-z0-9]+)*  # Get words such as far-reaching and short-term
    | \$?(\d)+(\.\d+)?%?            # Digits, currency and percentages
    | \.\.\.                        # Ellipsis
    | (&lt;)+                       # Lower than sign
    | [][.,;"'?():-_`]              # Some tokens
    '''

    results = []

    for sentence in sentences:
        results.append(nltk.regexp_tokenize(sentence, pattern))

    return results

# We split the raw text into sentences based on new rules, ! and ?-marks.
# This is the most reliable way without making it too complex.
def split_raw_to_sentences(raw):
    r = re.compile("(\n|\!|\?)")
    sentences = r.split(raw)

    return sentences

'''
This method makes sure that our testdata is in the good format.
It tags all our data files and stores the tagged data into separate files.
Finally, by default this method gives the NE tag 'O'.
The O tags are where necessary manually changed to B-Loc and I-Loc.
'''
def createTaggedFiles(input_directory, output_directory):
    global results
    results = str()
    # Open the email files to tag and chunk
    filelist = getFilenames(input_directory)

    # Loop through all the email files we have
    for file in filelist:
        fileName = file
        raw = open(input_directory + fileName, 'rU').read()
        
        # Split file into sentences
        sentences = split_raw_to_sentences(raw)
        # Tokenize the sentences
        tokenized_email = tokenize_sentences(sentences)
        # Tag the sentences
        tagged_email = tag_sentences(tokenized_email)

        tagList = []

        # Merge the text, tag and IOB tag into one line (just like conll2000 train.txt)
        for sentence in tagged_email:
            if len(sentence) > 0:
                for tags in sentence:
                    tags += ("O",)
                    tagList.append(' '.join(tags))
                tagList.append("")

        #Merge every pair in one document and place every pair on a new line
        tagged_doc = '\n'.join(tagList)

        # write the results to a file
        g = open(output_directory + file, "w")
        g.write(tagged_doc)
        g.close()

def getFilenames(input_directory):
    filelist = sorted([file for file in os.listdir(input_directory) if file.endswith('.txt')])
    return filelist

input_dir = "/Users/dirkdewit/Documents/School/Master HTI/Internationaal Semester/Applied Natural Language Processing/Assignments/Chatbot/chatbot/testdata/"
output_dir = "/Users/dirkdewit/Documents/School/Master HTI/Internationaal Semester/Applied Natural Language Processing/Assignments/Chatbot/chatbot/testdata/tagged/"

'''
This method is used to create the items (only ran once, now commented out)
'''
# createTaggedFiles(input_dir, output_dir)

'''
The following methods train a chunker and test it on the data.
'''
chunker = train_chunker(output_dir)
test_chunker(input_dir, chunker)
