import nltk
import random
import re

from nltk import TaggerI, untag
from nltk.chunk import ChunkParserI, tree2conlltags
from nltk.chunk.util import conlltags2tree

from nltk.corpus import conll2000
from nltk.classify import MaxentClassifier

from nltk.corpus import CategorizedPlaintextCorpusReader as CPCR

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

def extract_categories(filesDir):
	emails = CPCR(filesDir, r'.*\.txt', cat_pattern=r'(.*)\.txt')
	return emails

def train_chunker(filesDir):
	emails = extract_categories(filesDir)
    	
	sentences = split_raw_to_sentences(emails.raw())

	tokenized_emails = tokenize_sentences(sentences)
	train_sents = tag_sentences(tokenized_emails)
	
	chunker = ConsecutiveNPChunker(train_sents, featx, MaxentClassifier, 
        minlldelta=0.01, max_iter=10)

	return chunker

def tag_sentences(sentences):
	tagged_emails = []
	
	for e in sentences:
		tagged_emails.append(nltk.pos_tag(e))

	return tagged_emails

def tokenize_sentences(sentences):
	pattern = r'''(?x)				# Allow verbose regular expressions
	  ([a-zA-Z]\.)+					# Normal abbreviations
	| [A-Za-z]+(['-][A-Za-z0-9]+)*	# Get words such as far-reaching and short-term
	| \$?(\d)+(\.\d+)?%?			# Digits, currency and percentages
	| \.\.\. 						# Ellipsis
	| (&lt;)+						# Lower than sign
	| [][.,;"'?():-_`]				# Some tokens
	'''

	results = []

	for sentence in sentences:
		results.append(nltk.regexp_tokenize(sentence, pattern))

	return results

def split_raw_to_sentences(raw):
	sentences = raw.split('\n')

	return sentences

def test_classifier(filesDir, classifier):
	products = extract_categories(filesDir)
	documents = createDocuments(products)
	test_list = wordList(products, documents)

	print nltk.classify.accuracy(classifier, test_list)

output_dir = "/Users/dirkdewit/Documents/School/Master HTI/Internationaal Semester/Applied Natural Language Processing/Assignments/Chatbot/chatbot/testdata/"
#classifier = train_chunker(output_dir)
print conll2000.chunked_sents("train.txt")

