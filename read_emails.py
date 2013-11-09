import nltk
import random
import re

from nltk.corpus import CategorizedPlaintextCorpusReader as CPCR

class ConsecutiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

    def tag(self, sentence):
        history = []
    	for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return conlltags2tree(conlltags)

def npchunk_features(sentence, i, history):
    word, pos = sentence[i]

    if i == 0:
    	prevword, prevpos = "<START>", "<START>"
    else:
    	prevword, prevpos = sentence[i-1]

    if i == len(sentence)-1:
    	nextword, nextpos = "<END>", "<END>"
    else:
    	nextword, nextpos = sentence[i+1]

    return {"pos": pos, 
    		"word": word,
    		"prevpos": prevpos,
    		"nextpos": nextpos,
    		"prevpos+pos": "%s+%s" % (prevpos, pos),
    		"pos+nextpos": "%s+%s" % (pos, nextpos),
    		"tags-since-dt": tags_since_dt(sentence, i)}

def tags_since_dt(sentence, i):
	tags = set()

	for word, pos in sentence[:i]:
		if pos == 'DT':
			tags = set()
		else:
			tags.add(pos)

	return '+'.join(sorted(tags))

def extract_categories(filesDir):
	emails = CPCR(filesDir, r'.*\.txt', cat_pattern=r'(.*)\.txt')
	return emails

def train_chunker(filesDir):
	emails = extract_categories(filesDir)
	
	sentences = split_raw_to_sentences(emails.raw())

	tokenized_emails = tokenize_sentences(sentences)
	tagged_emails = tag_sentences(tokenized_emails)

	print tagged_emails
	# Return the chunker
	chunker = ConsecutiveNPChunker(tagged_emails)

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
classifier = train_chunker(output_dir)

