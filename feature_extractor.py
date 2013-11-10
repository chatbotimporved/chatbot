from nltk.corpus import wordnet

'''
Features that we think are important:
-	Words have capitals except for stopwords
-	There are numbers included
-	Most words are nouns
-	There are brackets ()
-	Most words won’t be in the dictionary
-	Most words are names NP
-	The locations often include hall, school or house

'''
'''#what do we mean by history'''

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
    return {'pos': pos, 'word': word}

 #gets the part of speech of the previous word, and the curent words
def prev_pos(sent, i, history):
    word, pos = sent[i]
 
    if i == 0:
        prevword, prevpos = '<START>', '<START>'
    else:
        prevword, prevpos = sent[i-1]
 
    return {'pos': pos, 'prevpos': prevpos}
 

 #gets the part of speech of the previous word, the current word, and the word
def prev_pos_word(sent, i, history):
    word, pos = sent[i]
 
    if i == 0:
        prevword, prevpos = '<START>', '<START>'
    else:
        prevword, prevpos = sent[i-1]
 
    return {'pos': pos, 'prevpos': prevpos, 'word': word}
 
 #gets the part of speech of the next word, and the current word
def next_pos(sent, i, history):
    word, pos = sent[i]
 
    if i == len(sent) - 1:
        nextword, nextpos = '<END>', '<END>'
    else:
        nextword, nextpos = sent[i+1]
 
    return {'pos': pos, 'nextpos': nextpos}
 
#get the part of speech of the current and next words and the actual next word
def next_pos_word(sent, i, history):
    word, pos = sent[i]
 
    if i == len(sent) - 1:
        nextword, nextpos = '<END>', '<END>'
    else:
        nextword, nextpos = sent[i+1]
 
    return {'pos': pos, 'nextpos': nextpos, 'word': word}
 
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
 
    return {'pos': pos, 'nextpos': nextpos, 'prevpos': prevpos}
 

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
 
    return {'pos': pos, 'nextpos': nextpos, 'word': word, 'prevpos': prevpos}

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
def is_capitalized(sent, i, history):
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

