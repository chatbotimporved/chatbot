# Natural Language Toolkit: Eliza
#
# Copyright (C) 2001-2013 NLTK Project
# Authors: Steven Bird <stevenbird1@gmail.com>
#          Edward Loper <edloper@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

# Based on an Eliza implementation by Joe Strout <joe@strout.net>,
# Jeff Epler <jepler@inetnebr.com> and Jez Higgins <mailto:jez@jezuk.co.uk>.

# a translation table used to convert things you say into things the
# computer says back, e.g. "I am" --> "you are"

import nltk

reflections = {
    "am" : "are",
    "was" : "were"
}

pairs = (
    # suggestions
    (
        r"I'm hungry",
        ( "I'm hungry too!",
          "Let's get dinner!",
          "Let's go to the Doggy Diner!")),
    (r"Where is(.*)",
     ( "Those teethmarks on %1 are not mine!",
       "I bet you didn't know %1 was edible...")),
    # anything else
    (r'(.*)',
     (
         "(Looking soulfully) treat please...",
         "Did you say 'Let's go for a walk?'",
         "Yes! Yes! Nap time!",
     )
    )
)


dog_chatbot = nltk.chat.Chat(pairs, reflections)

def dog_chat():
    print "Emoting cutely ..."
    dog_chatbot.converse()

def demo():
    dog_chat()

if __name__ == "__main__":
    demo()