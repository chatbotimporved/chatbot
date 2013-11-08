chatbot
=======
For this assignment, you have a choice of two options.   You are asked to work in pairs, with someone else from your current J-group.  I'm assuming you will decide who will work together based on which option you want to choose below.  It's ok for both pairs of a group to work on the same option but each pair should do their own work..

(A) ChatBot Option:

As you can see in the NLTK chatbot code, there are several different chabots with different personalities. You can easily create a new chatbot in NLTK as the attached code shows. The simple way to modify this code is to just make a new list of rules. But the NLTK rule pairs are very simple by classic Eliza standards and don't lead to very interesting conversations as currently written. This is in part because the substitutions are not very complicated compared to older versions of this program, such as the doctor.el program in emacs.
(http://www.csee.umbc.edu/courses/471/papers/emacs-doctor.shtml )

If you choose to do this option, your goal is to make a chatbot that is better than the one currently in NLTK.   You will probably want to give it a different personality than Eliza. The sample code I've attached shows the beginnings of a dog chatbot and the NLTK code shows other examples. (dog_chat.pyView in a new window)

Your task is to modify the code in a more significant way than just adding new comment/response pairs. There are several different things you might do. Here are some suggestions; you would probably only do one of these in the time allotted, but a combination may work well:

Suggestion 1: make it do something useful.  Maybe it can recommend chess moves or tutor on a topic. For example, this is a rather effective grammar tutor chatbot:  http://www.eslfast.com/robot/english_tutor.htm 

Suggestion 2: make it a better conversationalist by learning a large set of conversation pairs from an existing conversation collection. You'll need to decide what kinds of variables to insert where.

Suggestion 3: The NLTK chatbot is rather uninteresting because it only does a very limited kind of variable  substitution between the user statement and the chatbot response. Modify the chatbot in two ways. First, make the variable matching richer, and second, keep track of some discourse entities across the conversation, and resurface concepts that have been introduced earlier in the conversation.

No matter which suggestion you choose, you should test out the bot with other classmates. Try it out at the very least with the other two members of your group and record the transcript of their interaction with the chatbot.

To turn in:

(i) A text description of what you did, including which suggestion(s) above you did and the motivation behind the code and what was done by each project partner. (5 points)
(ii) The transcript of people using your chatbot (2 points)
(iii) The code itself, commented (8 points)
