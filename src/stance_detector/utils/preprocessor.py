"""
This is a copy from https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py
This is one kind of pre-processing of text pipeline which has been used in this project
No changes have been made to it by us. 
Info about pre-processing from BertTweet paper (https://aclanthology.org/2020.emnlp-demos.2.pdf), 
    "We tokenize those English Tweets using “TweetTokenizer” from the NLTK toolkit (Bird et al.,
    2009) and use the emoji package to translate
    emotion icons into text strings (here, each icon
    is referred to as a word token).2 We also normalize the Tweets by converting user mentions and
    web/url links into special tokens @USER and
    HTTPURL, respectively."
Copy-paster: Rachith
"""

from nltk.tokenize import TweetTokenizer
from emoji import demojize


tokenizer = TweetTokenizer()


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())
