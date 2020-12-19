import re

from sklearn.feature_extraction.text import CountVectorizer
#import nltk
import spacy

spacy_nlp = spacy.load('en_core_web_sm')
spacy_nlp.pipeline = []  # tokenize only

# nltk.download('punkt')

sklearn_analyzer = CountVectorizer().build_analyzer()


def ptb_analyzer(s):
    return map(str, spacy_nlp(s))


###def whitespace_and_strip_analyzer(s):
###    # should be basically equivalent to splitting on whitespace then stripping any non-word chars
###    return re.findall(r'\w+(?:\S+\w+)*', s, re.UNICODE)


def make_lucene():
    import lucene
    from java.io import StringReader
    from org.apache.lucene.analysis.standard import StandardTokenizer
    from org.apache.lucene.analysis.tokenattributes import CharTermAttribute

    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    # Basic tokenizer example.
    tokenizer = StandardTokenizer()
    charTermAttrib = tokenizer.getAttribute(CharTermAttribute.class_)

    def analyze(s):
        tokenizer.setReader(StringReader(s))
        tokenizer.reset()
        tokens = []
        while tokenizer.incrementToken():
            tokens.append(charTermAttrib.toString())
        tokenizer.close()
        return tokens

    return analyze


lucene_analyzer = make_lucene()


if __name__ == '__main__':
    import sys
    for l in sys.stdin:
        print('skl:', sklearn_analyzer(l))
        print('ptb:', ptb_analyzer(l))
###        print('wss:', whitespace_and_strip_analyzer(l))
        print('luc:', lucene_analyzer(l))
