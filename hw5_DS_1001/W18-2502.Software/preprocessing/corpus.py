import logging
import glob
import re
import os
import functools

from sklearn.datasets import fetch_20newsgroups


logger = logging.getLogger(__name__)


SGML_ENTITIES = {
    'amp': '&',
    'lt': '<',
    'gt': '>',
    'nbsp': ' ',
}

def _sgml_unescape_cb(match):
    entity = match.group(1).lower()
    return SGML_ENTITIES[entity]


def generate_gigaword_texts(root=None, publisher_glob='*', types={'story',}, lang='eng'):
    if root is None:
        root = os.environ['GIGAWORD_ROOT']
    type_re = '|'.join('type="{}"'.format(t) for t in types)
    paths = glob.glob(os.path.join(root, 'data', publisher_glob + '_' + lang, '*'))
    for path in paths:
        logger.info('reading %s', path)
        with open(path) as f:
            matches = re.finditer('<DOC.*?</DOC>', f.read(), re.DOTALL)
        parsed = yielded = 0
        for match in matches:
            parsed += 1
            doc = match.group()
            doc_tag = doc.partition('>')[0]
            if not re.search(type_re, doc_tag):
                continue
            text_match = re.search('<TEXT>.*?</TEXT>', doc, re.DOTALL)
            text_sgml = text_match.group()
            logger.debug('sgml: %r', text_sgml)
            # note that markup is usually only block markup, but newlines are also present.
            text = re.sub('<[^>]*>', '', text_sgml)
            logger.debug('text: %r', text)
            assert '<' not in text
            text = re.sub('&(.*?);', _sgml_unescape_cb, text)
            yield text
            yielded += 1
        logger.info('got %d texts from %d docs from %s', yielded, parsed, path)


for publisher in ['afp', 'apw', 'cna', 'ltw', 'nyt', 'wpb', 'xin']:
    locals()['generate_gigaword_%s_texts' % publisher] = functools.partial(generate_gigaword_texts, publisher_glob=publisher)


def generate_20newsgroups_texts():
    return fetch_20newsgroups(remove=('headers', 'footers', 'quotes')).data
