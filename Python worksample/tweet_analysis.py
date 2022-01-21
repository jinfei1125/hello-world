'''
Election Candidates Tweet Analysis

Author: Jinfei Zhu

'''

def keep_chr(ch):
    '''
    Find all characters that are classifed as punctuation in Unicode
    (except #, @, &) and combine them into a single string.
    '''
    return unicodedata.category(ch).startswith('P') and \
        (ch not in ("#", "@", "&"))

PUNCTUATION = " ".join([chr(i) for i in range(sys.maxunicode)
                        if keep_chr(chr(i))])

# When processing tweets, ignore these words
STOP_WORDS = ["a", "an", "the", "this", "that", "of", "for", "or",
              "and", "on", "to", "be", "if", "we", "you", "in", "is",
              "at", "it", "rt", "mt", "with"]

# When processing tweets, words w/ a prefix that appears in this list
# should be ignored.
STOP_PREFIXES = ("@", "#", "http", "&amp")



def find_top_k_entities(tweets, entity_desc, k):
    '''
    Find the k most frequently occuring entitites

    Inputs:
        tweets: a list of tweets
        entity_desc: a triple ("hashtags", "text", True),
          ("user_mentions", "screen_name", False), etc
        k: integer

    Returns: list of entities
    '''


    lst = []
    (key, subkey, case_sensitive) = entity_desc
    for tweet in tweets:
        key_lst = tweet['entities'][key]
        for i in key_lst:
            v = i[subkey]
            if case_sensitive:
                lst.append(v)
            if not case_sensitive:
                lst.append(v.lower())

    top_k = find_top_k(lst, k)

    return top_k


def find_min_count_entities(tweets, entity_desc, min_count):
    '''
    Find the entitites that occur at least min_count times.

    Inputs:
        tweets: a list of tweets
        entity_desc: a triple ("hashtags", "text", True),
          ("user_mentions", "screen_name", False), etc
        min_count: integer

    Returns: set of entities
    '''

    lst = []
    (key, subkey, case_sensitive) = entity_desc
    for tweet in tweets:
        key_lst = tweet['entities'][key]
        for i in key_lst:
            v = i[subkey]
            if case_sensitive:
                lst.append(v)
            if not case_sensitive:
                lst.append(v.lower())

    least_k = find_min_count(lst, min_count)    

    return set(least_k)



# Helper Functions

def pre_processing_words(text, PUNCTUATION, case_sensitive, STOP_PREFIXES, STOP_WORDS = False):
    """
    Convert the abridged text into a list of strings:

    1. Trun the abridged text of a tweet into a list of words
    2. Remove any leading and trailing punctuation from each word
    3. For tasks that are not case sensitive, convert the word to lower case.
    4. For the tasks that require it, eliminate any word from the list of words 
       that is included in STOP_WORDS
    5. Remove URLs, hashtags, and mentions

    Inputs:
        text: String 
        PUNCTUATION : Constant
        case_sensitive: Boolean
        STOP_PREFIXES: Constant
        STOP_WORDS: Constant (Default = False)
    
    Return: a list of processed words
    """
    # 1. Split a tweet into words
    words = text.split()
    new_words = []
    for word in words:
        # 2. Remove any leading and trailing punctuation
        word = word.strip(PUNCTUATION)
        if len(word) != 0:
            # 3. Check case sensitive
            if not case_sensitive:
                word = word.lower()
            # 4. If needed, only include words not in STOP_WORDS
            if STOP_WORDS:
                if word not in STOP_WORDS:
                    # 5. Remove URLs, hashtags, and mentions
                    if not word.startswith(STOP_PREFIXES):
                        new_words.append(word)

            # 4. If no need to exclude STOP_WORDS
            if not STOP_WORDS:
                # 5. Remove URLs, hashtags, and mentions
                if not word.startswith(STOP_PREFIXES):
                    new_words.append(word)
                    
    return new_words

def gen_n_gram(lst, n):
    """
    Generate N-gram for a list of words of a tweets

    Input:
        lst: list of words
        N: int

    Return:
        a new list of tuples

    """
    new_lst = []
    for i in range(len(lst)-n+1):
        n_gram = lst[i:i+n]
        new_lst.append(tuple(n_gram))

    return new_lst


def gen_ngram_lst(tweets, n, case_sensitive):
    """
    Generate a list of n-gram tuples.

    Inputs:
        tweets: a list of tweets
        n: integer
        case_sensitive: boolean

    Returns: list of n-grams

    """

    # Create a list to store pre-processed words
    texts = []
    for tweet in tweets:
        ori_texts = tweet["abridged_text"]
        processed_texts = pre_processing_words(ori_texts, PUNCTUATION, case_sensitive, STOP_PREFIXES, STOP_WORDS)
        n_gram = gen_n_gram(processed_texts, n)

        # Add n_gram tuples to the list (append method will create a list of lists)
        texts = texts + n_gram

    return texts    

def find_top_k_ngrams(tweets, n, case_sensitive, k):
    '''
    Find k most frequently occurring n-grams

    Inputs:
        tweets: a list of tweets
        n: integer
        case_sensitive: boolean
        k: integer

    Returns: list of n-grams
    '''

    texts = gen_ngram_lst(tweets, n, case_sensitive)

    top_k_ngram = find_top_k(texts, k)

    return top_k_ngram




def find_min_count_ngrams(tweets, n, case_sensitive, min_count):
    '''
    Find n-grams that occur at least min_count times.

    Inputs:
        tweets: a list of tweets
        n: integer
        case_sensitive: boolean
        min_count: integer

    Returns: set of n-grams
    '''

    texts = gen_ngram_lst(tweets, n, case_sensitive)
    least_ngram = find_min_count(texts, min_count)

    return set(least_ngram)

def find_salient_ngrams(tweets, n, case_sensitive, threshold):
    '''
    Find the salient n-grams for each tweet.

    Inputs:
        tweets: a list of tweets
        n: integer
        case_sensitive: boolean
        threshold: float

    Returns: list of sets of strings
    '''

    # Create a list of lists
    texts = []
    for tweet in tweets:
        ori_texts = tweet["abridged_text"]
        # Don't remove the stop words
        processed_texts = pre_processing_words(ori_texts, PUNCTUATION, case_sensitive, STOP_PREFIXES)
        n_gram = gen_n_gram(processed_texts, n)

        # Add n_gram tuples to the list (append method will create a list of lists)
        texts.append(n_gram)
    
    salient_ngram = find_salient(texts, threshold)

    return salient_ngram

import math
from util import sort_count_pairs

def count_tokens(tokens):
    '''
    Counts each distinct token (entity) in a list of tokens

    Inputs:
        tokens: list of tokens (must be immutable)

    Returns: dictionary that maps tokens to counts
    '''

    d = {}
    for i in tokens:
        if i in d:
            d[i] += 1
        if i not in d:
            d[i] = 1
    return d


def find_top_k(tokens, k):
    '''
    Find the k most frequently occuring tokens

    Inputs:
        tokens: list of tokens (must be immutable)
        k: a non-negative integer

    Returns: list of the top k tokens ordered by count.
    '''

    #Error checking 
    if k < 0:
        raise ValueError("In find_top_k, k must be a non-negative integer")
    
    d = count_tokens(tokens)
    lst = []
    top_k = []

    if len(tokens) != 0:
        for key, value in d.items():
            tpl = (key, value)
            lst.append(tpl) 

        sorted_lst = sort_count_pairs(lst)
        sorted_key = [i[0] for i in sorted_lst]
        top_k = sorted_key[:k]


    return top_k


def find_min_count(tokens, min_count):
    '''
    Find the tokens that occur *at least* min_count times

    Inputs:
        tokens: a list of tokens  (must be immutable)
        min_count: a non-negative integer

    Returns: set of tokens
    '''

    #Error checking 
    if min_count < 0:
        raise ValueError("min_count must be a non-negative integer")

    d = count_tokens(tokens)
    lst = []

    for key, value in d.items():
        tpl = (key, value)
        lst.append(tpl) 

    sorted_lst = sort_count_pairs(lst)

    min_lst = [x for x, v in sorted_lst if v >= min_count]

    return set(min_lst)


def find_salient(docs, threshold):
    '''
    Compute the salient words for each document.  A word is salient if
    its tf-idf score is strictly above a given threshold.

    Inputs:
      docs: list of list of tokens
      threshold: float

    Returns: list of sets of salient words
    '''

    sal_lst = []

    for doc in docs:
        sal_word = []
        for t in doc:
            if tf(t, doc)*idf(t, docs) > threshold:
                sal_word.append(t)
        sal_lst.append(set(sal_word))

    return sal_lst

def tf(t, doc):
    '''
    Compute the augumented frequency (tf score) 
    of a word t in a document

    Inputs:
        t: string
        doc: list of string

    Returns: tf score
    '''
    count_dict = count_tokens(doc)
    f_td = count_dict.get(t, 0)

    top_word = find_top_k(doc, 1)
    max_f_td = count_dict.get(top_word[0])

    tf = 0.5 + 0.5*(float(f_td)/float(max_f_td))

    return tf

def idf(t, docs):
    '''
    Compute the Vasic Inverse document frequency
    of term t in a collection of documents.
    
    Inputs:
        t: a string
        docs: a list of lists, the collection of documents
    
    Returns: idf score

    '''
    N = len(docs)
    num_d_docs = 0

    for doc in docs:
        if t in doc:
            num_d_docs += 1
    
    idf = math.log(N/num_d_docs)

    return idf