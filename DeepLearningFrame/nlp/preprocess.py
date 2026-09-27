import numpy as np

def preprocess(x:str)->tuple[list, dict,dict]:
    """
    :param x: the context you input
    :return: returns following in order: corpus(context translated using labels),word_to_id(a dictionary using word as key and id as value),id_to_word
    :process_principle: do not distinguish upper/lower case, keep all punctuation
    """
    if not isinstance(x, str):
        raise TypeError("input must be a string")
    x=x.lower()
    x = x.replace('.', ' . ')
    x=x.split()
    word_to_id={}
    id_to_word={}
    corpus=[]
    for word in x:
        if  word not in word_to_id:
            current_id=len(word_to_id)
            word_to_id[word]=current_id
            id_to_word[current_id]=word
        corpus.append(word_to_id[word])
    return corpus, word_to_id, id_to_word

def create_context(corpus: list[str], window_size:int=1)->tuple[list[list[int]],list[str]]:
    if window_size <= 0 or not isinstance(window_size, int):
        raise ValueError("window_size must be a positive integer")
    if 2*window_size>=len(corpus):
        raise ValueError(f"window size:{window_size} is bigger than the corpus size:({len(corpus)})!")
    #目标词列表
    target=corpus[window_size:-window_size]
    context=[]
    for center_index in range(window_size,len(corpus)-window_size):
        current_context = []
        for offset in range(-window_size, window_size+1):
            if offset==0:
                continue
            else:
                current_context.append(corpus[center_index+offset])
        context.append(current_context)
    return context,target

def count_words(corpus:list[int])->dict[int,int]:
    from collections import defaultdict
    word_count=defaultdict(int)
    for word_id in corpus:
        word_count[word_id]+=1
    return dict(word_count)

