# -*- coding: utf-8 -*-
import codecs,random
import numpy as np
class DataHelper(object):
    def __init__(self,filename,min_count=2):
        self.filename = filename
    
        self.unknown_token = "[unknown]"
        self.count,self.reverse_dictionary = self.build_dataset(min_count=min_count)
        self.vocabulary_size = len(self.reverse_dictionary)
    def data_iterator(self,dictionary_already=True):
        with codecs.open(self.filename,encoding="utf-8",errors='ignore') as f:
            for line in f:
                if not dictionary_already:
                    yield line.strip()
                else:
                    yield [self.dictionary[word] if word in self.dictionary else 0 for word in line.strip().split()]
    
    def build_dataset(self,min_count =2):
        count = dict()
        count[self.unknown_token]=min_count+1
        self.dictionary = dict()
        self.dictionary [self.unknown_token] = 0
        reversed_dictionary = [self.unknown_token]
        for line in self.data_iterator(dictionary_already=False):
            for word in line.split():
                if word not in count:                
                    count[word]=1
                else:
                    count[word]=count[word]+1
        for word,value in count.items():
            if value>min_count:
                self.dictionary[word]=len(self.dictionary)
                reversed_dictionary.append(word)
        return count,reversed_dictionary

    
    
    def generate_batch(self, batch_size, num_skips,skip_window, num_steps=2, cbow = False):
        if cbow != False:
            for batch in self.generate_batch_cbow(batch_size, num_skips,skip_window, num_steps=num_steps):
                yield batch 
            return
        x,y=[],[]
        context_candidates = [w for w in range(2 * skip_window + 1) if w != skip_window]
        for xx in range(num_steps):
            for line_tokens in self.data_iterator():
                for i in range(skip_window,len(line_tokens)-skip_window):
                    context_words=random.sample(context_candidates, num_skips)
                    for context_word in context_words:
                        x.append(line_tokens[i])
                        y.append([context_word])
                yield x,y
    def generate_batch_cbow(self, batch_size, num_skips,skip_window, num_steps=2):
        x,y=[],[]
        for xx in range(num_steps):
            for line_tokens in self.data_iterator():
                for i in range(skip_window,len(line_tokens)-skip_window):
                    x.append(line_tokens[i-skip_window:i]+ line_tokens[i+1:i+skip_window+1])
                    y.append(line_tokens[i])
                index=np.arange(len(x))
                choice = np.random.choice(index,batch_size)
                yield np.array(x)[choice],np.array(y)[choice]    


if __name__=="__main__":
    helper = DataHelper("demo.txt")
    generator = next(iter(helper.generate_batch(100, 3, 2)))
    batch_inputs, batch_labels = generator
    print( batch_inputs)
    print(batch_labels)
    generator = next(iter(helper.generate_batch_cbow(100, 3, 2)))
    batch_inputs, batch_labels = generator
    print( batch_inputs)
    print(batch_labels)

    
    