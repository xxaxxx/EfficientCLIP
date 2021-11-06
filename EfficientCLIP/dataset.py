import numbers
import os
import queue as Queue
import threading

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
import random

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch
    
class Img_Text_Dataset(Dataset):
    def __init__(self, text_data_mmap,feature_data_mmap,tokenize,datas_num,local_rank,indexes=None,scores=None,value=0.8):
        super(Img_Text_Dataset, self).__init__()
        self.text_mmap=text_data_mmap
        self.feature_mmap=feature_data_mmap
        self.datas_num=datas_num
        self.local_rank=local_rank
        if indexes is not None:
            self.indexes=indexes
        else:
            self.indexes=torch.arange(datas_num)
        self.tokenizer=tokenize
        self.scores=torch.zeros([datas_num,]) if scores is None else scores
        self.filtered_num=int(datas_num*value)

    def __getitem__(self, item):
        
        #map the index of text data
        index=self.indexes[item]
        
        # preproce the text data
        sample_zh=self.text_mmap.get(index).tolist()
        sample_zh=[self.tokenizer.encoder['<s>']]+sample_zh[:75]+[self.tokenizer.encoder['<eod>']]
        sample_zh+=[self.tokenizer.encoder['<pad>']]*(77-len(sample_zh))
        
        
        img_feature=np.frombuffer(self.feature_mmap,dtype='float16',offset=index*2*512,count=512)
        
        return torch.LongTensor(sample_zh),torch.from_numpy(img_feature).detach(),index


    def __len__(self):
        return len(self.indexes)

class Text_Dataset(Dataset):
    def __init__(self, data_mmap,tokenize,local_rank,datas_num):
        super(Text_Dataset, self).__init__()
        self.data_mmap=data_mmap
        self.tokenize_zh = tokenize
        self.datas_num=datas_num
        
    def __getitem__(self, index):
        sample_zh= self.data_mmap.get(index).tolist()[:75]

        mlm_labels=[self.tokenize_zh.encoder['<pad>']]
        for i in range(len(sample_zh)):
            prob=random.random()
            if prob<0.15:
                prob/=0.15
                mlm_labels.append(sample_zh[i])
                if prob<0.8:
                    sample_zh[i]=self.tokenize_zh.encoder['<mask>']
                else:
                    sample_zh[i]=random.randint(0,len(self.tokenize_zh.encoder)-1)
                
            else:
                mlm_labels.append(self.tokenize_zh.encoder['<pad>'])
        sample_zh=[self.tokenize_zh.encoder['<s>']]+sample_zh+[self.tokenize_zh.encoder['<eod>']]
        sample_zh=sample_zh+[self.tokenize_zh.encoder['<pad>']]*(77-len(sample_zh))
        sample_zh=torch.LongTensor(sample_zh)
        mlm_labels=mlm_labels+[self.tokenize_zh.encoder['<pad>']]*(77-len(mlm_labels))
        mlm_labels=torch.LongTensor(mlm_labels)
        
        return sample_zh,mlm_labels
    
    def __len__(self):
        return self.datas_num

def get_batch(dataloader):
    sample_zh,mlm_labels=dataloader.__iter__().__next__()
    return sample_zh,mlm_labels

def build_dataloader(args,text_data_mmap,feature_data_mmap,tokenizer,datas_num,indexes=None,scores=None,value=0.9):
    dataset = Img_Text_Dataset(text_data_mmap,feature_data_mmap,tokenizer,datas_num,\
                        local_rank=args.local_rank,indexes=indexes,scores=scores,value=0.9)
    sampler=DistributedSampler(dataset,shuffle=True)
    dataloader = DataLoaderX(dataset=dataset,batch_size=args.pairs_bs,local_rank=args.local_rank,
                             num_workers=args.num_workers,shuffle=False,sampler=sampler) 
    
    return dataset,dataloader

def build_mlm_dataloader(data_mmap,tokenize,args,datas_num):
    dataset=Text_Dataset(data_mmap,tokenize,args.local_rank,datas_num)
    dataloader = DataLoaderX(dataset=dataset,batch_size=args.text_bs,local_rank=args.local_rank,
                             num_workers=0,shuffle=True) 
    return dataloader



