
from torch import nn
from sklearn.metrics import f1_score
from transformers.optimization import AdamW, get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from tqdm import tqdm
import torch.distributed as dist
from dataset import *
from data_utils.tokenization_gpt2 import GPT2Tokenizer
from torch.cuda.amp import autocast,GradScaler
from tensorboardX import SummaryWriter
from copy import deepcopy as c
from megatron.data.gpt_dataset import *
from utils import get_sentence_feature, load_model, get_args, prepare_optimizer_parameters
import deepspeed

def train(args):
    local_rank=args.local_rank
    
    # Initialize the distribution
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:'+str(local_rank))
    
    text_data_path=args.text_data_path
    feature_data_path=args.feature_data_path
    mlm_data_path=args.mlm_data_path
    datas_num=args.num_datas
    
    # Build tokenizer
    tokenizer_path='bpe_3w_new/'
    tokenizer = GPT2Tokenizer(os.path.join(tokenizer_path, 'vocab.json'), os.path.join(tokenizer_path, 'chinese_vocab.model'))
    
    if local_rank==0:
        print('start process datas...')
    
    # Build data loader
    text_data_mmap=get_indexed_dataset_(text_data_path,'mmap',False)
    feature_data_mmap=np.memmap(feature_data_path,order='C',mode='r')
    mlm_data_mmap=get_indexed_dataset_(mlm_data_path,'mmap',False)
    mlm_dataloader=build_mlm_dataloader(mlm_data_mmap,tokenizer,args,args.text_num_datas)
    dataset,dataloader=build_dataloader(args,text_data_mmap,feature_data_mmap,tokenizer,datas_num)
    
    # Get the training and scoring model
    model,tmp_model=load_model(device,tokenizer)
    model.forward=model.forward_mlm
    
    img_queue=torch.Tensor([])
    num_epochs = args.num_epochs
    
    # If using the deepspeed, initialize the deepspeed model
    if args.deepspeed:
        params=prepare_optimizer_parameters(args,model)
        model,optim,_,_=deepspeed.initialize(args=args,
                                                model=model,
                                                model_parameters=params)
        
    # Use the torch DDP
    else:
        model = nn.parallel.DistributedDataParallel(model,broadcast_buffers=False,device_ids=[local_rank],find_unused_parameters=True)
        scaler = GradScaler()
        optim = AdamW(model.parameters(),lr=args.lr, eps=1e-8, betas=(0.9, 0.95))
        scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=0.1 * len(dataloader), num_training_steps=len(dataloader)*num_epochs)
        
    # Build the writer
    writer=SummaryWriter(log_dir='scalar/')
    steps=0
    max_norm=1.0
    mlm_inputs,mlm_labels=None,None
    for epoch in range(num_epochs):
        for sample_zh, img_features,indexes in tqdm(dataloader):

            b=sample_zh.shape[0]
            sample_zh = sample_zh.to(device)
            
            mlm_inputs,mlm_labels=get_batch(mlm_dataloader)
            text=(sample_zh,mlm_inputs.to(device),mlm_labels.to(device),)
            img_features = img_features.to(device)
            
            if not args.deepspeed:        
                optim.zero_grad()
            
            if args.deepspeed: 
                with torch.no_grad():
                    # Compute the score of the data of mini-batch
                    _,text_features=tmp_model.encode_text(sample_zh,boolean=True)
                    scores=torch.einsum('ab,ab->a',text_features/text_features.norm(dim=-1,keepdim=True),\
                                        (img_features/img_features.norm(dim=-1,keepdim=True)).float()).cpu()
                    
                    # Record the scores computed by scoring model
                    dataset.scores[indexes]=dataset.scores[indexes]+scores
                    
                    
                # Forward, get the loss and update the queue
                loss,img_queue=model(text=text,image=img_features,img_queue=img_queue,queue_len=args.queue_len,
                                     steps=steps,pad_id=tokenizer.encoder['<pad>'])
            else:
                # Use the fixed precision
                with autocast():
                    with torch.no_grad():
                        _,text_features=tmp_model.encode_text(sample_zh,boolean=True)
                        scores=torch.einsum('ab,ab->a',text_features/text_features.norm(dim=-1,keepdim=True)\
                                            ,img_features/img_features.norm(dim=-1,keepdim=True)).cpu()
                        dataset.scores[indexes]=dataset.scores[indexes]+scores
                    loss,img_queue=model(text=text,image=img_features,img_queue=img_queue,queue_len=args.queue_len,
                                     steps=steps,pad_id=tokenizer.encoder['<pad>'])

            if local_rank ==0:
                print(loss,img_queue.shape)
                # Write the loss
                writer.add_scalar('scalar/loss', loss, steps)
                
            if dist.get_rank()==0 and steps%100==0:
           #    Write the validation score
                writer.add_scalar('scalar/f1_score', model.module.f1_score,steps)
                
            if args.deepspeed:
                model.backward(loss)
                model.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optim)
                scaler.update()
                scheduler.step()
            steps+=1
            
            # Save the model
            if dist.get_rank()==0 and steps%1000==0:
                model.eval()
                val_score=evaluate(args,model,tokenizer)
                writer.add_scalar('scalar/val_score', val_score,steps)
                torch.save(model.module.state_dict(),args.save_path+'_'+str(epoch)+'.pth')
                model.train()
        # Update the dataset and scoring model
        
        scores,next_indexes=torch.topk(dataset.scores[dataset.indexes],k=dataset.filtered_num,sorted=True,largest=True)
        next_indexes=dataset.indexes[next_indexes]
        dataset,dataloader=build_dataloader(args,text_data_mmap,feature_data_mmap,tokenizer,\
                                            len(next_indexes),indexes=next_indexes,scores=dataset.scores)
        tmp_model=c(model.module)
        del tmp_model.visual
        tmp_model.eval()

    dist.destroy_process_group()

def evaluate(args,model,tokenizer):
    image_features=torch.load(args.val_img_path)
    texts=open(args.val_text_path)
    text_features=get_sentence_feature(texts,model,tokenizer)
    
    labels = torch.arange(image_features.shape[0])
    scores = torch.einsum('ab,cb->ac', text_features, image_features)
    
    pred = scores.argmax(dim=-1)
    f1 = f1_score(labels.cpu(), pred.cpu(), average='micro')
    
    return f1
    
def main():
    args=get_args()
    train(args)


if __name__ == "__main__":
    
    main()
