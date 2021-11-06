import json
import clip
import torch.nn as nn
from transformer import Transformer
import torch
import argparse
from copy import deepcopy as c

def load_model(device,tokenizer):
    model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
    model.token_embedding = nn.Embedding(tokenizer.vocab_size,model.token_embedding.embedding_dim)
    model.transformer2=Transformer(dim=512,depth=20,seq_len=77,causal = False)
    model.eot_token = tokenizer.encoder['<eod>']
    model.output=nn.Linear(512,len(tokenizer.encoder))
    model = model.float()
    model=model.to(device)
    tmp_model=c(model)
    del tmp_model.visual
    tmp_model.eval()
    model.train()
    return model,tmp_model

def get_sentence_feature(sent, model, tokenizer):
    batchsize = 1000
    ret = []
    for start in tqdm(range(0, len(sent), batchsize)):
        input_ids = tokenizer.batch_encode_plus(
            sent[start:start+batchsize], max_length=77, truncation='only_first', padding='max_length', return_tensors='pt')['input_ids']
        input_ids = input_ids.to(device)
        _,text_features = model.encode_text(input_ids,boolean=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        ret.append(text_features.cpu())
    ret = torch.cat(ret, dim=0)
    assert len(sent)==ret.size()[0]
    return ret.float()


def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--pairs_bs',type=int,default=256,help='The batch size of image-text pairs')
    parser.add_argument('--text_bs',type=int,default=40,help='The batch size of text data')
    parser.add_argument('--num_workers',type=int,default=8,help='Number of workers')
    parser.add_argument('--num_datas',type=int,default=8,help='Number of image-text pairs')
    parser.add_argument('--text_num_datas',type=int,default=8,help='Number of documents')
    parser.add_argument('--feature_data_path',type=str,help='Path of image feature')
    parser.add_argument('--text_data_path',type=str,help='Path of text branch of image-text pairs')
    parser.add_argument('--mlm_data_path',type=str,default='',help='Path of documents')
    parser.add_argument('--pretrain_model_path',type=str,help='')
    parser.add_argument('--num_epochs',type=int,default=20,help='')
    parser.add_argument('--lr',type=float,default=0.0002,help='Learning rate')
    parser.add_argument('--deepspeed',type=bool,default=False,help='If using deepspeed')
    parser.add_argument('--deepspeed_config',type=str,default=False,help='Config of deepspeed if using deepspeed')
    parser.add_argument('--save_path',type=str,help='Path to save pretrain models')
    parser.add_argument('--val_img_path',type=str,help='Path of validation image feature')
    parser.add_argument('--val_text_path',type=str,help='Path of validation text')
    parser.add_argument('--queue_len',type=int,help='The length of memory queue')
    args = parser.parse_args()
    return args


def prepare_optimizer_parameters(args, model):
    deepspeed_config = json.load(
        open(args.deepspeed_config, 'r', encoding='utf-8'))
    weight_decay=0.0
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    if deepspeed_config["optimizer"]["type"] not in [
            "OneBitAdam", "OneBitLamb"
    ]:
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            weight_decay
        }, {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]
    else:
        # Because 1-bit compression cannot represent exact zero, it is required to
        # provide a momentum mask for those params that have constant exact zeros in their
        # momentums, otherwise the compression error would keep accumulating.
        # For example, for bert pre-training seq 128, bert.embeddings.position_embeddings.weight
        # always have exact zeros in its momentum for row 129 to 512, because it only
        # learns up to seq length 128 while the model supports up to 512 seq length.
        need_mask = ['position_embeddings.weight']
        need_mask_p = []
        need_mask_decay = []
        masks = []
        for n, p in param_optimizer:
            if any(nd in n for nd in need_mask):
                mask = torch.zeros_like(p.data)
                for position in range(max_len):
                    for col in range(p.size()[1]):
                        mask[position][col] += 1
                if deepspeed_config["optimizer"]["type"] == "OneBitAdam":
                    mask = torch.flatten(mask)
                masks.append(mask)
                need_mask_p.append(p)
                if any(nd in n for nd in no_decay):
                    need_mask_decay.append(0.0)
                else:
                    need_mask_decay.append(weight_decay)

        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay + need_mask)
            ],
            'weight_decay':
            weight_decay
        }, {
            'params': [
                p for n, p in param_optimizer
                if (any(nd in n
                        for nd in no_decay) and not any(nd in n
                                                        for nd in need_mask))
            ],
            'weight_decay':
            0.0
        }]

        for i_mask in range(len(need_mask_p)):
            optimizer_grouped_parameters.append({
                'params': [need_mask_p[i_mask]],
                'weight_decay':
                need_mask_decay[i_mask],
                'exp_avg_mask':
                masks[i_mask]
            })

    return optimizer_grouped_parameters