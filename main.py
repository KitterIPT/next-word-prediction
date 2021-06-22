# %%
import torch
import string

# Original index
## Base BERT
from transformers import BertTokenizer, BertForMaskedLM
## Base XLNet
from transformers import XLNetTokenizer, XLNetLMHeadModel
xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
xlnet_model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased').eval()
## Base XLMRoberta
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
xlmroberta_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
xlmroberta_model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base').eval()
## Base Bart
from transformers import BartTokenizer, BartForConditionalGeneration
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large').eval()
## Base Eletra
from transformers import ElectraTokenizer, ElectraForMaskedLM
electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
electra_model = ElectraForMaskedLM.from_pretrained('google/electra-small-generator').eval()
## Base Roberta
from transformers import RobertaTokenizer, RobertaForMaskedLM
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForMaskedLM.from_pretrained('roberta-base').eval()

## Bert VNese base uncased
from transformers import AutoTokenizer, AutoModelForMaskedLM
phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
phobert_model = AutoModelForMaskedLM.from_pretrained("vinai/phobert-base")
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
# phobert_model = AutoModelForMaskedLM.from_pretrained("vinai/phobert-base")

# from transformers import BertModel # BertTokenizer, BertModel, BertForMaskedLM
# vi_tokenizer = BertTokenizer.from_pretrained('trituenhantaoio/bert-base-vietnamese-uncased')
# vi_model = BertModel.from_pretrained('trituenhantaoio/bert-base-vietnamese-uncased').eval()
## Bert VNese base diacritics
# vi2_tokenizer = BertTokenizer.from_pretrained('trituenhantaoio/bert-base-vietnamese-diacritics-uncased')
# vi2_model = BertModel.from_pretrained('trituenhantaoio/bert-base-vietnamese-diacritics-uncased').eval()
## Bert VNese Pho :)
# from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
# vi3_tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
# vi3_model = AutoModelForMaskedLM.from_pretrained('vinai/phobert-base').eval()

from transformers import PhobertTokenizer, RobertaModel
bert_tokenizer = PhobertTokenizer.from_pretrained('vinai/phobert-base')
bert_model = RobertaModel.from_pretrained('vinai/phobert-base').eval()

top_k = 10


def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


def get_all_predictions(text_sentence, top_clean=5):
    # ========================= BERT =================================
    print(text_sentence)
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bert_model(input_ids)[0]
    bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= XLNET LARGE =================================
    input_ids, mask_idx = encode(xlnet_tokenizer, text_sentence, False)
    perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
    perm_mask[:, :, mask_idx] = 1.0  # Previous tokens don't see last token
    target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
    target_mapping[0, 0, mask_idx] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

    with torch.no_grad():
        predict = xlnet_model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)[0]
    xlnet = decode(xlnet_tokenizer, predict[0, 0, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= XLM ROBERTA BASE =================================
    input_ids, mask_idx = encode(xlmroberta_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = xlmroberta_model(input_ids)[0]
    xlm = decode(xlmroberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= BART =================================
    input_ids, mask_idx = encode(bart_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = bart_model(input_ids)[0]
    bart = decode(bart_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= ELECTRA =================================
    input_ids, mask_idx = encode(electra_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = electra_model(input_ids)[0]
    electra = decode(electra_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # ========================= ROBERTA =================================
    input_ids, mask_idx = encode(roberta_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = roberta_model(input_ids)[0]
    roberta = decode(roberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

     # ========================= PHOBERT =================================
    input_ids, mask_idx = encode(phobert_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = phobert_model(input_ids)[0]
    phobert = decode(phobert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
            
    # ========================= VI_BERT =================================
    # input_ids, mask_idx = encode(vi_tokenizer, text_sentence)
    # with torch.no_grad():
    #     predict = vi_model(input_ids)[0]
    # vi_bert = decode(vi_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # # ========================= VI2_BERT ================================
    # input_ids, mask_idx = encode(vi2_tokenizer, text_sentence)
    # with torch.no_grad():
    #     predict = vi2_model(input_ids)[0]
    # vi2_bert = decode(vi2_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # # ========================= VI3 =====================================
    # input_ids, mask_idx = encode(vi2_tokenizer, text_sentence)
    # with torch.no_grad():
    #     predict = vi3_model(input_ids)[0]
    # vi3_bert = decode(vi3_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    return {'phobert': phobert,
            'bert': bert,
            'xlnet': xlnet,
            'xlm': xlm,
            'bart': bart,
            'electra': electra,
            'roberta': roberta
            # 'vi_bert': vi_bert,
            # 'vi2_bert': vi2_bert,
            # 'vi3_bert': vi3_bert
            }
