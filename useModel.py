import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

# CPU 사용
device = torch.device("cpu")
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

class BERTClassifier_2(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 validation_split=0.1,
                 params=None):
        super(BERTClassifier_2, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=5,
                 dr_rate=None,
                 validation_split=0.1,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


## 모델 불러오기
# Setting max len
max_len = 50

class useDataset_2(Dataset):
    def __init__(self, line, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([line])]
        self.labels = [np.int32(0)]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

class useDataset_5(Dataset):
    def __init__(self, line, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([line])]
        self.labels = [np.int32(0)]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


# model use
def useModel_2(line):
    model_emo2 = './model/model_emo2.pt'
    model_2 = BERTClassifier_2(bertmodel, dr_rate=0.5)
    model_2.load_state_dict(torch.load(model_emo2, map_location=device))
    model_2.eval()
    model_2 = model_2.to(device)

    emotion_label = ['긍정', '부정']
    tmp = useDataset_2(line, 0, 1, tok, max_len, True, False)
    tmpLoader = DataLoader(tmp, batch_size=1)

    tmpIter = iter(tmpLoader)
    (token_ids, valid_length, segment_ids, label) = tmpIter.next()

    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length = valid_length
    label = label.long().to(device)

    out = model_2(token_ids, valid_length, segment_ids)
    logits = out[0]
    logits = logits.detach().cpu().numpy()
    logits = np.argmax(logits)

    return emotion_label[logits]

def useModel_5(line):
    # 부정 5가지
    model5_path = './model/model_emo5.pt'
    model5 = BERTClassifier(bertmodel, dr_rate=0.5)
    model5.load_state_dict(torch.load(model5_path, map_location=device))
    model5.eval()
    model_5 = model5.to(device)
    emotion_label = ["공포", "놀람", "분노", "슬픔", "혐오"]
    tmp = useDataset_5(line, 0, 1, tok, max_len, True, False)
    tmpLoader = DataLoader(tmp, batch_size=1)

    tmpIter = iter(tmpLoader)
    (token_ids, valid_length, segment_ids, label) = tmpIter.next()

    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length = valid_length
    label = label.long().to(device)

    out = model_5(token_ids, valid_length, segment_ids)
    logits = out[0]
    logits = logits.detach().cpu().numpy()

    print(line)
    print(logits)
    print(sum(logits))

    print("===========================================")
    print("label :", np.argmax(logits))
    print("emotion : ", emotion_label[np.argmax(logits)])

    return emotion_label[np.argmax(logits)]

def predict_emo(line):
    emo2_Res = useModel_2(line)
    if emo2_Res.startswith("부정"):
        emo5_Res = useModel_5(line)
        print(emo5_Res)
        return [emo2_Res, emo5_Res]
    else:
        return [emo2_Res]

