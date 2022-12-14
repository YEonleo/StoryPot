import torch
from torch import nn

from transformers import AutoModel, AutoModelForSequenceClassification, T5ForConditionalGeneration,AutoTokenizer,BartForConditionalGeneration,AutoModelForSeq2SeqLM

class ACD_model(nn.Module):
    def __init__(self,args, num_label,len_tokenizer):
        super(ACD_model, self).__init__()
        self.num_label = num_label
        self.model_PLM = AutoModelForSeq2SeqLM.from_pretrained('paust/pko-t5-base')
        #self.model_PLM.resize_token_embeddings(len_tokenizer)
        self.lf = nn.CrossEntropyLoss()


    def forward(self, e_input_ids, e_attention_mask, d_input_ids, d_attention_mask,labels):
        outputs = self.model_PLM(
            input_ids=e_input_ids,
            attention_mask=e_attention_mask,
            decoder_input_ids=d_input_ids,
            decoder_attention_mask=d_attention_mask
        )
        
        logits = outputs.logits
        
        label = labels[:,1:]
        
        pred = logits[:,:-1,:]
        vocab_num = pred.size()[-1]
        loss = self.lf(pred.reshape(-1,vocab_num),label.reshape(-1))
        
        return loss, logits
    
    def test(self, e_input_ids, e_attention_mask):
        
        outputs = self.model_PLM.generate(
            input_ids=e_input_ids,
            attention_mask=e_attention_mask,
            num_beams=5, max_length=256, early_stopping=True)
        
        return outputs
    
