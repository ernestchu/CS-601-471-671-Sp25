import pandas as pd
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModel


class CustomModelforSequenceClassification(nn.Module):

    def __init__(self, model_name, num_labels=2, type="full", prefix_length=128):
        super(CustomModelforSequenceClassification, self).__init__()
        self.prefix_length = prefix_length
        self.model = AutoModel.from_pretrained(model_name)
        self.type = type
        self.num_labels = num_labels
        self.prefix = torch.nn.Parameter(torch.randn(prefix_length, self.model.config.hidden_size, requires_grad=True).to('cuda'))
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        if self.type == "full":
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = output.last_hidden_state
            mean = torch.mean(last_hidden_state, dim=1)
            logits = self.classifier(mean)
        elif self.type == "head":
            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = output.last_hidden_state
                mean = torch.mean(last_hidden_state, dim=1)
            logits = self.classifier(mean)
        elif self.type == 'prefix':
            BS = input_ids.shape[0]
            prefix = self.prefix.expand(BS, -1, -1)
            inputs_embeds = self.model.embeddings(input_ids)
            inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1).to('cuda')
            attention_mask = torch.cat([torch.ones(BS, self.prefix_length).to('cuda'), attention_mask], dim=1)
            output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            last_hidden_state = output.last_hidden_state
            mean = torch.mean(last_hidden_state, dim=1)
            logits = self.classifier(mean)
        return {"logits": logits}

with open('profile.txt', 'w') as f:
    for type in ['full']:
    # for type in ['full', 'head', 'prefix']:
        mymodel = CustomModelforSequenceClassification('RoBERTa-base', num_labels=2, type=type)
        mymodel.to('cuda')
        mymodel.train()

        # here, we use the AdamW optimizer. Use torch.optim.AdamW
        lr = 1e-3
        if mymodel.type == "full":
            optimizer = torch.optim.SGD(mymodel.parameters(), lr=lr)
        if mymodel.type == "head":
            optimizer = torch.optim.SGD(mymodel.classifier.parameters(), lr=lr)
        elif mymodel.type == "prefix":
            prefix_params = mymodel.prefix
            classifier_params = mymodel.classifier.parameters()
            optimizer = torch.optim.SGD([prefix_params] + list(classifier_params), lr=lr)
        
        # dummy input
        input_ids = torch.randint(0, 100, (1, 128)).to('cuda')
        attention_mask = torch.randint(0, 2, (1, 128)).to('cuda')
        labels = torch.randint(0, 2, (1,)).to('cuda')


        with profile(profile_memory=True) as prof:
            with record_function('forward'):
                output = mymodel(input_ids, attention_mask)
                loss = torch.nn.functional.cross_entropy(output['logits'], labels)
        print(f'{type}, forward', file=f)
        # print([f for f in prof.key_averages() if f.key == 'forward'][0].cuda_memory_usage, file=f)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1), file=f)

        with profile(profile_memory=True) as prof:
            with record_function("backward"):
                loss.backward()
                optimizer.step()
        print(f'{type}, backward', file=f)
        # print([f for f in prof.key_averages() if f.key == 'backward'][0].cuda_memory_usage, file=f)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1), file=f)

