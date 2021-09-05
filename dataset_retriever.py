
import torch

from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)


class DatasetRetriever(Dataset):
    def __init__(self, data, tokenizer, max_len, is_test=False):
        self.data = data
        if 'excerpt' in self.data.columns:
            self.excerpts = self.data.excerpt.values.tolist()
        else:
            self.excerpts = self.data.text.values.tolist()
        self.targets = self.data.target.values.tolist()
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        excerpt, label = self.excerpts[item], self.targets[item]
        features = convert_examples_to_features(
            excerpt, self.tokenizer,
            self.max_len, self.is_test
        )
        return {
            'input_ids' :torch.tensor(features['input_ids'], dtype=torch.long),
            'token_type_ids' :torch.tensor(features['token_type_ids'], dtype=torch.long),
            'attention_mask' :torch.tensor(features['attention_mask'], dtype=torch.long),
            'label' :torch.tensor(label, dtype=torch.double),
        }

def convert_examples_to_features(data, tokenizer, max_len, is_test=False):
    data = data.replace('\n', '')
    tok = tokenizer.encode_plus(
        data,
        max_length=max_len,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True
    )
    curr_sent = {}
    padding_length = max_len - len(tok['input_ids'])
    curr_sent['input_ids'] = tok['input_ids'] + ([tokenizer.pad_token_id] * padding_length)
    curr_sent['token_type_ids'] = tok['token_type_ids'] + \
        ([0] * padding_length)
    curr_sent['attention_mask'] = tok['attention_mask'] + \
        ([0] * padding_length)
    return curr_sent


def make_loader(
        data,
        tokenizer,
        max_len,
        batch_size,
):
    test_dataset = DatasetRetriever(data, tokenizer, max_len, is_test=True)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size // 2,
        sampler=test_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=0
    )

    return test_loader
