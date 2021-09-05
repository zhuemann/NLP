
import torch


class Evaluator:

    def __init__(self, model, scalar=None):
        self.model = model
        self.scalar = scalar

    def evaluate(self, data_loader, tokenizer):
        # holds all the predictions
        preds = []
        # puts the model in evaluation mode
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                # gets the information from the data loader and makes them cuda objects for gpu use
                input_ids, attention_mask, token_type_ids = batch_data['input_ids'], \
                    batch_data['attention_mask'], batch_data['token_type_ids']
                input_ids, attention_mask, token_type_ids = input_ids.cuda(), \
                    attention_mask.cuda(), token_type_ids.cuda()

                outputs = self.model(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    token_type_ids = token_type_ids
                    )

                logits = outputs[0].detach().cpu().numpy().squeeze().tolist()
                preds += logits
            return preds
