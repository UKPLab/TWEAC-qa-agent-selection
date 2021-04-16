import torch
from torch import nn

class TransformerModel(nn.Module):
    def __init__(self, config, bert):
        super(TransformerModel, self).__init__()
        self.config = config
        self.model_config = config["model"]
        self.agents_extended = config.get("agents_extended", 0)
        self.num_labels = len(config["all_agents"])-self.agents_extended
        self.bert = bert
        self.dropout = nn.Dropout(self.model_config.get("dropout", 0.1))
        self.classifier = nn.Conv1d(1, self.num_labels, bert.config.hidden_size)
        if self.agents_extended > 0:
            self.extend_classifier = nn.Conv1d(1, self.agents_extended, bert.config.hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, attention_mask=None, labels=None, pos_weight=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)

        pooled_output = bert_outputs[0][:,0]
        pooled_output = self.dropout(pooled_output) # shape (batch_size, class_size)
        pooled_output = torch.unsqueeze(pooled_output, 1)  # shape (batch_size, 1, hidden_size) for convolution
        logits = self.classifier(pooled_output).squeeze(dim=2)  # shape (batch_size, num_labels)
        if self.agents_extended > 0:
            ext_logits = self.extend_classifier(pooled_output).squeeze(dim=2)
            logits = torch.cat((logits, ext_logits), dim=1)
        outputs = (self.sigmoid(logits),)

        if labels is not None:
            # against class imbalances
            if pos_weight is None:
                pos_weight = torch.ones(logits.size()[1]).float()
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
            loss = loss_fct(logits, labels)
            outputs = outputs + (loss,)

        return outputs  # sigmoid(logits), (loss)


class TransformerModelV2(nn.Module):
    def __init__(self, config, bert):
        super(TransformerModelV2, self).__init__()
        self.config = config
        self.model_config = config["model"]
        self.agents_extended = config.get("agents_extended", 0)
        self.num_labels = len(config["all_agents"])-self.agents_extended
        self.bert = bert
        self.dropout = nn.Dropout(self.model_config.get("dropout", 0.1))
        class_dim = self.model_config.get("classification_dim", 756)
        self.adapter = nn.Conv1d(1, self.num_labels*class_dim, bert.config.hidden_size)
        self.classifier = nn.Conv1d(self.num_labels*class_dim, self.num_labels, 1, groups=self.num_labels)
        if self.agents_extended > 0:
            self.extend_adapter = nn.Conv1d(1, self.agents_extended*class_dim, bert.config.hidden_size)
            self.extend_classifier = nn.Conv1d(self.agents_extended*class_dim, self.agents_extended, 1, groups=self.agents_extended)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids=None, attention_mask=None, labels=None, pos_weight=None, reduction="mean"):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)

        pooled_output = bert_outputs[0][:,0]
        pooled_output = self.dropout(pooled_output) # shape (batch_size, hidden_size)
        if self.agents_extended > 0:
            ext_output = nn.GELU()(self.extend_adapter(pooled_output.unsqueeze(1)))
            ext_output = self.dropout(ext_output)
        pooled_output = nn.GELU()(self.adapter(pooled_output.unsqueeze(1)))
        pooled_output = self.dropout(pooled_output) # shape (batch_size, class_size)
        logits = self.classifier(pooled_output).squeeze(dim=2)  # shape (batch_size, num_labels)
        if self.agents_extended > 0:
            ext_logits = self.extend_classifier(ext_output).squeeze(dim=2)
            logits = torch.cat((logits, ext_logits), dim=1)
        outputs = (self.sigmoid(logits),)

        if labels is not None:
            # against class imbalances
            if pos_weight is None:
                pos_weight = torch.ones(logits.size()[1]).float()
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
            loss = loss_fct(logits, labels)
            outputs = outputs + (loss,)

        return outputs  # sigmoid(logits), (loss)


class TransformerModelSoftmax(nn.Module):
    def __init__(self, config, bert):
        super(TransformerModelSoftmax, self).__init__()
        self.config = config
        self.model_config = config["model"]
        self.agents_extended = config.get("agents_extended", 0)
        self.num_labels = len(config["all_agents"])-self.agents_extended
        self.bert = bert
        self.dropout = nn.Dropout(self.model_config.get("dropout", 0.1))
        class_dim = self.model_config.get("classification_dim", 756)
        self.adapter = nn.Conv1d(1, self.num_labels*class_dim, bert.config.hidden_size)
        self.classifier = nn.Conv1d(self.num_labels*class_dim, self.num_labels, 1, groups=self.num_labels)
        if self.agents_extended > 0:
            self.extend_adapter = nn.Conv1d(1, self.agents_extended*class_dim, bert.config.hidden_size)
            self.extend_classifier = nn.Conv1d(self.agents_extended*class_dim, self.agents_extended, 1, groups=self.agents_extended)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids=None, attention_mask=None, labels=None, pos_weight=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)

        pooled_output = bert_outputs[0][:,0]
        pooled_output = self.dropout(pooled_output) # shape (batch_size, hidden_size)
        if self.agents_extended > 0:
            ext_output = nn.GELU()(self.extend_adapter(pooled_output.unsqueeze(1)))
            ext_output = self.dropout(ext_output)
        pooled_output = nn.GELU()(self.adapter(pooled_output.unsqueeze(1)))
        pooled_output = self.dropout(pooled_output) # shape (batch_size, class_size)
        logits = self.classifier(pooled_output).squeeze(dim=2)  # shape (batch_size, num_labels)
        if self.agents_extended > 0:
            ext_logits = self.extend_classifier(ext_output).squeeze(dim=2)
            logits = torch.cat((logits, ext_logits), dim=1)
        outputs = (logits,)

        if labels is not None:
            # against class imbalances
            if pos_weight is None:
                pos_weight = torch.ones(logits.size()[1]).float()
            loss_fct = nn.CrossEntropyLoss(weight=pos_weight, reduction="mean")
            loss = loss_fct(logits, labels)
            outputs = outputs + (loss,)

        return outputs  # sigmoid(logits), (loss)


class TransformerModelMSE(nn.Module):
    def __init__(self, config, bert):
        super(TransformerModelMSE, self).__init__()
        self.config = config
        self.model_config = config["model"]
        self.agents_extended = config.get("agents_extended", 0)
        self.num_labels = len(config["all_agents"])-self.agents_extended
        self.bert = bert
        self.dropout = nn.Dropout(self.model_config.get("dropout", 0.1))
        class_dim = self.model_config.get("classification_dim", 756)
        self.adapter = nn.Conv1d(1, self.num_labels*class_dim, bert.config.hidden_size)
        self.classifier = nn.Conv1d(self.num_labels*class_dim, self.num_labels, 1, groups=self.num_labels)
        if self.agents_extended > 0:
            self.extend_adapter = nn.Conv1d(1, self.agents_extended*class_dim, bert.config.hidden_size)
            self.extend_classifier = nn.Conv1d(self.agents_extended*class_dim, self.agents_extended, 1, groups=self.agents_extended)

    def forward(self, input_ids=None, attention_mask=None, labels=None, pos_weight=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)

        pooled_output = bert_outputs[0][:,0]
        pooled_output = self.dropout(pooled_output) # shape (batch_size, hidden_size)
        if self.agents_extended > 0:
            ext_output = nn.GELU()(self.extend_adapter(pooled_output.unsqueeze(1)))
            ext_output = self.dropout(ext_output)
        pooled_output = nn.GELU()(self.adapter(pooled_output.unsqueeze(1)))
        pooled_output = self.dropout(pooled_output) # shape (batch_size, class_size)
        logits = self.classifier(pooled_output).squeeze(dim=2)  # shape (batch_size, num_labels)
        if self.agents_extended > 0:
            ext_logits = self.extend_classifier(ext_output).squeeze(dim=2)
            logits = torch.cat((logits, ext_logits), dim=1)
        outputs = (logits,)

        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)
            outputs = outputs + (loss,)

        return outputs  # sigmoid(logits), (loss)

class TransformerModelPairwise(nn.Module):
    def __init__(self, config, bert):
        super(TransformerModelPairwise, self).__init__()
        self.config = config
        self.model_config = config["model"]
        self.agents_extended = config.get("agents_extended", 0)
        self.num_labels = len(config["all_agents"])-self.agents_extended
        self.bert = bert
        self.dropout = nn.Dropout(self.model_config.get("dropout", 0.1))
        class_dim = self.model_config.get("classification_dim", 756)
        self.adapter = nn.Conv1d(1, self.num_labels*class_dim, bert.config.hidden_size)
        self.classifier = nn.Conv1d(self.num_labels*class_dim, self.num_labels, 1, groups=self.num_labels)
        if self.agents_extended > 0:
            self.extend_adapter = nn.Conv1d(1, self.agents_extended*class_dim, bert.config.hidden_size)
            self.extend_classifier = nn.Conv1d(self.agents_extended*class_dim, self.agents_extended, 1, groups=self.agents_extended)

    def forward(self, input_ids=None, attention_mask=None, labels=None, pos_weight=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)

        pooled_output = bert_outputs[0][:,0]
        pooled_output = self.dropout(pooled_output) # shape (batch_size, hidden_size)
        if self.agents_extended > 0:
            ext_output = nn.GELU()(self.extend_adapter(pooled_output.unsqueeze(1)))
            ext_output = self.dropout(ext_output)
        pooled_output = nn.GELU()(self.adapter(pooled_output.unsqueeze(1)))
        pooled_output = self.dropout(pooled_output) # shape (batch_size, class_size)
        logits = self.classifier(pooled_output).squeeze(dim=2)  # shape (batch_size, num_labels)
        if self.agents_extended > 0:
            ext_logits = self.extend_classifier(ext_output).squeeze(dim=2)
            logits = torch.cat((logits, ext_logits), dim=1)
        outputs = (logits,)

        if labels is not None:
            loss_fct = nn.MultiLabelMarginLoss()
            loss = loss_fct(logits, labels)
            outputs = outputs + (loss,)

        return outputs  # sigmoid(logits), (loss)


class TransformerModelV3(nn.Module):
    def __init__(self, config, bert):
        super(TransformerModelV3, self).__init__()
        self.config = config
        self.model_config = config["model"]
        self.agents_extended = config.get("agents_extended", 0)
        self.num_labels = len(config["all_agents"])-self.agents_extended
        self.bert = bert
        self.dropout = nn.Dropout(self.model_config.get("dropout", 0.1))
        class_dim = self.model_config.get("classification_dim", 756)
        self.preclass1 = nn.Linear(bert.config.hidden_size, class_dim//2)
        self.preclass2 = nn.Linear(class_dim//2, class_dim)
        self.embedding = nn.Parameter(torch.FloatTensor(self.num_labels, class_dim).uniform_(-1, 1))
        if self.agents_extended > 0:
            self.extend_embedding = nn.Parameter(torch.FloatTensor(self.agents_extended, class_dim).uniform_(-1, 1))
        self.cosine = nn.CosineSimilarity(dim=2)

    def forward(self, input_ids=None, attention_mask=None, labels=None, pos_weight=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)

        pooled_output = bert_outputs[0][:,0]
        pooled_output = self.dropout(pooled_output) # shape (batch_size, hidden_size)
        pooled_output = nn.GELU()(self.preclass1(pooled_output))
        pooled_output = self.dropout(pooled_output) # shape (batch_size, hidden_size)
        pooled_output = nn.GELU()(self.preclass2(pooled_output))
        cosine = self.cosine(self.embedding.unsqueeze(dim=0).repeat(pooled_output.size()[0], 1, 1), pooled_output.unsqueeze(1).repeat(1, self.num_labels, 1))
        if self.agents_extended > 0:
            ext_cosine = self.cosine(self.extend_embedding.unsqueeze(dim=0).repeat(pooled_output.size()[0], 1, 1), pooled_output.unsqueeze(1).repeat(1, self.agents_extended, 1))
            cosine = torch.cat((cosine, ext_cosine), dim=1)
        outputs = (cosine,)

        if labels is not None:
            # against class imbalances
            if pos_weight is None:
                pos_weight = torch.ones(cosine.size()[1]).float()
            pos_weight = torch.clamp(pos_weight.repeat(cosine.size()[0], 1) * labels, 1, 1000)
            loss_fct = nn.HingeEmbeddingLoss(reduction="none")
            cos_dist = 1-cosine
            labels = labels*2 - 1 # transform to -1, 1 labels
            hinges = torch.cat([loss_fct(cos_dist[:, i], labels[:, i]) for i in range(cos_dist.size()[1])]).reshape(cos_dist.size()[0], -1)
            loss = torch.mean(pos_weight*hinges)
            outputs = outputs + (loss,)

        return outputs  # sigmoid(logits), (loss)



class TransformerModelPretrainQC(nn.Module):
    def __init__(self, config, bert):
        super(TransformerModelPretrainQC, self).__init__()
        self.config = config
        self.model_config = config["model"]
        self.num_labels = len(config["all_agents"])
        self.bert = bert
        self.dropout = nn.Dropout(self.model_config.get("dropout", 0.1))
        class_dim = self.model_config.get("classification_dim", 756)
        self.preclass = nn.Linear(bert.config.hidden_size, class_dim)
        self.classifier = nn.Linear(class_dim, self.num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids=None, attention_mask=None, labels=None, weights=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)

        pooled_output = bert_outputs[0][:,0]
        pooled_output = self.dropout(pooled_output) # shape (batch_size, hidden_size)
        pooled_output = nn.Tanh()(self.preclass(pooled_output))
        pooled_output = self.dropout(pooled_output) # shape (batch_size, class_size)
        logits = self.classifier(pooled_output)  # shape (batch_size, num_labels)
        outputs = (self.softmax(logits),)

        if labels is not None:
            # against class imbalances
            if weights is None:
                weights = torch.ones(logits.size()[1]).float()
            loss_fct = nn.CrossEntropyLoss(weight=weights, reduction="mean")
            loss = loss_fct(logits, labels)
            outputs = outputs + (loss,)

        return outputs  # sigmoid(logits), (loss)
