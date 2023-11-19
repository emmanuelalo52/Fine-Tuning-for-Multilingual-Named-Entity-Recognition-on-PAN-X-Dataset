# XLM-Roberta Fine-Tuning for Multilingual Named Entity Recognition on PAN-X Dataset

## Introduction

Named Entity Recognition (NER) is a vital natural language processing task involving the identification and classification of entities (such as persons, organizations, and locations) within text. This repository provides a comprehensive guide and code for fine-tuning the XLM-Roberta model on the PAN-X dataset, a multilingual cross-domain NER dataset.

## XLM-Roberta Overview

XLM-Roberta is a multilingual extension of the RoBERTa (Robustly optimized BERT approach) model. It employs a transformer-based architecture, utilizing self-attention mechanisms to process input data in parallel. The model is pre-trained on a massive amount of multilingual text data, making it effective for various natural language understanding tasks.

## Setup

Ensure the necessary Python libraries are installed by running:

```bash
pip install transformers datasets seqeval torch matplotlib
```

## Data Loading

The PAN-X dataset is imported using the Hugging Face `datasets` library. The script explores available configurations and subsets related to PAN-X.

```python
from datasets import get_dataset_config_names
xtreme_subset = get_dataset_config_names("xtreme")
print(f"xtreme has {len(xtreme_subset)} configurations")
panx_dataset = [s for s in xtreme_subset if s.startswith("PAN")]
panx_dataset[:3]
```

## Tokenization

Tokenization is a crucial step in NER tasks. The code demonstrates tokenization using both BERT and XLM-Roberta tokenizers. SentencePiece is used for special tokenization.

```python
from transformers import AutoTokenizer

bert_model = "bert-base-cased"
xlmr_model_name = "xlm-roberta-base"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)

text = "it is hell in here"
bert_tokens = bert_tokenizer(text).tokens()
xlmr_token = xlmr_tokenizer(text).tokens()
```

## XLM-Roberta for Token Classification

The script provides a custom XLM-Roberta token classification model for NER tasks. It includes the model's architecture, tokenization, and post-processing steps.

```python
from transformers import XLMRobertaConfig, AutoConfig

class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
    # ... (model architecture details)
```

## Training

Fine-tuning XLM-Roberta for NER on the PAN-X dataset is accomplished using the `Trainer` class from the `transformers` library.

```python
from transformers import Trainer, TrainingArguments

trainer = Trainer(model_init=model_init, args=training_args,
                  data_collator=data_collator, compute_metrics=compute_metrics,
                  train_dataset=panx_de_encoded["train"],
                  eval_dataset=panx_de_encoded["validation"],
                  tokenizer=xlmr_tokenizer)
trainer.train()
```

## Evaluation

The script evaluates the model using the `seqeval` library, providing functions for confusion matrix visualization and error analysis.

```python
from seqeval.metrics import classification_report

y_true = [["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"],
          ["B-PER", "I-PER", "O"]]
y_pred = [["O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"],
          ["B-PER", "I-PER", "O"]]

print(classification_report(y_true, y_pred))
```

## F1 Score

The F1 score is a key metric for evaluating the model's performance on NER tasks. It is the harmonic mean of precision and recall, providing a balanced measure.

```python
from seqeval.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")
```

## Zero-Shot Transfer Learning

The code explores zero-shot transfer learning by fine-tuning the model on a subset of the French dataset and evaluating performance on German.

## Multilingual Fine-Tuning

The repository includes code for fine-tuning the model on multiple languages simultaneously. It demonstrates the performance of the fine-tuned model on each language individually.

## Results and Analysis

The script outputs F1 scores for the fine-tuned model on each language and provides a detailed analysis of model performance, including error analysis and confusion matrix visualization.

## XLM-Roberta and NER

XLM-Roberta's effectiveness in NER tasks arises from its ability to capture contextual information across multiple languages. The model's pre-training on a diverse set of languages enables it to generalize well to various linguistic patterns, making it suitable for multilingual NER applications.
