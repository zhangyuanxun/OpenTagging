from collections import defaultdict
import torch
from metrics import SeqEntityScore
from utils import get_entities, extract_text_from_bert_padding

import json
import os


def evaluate(args, model, tokenizer, dataloader, labels_list):
    print("Start to evaluate the model...")
    args.id2label = {i: label for i, label in enumerate(labels_list)}
    args.label2id = {label: i for i, label in enumerate(labels_list)}
    metric = SeqEntityScore(args.id2label, markup='bio')

    eval_loss = 0.0
    nb_eval_steps = 0
    all_predictions = defaultdict(dict)
    all_labels = defaultdict(dict)
    all_context_input_ids = defaultdict(dict)
    all_attribute_input_ids = defaultdict(dict)
    idx = 0
    for batch in dataloader:
        model.eval()

        inputs = {k: v.to(args.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs['logits']
            eval_loss += outputs['loss'].item()
            pred_labels, _ = model.crf.obtain_labels(logits, args.id2label, inputs['context_input_len'])

        true_label_ids = inputs['label_ids'].cpu().numpy().tolist()
        context_input_ids = inputs['context_input_ids']
        attribute_input_ids = inputs['attribute_input_ids']

        for i in range(len(pred_labels)):
            all_predictions[idx] = get_entities(pred_labels[i], args.id2label, 'bio')
            all_labels[idx] = get_entities(true_label_ids[i], args.id2label, 'bio')
            all_context_input_ids[idx] = context_input_ids[i]
            all_attribute_input_ids[idx] = attribute_input_ids[i]
            idx += 1

        nb_eval_steps += 1
        for i, labels in enumerate(true_label_ids):
            true_label_path = []
            pred_label_path = []
            for j, m in enumerate(labels):
                if j == 0:
                    continue
                elif true_label_ids[i][j] == args.label2id['[SEP]']:
                    metric.update(pred_paths=[pred_label_path], label_paths=[true_label_path])
                    break
                else:
                    true_label_path.append(args.id2label[true_label_ids[i][j]])
                    pred_label_path.append(pred_labels[i][j])

    assert len(all_predictions) == len(all_context_input_ids) == len(all_labels) == len(all_attribute_input_ids)

    prediction_results = list()

    for i in range(idx):
        predictions = all_predictions[i]
        labels = all_labels[i]
        context_input_ids = all_context_input_ids[i]
        attribute_input_ids = all_attribute_input_ids[i]

        context = tokenizer.convert_ids_to_tokens(context_input_ids)
        context = tokenizer.convert_tokens_to_string(context)

        attribute = tokenizer.convert_ids_to_tokens(attribute_input_ids)
        attribute = tokenizer.convert_tokens_to_string(attribute)

        true_label_tokens = [tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(context_input_ids[label[1]: label[2] + 1])) for label in labels]
        pred_label_tokens = [tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(context_input_ids[pred[1]: pred[2] + 1])) for pred in predictions]

        context = extract_text_from_bert_padding(context)
        attribute = extract_text_from_bert_padding(attribute)
        prediction_results.append({'context': context, 'attribute': attribute,
                                   'true value': ' '.join(true_label_tokens),
                                   'predict value': ' '.join(pred_label_tokens),
                                   'predictions': predictions})

    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    performance_results = {f'{key}': value for key, value in eval_info.items()}
    performance_results['loss'] = eval_loss

    print("***** Eval results *****")
    info = "-".join([f' {key}: {value:.4f} ' for key, value in performance_results.items()])
    print(info)

    # save the file
    with open(os.path.join(args.output_dir, "performance_results.json"), "w") as f:
        json.dump(performance_results, f)

    with open(os.path.join(args.output_dir, "prediction_results.json"), "w") as f:
        json.dump(prediction_results, f, indent=4)