#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.
import json
import logging
import os
import torch
from transformers import cached_path

logger = logging.getLogger(__file__)

def get_dataset_only_train_dev(tokenizer, train_dataset_path, train_dataset_cache, dev_dataset_path, dev_dataset_cache):

    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)

    train_dataset_cache = train_dataset_cache + '_train_focus_' + type(tokenizer).__name__
    dev_dataset_cache = dev_dataset_cache + '_dev_focus_' + type(tokenizer).__name__
    if train_dataset_cache and os.path.isfile(train_dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", train_dataset_cache)
        train_dataset = torch.load(train_dataset_cache)
        dev_dataset = torch.load(dev_dataset_cache)
        all_dataset = dict()
        all_dataset["train"] = train_dataset["train"]
        all_dataset["valid"] = dev_dataset["valid"]
    else:
        logger.info("Process dataset from %s", train_dataset_path)
        plan_file_train = cached_path(train_dataset_path)
        plan_file_dev = cached_path(dev_dataset_path)
        file_dict = {"train": plan_file_train, "valid": plan_file_dev}
        all_dataset = dict()

        for name, file in file_dict.items():
            with open(file, "r", encoding="utf-8") as f:
                dataset = json.loads(f.read())
                dataset_enc = dict()
                dataset_enc[name] = list()
                for dialogue in dataset["data"]:
                    ID = dialogue["dialogID"]
                    persona = dialogue["persona"]
                    knowledge = dialogue["knowledge"]
                    utterance = dialogue["utterance"]
                    new_dialogue = dict()
                    new_dialogue["utterance"] = list()
                    for i, utt in enumerate(utterance):
                        key = "dialogue" + str(i+1)
                        dial = utt[key]
                        dial_new = dict()
                        persona_can = utt["persona_candidate"]
                        if len(persona_can) > 5:
                            persona_can = persona_can[:5]
                        persona_ground = utt["persona_grounding"]
                        if len(persona_ground) > 5:
                            persona_ground = persona_ground[:5]
                        knowledge_can = utt["knowledge_candidates"]
                        knowledge_answer = utt["knowledge_answer_index"]
                        dial_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in dial]
                        persona_ground_enc = [1 if item==True else 0 for item in persona_ground]
                        knowledge_can_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in knowledge_can]
                        persona_can_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in persona_can]
                        dial_new["dialog"] = dial_enc
                        dial_new["persona_grounding"] = persona_ground_enc
                        dial_new["persona_candidates"] = persona_can_enc
                        dial_new["knowledge_candidates"] = knowledge_can_enc
                        dial_new["knowledge_answer_index"] = knowledge_answer
                        new_dialogue["utterance"].append(dial_new)
                    persona_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in persona]
                    knowledge_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in knowledge]
                    new_dialogue["persona"] = persona_enc
                    new_dialogue["knowledge"] = knowledge_enc
                    new_dialogue["dialogID"] = ID
                    dataset_enc[name].append(new_dialogue)
            logger.info("Tokenize and encode the dataset")
            dataset = dataset_enc
            all_dataset[name] = dataset_enc[name]
            if name == 'train':
                torch.save(dataset, train_dataset_cache)
            else:
                torch.save(dataset, dev_dataset_cache)

    return all_dataset

def get_dataset_only_test(tokenizer, dataset_path, dataset_cache):

    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)

    dataset_cache = dataset_cache + '_test_focus_' + type(tokenizer).__name__

    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Process dataset from %s", dataset_path)
        plan_file = cached_path(dataset_path)
        with open(plan_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())
            dataset_enc = dict()
            dataset_enc["test"] = list()
            for dialogue in dataset["data"]:
                ID = dialogue["dialogID"]
                persona = dialogue["persona"]
                knowledge = dialogue["knowledge"]
                utterance = dialogue["utterance"]
                new_dialogue = dict()
                new_dialogue["utterance"] = list()
                for i, utt in enumerate(utterance):
                    key = "dialogue" + str(i+1)
                    dial = utt[key]
                    dial_new = dict()
                    persona_can = utt["persona_candidate"]
                    if len(persona_can) > 5:
                        persona_can = persona_can[:5]
                    persona_ground = utt["persona_grounding"]
                    if len(persona_ground) > 5:
                        persona_ground = persona_ground[:5]
                    knowledge_can = utt["knowledge_candidates"]
                    knowledge_answer = utt["knowledge_answer_index"]
                    dial_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in dial]
                    persona_ground_enc = [1 if item==True else 0 for item in persona_ground]
                    knowledge_can_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in knowledge_can]
                    persona_can_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in persona_can]
                    dial_new["dialog"] = dial_enc
                    dial_new["persona_candidates"] = persona_can_enc
                    dial_new["persona_grounding"] = persona_ground_enc
                    dial_new["knowledge_candidates"] = knowledge_can_enc
                    dial_new["knowledge_answer_index"] = knowledge_answer
                    new_dialogue["utterance"].append(dial_new)
                persona_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in persona]
                knowledge_enc = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence.strip())) for sentence in knowledge]
                new_dialogue["persona"] = persona_enc
                new_dialogue["knowledge"] = knowledge_enc
                new_dialogue["dialogID"] = ID
                dataset_enc["test"].append(new_dialogue)
        logger.info("Tokenize and encode the dataset")
        dataset = dataset_enc
        torch.save(dataset, dataset_cache)
    return dataset

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_focus_logdir(dir_name: str):
    logdir = os.path.join(
        './models', dir_name
    )
    return logdir