#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.
import logging
from collections import defaultdict
from itertools import chain

import torch
from torch.utils.data import DataLoader, TensorDataset
from python_tf_idf.tfidf import TfIdf

from utils_focus import get_dataset_only_train_dev, get_dataset_only_test

SPECIAL_TOKENS = ["<machine>", "<human>", "<persona>", "<knowledge>"]
ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': ['<machine>', '<human>', '<persona>', '<knowledge>']}
BART_MODEL_INPUTS = ["input_ids", "decoder_input_ids", "lm_labels", "token_type_ids", "mc_token_ids",
                "persona_candidates", "persona_can_idx", "persona_grounding",
                "knowledge_candidates", "knowledge_can_idx", "knowledge_grounding", "reply"]
GPT2_MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids", "mc_token_ids",
                "persona_candidates", "persona_can_idx", "persona_grounding",
                "knowledge_candidates", "knowledge_can_idx", "knowledge_grounding", "reply"]
GPT2_CTXT_MODEL_INPUTS = ["input_ids", "input_eos", "lm_labels", "token_type_ids", "mc_token_ids",
                "persona_candidates", "persona_can_idx", "persona_grounding",
                "knowledge_candidates", "knowledge_can_idx", "knowledge_grounding",
                "tot_knowledge", "tot_knowledge_token_ids", "tot_knowledge_eos", "reply", "dialog", "dialog_tti"]
BART_CTXT_MODEL_INPUTS = ["input_ids", "input_eos", "decoder_input_ids", "lm_labels", "token_type_ids", "mc_token_ids",
                          "persona_candidates", "persona_can_idx", "persona_grounding",
                          "knowledge_candidates", "knowledge_can_idx", "knowledge_grounding",
                          "tot_knowledge", "tot_knowledge_eos", "reply", "dialog"]
BART_PADDED_INPUTS = ["decoder_input_ids", "lm_labels", "token_type_ids"]
GPT2_PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)

num_persona = 5
num_knowledge = 10

def pad_dataset_gpt2_inctxt(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    max_l_reply = max(len(x) for x in dataset["reply"])
    # print([x for x in dataset])
    # exit()
    ############### to delete examples with too long knowledge candidates ######################
    remove_list = list()
    persona_nonlist = list()
    #print("knowledge candidates: ", dataset["knowledge_candidates"])
    for idx_1, x in enumerate(dataset["knowledge_candidates"]):
        for idx_2, i in enumerate(x):
            if type(i) != list:
                remove_list.append(idx_1)
            elif len(i) > 500:
                dataset["knowledge_candidates"][idx_1][idx_2] = i[:500]


    for idx_1, x in enumerate(dataset["persona_candidates"]):
        #print("? ", len(x))
        if len(x) != num_persona or type(x) != list:
            remove_list.append(idx_1)
            persona_nonlist.append(idx_1)
        for idx_2, i in enumerate(x):
            if len(i) > 500 or type(i) != list:
                remove_list.append(idx_1)


    for idx_1, x in enumerate(dataset["tot_knowledge"]):
        for idx_2, i in enumerate(x):
            if type(i) != list:
                remove_list.append(idx_1)
            elif len(i) > 200:
                dataset["tot_knowledge"][idx_1][idx_2] = i[:200]


    remove_list = list(set(remove_list))
    print("remove list: ", len(remove_list))

    if len(remove_list) != 0:
        new_dataset = defaultdict(list)
        for input in GPT2_MODEL_INPUTS:
            for i, element in enumerate(dataset[input]):
                if i in remove_list:
                    continue
                else:
                    if input == 'persona_candidates':
                        assert len(element) == num_persona
                    new_dataset[input].append(element)
    else:
        new_dataset = dataset

    max_l_knowledge_cans = max([len(i) for x in new_dataset["knowledge_candidates"] for i in x])
    max_l_tot_knowledge = max([len(i) for x in new_dataset["tot_knowledge"] for i in x])
    max_l_persona_cans = max([len(i) for x in new_dataset["persona_candidates"] for i in x])
    max_l_dialog = max(len(x) for x in new_dataset["dialog"])

    for name in GPT2_PADDED_INPUTS:
        new_dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in
                             new_dataset[name]]
    new_dataset["reply"] = [x + [padding if name != "lm_labels" else -100] * (max_l_reply - len(x)) for x in
                            new_dataset["reply"]]

    knowledge_list = list()
    for i, knowledges in enumerate(new_dataset["knowledge_candidates"]):
        candidates_list = list()
        for candidates in knowledges:
            padded_candidate = candidates + [padding] * (max_l_knowledge_cans - len(candidates))
            candidates_list.append(padded_candidate)
        knowledge_list.append(candidates_list)
    new_dataset["knowledge_candidates"] = knowledge_list

    persona_list = list()
    for i, personas in enumerate(new_dataset["persona_candidates"]):
        candidates_list = list()
        for candidates in personas:
            padded_candidate = candidates + [padding] * (max_l_persona_cans - len(candidates))
            candidates_list.append(padded_candidate)
        persona_list.append(candidates_list)
    new_dataset["persona_candidates"] = persona_list

    tot_knowledge_list = list()
    for i, tot_kn in enumerate(new_dataset["tot_knowledge"]):
        candidates_list = list()
        for candidates in tot_kn:
            padded_candidate = candidates + [padding] * (max_l_tot_knowledge - len(candidates))
            candidates_list.append(padded_candidate)

        tot_knowledge_list.append(candidates_list)
    new_dataset["tot_knowledge"] = tot_knowledge_list

    tot_knowledge_token_ids_list = list()
    for i, tot_kn_ids in enumerate(new_dataset["tot_knowledge_token_ids"]):
        candidates_list = list()
        for candidates in tot_kn_ids:
            padded_candidate = candidates + [padding] * (max_l_tot_knowledge - len(candidates))
            candidates_list.append(padded_candidate)

        tot_knowledge_token_ids_list.append(candidates_list)
    new_dataset["tot_knowledge_token_ids"] = tot_knowledge_token_ids_list

    new_dataset["dialog"] = [x + [padding] * (max_l_dialog - len(x)) for x in new_dataset["dialog"]]
    new_dataset["dialog_tti"] = [x + [padding] * (max_l_dialog - len(x)) for x in new_dataset["dialog_tti"]]
    #print("dial: ", new_dataset["dialog"][0], "dial_tti", new_dataset["dialog_tti"][0])
    return new_dataset

def pad_dataset_bart_inctxt(dataset, padding=1):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_enc_l = max(len(x) for x in dataset["input_ids"])
    max_l = max(len(x) for x in dataset["decoder_input_ids"])
    max_l_reply = max(len(x) for x in dataset["reply"])

    ###############to delete examples with too long knowledge candidates######################
    remove_list = list()
    persona_nonlist = list()
    for idx_1, x in enumerate(dataset["knowledge_candidates"]):
        for idx_2, i in enumerate(x):
            if len(i) > 500 or type(i) != list:
                print("knowledge", len(i), type(i))
                remove_list.append(idx_1)

    for idx_1, x in enumerate(dataset["persona_candidates"]):
        if len(x) != num_persona or type(x) != list:
            remove_list.append(idx_1)
            persona_nonlist.append(idx_1)
        for idx_2, i in enumerate(x):
            if len(i) > 500 or type(i) != list:
                remove_list.append(idx_1)

    for idx_1, x in enumerate(dataset["tot_knowledge"]):
        for idx_2, i in enumerate(x):
            if type(i) != list:
                remove_list.append(idx_1)
            elif len(i) > 200:
                dataset["tot_knowledge"][idx_1][idx_2] = i[:200]

    remove_list = list(set(remove_list))
    print("remove list: ", len(remove_list))

    if len(remove_list) != 0:
        new_dataset = defaultdict(list)
        for input in BART_MODEL_INPUTS:
            for i, element in enumerate(dataset[input]):
                if i in remove_list:
                    continue
                else:
                    if input == 'persona_candidates':
                        assert len(element) == num_persona
                    new_dataset[input].append(element)
    else:
        new_dataset = dataset

    max_l_knowledge_cans = max([len(i) for x in new_dataset["knowledge_candidates"] for i in x])
    max_l_tot_knowledge = max([len(i) for x in new_dataset["tot_knowledge"] for i in x])
    max_l_persona_cans = max([len(i) for x in new_dataset["persona_candidates"] for i in x])
    max_l_dialog = max(len(x) for x in new_dataset["dialog"])


    for name in BART_PADDED_INPUTS:
        new_dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in new_dataset[name]]
    new_dataset["input_ids"] = [x + [padding if name != "lm_labels" else -100] * (max_enc_l - len(x)) for x in new_dataset["input_ids"]]
    new_dataset["reply"] = [x + [padding if name != "lm_labels" else -100] * (max_l_reply - len(x)) for x in new_dataset["reply"]]

    knowledge_list = list()
    for i, knowledges in enumerate(new_dataset["knowledge_candidates"]):
        candidates_list = list()
        for candidates in knowledges:
            padded_candidate = candidates + [padding] * (max_l_knowledge_cans - len(candidates))
            candidates_list.append(padded_candidate)
        knowledge_list.append(candidates_list)
    new_dataset["knowledge_candidates"] = knowledge_list

    persona_list = list()
    for i, personas in enumerate(new_dataset["persona_candidates"]):
        candidates_list = list()
        for candidates in personas:
            padded_candidate = candidates + [padding] * (max_l_persona_cans - len(candidates))
            candidates_list.append(padded_candidate)
        persona_list.append(candidates_list)
    new_dataset["persona_candidates"] = persona_list

    tot_knowledge_list = list()
    for i, tot_kn in enumerate(new_dataset["tot_knowledge"]):
        candidates_list = list()
        for candidates in tot_kn:
            padded_candidate = candidates + [padding] * (max_l_tot_knowledge - len(candidates))
            candidates_list.append(padded_candidate)

        tot_knowledge_list.append(candidates_list)
    new_dataset["tot_knowledge"] = tot_knowledge_list

    tot_knowledge_token_ids_list = list()
    for i, tot_kn_ids in enumerate(new_dataset["tot_knowledge_token_ids"]):
        candidates_list = list()
        for candidates in tot_kn_ids:
            padded_candidate = candidates + [padding] * (max_l_tot_knowledge - len(candidates))
            candidates_list.append(padded_candidate)

        tot_knowledge_token_ids_list.append(candidates_list)
    new_dataset["tot_knowledge_token_ids"] = tot_knowledge_token_ids_list

    new_dataset["dialog"] = [x + [padding] * (max_l_dialog - len(x)) for x in new_dataset["dialog"]]
    new_dataset["dialog_tti"] = [x + [padding] * (max_l_dialog - len(x)) for x in new_dataset["dialog_tti"]]

    return new_dataset

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    if type(tokenizer).__name__ == 'GPT2Tokenizer':
        ATTR_TO_SPECIAL_TOKEN['pad_token'] = '<pad>'
        print('<pad> token added!')
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    print("orig num", orig_num_tokens, "num_added", num_added_tokens) #50265, 4
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def choose_knowledge(knowledge, question):
    table = TfIdf()
    for i, paragraph in enumerate(knowledge):
        table.add_document(i, paragraph)
    results = table.similarities(question)
    results = sorted(results, key=lambda x: x[1], reverse=True)
    result_idx = [i[0] for i in results[:5]]
    chosen_knowledge = [knowledge[ri] for ri in result_idx]
    return chosen_knowledge


def build_input_from_segments_bart_inctxt(persona, knowledge, history, persona_cans, persona_grounding, knowledge_cans, knowledge_answer_idx, ID, tokenizer, lm_labels=False, testset=False, inference=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    machine_st, human_st, persona_st, knowledge_st = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    #machine: 50265 human: 50266 persona: 50267 knowledge: 50268 padding: 1 bos: 0 eos: 2

    history, reply = history[:-1], history[-1]
    history = [[human_st if i % 2 == 0 else machine_st] + s for i, s in enumerate(history)]
    reply = reply
    reply_tti = [machine_st] * (len(reply)+2)

    if inference == False:
        if len(knowledge) > 1:
            chosen_knowledge = choose_knowledge(knowledge, history[-1])
        else:
            chosen_knowledge = knowledge[0:5]
    else:
        chosen_knowledge = knowledge_cans[knowledge_answer_idx]

    paragraphs = []
    for para in chosen_knowledge:
        #for para in knowledge:
        if len(para) > 100:
            short_para = para[:100]
        else:
            short_para = para
        paragraphs.append(short_para)


    if testset == False:
        if len(history) == 1:
            enc_sequence = [[bos]] + [[persona_st] + list(chain(*persona))] + history
            dec_sequence = [bos] + reply + ([eos] if with_eos else [])
            dialog = [[bos]] + history

        else:
            enc_sequence = [[bos]] + [[persona_st] + list(chain(*persona))] + [list(chain(*history))]
            dec_sequence = [bos] + reply + ([eos] if with_eos else [])
            dialog = [[bos]] + [list(chain(*history))]
    else:
        if len(history) == 1:
            enc_sequence = [[bos]] + [[persona_st] + list(chain(*persona))] + history
            dec_sequence = [bos] + [machine_st]
            reply_tti = [machine_st]
            dialog = [[bos]] + history
        else:
            enc_sequence = [[bos]] + [[persona_st] + list(chain(*persona))] + [list(chain(*history))]
            dec_sequence = [bos] + [machine_st]
            reply_tti = [machine_st]
            dialog = [[bos]] + [list(chain(*history))]

    max_tot_k_token_ids = max([len(i) for i in paragraphs])

    instance = {}
    instance["input_ids"] = list(chain(*enc_sequence))
    instance["input_eos"] = len(list(chain(*enc_sequence)))
    instance["dialog"] = list(chain(*dialog))
    instance["decoder_input_ids"] = dec_sequence
    instance["token_type_ids"] = reply_tti
    if lm_labels:
        if len(dec_sequence) > 1:
            instance["lm_labels"] = [-100] + dec_sequence[1:]
        else:
            instance["lm_labels"] = [-100]
    instance["persona_candidates"] = [[bos] + can + [persona_st] + [eos] for can in persona_cans]
    instance["persona_can_idx"] = [len(can)-1 for can in instance["persona_candidates"]]
    instance["persona_grounding"] = persona_grounding
    instance["knowledge_candidates"] = [[bos] + can[:100] + [knowledge_st] + [eos] if len(can) > 100 else [bos] + can + [knowledge_st] + [eos] for can in knowledge_cans]
    instance["knowledge_can_idx"] = [len(can)-1 for can in instance["knowledge_candidates"]]
    instance["knowledge_grounding"] = knowledge_answer_idx
    instance["mc_token_ids"] = 0
    instance["dialog_ID"] = ID
    instance["reply"] = reply[1:]
    instance['tot_knowledge'] = paragraphs
    instance["tot_knowledge_token_ids"] = [[knowledge_st] * max_tot_k_token_ids + [tokenizer.pad_token_id] * (100 - max_tot_k_token_ids) for _ in range(5)]
    instance['tot_knowledge_eos'] = [len(p)-1 for p in paragraphs]
    assert len(instance["decoder_input_ids"]) == len(instance["lm_labels"])

    return instance

def build_input_from_segments_gpt2_inctxt(persona, knowledge, history, persona_cans, persona_grounding, knowledge_cans, knowledge_answer_idx, ID, tokenizer, lm_labels=False, testset=False, inference=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    machine_st, human_st, persona_st, knowledge_st = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    # machine: 50257
    # human: 50258
    # persona: 50259
    # knowledge: 50260
    # padding: 50261
    # bos: 50256
    # eos: 50256

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    history, reply = history[:-1], history[-1]
    history = [[human_st if i % 2 == 0 else machine_st] + s for i, s in enumerate(history)]
    history_tti = [[sent[0]] * len(sent) for sent in history]
    reply = [machine_st] + reply
    reply_tti = [machine_st] * (len(reply)+1)

    if inference == False:
        if len(knowledge) > 1:
            chosen_knowledge = choose_knowledge(knowledge, history[-1])
        else:
            chosen_knowledge = knowledge[0:5]
    else:
        chosen_knowledge = knowledge_cans[knowledge_answer_idx]

    paragraphs = []
    for para in chosen_knowledge:
        #for para in knowledge:
        if len(para) > 100:
            short_para = para[:100]
        else:
            short_para = para
        paragraphs.append(short_para)


    if testset == False:
        if len(history) == 1:
            sequence = [[bos]] + [[persona_st] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
            dialog = [[bos]] + history + [reply + ([eos] if with_eos else [])]
        else:
            sequence = [[bos]] + [[persona_st] + list(chain(*persona))] + [list(chain(*history))] + [reply + ([eos] if with_eos else [])]
            dialog = [[bos]] + [list(chain(*history))] + [reply + ([eos] if with_eos else [])]
        persona_tti = [persona_st] * (len(list(chain(*persona))) + 2) # bos, eos
        tti = persona_tti + list(chain(*history_tti)) + reply_tti
        dialog_tti = [history_tti[0][0]] + list(chain(*history_tti)) + reply_tti

    else:
        if len(history) == 1:
            sequence = [[bos]] + [[persona_st] + list(chain(*persona))] + history + [[machine_st]]
            dialog = [[bos]] + history + [[machine_st]]
        else:
            sequence = [[bos]] + [[persona_st] + list(chain(*persona))] + [list(chain(*history))] + [[machine_st]]
            dialog = [[bos]] + [list(chain(*history))] + [[machine_st]]
        persona_tti = [persona_st] * (len(list(chain(*persona))) + 2)
        tti = persona_tti + list(chain(*history_tti)) + [machine_st]
        dialog_tti = [history_tti[0][0]] + list(chain(*history_tti)) + [machine_st]

    mc_token_ids = list(chain(*sequence))
    mc_list = [x for x, y in enumerate(mc_token_ids) if y == machine_st]
    max_tot_k_token_ids = max([len(i) for i in paragraphs])
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["input_eos"] = len(tti)-1
    instance["token_type_ids"] = tti
    instance["dialog"] = list(chain(*dialog))
    instance["dialog_tti"] = dialog_tti
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in dialog[:-1])) + [-100] + dialog[-1][1:]
    instance["persona_candidates"] = [can + [persona_st] + [eos] for can in persona_cans]
    instance["persona_can_idx"] = [len(can)-1 for can in instance["persona_candidates"]]
    instance["persona_grounding"] = persona_grounding
    instance["knowledge_candidates"] = [[bos] + can[:100] + [knowledge_st] + [eos] if len(can) > 100 else [bos] + can + [knowledge_st] + [eos] for can in knowledge_cans]
    instance["knowledge_can_idx"] = [len(can)-1 for can in instance["knowledge_candidates"]]
    instance["knowledge_grounding"] = knowledge_answer_idx
    instance['tot_knowledge'] = paragraphs
    instance["mc_token_ids"] = mc_list[-1]
    instance["dialog_ID"] = ID
    instance["reply"] = reply[1:]
    instance["tot_knowledge_token_ids"] = [[knowledge_st] * max_tot_k_token_ids + [tokenizer.pad_token_id] * (100 - max_tot_k_token_ids) for _ in range(5)]
    instance['tot_knowledge_eos'] = [len(p)-1 for p in paragraphs]

    assert len(instance["input_ids"]) == len(instance["token_type_ids"])
    assert len(instance["dialog"]) == len(instance["lm_labels"]) == len(instance["dialog_tti"])

    return instance

def get_data_loaders(args, tokenizer, generation=False):
    """ Prepare the dataset for training and evaluation """

    plan = get_dataset_only_train_dev(tokenizer, args.train_dataset_path, args.train_dataset_cache, args.dev_dataset_path, args.dev_dataset_cache)

    model_name = args.model_name

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}

    for dataset_name, dataset in plan.items():
        print(dataset_name, len(dataset))
        if generation == True:
            testset = True
        else:
            testset = False
        for dialog in dataset:
            ID = dialog["dialogID"]
            persona = dialog['persona']
            knowledge = dialog['knowledge']
            utterance = dialog['utterance']
            for i, utt in enumerate(utterance):
                history = utt['dialog'][-(2*args.max_history):]
                persona_cans = utt['persona_candidates']
                persona_grouding = utt['persona_grounding']
                knowledge_cans = utt['knowledge_candidates']
                knowledge_answer_idx = utt['knowledge_answer_index']


                if model_name == 'GPT2' or model_name == 'transformer-decoder':
                    instance = build_input_from_segments_gpt2_inctxt(persona, knowledge, history, persona_cans, persona_grouding,
                                                                     knowledge_cans, knowledge_answer_idx, ID, tokenizer,
                                                                     lm_labels=True, testset=testset, inference=args.inference)
                elif model_name == 'BART' or model_name == 'transformer-encdec':
                    instance = build_input_from_segments_bart_inctxt(persona, knowledge, history, persona_cans, persona_grouding,
                                                              knowledge_cans, knowledge_answer_idx, ID, tokenizer,
                                                              lm_labels=True, testset=testset, inference=args.inference)

                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(input_array)

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}

    for dataset_name, dataset in datasets.items():
        if model_name == 'GPT2' or model_name == 'transformer-decoder':
            dataset = pad_dataset_gpt2_inctxt(dataset, padding=tokenizer.pad_token_id)
            MODEL_INPUTS = GPT2_CTXT_MODEL_INPUTS
        elif model_name == 'BART' or model_name == 'transformer-encdec':
            dataset = pad_dataset_bart_inctxt(dataset, padding=tokenizer.pad_token_id)
            MODEL_INPUTS = BART_CTXT_MODEL_INPUTS
        for input_name in MODEL_INPUTS:
            #print("tensor: ", input_name, "len: ", len(dataset[input_name]))
            tensor = torch.tensor(dataset[input_name], device=args.device)
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)
    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape)) #[131438, 4, 280]
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape)) #[7801, 20, 184]

    return train_loader, valid_loader, train_sampler, valid_sampler


def get_testdata_loaders(args, tokenizer, generation=True):
    """ Prepare the dataset for training and evaluation """

    plan = get_dataset_only_test(tokenizer, args.test_dataset_path, args.test_dataset_cache)

    model_name = args.model_name

    logger.info("Build inputs and labels")
    datasets = {"test": defaultdict(list)}
    for dataset_name, dataset in plan.items():
        print(dataset_name, len(dataset))
        if generation == True:
            testset = True
        else:
            testset = False
        for dialog in dataset:
            ID = dialog["dialogID"]
            persona = dialog['persona']
            knowledge = dialog['knowledge']
            utterance = dialog['utterance']
            for i, utt in enumerate(utterance):
                history = utt['dialog'][-(2*args.max_history):]
                persona_cans = utt['persona_candidates']
                persona_grouding = utt['persona_grounding']
                knowledge_cans = utt['knowledge_candidates']
                knowledge_answer_idx = utt['knowledge_answer_index']

                if model_name == 'GPT2' or model_name == 'transformer-decoder':
                    instance = build_input_from_segments_gpt2_inctxt(persona, knowledge, history, persona_cans,
                                                                     persona_grouding,
                                                                     knowledge_cans, knowledge_answer_idx, ID,
                                                                     tokenizer,
                                                                     lm_labels=True, testset=testset,
                                                                     inference=args.inference)
                elif model_name == 'BART' or model_name == 'transformer-encdec':
                    instance = build_input_from_segments_bart_inctxt(persona, knowledge, history, persona_cans, persona_grouding,
                                                         knowledge_cans, knowledge_answer_idx, ID, tokenizer,
                                                         lm_labels=True, testset=testset)

                for input_name, input_array in instance.items():
                    datasets[dataset_name][input_name].append(input_array)

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"test": []}

    for dataset_name, dataset in datasets.items():
        if model_name == 'GPT2' or model_name == 'transformer-decoder':
            dataset = pad_dataset_gpt2_inctxt(dataset, padding=tokenizer.pad_token_id)
            MODEL_INPUTS = GPT2_CTXT_MODEL_INPUTS
        elif model_name == 'BART' or model_name == 'transformer-encdec':
            dataset = pad_dataset_bart_inctxt(dataset, padding=tokenizer.pad_token_id)
            MODEL_INPUTS = BART_CTXT_MODEL_INPUTS
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name], device=args.device)
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    test_dataset = TensorDataset(*tensor_datasets["test"])
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if args.distributed else None
    test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size, shuffle=False)
    logger.info("Test dataset (Batch, Candidates, Seq length): {}".format(test_dataset.tensors[0].shape))

    return test_loader, test_sampler