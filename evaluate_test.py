#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.
import logging
import random
from argparse import ArgumentParser
from pprint import pformat
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import Sigmoid, Softmax, CrossEntropyLoss
from data_utils import get_testdata_loaders, add_special_tokens_
from inference_test import sample_sequence
from ignite.metrics import Bleu, RougeL, RougeN, Accuracy
from ignite.metrics.precision import CharPrecision
from ignite.metrics.recall import CharRecall
logger = logging.getLogger(__file__)

SPECIAL_TOKENS = ["<machine>", "<human>", "<persona>", "<knowledge>"]

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits




def run():
    parser = ArgumentParser()
    parser.add_argument("--test_dataset_path", type=str, default="data/test_focus.json", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--test_dataset_cache", type=str, default='data/focus_cache.tar.gz', help="Path or url of the dataset cache")
    parser.add_argument("--model_name", type=str, default="", help="{GPT2, BART, transformer-decoder, transformer-encdec}")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous utterances to keep in history")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--inference", action='store_true', help="If true, inference with gold knowledge")
    parser.add_argument("--seed", type=int, default=19950604, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))
    args.distributed = (args.local_rank != -1)

    if args.seed != 0:
    	random.seed(args.seed)
    	torch.random.manual_seed(args.seed)
    	torch.cuda.manual_seed(args.seed)

    logger.info("Get model and tokenizer")

    if args.model_name == 'GPT2':
        from transformers import GPT2Tokenizer
        from classification_modules import GPT2PK_ctxt
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
        model = GPT2PK_ctxt.from_pretrained(args.model_checkpoint)
        model.to(args.device)
        add_special_tokens_(model, tokenizer)

    elif args.model_name == 'BART':
        from transformers import BartTokenizer
        from classification_modules import BARTPK_ctxt
        tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint)
        model = BARTPK_ctxt.from_pretrained(args.model_checkpoint)
        model.to(args.device)
        add_special_tokens_(model, tokenizer)

    elif args.model_name == 'transformer-decoder':
        from transformers import GPT2Tokenizer
        from classification_modules import GPT2PK_ctxt
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
        model = GPT2PK_ctxt.from_pretrained(args.model_checkpoint)
        model.to(args.device)
        add_special_tokens_(model, tokenizer)

    elif args.model_name == 'transformer-encdec':
        from transformers import BartTokenizer
        from classification_modules import BARTPK_ctxt
        tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint)
        model = BARTPK_ctxt.from_pretrained(args.model_checkpoint)
        model.to(args.device)
        add_special_tokens_(model, tokenizer)

    else:
        raise NotImplementedError

    #dataset = get_dataset_only_train(tokenizer, args.dataset_path, args.dataset_cache)
    logger.info("Prepare datasets")
    test_loader, test_sampler = get_testdata_loaders(args, tokenizer, generation=True)

    with torch.no_grad():
        r1 = RougeN(ngram=1)
        r2 = RougeN(ngram=2)
        rl = RougeL()
        b1 = Bleu(ngram=1)
        b2 = Bleu(ngram=2)
        b3 = Bleu(ngram=3)
        b4 = Bleu(ngram=4)
        pre = CharPrecision()
        rec = CharRecall()
        pg = Accuracy()
        kg = Accuracy()

        for test_data in tqdm(test_loader):
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            if model.config.model_type == 'gpt2':
                input_ids, input_eos, lm_labels, token_type_ids, mc_token_ids, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
                knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_token_ids, tot_knowledge_eos, reply, dialog, dialog_tti = test_data

            elif model.config.model_type == 'bart':
                input_ids, input_eos, decoder_input_ids, lm_labels, token_type_ids, mc_token_ids, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
                knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_eos, reply, dialog = test_data
            else:
                raise NotImplementedError
            mask = (reply != tokenizer.pad_token_id)
            reply = reply[mask]

            if model.config.model_type == 'gpt2':
                output = model(
                    input_ids=input_ids,
                    input_eos=input_eos,
                    token_type_ids=token_type_ids,
                    only_dial_input_ids=dialog,
                    only_dial_token_type_ids=dialog_tti,
                    persona_input_ids=persona_candidates,
                    knowledge_input_ids=knowledge_candidates,
                    persona_can_idx=persona_can_idx,
                    knowledge_can_idx=knowledge_can_idx,
                    tot_knowledge=tot_knowledge,
                    tot_knowledge_token_ids=tot_knowledge_token_ids,
                    tot_knowledge_eos=tot_knowledge_eos,
                    training=False,
                    mc_token_ids=mc_token_ids
                )
                lm_labels, lm_logits, knowledge_logits, persona_logits = output[0], output[1], output[2], output[3]

                machine, human, persona, knowledge, padding, bos = 50257, 50258, 50259, 50260, 50261, 50256
                device = input_ids.get_device()

                machine_tensor = torch.tensor([machine]).cuda(device)
                persona_tensor = torch.tensor([persona]).cuda(device)
                knowledge_tensor = torch.tensor([knowledge]).cuda(device)
                bos_tensor = torch.tensor([bos]).cuda(device)


                sigmoid = Sigmoid()
                persona_pred_sigmoid = sigmoid(persona_logits)
                persona_pred_sigmoid = (persona_pred_sigmoid > 0.5).float()
                all_persona_pred = []
                selected_persona_idx = list()
                for batch_idx, persona_batch in enumerate(torch.eq(persona_pred_sigmoid, 1)):
                    batch_list_idx = list()
                    batch_list = list()
                    for i, can in enumerate(persona_batch):
                        if can == True:
                            batch_list_idx.append(can)
                            persona_selected_now = persona_candidates[batch_idx][i]
                            mask_persona = torch.ne(persona_selected_now, padding)
                            persona_selected_now = torch.masked_select(persona_selected_now, mask_persona)
                            batch_list.append(persona_selected_now[:-2])
                    all_persona_pred.append(batch_list)
                    selected_persona_idx.append(batch_list_idx)
                p_index_cvtd = persona_pred_sigmoid


                softmax = Softmax(dim=-1)
                knowledge_pred = softmax(knowledge_logits)
                _, k_index_1 = torch.topk(knowledge_pred, k=1, dim=-1)
                all_knowledge_pred = []
                for batch_i in range(args.test_batch_size):
                    knowledge_pred_idx = k_index_1[batch_i]
                    knowledge_pred = knowledge_candidates[batch_i][knowledge_pred_idx]
                    mask_knowledge = torch.ne(knowledge_pred, padding)
                    knowledge_pred = torch.masked_select(knowledge_pred, mask_knowledge)
                    knowledge_pred = knowledge_pred[1:-2]
                    all_knowledge_pred.append(knowledge_pred) #delete bos, knowledge_st, eos


                k_index_1 = k_index_1.squeeze(0)
                k_index_cvtd = torch.tensor([1 if num in k_index_1 else 0 for num in range(10)], device=args.device)

                final_input_list = []
                final_input_tti_list = []
                for batch_i in range(args.test_batch_size):
                    only_dial_input_ids_batch = dialog[batch_i]
                    only_dial_token_type_ids_batch = dialog_tti[batch_i]
                    mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
                    mask_only_dial_tti_batch = torch.ne(only_dial_token_type_ids_batch, padding)
                    only_dial_input_ids_batch = torch.masked_select(only_dial_input_ids_batch, mask_only_dial_input_ids_batch)
                    only_dial_token_type_ids_batch = torch.masked_select(only_dial_token_type_ids_batch, mask_only_dial_tti_batch)

                    if len(all_persona_pred[batch_i]) > 0:
                        concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                        new_persona = torch.cat([persona_tensor, concat_persona], dim=-1)
                        new_persona_tti = torch.tensor([persona] * (new_persona.size()[0])).cuda(device)

                    else:
                        new_persona = None
                        new_persona_tti = None


                    new_knowledge = torch.cat([knowledge_tensor, all_knowledge_pred[batch_i]], dim=-1)
                    new_knowledge_tti = torch.tensor([knowledge] * (new_knowledge.size()[0])).cuda(device)

                    only_dial_input_ids_batch = only_dial_input_ids_batch[1:-1]
                    only_dial_token_type_ids_batch = only_dial_token_type_ids_batch[1:]
                    if new_persona is not None:
                        new_input = torch.cat([bos_tensor, new_knowledge, new_persona, only_dial_input_ids_batch, machine_tensor], dim=-1)
                        new_input_tti = torch.cat([knowledge_tensor, new_knowledge_tti, new_persona_tti, only_dial_token_type_ids_batch], dim=-1)
                    else:
                        new_input = torch.cat([bos_tensor, new_knowledge, only_dial_input_ids_batch, machine_tensor], dim=-1)
                        new_input_tti = torch.cat([knowledge_tensor, new_knowledge_tti, only_dial_token_type_ids_batch], dim=-1)

                    final_input_list.append(new_input)
                    final_input_tti_list.append(new_input_tti)
                final_input_tensor = torch.stack(final_input_list)
                final_input_tti_tensor = torch.stack(final_input_tti_list)

                out_ids = sample_sequence(final_input_tensor, token_type_ids=final_input_tti_tensor, decoder_input_ids=None, tokenizer=tokenizer, model=model, args=args, current_output=None)

            elif model.config.model_type == 'bart':
                output = model(
                    input_ids=input_ids,
                    input_eos=input_eos,
                    only_dial_input_ids=dialog,
                    decoder_input_ids=decoder_input_ids,
                    persona_input_ids=persona_candidates,
                    knowledge_input_ids=knowledge_candidates,
                    persona_can_idx=persona_can_idx,
                    knowledge_can_idx=knowledge_can_idx,
                    tot_knowledge=tot_knowledge,
                    tot_knowledge_eos=tot_knowledge_eos,
                    training=False,
                    mc_token_ids=mc_token_ids
                )
                lm_logits, knowledge_logits, persona_logits = output[0], output[1], output[2]

                persona, knowledge = 50267, 50268
                bos, padding, eos = 0, 1, 2
                device = input_ids.get_device()

                persona_tensor = torch.tensor([persona]).cuda(device)
                knowledge_tensor = torch.tensor([knowledge]).cuda(device)
                bos_tensor = torch.tensor([bos]).cuda(device)
                eos_tensor = torch.tensor([eos]).cuda(device)
                max_position = 1024

                sigmoid = Sigmoid()
                persona_pred_sigmoid = sigmoid(persona_logits)
                persona_pred_sigmoid = (persona_pred_sigmoid > 0.5).float()
                all_persona_pred = []
                selected_persona_idx = list()
                for batch_idx, persona_batch in enumerate(torch.eq(persona_pred_sigmoid, 1)):
                    batch_list_idx = list()
                    batch_list = list()
                    for i, can in enumerate(persona_batch):
                        if can == True:
                            batch_list_idx.append(can)
                            persona_selected_now = persona_candidates[batch_idx][i]
                            mask_persona = torch.ne(persona_selected_now, padding)
                            persona_selected_now = torch.masked_select(persona_selected_now, mask_persona)
                            batch_list.append(persona_selected_now[:-2])
                    all_persona_pred.append(batch_list)
                    selected_persona_idx.append(batch_list_idx)

                p_index_cvtd = persona_pred_sigmoid

                softmax = Softmax(dim=-1)
                knowledge_softmax = softmax(knowledge_logits)
                _, k_index_1 = torch.topk(knowledge_softmax, k=1, dim=-1)
                all_knowledge_pred = []
                for batch_i in range(args.test_batch_size):
                    knowledge_pred_idx = k_index_1[batch_i]
                    knowledge_pred = knowledge_candidates[batch_i][knowledge_pred_idx]
                    mask_knowledge = torch.ne(knowledge_pred, padding)
                    knowledge_pred = torch.masked_select(knowledge_pred, mask_knowledge)
                    knowledge_pred = knowledge_pred[1:-2]
                    all_knowledge_pred.append(knowledge_pred) #delete bos, knowledge_st, eos

                k_index_1 = k_index_1.squeeze(0)
                k_index_cvtd = torch.tensor([1 if num in k_index_1 else 0 for num in range(10)], device=args.device)

                final_input_list = []
                for batch_i in range(args.test_batch_size):
                    only_dial_input_ids_batch = dialog[batch_i]
                    mask_only_dial_input_ids_batch = torch.ne(only_dial_input_ids_batch, padding)
                    only_dial_input_ids_batch = torch.masked_select(only_dial_input_ids_batch, mask_only_dial_input_ids_batch)
                    if len(all_persona_pred[batch_i])>0:
                        concat_persona = torch.cat(all_persona_pred[batch_i], dim=-1)
                        new_persona = torch.cat([persona_tensor, concat_persona], dim=-1)
                    else:
                        new_persona = None
                    new_knowledge = torch.cat([knowledge_tensor, all_knowledge_pred[batch_i]], dim=-1)

                    if new_persona is not None:
                        new_input = torch.cat([bos_tensor, new_knowledge, new_persona, only_dial_input_ids_batch, eos_tensor], dim=-1)
                    else:
                        new_input = torch.cat([bos_tensor, new_knowledge, only_dial_input_ids_batch, eos_tensor], dim=-1)
                    new_input_size = new_input.size()[0]

                    if new_input_size < int(max_position) :
                        padding_size = int(max_position) -new_input_size
                        add_padding = torch.tensor([padding]*padding_size).cuda(device)
                        final_input = torch.cat([new_input, add_padding], dim=-1)
                    final_input_list.append(final_input)
                final_input_tensor = torch.stack(final_input_list)
                decoder_input_ids = bos_tensor.unsqueeze(0)
                out_ids = sample_sequence(final_input_tensor, token_type_ids=None, decoder_input_ids=decoder_input_ids, tokenizer=tokenizer, model=model, args=args, current_output=None)

            machine, human, persona, knowledge = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
            special_tokens_list = [machine, human, persona, knowledge, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]


            gold_reply = reply

            for special_token in special_tokens_list:
                    out_ids = [value for value in out_ids if value != special_token]
            gold_reply = gold_reply.tolist()
            pred_reply = out_ids

            #ROUGE
            r1.update((pred_reply, [gold_reply]))
            r2.update((pred_reply, [gold_reply]))
            rl.update((pred_reply, [gold_reply]))
            r1_res = r1.compute()
            r2_res = r2.compute()
            rl_res = rl.compute()

            #BLEU1,2,3,4 / BLEU avg
            b1.update((pred_reply, [gold_reply]))
            b2.update((pred_reply, [gold_reply]))
            b3.update((pred_reply, [gold_reply]))
            b4.update((pred_reply, [gold_reply]))
            b1_res = b1.compute()
            b2_res = b2.compute()
            b3_res = b3.compute()
            b4_res = b4.compute()

            #CharF1
            tensor_pred = torch.tensor(pred_reply).type(torch.cuda.FloatTensor)
            tensor_gold = torch.tensor(gold_reply).type(torch.cuda.FloatTensor)
            pre.update((tensor_pred, tensor_gold))
            rec.update((tensor_pred, tensor_gold))
            pre_res = pre.compute()
            rec_res = rec.compute()

            # PG
            p_label_cvtd = torch.tensor([1 if num in persona_grounding else 0 for num in range(5)], device=args.device)
            pg.update((p_index_cvtd.squeeze(), p_label_cvtd))
            pg_res = pg.compute()

            # KG
            k_label_cvtd = torch.tensor([1 if num in knowledge_grounding else 0 for num in range(10)], device=args.device)
            kg.update((k_index_cvtd, k_label_cvtd))
            kg_res = kg.compute()

            bleu_res = (b1_res.item() + b2_res.item() + b3_res.item() + b4_res.item())/4

            precision = pre_res.item()
            recall = rec_res.item()
            f1_res = (1.0 + 1 ** 2) * precision * recall / (1 ** 2 * precision + recall + 1e-15)

        print("F1: ", f1_res)
        print("ROUGE1", r1_res)
        print("ROUGE2", r2_res)
        print("ROUGEL", rl_res)
        print("avg BLEU: ", bleu_res)
        print("PG: ", pg_res)
        print("KG: ", kg_res)


if __name__ == "__main__":
    run()
