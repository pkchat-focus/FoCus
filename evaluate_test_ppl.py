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

    logger.info("Prepare datasets")
    test_loader, test_sampler = get_testdata_loaders(args, tokenizer, generation=False)


    with torch.no_grad():


        ppl_avg = []
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
                lm_logits_flat_shifted = lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[:, 1:].contiguous().view(-1)
                lm_loss = loss_fct(lm_logits_flat_shifted, lm_labels_flat_shifted)


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
                lm_logits_flat_shifted = lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.size(-1))
                lm_labels_flat_shifted = lm_labels[:, 1:].contiguous().view(-1)
                lm_loss = loss_fct(lm_logits_flat_shifted, lm_labels_flat_shifted)

            #ppl
            ppl = torch.exp(lm_loss).item()
            ppl_avg.append(ppl)
        print("ppl: ", sum(ppl_avg) / len(ppl_avg))


if __name__ == "__main__":
    run()
