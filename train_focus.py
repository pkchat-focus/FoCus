#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
import torch
from torch.nn import Sigmoid, Softmax
from torch.nn.parallel import DistributedDataParallel
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss, MetricsLambda, RunningAverage, Precision, CharFbeta, Recall, Accuracy
from ignite.metrics import Bleu, RougeL, RougeN
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, WEIGHTS_NAME, CONFIG_NAME)
from utils_focus import make_focus_logdir
from data_utils import get_data_loaders, add_special_tokens_

logger = logging.getLogger(__file__)


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def train():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="",
                        help="{GPT2, BART, transformer-decoder, transformer-encdec}")
    parser.add_argument("--gpt2_model_path", type=str, default="gpt2",
                        help="pre-trained model path for decoder only models")  # gpt2-medium
    parser.add_argument("--bart_model_path", type=str, default="facebook/bart-base",
                        help="pre-trained model path for encoder-decoder models")  # facebook/bart-large
    parser.add_argument("--train_dataset_path", type=str, default="data/train_focus.json",
                        help="Path or url of the dataset.")
    parser.add_argument("--train_dataset_cache", type=str, default='data/focus_cache.tar.gz',
                        help="Path or url of the dataset cache")
    parser.add_argument("--dev_dataset_path", type=str, default="data/valid_focus.json",
                        help="Path or url of the dataset.")
    parser.add_argument("--dev_dataset_cache", type=str, default='data/focus_cache.tar.gz',
                        help="Path or url of the dataset cache")
    parser.add_argument("--ps_coef", type=float, default=1.0, help="Coefficient for persona loss")
    parser.add_argument("--kn_coef", type=float, default=1.0, help="Coefficient for knowledge loss")
    parser.add_argument("--lm_coef", type=float, default=10.0, help="Coefficient for LM loss")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--inference", action='store_true', help="If true, inference with gold knowledge")
    parser.add_argument("--test_infer", action='store_true', help="If true, test inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--gpu_start_num", type=int, default=1, help="Start number of GPU")
    parser.add_argument("--flag", type=str, default="", help="Assign the name of the folder")
    parser.add_argument("--seed", type=int, default=19950604)
    parser.add_argument("--random_knowledge", action='store_true',
                        help="If true, the model choose the knowledge randomly")
    parser.add_argument("--incontext", action='store_true', help="If true, it will use incontext structure")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("Arguments: %s", pformat(args))

    args.distributed = (args.local_rank != -1)
    if args.distributed:
        local_rank = args.local_rank + args.gpu_start_num
        print("args local rank: ", args.local_rank, " local rank: ", local_rank)
        torch.cuda.set_device(local_rank)
        args.device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer.")

    if args.model_name == 'GPT2':
        from transformers import GPT2Tokenizer
        from classification_modules import GPT2PK_ctxt as gpt2model
        tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_model_path)
        model = gpt2model.from_pretrained(args.gpt2_model_path)
        model.to(args.device)
        model.eval()
        if args.gpt2_model_path == 'gpt2' or 'gpt2-medium':
            add_special_tokens_(model, tokenizer)

    elif args.model_name == 'BART':
        from transformers import BartTokenizer
        from classification_modules import BARTPK_ctxt as bartmodel
        tokenizer = BartTokenizer.from_pretrained(args.bart_model_path)
        model = bartmodel.from_pretrained(args.bart_model_path)
        model.to(args.device)
        model.eval()
        if args.bart_model_path == "facebook/bart-base" or "facebook/bart-large":
            add_special_tokens_(model, tokenizer)

    elif args.model_name == 'transformer-decoder':
        from transformers import GPT2Tokenizer, GPT2Config
        from classification_modules import GPT2PK_ctxt as gpt2model
        tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_model_path)
        model_config = GPT2Config.from_pretrained(args.gpt2_model_path)
        model = gpt2model(model_config)
        model.to(args.device)
        if args.gpt2_model_path == 'gpt2' or 'gpt2-medium':
            add_special_tokens_(model, tokenizer)

    elif args.model_name == 'transformer-encdec':
        from transformers import BartTokenizer, BartConfig
        from classification_modules import BARTPK_ctxt as bartmodel
        tokenizer = BartTokenizer.from_pretrained(args.bart_model_path)
        model_config = BartConfig.from_pretrained(args.bart_model_path)
        model = bartmodel(model_config)
        model.to(args.device)
        if args.bart_model_path == "facebook/bart-base" or "facebook/bart-large":
            add_special_tokens_(model, tokenizer)

    else:
        raise NotImplementedError

    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        if model.config.model_type == 'gpt2':
            input_ids, input_eos, lm_labels, token_type_ids, mc_token_ids, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
            knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_token_ids, tot_knowledge_eos, reply, dialog, dialog_tti = batch
            output = model(
                input_ids=input_ids,
                input_eos=input_eos,
                token_type_ids=token_type_ids,
                only_dial_input_ids=dialog,
                only_dial_token_type_ids=dialog_tti,
                persona_input_ids=persona_candidates,
                knowledge_input_ids=knowledge_candidates,
                persona_can_idx=persona_can_idx,
                persona_grounding=persona_grounding,
                knowledge_can_idx=knowledge_can_idx,
                knowledge_grounding=knowledge_grounding,
                tot_knowledge=tot_knowledge,
                tot_knowledge_token_ids=tot_knowledge_token_ids,
                tot_knowledge_eos=tot_knowledge_eos,
                training=True,
                mc_token_ids=mc_token_ids
            )
        elif model.config.model_type == 'bart':
            input_ids, input_eos, decoder_input_ids, lm_labels, token_type_ids, mc_token_ids, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
            knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_eos, reply, dialog = batch
            output = model(
                input_ids=input_ids,
                input_eos=input_eos,
                only_dial_input_ids=dialog,
                decoder_input_ids=decoder_input_ids,
                persona_input_ids=persona_candidates,
                knowledge_input_ids=knowledge_candidates,
                persona_can_idx=persona_can_idx,
                persona_grounding=persona_grounding,
                knowledge_can_idx=knowledge_can_idx,
                knowledge_grounding=knowledge_grounding,
                tot_knowledge=tot_knowledge,
                tot_knowledge_eos=tot_knowledge_eos,
                lm_labels=lm_labels,
                training=True,
                mc_token_ids=mc_token_ids
            )
        else:
            raise NotImplementedError
        lm_loss, knowledge_loss, persona_loss = output[0], output[1], output[2]
        loss = (lm_loss * args.lm_coef + knowledge_loss * args.kn_coef + persona_loss * args.ps_coef) / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return (lm_loss.item(), knowledge_loss.item(), persona_loss.item())

    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            if model.config.model_type == 'gpt2':
                input_ids, input_eos, lm_labels, token_type_ids, mc_token_ids, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
                knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_token_ids, tot_knowledge_eos, reply, dialog, dialog_tti = batch

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


            elif model.config.model_type == 'bart':
                input_ids, input_eos, decoder_input_ids, lm_labels, token_type_ids, mc_token_ids, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
                knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_eos, reply, dialog = batch
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


            else:
                raise NotImplementedError

            lm_logits_flat_shifted = lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[:, 1:].contiguous().view(-1)

            persona_logits = persona_logits.squeeze()
            persona_grounding = persona_grounding.type_as(persona_logits).squeeze()

            sigmoid = Sigmoid()
            persona_pred_sigmoid = sigmoid(persona_logits)
            persona_pred_sigmoid = (persona_pred_sigmoid > 0.5).float()

            softmax = Softmax(dim=-1)
            knowledge_pred = softmax(knowledge_logits)
            _, k_index_1 = torch.topk(knowledge_pred, k=1, dim=-1)
            _, k_index_5 = torch.topk(knowledge_pred, k=5, dim=-1)
            k_index_1, k_index_5 = k_index_1.squeeze(0), k_index_5.squeeze(0)
            k_index_1_cvtd = torch.tensor([1 if num in k_index_1 else 0 for num in range(10)], device=args.device)
            k_index_5_cvtd = torch.tensor([1 if num in k_index_5 else 0 for num in range(10)], device=args.device)
            k_label_cvtd = torch.tensor([1 if num in knowledge_grounding else 0 for num in range(10)],
                                        device=args.device)
            persona_pred = softmax(persona_logits)
            _, p_index_1 = torch.topk(persona_pred, k=1, dim=-1)
            _, p_index_5 = torch.topk(persona_pred, k=5, dim=-1)
            p_index_1, p_index_5 = p_index_1.squeeze(0), p_index_5.squeeze(0)
            p_index_1_cvtd = torch.tensor([1 if num in p_index_1 else 0 for num in range(11)], device=args.device)
            p_index_5_cvtd = torch.tensor([1 if num in p_index_5 else 0 for num in range(11)], device=args.device)
            p_label_cvtd = torch.tensor([1 if num in persona_grounding else 0 for num in range(11)], device=args.device)

            lm_pred = softmax(lm_logits_flat_shifted)
            lm_val, lm_idx = torch.topk(lm_pred, k=1, dim=-1)
            lm_idx = lm_idx.squeeze(-1)

            mask = (lm_labels_flat_shifted != -100)
            lm_labels_only = [lm_labels_flat_shifted[mask].tolist()]
            lm_idx_only = lm_idx[mask].tolist()

            return (lm_logits_flat_shifted, knowledge_logits, persona_logits, persona_pred_sigmoid, k_index_1_cvtd,
                    k_index_5_cvtd, p_index_1_cvtd, p_index_5_cvtd, lm_idx_only), \
                   (lm_labels_flat_shifted, knowledge_grounding, persona_grounding.type_as(persona_logits), k_label_cvtd,
                   p_label_cvtd, lm_labels_only)

    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "lm_loss")
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "knowledge_loss")
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, "persona_loss")

    metrics = {
        "lm_loss": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0])),
        "knowledge_loss": Loss(torch.nn.CrossEntropyLoss(), output_transform=lambda x: (x[0][1], x[1][1])),
        "persona_loss": Loss(torch.nn.BCEWithLogitsLoss(), output_transform=lambda x: (x[0][2], x[1][2])),
        "Knowledge_acc": Accuracy(output_transform=lambda x: (x[0][4], x[1][3])),
        "Persona_acc":Accuracy(output_transform=lambda x:(x[0][3], x[1][2])),
        "BLEU1": Bleu(ngram=1, output_transform=lambda x: (x[0][8], x[1][5])),
        "BLEU2": Bleu(ngram=2, output_transform=lambda x: (x[0][8], x[1][5])),
        "BLEU3": Bleu(ngram=3, output_transform=lambda x: (x[0][8], x[1][5])),
        "BLEU4": Bleu(ngram=4, output_transform=lambda x: (x[0][8], x[1][5])),
        "Rouge1": RougeN(ngram=1, output_transform=lambda x: (x[0][8], x[1][5])),
        "Rouge2": RougeN(ngram=2, output_transform=lambda x: (x[0][8], x[1][5])),
        "RougeL": RougeL(output_transform=lambda x: (x[0][8], x[1][5])),
        "F1": CharFbeta(beta=1, output_transform=lambda x: (torch.tensor(x[0][8]), torch.tensor(x[1][5][0])))}

    metrics.update({"average_lm_loss": MetricsLambda(average_distributed_scalar, metrics["lm_loss"], args),
                    "average_knowledge_loss": MetricsLambda(average_distributed_scalar, metrics["knowledge_loss"],args),
                    "average_persona_loss": MetricsLambda(average_distributed_scalar, metrics["persona_loss"], args),
                    "average_Knowledge_acc": MetricsLambda(average_distributed_scalar, metrics["Knowledge_acc"], args),
                    "average_Persona_acc": MetricsLambda(average_distributed_scalar, metrics["Persona_acc"], args),
                    "average_BLEU1": MetricsLambda(average_distributed_scalar, metrics["BLEU1"], args),
                    "average_BLEU2": MetricsLambda(average_distributed_scalar, metrics["BLEU2"], args),
                    "average_BLEU3": MetricsLambda(average_distributed_scalar, metrics["BLEU3"], args),
                    "average_BLEU4": MetricsLambda(average_distributed_scalar, metrics["BLEU4"], args),
                    "average_Rouge1": MetricsLambda(average_distributed_scalar, metrics["Rouge1"], args),
                    "average_Rouge2": MetricsLambda(average_distributed_scalar, metrics["Rouge2"], args),
                    "average_RougeL": MetricsLambda(average_distributed_scalar, metrics["RougeL"], args),
                    "average_F1": MetricsLambda(average_distributed_scalar, metrics["F1"], args)
                    })

    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_lm_loss"])

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["lm_loss"])
        evaluator.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        dir_name = str(os.path.basename(__file__))[:-3] + "_" + args.model_name + "_" + args.flag
        log_dir = make_focus_logdir(dir_name)
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["lm_loss", "knowledge_loss", "persona_loss",
                                                                          "knowledge_accuracy", "persona_accuracy", "f1_score"]),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys())),
                         event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys())),
                         event_name=Events.EPOCH_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys())),
                         event_name=Events.ITERATION_COMPLETED(every=5000))
        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))
        tb_logger.close()


if __name__ == "__main__":
    train()
