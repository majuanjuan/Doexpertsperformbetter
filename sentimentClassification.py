import os
import re
import codecs
import logging
import argparse
import configparser
import random
from tqdm import tqdm, trange
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import pandas as pd
import fnmatch

config = configparser.ConfigParser()
config.read('../config.ini')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BertForSmooth(BertPreTrainedModel):

    def __init__(self, config, num_labels=72):
        super(BertForSmooth, self).__init__(config)
        self.num_labels = num_labels
        self.dropout = torch.nn.Dropout(0.2)
        self.bert = BertModel(config)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.loss = torch.nn.CrossEntropyLoss()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids, input_mask, labels=None):
        _, pooled_output = self.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        if labels is not None:
            pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            return self.loss(logits, labels)
        else:
            return logits


class InputExample(object):

    def __init__(self, sentence, label=None):
        self.sentence = sentence
        self.label = label


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label


class DataProcessor(object):

    @staticmethod
    def get_train_examples(data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        examples = []
        for line in open(os.path.join(data_dir, 'sentiment.train'), 'r', encoding="utf8"):
            tokens = line.strip('\n').split('\t')
            examples.append(InputExample("".join(tokens[:-1]), int(tokens[-1])))
        return examples

    @staticmethod
    def get_dev_examples(data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        examples = []
        for line in open(os.path.join(data_dir, 'sentiment.train'), 'r', encoding="utf8"):
            tokens = line.strip('\n').split('\t')
            examples.append(InputExample("".join(tokens[:-1]), int(tokens[-1])))
        return examples

    @staticmethod
    def get_test_examples(eachFileName):
        """Gets a collection of `InputExample`s for prediction."""
        ids, examples, publish_date, user_id = [], [], [], []
        i = 0
        currentFilename = '../dataset/sentiment.test.' + eachFileName
        for line in open(currentFilename, 'r', encoding='utf-8'):
            tokens = line.strip('\n').split('\t')
            if len(tokens) < 3:
                pass
            else:
                examples.append(InputExample("".join(tokens[0])))
                publish_time = tokens[-1]
                publish_date.append(publish_time.split(' ')[0])
                ids.append(i)
                user_id.append(tokens[1])
                i = i + 1
        return ids, examples, publish_date, user_id


def convert_examples_to_features(examples, max_seq_length, tokenizer, has_label=True):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for index, example in enumerate(examples):
        tokens = ['[CLS]']
        if has_label:
            label = example.label
        else:
            label = None
        chars = tokenizer.tokenize(example.sentence)
        if not chars:  # 不可见字符导致返回空列表
            chars = ['[UNK]']
        tokens.extend(chars)
        if len(tokens) > max_seq_length:
            logging.debug('Example {} is too long: {}'.format(index, len(tokens)))
            tokens = tokens[0: max_seq_length]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        zero_padding = [0] * padding_length
        input_ids += zero_padding
        input_mask += zero_padding
        segment_ids += zero_padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label=label))
    return features


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def features_to_tensor(ds_features, need_labels=True):
    ds_input_ids = torch.tensor([f.input_ids for f in ds_features], dtype=torch.long)
    ds_input_mask = torch.tensor([f.input_mask for f in ds_features], dtype=torch.long)
    ds_segment_ids = torch.tensor([f.segment_ids for f in ds_features], dtype=torch.long)
    if need_labels and ds_features[0].label is not None:
        ds_labels = torch.tensor([f.label for f in ds_features], dtype=torch.long)
        ds_data = TensorDataset(ds_input_ids, ds_input_mask, ds_segment_ids, ds_labels)
    else:
        ds_data = TensorDataset(ds_input_ids, ds_input_mask, ds_segment_ids)
    return ds_data


def do_predict(dataloader, model, device):
    model.eval()
    class_probas = []
    predictions = []
    for batch in tqdm(dataloader, desc="Iteration"):
        logger.info(" iterating------------")
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids = batch
        logits = model(input_ids, segment_ids, input_mask)

        class_proba = torch.nn.functional.softmax(logits, 1)
        class_proba = class_proba.detach().cpu().numpy()
        class_probas.extend(class_proba.tolist())
        prediction = np.argmax(class_proba, -1).tolist()
        predictions.extend(prediction)
    return predictions, class_probas


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_dir",
                        default='../dataset',
                        type=str,
                        required=None,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model_dir",
                        default='../models/chinese_L-12_H-768_A-12',
                        type=str, required=None,
                        help="bert pre-trained model dir")
    parser.add_argument("--output_dir",
                        default='../dataset',
                        type=str,
                        required=None,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # Optional parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        default=True,
                        action='store_true',
                        help="Whether to run the model in inference mode on the test set.")
    parser.add_argument("--checkpoint",
                        default=None,
                        type=str,
                        help="Specify the ckeckpoint to load.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--predict_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for predict.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    args = parser.parse_args()

    device = torch.device("cpu")
    use_gpu = False
    logger.info("device: {}".format(device))

    os.makedirs(args.output_dir, exist_ok=True)
    ckpts = [(int(filename.split('-')[1]), filename) for filename in os.listdir(args.output_dir) if
             re.fullmatch('checkpoint-\d+', filename)]
    ckpts = sorted(ckpts, key=lambda x: x[0])
    if args.checkpoint or ckpts:
        if args.checkpoint:
            model_file = args.checkpoint
        else:
            model_file = os.path.join(args.output_dir, ckpts[-1][1])
        logging.info('Load %s' % model_file)
        checkpoint = torch.load(model_file, map_location='cpu')
        global_step = checkpoint['step']
        max_seq_length = checkpoint['max_seq_length']
        lower_case = checkpoint['lower_case']
        model = BertForSmooth.from_pretrained(args.bert_model_dir, state_dict=checkpoint['model_state'])
    else:
        global_step = 0
        max_seq_length = args.max_seq_length
        lower_case = args.do_lower_case
        model = BertForSmooth.from_pretrained(args.bert_model_dir, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
    # 分词器
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir, do_lower_case=lower_case)
    model.to(device)

    # train
    if args.do_train:

        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps))

        args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if use_gpu:
            torch.cuda.manual_seed_all(args.seed)

        train_examples = DataProcessor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion,
                             t_total=num_train_steps)

        train_features = convert_examples_to_features(train_examples, max_seq_length, tokenizer)
        train_data = features_to_tensor(train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        dev_examples = DataProcessor.get_dev_examples(args.data_dir)
        dev_features = convert_examples_to_features(dev_examples, max_seq_length, tokenizer)
        dev_data = features_to_tensor(dev_features, False)
        dev_labels = [example.label for example in dev_examples]
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.predict_batch_size)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        sw = SummaryWriter()  # tensorboard显示数据收集
        top_ckpts = []
        threshold = 0
        start_epoch = int(
            global_step / (len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps))
        residue_step = global_step % (len(
            train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.gradient_accumulation_steps
        model.train()
        for epoch in trange(start_epoch, args.num_train_epochs, desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if epoch == start_epoch and step <= residue_step:
                    continue
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, labels = batch

                loss = model(input_ids, segment_ids, input_mask, labels)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                sw.add_scalar('loss', loss.clone().cpu().data.numpy().mean(), global_step)

                # 并行划分数据到gpu的情况下, 每块gpu都会返回一个loss
                loss = loss.mean()
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_steps,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                if global_step > 20000 and (step + 1) % 2000 == 0:
                    dev_predictions, _ = do_predict(dev_dataloader, model, device)
                    model.train()
                    precision, recall, f1, _ = precision_recall_fscore_support(dev_labels, dev_predictions,
                                                                               average='macro')
                    logger.info(f'global step: {global_step}, F1 value: {f1}')
                    sw.add_scalar('precision', precision, global_step)
                    sw.add_scalar('recall', recall, global_step)
                    sw.add_scalar('f1', f1, global_step)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    torch.save({'step': global_step, 'model_state': model_to_save.state_dict(),
                                'max_seq_length': max_seq_length, 'lower_case': lower_case},
                               os.path.join(args.output_dir, 'checkpoint-%d' % global_step))

                    top_ckpts.append(global_step)
                    if len(top_ckpts) > 5:
                        os.system('rm %s' % os.path.join(args.output_dir, 'checkpoint-%d' % top_ckpts[0]))
                        top_ckpts.pop(0)

    if args.do_predict:
        logger.info(" doing predict ------------")
        patten = '*-weibo.csv'
        weibo_file_list = fnmatch.filter(os.listdir('../dataset'), patten)
        for eachFile in weibo_file_list:
            type = eachFile.replace('-weibo.csv', '')
            ids, predict_examples, publish_date, user_id = DataProcessor.get_test_examples(type)
            predict_features = convert_examples_to_features(predict_examples, max_seq_length, tokenizer, False)
            predict_data = features_to_tensor(predict_features)
            predict_sampler = SequentialSampler(predict_data)
            predict_dataloader = DataLoader(predict_data, sampler=predict_sampler, batch_size=args.predict_batch_size)
            logger.info(" predict start ------------")
            predictions, class_probas = do_predict(predict_dataloader, model, device)
            logger.info(" predict finished ------------")
            # print("len of prediction: " + str(len(predictions)) + "len of class probability: " + str(len(
            # class_probas)))

            eachFileResult = 'ClassificationResult-' + type + '-normal.csv'
            writer = open(os.path.join(args.output_dir, eachFileResult), 'w', encoding='utf-8')
            writer.write('ID,user_id,Text,Date,Expected\n')
            for _id, label in zip(ids, predictions):
                # print("id :"+ str(_id) + "prediction: " +str(predictions[_id]))
                writer.write(str(_id) + ',' + str(user_id[_id]) + ',' + str(predict_examples[_id].sentence.replace(',', '，'))
                             + ',' + str(publish_date[_id]) + ',' + str(label) + '\n')
            writer.close()


if __name__ == "__main__":
    main()
