from typing import Dict, Union, Optional, List
import logging
from overrides import overrides
import torch
from pytorch_pretrained_bert.modeling import BertModel

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, FeedForward
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from allennlp.nn import InitializerApplicator, RegularizerApplicator, Activation
from metric.SentACC import SentenceAccuracy
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
from allennlp.modules.scalar_mix import ScalarMix
logger = logging.getLogger(__name__)

@Model.register("bert_multitask")
class Bert_Multitask(Model):
    """
    An AllenNLP Model that runs pretrained BERT,
    takes the pooled output, and adds a Linear layer on top.
    If you want an easy way to use BERT for classification, this is it.
    Note that this is a somewhat non-AllenNLP-ish model architecture,
    in that it essentially requires you to use the "bert-pretrained"
    token indexer, rather than configuring whatever indexing scheme you like.

    See `allennlp/tests/fixtures/bert/bert_for_classification.jsonnet`
    for an example of what your config might look like.

    Parameters
    ----------
    vocab : ``Vocabulary``

    initializer : ``InitializerApplicator``, optional
        If provided, will be used to initialize the final linear layer *only*.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 bert_model: Union[str, BertModel],
                 index: str = "bert",
                 trainable: bool = True,
                 dropout_prob: float = 0.0,
                 domain_feedforward: FeedForward = None,
                 intent_feedforward: FeedForward = None,
                 slot_feedforward: FeedForward = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Bert_Multitask, self).__init__(vocab, regularizer)

        if isinstance(bert_model, str):
            self.bert_model = PretrainedBertModel.load(bert_model)
        else:
            self.bert_model = bert_model
        for param in self.bert_model.parameters():
            param.requires_grad = trainable

        self.embedder_dim = self.bert_model.config.hidden_size




        self.num_domain_labels = self.vocab.get_vocab_size("domain_labels")
        self.num_intent_labels = self.vocab.get_vocab_size("intent_labels")
        self.num_slot_labels = self.vocab.get_vocab_size("slot_labels")


        self.domain_feedforward = domain_feedforward or \
                                    FeedForward(self.embedder_dim, 1,
                                                self.num_domain_labels,
                                                Activation.by_name('relu')())
        self.intent_feedforward = intent_feedforward or \
                                    FeedForward(self.embedder_dim, 1,
                                                self.num_intent_labels,
                                                Activation.by_name('relu')())
        self.slot_feedforward = slot_feedforward or \
                                    FeedForward(self.embedder_dim, 1,
                                                self.num_slot_labels,
                                                Activation.by_name('relu')())

        self._dropout = torch.nn.Dropout(dropout_prob)

        self._domain_loss = torch.nn.CrossEntropyLoss()
        self._intent_loss = torch.nn.CrossEntropyLoss()
        self._domain_acc = CategoricalAccuracy()
        self._intent_acc = CategoricalAccuracy()
        self._slot_f1 = SpanBasedF1Measure(vocab, tag_namespace="slot_labels")
        self._sent_acc = SentenceAccuracy()
        self._index = index
        # self.seq_scalar_mix = ScalarMix(12, do_layer_norm=False)
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                domain_labels: torch.LongTensor = None,
                intent_labels: torch.LongTensor = None,
                slot_labels: torch.LongTensor = None,
                metadata: List[str] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField`` (that has a bert-pretrained token indexer)
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        input_ids = tokens[self._index]
        token_type_ids = tokens[f"{self._index}-type-ids"]
        input_mask = (input_ids != 0).long()
        # [batch, layer, seq_len, dim]
        sequence_output, pooled = self.bert_model(input_ids=input_ids,
                                                  token_type_ids=token_type_ids,
                                                  attention_mask=input_mask)
        # sequence_output = sequence_output[-1]
        sequence_output = sequence_output[-1]
        sequence_output = sequence_output[:, 1:-1, :]


        pooled = self._dropout(pooled)
        # print(pooled.shape)
        # apply classification layer
        domain_logits = self.domain_feedforward(pooled)
        domain_probs = torch.nn.functional.softmax(domain_logits, dim=-1)

        intent_logits = self.intent_feedforward(pooled)
        intent_probs = torch.nn.functional.softmax(intent_logits, dim=-1)

        slot_logits = self.slot_feedforward(sequence_output)
        slot_probs = torch.nn.functional.softmax(slot_logits, dim=-1)
        output_dict = {"domain_logits": domain_logits, "domain_probs": domain_probs, \
                       "intent_logits": intent_logits, "intent_probs": intent_probs, \
                       "slot_logits": slot_logits, "slot_probs": slot_probs}

        if domain_labels is not None:
            domain_loss = self._domain_loss(domain_logits, domain_labels.long().view(-1))
            output_dict["domain_loss"] = domain_loss
            # print(domain_loss)
            self._domain_acc(domain_logits, domain_labels)

        if intent_labels is not None:
            intent_loss = self._intent_loss(intent_logits, intent_labels.long().view(-1))
            output_dict["intent_loss"] = intent_loss
            # print(intent_loss)
            self._intent_acc(intent_logits, intent_labels)

        if slot_labels is not None:

            slot_loss = sequence_cross_entropy_with_logits(slot_logits, slot_labels, input_mask[:, 1:-1])
            output_dict["slot_loss"] = slot_loss
            # print(slot_loss)
            self._slot_f1(slot_logits, slot_labels)

        if domain_labels is not None and intent_labels is not None and slot_labels is not None:
            output_dict["loss"] = output_dict["domain_loss"] + output_dict["intent_loss"] + output_dict["slot_loss"]
            self._sent_acc(domain_logits, domain_labels, intent_logits, intent_labels, slot_logits, slot_labels, input_mask[:, 1:-1])
        # print(output_dict['loss'])
        output_dict["metadata"] = metadata
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        domain_pred = output_dict["domain_probs"].cpu().detach().numpy()
        if domain_pred.ndim == 2:
            domain_pred_list = [domain_pred[i] for i in range(domain_pred.shape[0])]
        else:
            domain_pred_list = [domain_pred]
        classes = []
        for pred in domain_pred_list:
            label_idx = pred.argmax(-1)
            label_str = self.vocab.get_token_from_index(label_idx, "domain_labels")
            classes.append(label_str)
        output_dict['domain_labels'] = classes
        intent_pred = output_dict["intent_probs"].cpu().detach().numpy()
        if intent_pred.ndim == 2:
            intent_pred_list = [intent_pred[i] for i in range(intent_pred.shape[0])]
        else:
            intent_pred_list = [intent_pred]
        classes = []
        for pred in intent_pred_list:
            label_idx = pred.argmax(-1)
            label_str = self.vocab.get_token_from_index(label_idx, "intent_labels")
            classes.append(label_str)
        output_dict["intent_labels"] = classes

        slot_pred = output_dict["slot_probs"].cpu().detach().numpy()
        if slot_pred.ndim == 3:
            slot_pred_list = [slot_pred[i] for i in range(slot_pred.shape[0])]
        else:
            slot_pred_list = [slot_pred]
        all_tags = []
        metadata = output_dict["metadata"]
        for idx, pred in enumerate(slot_pred_list):
            label_idx = pred.argmax(-1)
            tags = [self.vocab.get_token_from_index(x, namespace="slot_labels")
                    for x in label_idx]
            spans = bio_tags_to_spans(tags)
            tmp = {}
            print("tags")
            print(tags)
            print("spans")
            print(spans)
            for (tag, (bg, ed)) in spans:
                tmp[tag] = metadata[idx][bg:ed+1]
            all_tags.append(tmp)
        output_dict["slot_labels"] = all_tags
        # print(output_dict['domain_labels'])
        # print(output_dict['intent_labels'])
        # print(output_dict['slot_labels'])
        # print(output_dict['metadata'])
        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        metrics["domain_acc"] = self._domain_acc.get_metric(reset)
        # print(metrics["domain_acc"])
        metrics["intent_acc"] = self._intent_acc.get_metric(reset)
        # print(metrics["intent_acc"])
        metrics["slot_f1"] = self._slot_f1.get_metric(reset)["f1-measure-overall"]
        # print(metrics["slot_f1"])
        metrics["sent_acc"] = self._sent_acc.get_metric(reset)
        # print(metrics["sent_acc"])
        return metrics