from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("sentence_accuracy")
class SentenceAccuracy(Metric):

	def __init__(self) -> None:
		self.correct_count = 0.0
		self.total_count = 0.0

	def __call__(self,
				 domain_pred: torch.Tensor = None,
				 domain_gold: torch.Tensor = None,
				 intent_pred: torch.Tensor = None,
				 intent_gold: torch.Tensor = None,
				 slot_pred: torch.Tensor = None,
				 slot_gold: torch.Tensor = None,
				 mask: Optional[torch.Tensor] = None):
		"""
		Parameters
		----------
		predictions : ``torch.Tensor``, required.
			A tensor of predictions of shape (batch_size, k, sequence_length).
		gold_labels : ``torch.Tensor``, required.
			A tensor of integer class label of shape (batch_size, sequence_length).
		mask: ``torch.Tensor``, optional (default = None).
			A masking tensor the same size as ``gold_labels``.
		"""
		domain_pred, domain_gold, intent_pred, intent_gold, slot_pred, slot_gold, mask = \
			self.unwrap_to_tensors(domain_pred, domain_gold, intent_pred, intent_gold, \
								   slot_pred, slot_gold, mask)

		domain_acc = (torch.argmax(domain_pred, -1) == domain_gold)
		intent_acc = (torch.argmax(intent_pred, -1) == intent_gold)

		slot_acc = (torch.argmax(slot_pred, dim=-1) * mask == slot_gold * mask)
		slot_acc = torch.all(slot_acc, dim=-1)
		sent_acc = domain_acc * intent_acc * slot_acc

		self.total_count += domain_pred.size()[0]
		self.correct_count += sent_acc.sum()


	def get_metric(self, reset: bool = False):
		"""
		Returns
		-------
		The accumulated accuracy.
		"""
		if self.total_count > 0:
			accuracy = float(self.correct_count) / float(self.total_count)
		else:
			accuracy = 0

		if reset:
			self.reset()
		return accuracy


	@overrides
	def reset(self):
		self.correct_count = 0.0
		self.total_count = 0.0
