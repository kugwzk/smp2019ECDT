from typing import List, Callable, Dict
import json

from allennlp.predictors import Predictor

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance


@Predictor.register('bert_multitask')
class Bert_Multitask_Predictor(Predictor):
	# @overrides
	# def _json_to_instance(self, json_dict: JsonDict) -> Instance:
	#     words = json_dict['word']
	#     if isinstance(words, str):
	#         words = words.split()
	#     pos_tags = json_dict['pos']
	#     if isinstance(pos_tags, str):
	#         pos_tags = pos_tags.split()
	#     return self._dataset_reader.text_to_instance(words, pos_tags)
	@overrides
	def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
		"""
		If you don't want your outputs in JSON-lines format
		you can override this function to output them differently.
		"""
		ret = []
		ret.append(outputs)
		return json.dumps(outputs, ensure_ascii=False) + "\n"

	@overrides
	def predict_instance(self, instance: Instance) -> JsonDict:
		outputs = self._model.forward_on_instance(instance)
		ret = {
			'text': outputs['metadata'],
			'domain': outputs['domain_labels'],
			'intent': outputs['intent_labels'],
			'slots': outputs['slot_labels'],
		}
		return sanitize(ret)
