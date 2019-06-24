from typing import Dict, List, Sequence, Iterable
import itertools
import logging
import json
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers import Tokenizer, CharacterTokenizer
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

@DatasetReader.register('NLU')
class NLUReader(DatasetReader):
	"""
	NLU
	"""
	def __init__(self,
				 tokenizer: Tokenizer = None,
				 token_indexers: Dict[str, TokenIndexer] = None
				 ) -> None:
		super().__init__(lazy=False)

		self._tok = tokenizer or CharacterTokenizer()
		if 'bert' in token_indexers:
			wordpiece_tok = token_indexers['bert'].wordpiece_tokenizer
			token_indexers['bert'].wordpiece_tokenizer = \
				lambda s: ['[UNK]'] if s.isspace() else wordpiece_tok(s)
		self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

	def text_to_instance(self,
						 text: str,
						 domain: str,
						 intent: str,
						 slots: dict,
						 ):
		fields = {}
		tok_text = self._tok.tokenize(text)
		text_field = TextField(tok_text, self._token_indexers)
		fields['tokens'] = text_field
		if type(domain) == float:
			domain = str(domain)
		if type(intent) == float:
			intent = str(intent)
		if domain:
			fields['domain_labels'] = LabelField(domain, label_namespace='domain_labels')
		if intent:
			fields['intent_labels'] = LabelField(intent, label_namespace='intent_labels')
		#process for BIO
		if slots:
			slot_labels = ['O'] * len(tok_text)
			for key in slots:
				value = slots[key]
				idx = text.find(value)
				if idx == -1:
					print(text + '\n' + value)
					continue
				slot_labels[idx] = 'B-' + str(key)
				for i in range(idx+1, idx+len(value)):
					slot_labels[i] = 'I-' + str(key)
			fields['slot_labels'] = SequenceLabelField(labels=slot_labels, sequence_field=text_field, label_namespace='slot_labels')
		meta_fields_list = [x.text for x in tok_text]
		meta_fields_str = ''.join(meta_fields_list)
		fields['metadata'] = MetadataField(meta_fields_str)
		return Instance(fields)

	def _read(self, file_path: str) -> Iterable[Instance]:
		data = []
		# print("fuck")
		with open(file_path, "r") as fr:
			data = json.load(fr)
			for instance in data:
				text = instance["text"]
				domain = instance.get("domain")
				intent = instance.get("intent")
				slots = instance.get("slots")
				yield self.text_to_instance(text, domain, intent, slots)


if __name__ == "__main__":
	tmp = NLUReader()
	tmp.read("data/train.json")
