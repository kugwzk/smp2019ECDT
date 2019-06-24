local bert_model = "chinese_L-12_H-768_A-12/";
local NUM_EPOCHS = 10;
{
  "random_seed": 1,
  "numpy_seed": 1,
  "pytorch_seed": 1,
  "dataset_reader":{
    "type": "NLU",
    "token_indexers": {
      "bert":{
    	 "type": "bert-pretrained",
    	 "pretrained_model": bert_model
      }
    }
  },
  "train_data_path": "sample.json",
  "validation_data_path": "sample.json",
  "test_data_path": "sample.json",
  "model": {
    "type": "bert_multitask",
    "bert_model": bert_model,
    "dropout_prob": 0.2,
//    "initializer": [
//      [".*feedforward.*weight", {"type": "xavier_uniform"}],
//      [".*feedforward.*bias", {"type": "zero"}]
//    ]
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": 16,
  },
  "trainer": {
        "validation_metric": "+sent_acc",
        "num_serialized_models_to_keep": 1,
        "num_epochs": NUM_EPOCHS,
        "grad_clipping": 5.0,
        "patience": 6,
        "cuda_device": 0,
        "optimizer": {
          "type": "adam",
          "lr": 5e-5
        }
    }
}