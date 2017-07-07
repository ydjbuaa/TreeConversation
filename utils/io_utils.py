import torch


def load_dataset(train_path, valid_path, vocab_path):
    train_data = torch.load(train_path)
    valid_data = torch.load(valid_path)
    vocabs = torch.load(vocab_path)

    return {"train": train_data, "valid": valid_data, "vocabs": vocabs}


def load_configs(config_path):
    import json
    return json.load(open(config_path, 'r', encoding='utf-8'))


def print_configs(config):
    print("------------------model configurations----------------")
    for key in config:
        print('{}:\t{}'.format(key, config[key]))
    print()
