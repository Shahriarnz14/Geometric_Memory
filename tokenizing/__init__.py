import torch
from geometric_memory.tokenizing.numeral_tokenizer import NumeralTokenizer
from transformers import AutoTokenizer


class Tokenizer:
    """Tokenizer definition.
    
    Description:
        Encapsulates related behavior and state.
    """
    def __init__(self, encoder, decoder, vocab_size, name=None):
        """  init  .
        
        Args:
            encoder: Input parameter.
            decoder: Input parameter.
            vocab_size: Input parameter.
            name: Input parameter.
        
        Returns:
            object: Function return value.
        """
        self.encode = encoder
        self.decode = decoder
        self.vocab_size = vocab_size
        self.name = name

    def tokenize(self, data_list):
        """Takes a list of prefix/target pairs, tokenizes and concatenates them

        Args:
            data_list: Input parameter.

        Returns:
            object: Function return value.
        """
        prefix_len = len(self.encode(data_list[0][0]))
        target_len = len(self.encode(data_list[0][1]))
        same_len = True

        out = []
        for prefix, target in data_list:
            p = torch.tensor(self.encode(prefix))
            t = torch.tensor(self.encode(target))
            if not (len(p) == prefix_len and len(t) == target_len):
                same_len = False
            out.append(torch.concatenate((p, t), dim=-1).long())

        # check if all prefixes and all targets have the same length
        if same_len:
            print("Equal sequence lengths!")
        else:
            print("Not all prefixes or targets have the same length!!")

        return out, prefix_len, target_len


def get_tokenizer(args):
    """Get tokenizer.
    
    Args:
        args: Input parameter.
    
    Returns:
        object: Function return value.
    """
    if args.model_family == "gpt" or args.model_family == "mamba":
        numeral_tokenizer = NumeralTokenizer(args.total_nodes)
        tokenizer = Tokenizer(
            encoder=numeral_tokenizer.encode,
            decoder=numeral_tokenizer.decode,
            vocab_size=numeral_tokenizer.vocab_size,
            name="numeral",
        )
        tokenizer.pad_token_id = numeral_tokenizer.pad_token_id

    elif args.model_family.startswith("gpt2"):
        gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer = Tokenizer(
            encoder=gpt_tokenizer.encode,
            decoder=gpt_tokenizer.decode,
            vocab_size=50257,
            name="gpt2",
        )

    elif args.model_family.startswith("pythia"):
        pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/" + args.model_family)
        tokenizer = Tokenizer(
            encoder=pythia_tokenizer.encode,
            decoder=pythia_tokenizer.decode,
            vocab_size=50304,
            name="gpt2",
        )

    elif args.model_family.startswith("phi"):
        phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        tokenizer = Tokenizer(
            encoder=phi_tokenizer.encode,
            decoder=phi_tokenizer.decode,
            vocab_size=51200,
            name="phi",
        )

    else:
        raise ValueError(
            f"Unknown model family: {args.model_family}. Need to define tokenizer."
        )

    return tokenizer
