class NumeralTokenizer:
    """Tokenizer for numeral graph datasets with task and control tokens.

    Args:
        num_nodes: Number of graph-node ids to include in the vocabulary.

    Returns:
        NumeralTokenizer: Initialized tokenizer instance with `encode` and `decode` methods.
    """

    DELIMITER_TOKEN = "|"
    PAUSE_TOKEN = "[PAUSE]"
    PATH_SEPARATOR_TOKEN = "/"
    TEACHERLESS_TOKEN = "$"
    EDGE_FORWARD_TOKEN = ">"
    EDGE_BACKWARD_TOKEN = "<"
    PAD_TOKEN = "[PAD]"
    EDGE_TASK_TOKEN = "[EDGE]"
    PATH_TASK_TOKEN = "[PATH]"

    def __init__(self, num_nodes):
        """Builds encoder/decoder tables for numeral and special tokens.

        Args:
            num_nodes: Number of integer node tokens (from `0` to `num_nodes - 1`).

        Returns:
            None: Initializes tokenizer state in-place.
        """
        self.digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.num_nodes = num_nodes

        # encoder
        self.encoder = {str(i): i for i in range(num_nodes)}
        self.encoder[self.DELIMITER_TOKEN] = num_nodes
        self.encoder[self.PAUSE_TOKEN] = num_nodes + 1
        self.encoder[self.PATH_SEPARATOR_TOKEN] = num_nodes + 2
        self.encoder[self.TEACHERLESS_TOKEN] = num_nodes + 3

        # edge direction
        self.encoder[self.EDGE_FORWARD_TOKEN] = num_nodes + 4
        self.encoder[self.EDGE_BACKWARD_TOKEN] = num_nodes + 5

        # pad token
        self.pad_token_id = num_nodes + 6
        self.encoder[self.PAD_TOKEN] = self.pad_token_id

        # task-prefix tokens
        self.encoder[self.EDGE_TASK_TOKEN] = num_nodes + 7
        self.encoder[self.PATH_TASK_TOKEN] = num_nodes + 8

        # vocab size = nodes + specials
        self.vocab_size = num_nodes + 9

        # decoder
        self.decoder = {i: i for i in range(num_nodes)}
        self.decoder[num_nodes] = self.DELIMITER_TOKEN
        self.decoder[num_nodes + 1] = self.PAUSE_TOKEN
        self.decoder[num_nodes + 2] = self.PATH_SEPARATOR_TOKEN
        self.decoder[num_nodes + 3] = self.TEACHERLESS_TOKEN
        self.decoder[num_nodes + 4] = self.EDGE_FORWARD_TOKEN
        self.decoder[num_nodes + 5] = self.EDGE_BACKWARD_TOKEN
        self.decoder[num_nodes + 6] = self.PAD_TOKEN
        self.decoder[num_nodes + 7] = self.EDGE_TASK_TOKEN
        self.decoder[num_nodes + 8] = self.PATH_TASK_TOKEN

        # present in your original code
        self.decoder[-1] = ":"

    def encode(self, x):
        """Converts a token string into a list of token ids.

        Args:
            x: Input token string containing node ids and special tokens.

        Returns:
            list[int]: Encoded token ids.
        """
        out = []
        i = 0
        while i < len(x):
            if x[i] == ",":
                i += 1
                continue

            # bracketed task tokens like "[EDGE]" / "[PATH]"
            if x[i] == "[":
                s = ""
                j = 0
                while i + j < len(x) and x[i + j] != "]":
                    s += x[i + j]
                    j += 1
                s += x[i + j]
                i += j + 1
                out.append(self.encoder[s])
                continue

            # multi-digit numbers
            s = ""
            j = 0
            while i + j < len(x) and x[i + j] in self.digits:
                s += x[i + j]
                j += 1
            if s == "":
                s = x[i]
                i += 1
            else:
                i += j
            out.append(self.encoder[s])

        return out

    def decode(self, x):
        """Converts a list of token ids back to decoded token values.

        Args:
            x: Sequence of token ids.

        Returns:
            list[object]: Decoded token values.
        """
        return [self.decoder[i] for i in x]
