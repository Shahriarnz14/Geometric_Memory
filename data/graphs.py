from collections import deque
import random
import numpy as np
import torch
from torch.utils.data import Dataset


GRAPH_DIR = "geometric_memory/data/datasets/graphs/"


# --------------------------------------------------------------------------- #
#  STAR GRAPH
# --------------------------------------------------------------------------- #
def star_graph(degSource, pathLen, numNodes, reverse=False):
    """Original star graph implementation (unchanged).

    Args:
        degSource: Input parameter.
        pathLen: Input parameter.
        numNodes: Input parameter.
        reverse: Input parameter.

    Returns:
        object: Function return value.
    """
    # Pick distinct source and goal
    source = np.random.randint(0, numNodes, 1)[0]
    goal = np.random.randint(0, numNodes, 1)[0]
    while goal == source:
        goal = np.random.randint(0, numNodes, 1)[0]

    # Sample an internal path
    path = [source]
    edge_list = []

    # Choose random nodes along the path (excluding source and goal)
    for _ in range(pathLen - 2):
        node = np.random.randint(0, numNodes, 1)[0]
        while node in path or node == goal:
            node = np.random.randint(0, numNodes, 1)[0]
        path.append(node)

    # Add goal to the path
    path.append(goal)

    # Connect the path
    for i in range(len(path) - 1):
        edge_list.append([path[i], path[i + 1]])

    # Remaining nodes (not actually used in original implementation)
    remaining_nodes = []
    for i in range(numNodes):
        if i not in path:
            remaining_nodes.append(i)

    # Add degSource − 1 “spokes” starting at source
    i = 0
    deg_nodes = set()
    while i < degSource - 1:
        node = source
        next_node = np.random.randint(0, numNodes, 1)[0]
        l = 1
        while l < pathLen:
            if next_node not in deg_nodes and next_node not in path:
                edge_list.append([node, next_node])
                deg_nodes.add(next_node)
                node = next_node
                l += 1
            next_node = np.random.randint(0, numNodes, 1)[0]
        i += 1

    random.shuffle(edge_list)

    if reverse:
        path = path[::-1]

    return path, edge_list, source, goal


# --------------------------------------------------------------------------- #
#  BAT GRAPH (MULTIPLE DIAMOND-SHAPED WINGS)
# --------------------------------------------------------------------------- #
def bat_graph(deg, wing, path_len, num_nodes, reverse=False):
    """
    Generate a bat graph structure with multiple diamond-shaped wings.

    Args:
        deg:       total degree from root node (must be divisible by wing)
        wing:      width of each wing (number of nodes per intermediate layer)
        path_len:  length of path from root to leaf
        num_nodes: total number of nodes available for selection
        reverse:   whether to reverse the final path

    Returns:
        path:      list of nodes from start (root) to goal (leaf)
        edge_list: list of edges [node1, node2]
        start:     start node (root)
        goal:      goal node (randomly selected leaf)
    """
    # Ensure deg is divisible by wing
    assert deg % wing == 0, f"deg ({deg}) must be divisible by wing ({wing})"

    num_wings = deg // wing

    # Choose random root node
    root = np.random.randint(0, num_nodes, 1)[0]

    # Keep track of all used nodes
    used_nodes = {root}
    edge_list = []

    # Store nodes by wing and layer for easy access
    wings_nodes = []
    leaf_nodes = []

    # Create each wing
    for _ in range(num_wings):
        # Layer 0 is always the root
        wing_layers = [[root]]

        # Create intermediate layers (1 to path_len-2)
        for _layer in range(1, path_len - 1):
            layer_nodes = []
            for _ in range(wing):
                # Find next available node
                node = np.random.randint(0, num_nodes, 1)[0]
                while node in used_nodes:
                    node = np.random.randint(0, num_nodes, 1)[0]
                layer_nodes.append(node)
                used_nodes.add(node)
            wing_layers.append(layer_nodes)

        # Create leaf layer (path_len-1) – single node per wing
        leaf_node = np.random.randint(0, num_nodes, 1)[0]
        while leaf_node in used_nodes:
            leaf_node = np.random.randint(0, num_nodes, 1)[0]
        used_nodes.add(leaf_node)
        leaf_nodes.append(leaf_node)
        wing_layers.append([leaf_node])

        # Store wing structure
        wings_nodes.append(wing_layers)

    # Add edges within each wing (full bipartite between adjacent layers)
    for wing_layers in wings_nodes:
        for layer_idx in range(len(wing_layers) - 1):
            current_layer = wing_layers[layer_idx]
            next_layer = wing_layers[layer_idx + 1]
            for curr_node in current_layer:
                for next_node in next_layer:
                    edge_list.append([curr_node, next_node])

    # Randomly select one leaf as goal
    goal = random.choice(leaf_nodes)

    # Find which wing contains the goal
    goal_wing = None
    for wing_idx, wing_layers in enumerate(wings_nodes):
        if goal in wing_layers[-1]:
            goal_wing = wing_idx
            break

    # Build path by randomly selecting one node from each intermediate layer
    path = [root]
    for layer_idx in range(1, len(wings_nodes[goal_wing]) - 1):
        layer_nodes = wings_nodes[goal_wing][layer_idx]
        chosen_node = random.choice(layer_nodes)
        path.append(chosen_node)

    path.append(goal)

    random.shuffle(edge_list)

    if reverse:
        path = path[::-1]

    return path, edge_list, root, goal


# --------------------------------------------------------------------------- #
#  BALANCED TREE GRAPH WITH OPTIONAL PRUNING
# --------------------------------------------------------------------------- #
def tree_graph(degSource, pathLen, numNodes, reverse=False, degTree=None):
    """
    Generate a balanced n-ary tree graph with pruning capability.

    Args:
        degSource: Number of children from the root (source)
        pathLen:   Height of the tree (distance from root to leaves)
        numNodes:  Maximum number of nodes available (0 to numNodes-1)
        reverse:   Whether to reverse the final path
        degTree:   Branching factor for subtrees (defaults to degSource)

    Returns:
        path:      List of nodes from source (root) to goal (leaf)
        edge_list: List of edges in the tree
        source:    Root node (randomly selected)
        goal:      Selected leaf node (randomly selected)

    Structure:
        - Root has degSource children
        - Each of these degSource children becomes root of a balanced
          degTree-ary subtree
        - When degSource == degTree: same as original balanced tree
        - When degSource < degTree: pruned version (fewer subtrees from root)
    """
    # Handle degTree parameter
    if degTree is None:
        degTree = degSource

    # Calculate total nodes needed
    if pathLen == 1:
        # Just the root
        total_nodes_needed = 1
    elif pathLen == 2:
        # Root + degSource children (leaves)
        total_nodes_needed = 1 + degSource
    else:
        # Root + degSource subtrees, each being a degTree-ary tree of
        # height (pathLen-1)
        if degTree == 1:
            # Linear chain
            nodes_per_subtree = pathLen - 1
        else:
            nodes_per_subtree = (degTree ** (pathLen - 1) - 1) // (degTree - 1)
        total_nodes_needed = 1 + degSource * nodes_per_subtree

    # Throw error if insufficient nodes
    if total_nodes_needed > numNodes:
        raise ValueError(
            f"Cannot create tree with degSource={degSource}, "
            f"degTree={degTree}, height={pathLen}: "
            f"needs {total_nodes_needed} nodes but only "
            f"{numNodes} available. Please increase numNodes to at least "
            f"{total_nodes_needed}."
        )

    # Randomly select nodes for the tree (without replacement)
    available_nodes = list(range(numNodes))
    np.random.shuffle(available_nodes)
    tree_nodes = available_nodes[:total_nodes_needed]

    # First node becomes the source (root)
    source = tree_nodes[0]
    node_index = 1

    edge_list = []
    all_leaves = []

    if pathLen == 1:
        # Special case: just the root (also a leaf)
        all_leaves = [source]
    elif pathLen == 2:
        # Special case: root + degSource children (all children are leaves)
        for _ in range(degSource):
            child = tree_nodes[node_index]
            edge_list.append([source, child])
            all_leaves.append(child)
            node_index += 1
    else:
        # General case: root + degSource subtrees
        subtree_roots = []

        # Create degSource children of the root (these become subtree roots)
        for _ in range(degSource):
            subtree_root = tree_nodes[node_index]
            edge_list.append([source, subtree_root])
            subtree_roots.append(subtree_root)
            node_index += 1

        # Build each subtree as a balanced degTree-ary tree
        for subtree_root in subtree_roots:
            subtree_edges, subtree_leaves, node_index = build_balanced_subtree(
                subtree_root, degTree, pathLen - 1, tree_nodes, node_index
            )
            edge_list.extend(subtree_edges)
            all_leaves.extend(subtree_leaves)

    if not all_leaves:
        raise ValueError("No leaves generated - tree construction failed")

    # Randomly select one leaf as goal
    goal = random.choice(all_leaves)

    # Find path from root to selected leaf
    path = find_path_in_tree(edge_list, source, goal)

    # Verify path length matches expected
    if len(path) != pathLen:
        raise ValueError(f"Generated path length {len(path)} doesn't match expected " f"{pathLen}")

    # Shuffle edge list to prevent order-based shortcuts
    random.shuffle(edge_list)

    if reverse:
        path = path[::-1]

    return path, edge_list, source, goal


def build_balanced_subtree(root, degree, height, tree_nodes, start_index):
    """
    Build a balanced tree rooted at `root` with given degree and height.

    Args:
        root:       Root node of the subtree
        degree:     Branching factor
        height:     Height of the subtree
        tree_nodes: List of available nodes
        start_index: Index to start picking nodes from tree_nodes

    Returns:
        edges:     List of edges in the subtree
        leaves:    List of leaf nodes in the subtree
        node_index: Next available index in tree_nodes
    """
    edges = []
    leaves = []
    node_index = start_index

    if height == 1:
        # Root is a leaf
        leaves = [root]
        return edges, leaves, node_index

    # Build level by level
    current_level = [root]
    for _level in range(1, height):
        next_level = []
        for parent in current_level:
            for _child_idx in range(degree):
                if node_index >= len(tree_nodes):
                    raise ValueError("Ran out of tree nodes during subtree construction")
                child = tree_nodes[node_index]
                edges.append([parent, child])
                next_level.append(child)
                node_index += 1
        current_level = next_level

    # Current level contains all leaves
    leaves = current_level
    return edges, leaves, node_index


def find_path_in_tree(edge_list, source, goal):
    """Find path from source to goal in a tree using BFS.

    Args:
        edge_list: Input parameter.
        source: Input parameter.
        goal: Input parameter.

    Returns:
        object: Function return value.
    """
    # Build adjacency list
    graph = {}
    for edge in edge_list:
        parent, child = edge
        if parent not in graph:
            graph[parent] = []
        graph[parent].append(child)

        # Also add reverse direction for easier pathfinding
        if child not in graph:
            graph[child] = []
        graph[child].append(parent)

    # BFS to find path
    queue = deque([(source, [source])])
    visited = {source}

    while queue:
        node, path = queue.popleft()

        if node == goal:
            return path

        if node in graph:
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

    raise ValueError(f"No path found from {source} to {goal}")


# --------------------------------------------------------------------------- #
#  GRAPH GENERATION (WRITE TO DISK)
# --------------------------------------------------------------------------- #
def _write_sample(file_obj, path, edge_list, start, goal):
    """Helper: write one sample in prefix=target format.

    Args:
        file_obj: Input parameter.
        path: Input parameter.
        edge_list: Input parameter.
        start: Input parameter.
        goal: Input parameter.

    Returns:
        object: Function return value.
    """
    path_str = ",".join(str(node) for node in path)
    edge_str = "|".join(f"{e[0]},{e[1]}" for e in edge_list)
    prefix = f"{edge_str}/{start},{goal}="
    out = prefix + path_str
    file_obj.write(out + "\n")


def generate_and_save(n_train, n_test, degSource, pathLen, numNodes, reverse=False):
    """Generate a list of train and testing STAR graphs and save them.

    Args:
        n_train: Input parameter.
        n_test: Input parameter.
        degSource: Input parameter.
        pathLen: Input parameter.
        numNodes: Input parameter.
        reverse: Input parameter.

    Returns:
        object: Function return value.
    """
    train_fname = (
        GRAPH_DIR
        + "deg_"
        + str(degSource)
        + "_path_"
        + str(pathLen)
        + "_nodes_"
        + str(numNodes)
        + "_train_"
        + str(n_train)
        + ".txt"
    )

    test_fname = (
        GRAPH_DIR
        + "deg_"
        + str(degSource)
        + "_path_"
        + str(pathLen)
        + "_nodes_"
        + str(numNodes)
        + "_test_"
        + str(n_test)
        + ".txt"
    )

    with open(train_fname, "w") as f:
        for _ in range(n_train):
            path, edge_list, start, goal = star_graph(degSource, pathLen, numNodes, reverse=reverse)
            _write_sample(f, path, edge_list, start, goal)

    with open(test_fname, "w") as f:
        for _ in range(n_test):
            path, edge_list, start, goal = star_graph(degSource, pathLen, numNodes, reverse=reverse)
            _write_sample(f, path, edge_list, start, goal)

    return train_fname, test_fname


def generate_bat_and_save(n_train, n_test, degSource, pathLen, numNodes, reverse=False, wing=None):
    """Generate a list of train and testing BAT graphs and save them.

    Args:
        n_train: Input parameter.
        n_test: Input parameter.
        degSource: Input parameter.
        pathLen: Input parameter.
        numNodes: Input parameter.
        reverse: Input parameter.
        wing: Input parameter.

    Returns:
        object: Function return value.
    """
    if wing is None:
        wing = 2  # default; must divide degSource

    filename_base = f"bat_deg{degSource}_wing{wing}_path{pathLen}_nodes{numNodes}"

    train_fname = GRAPH_DIR + filename_base + "_train_" + str(n_train) + ".txt"
    test_fname = GRAPH_DIR + filename_base + "_test_" + str(n_test) + ".txt"

    # Training data
    with open(train_fname, "w") as train_file:
        for i in range(n_train):
            path, edge_list, start, goal = bat_graph(
                deg=degSource,
                wing=wing,
                path_len=pathLen,
                num_nodes=numNodes,
                reverse=reverse,
            )
            _write_sample(train_file, path, edge_list, start, goal)
            if (i + 1) % 100000 == 0:
                print(f"Generated {(i + 1)}/{n_train} bat training samples")

    # Test data
    with open(test_fname, "w") as test_file:
        for i in range(n_test):
            path, edge_list, start, goal = bat_graph(
                deg=degSource,
                wing=wing,
                path_len=pathLen,
                num_nodes=numNodes,
                reverse=reverse,
            )
            _write_sample(test_file, path, edge_list, start, goal)
            if (i + 1) % 100000 == 0:
                print(f"Generated {(i + 1)}/{n_test} bat test samples")

    return train_fname, test_fname


def generate_tree_and_save(
    n_train, n_test, degSource, pathLen, numNodes, reverse=False, degTree=None
):
    """Generate tree graphs with degSource and degTree parameters.

    Args:
        n_train: Input parameter.
        n_test: Input parameter.
        degSource: Input parameter.
        pathLen: Input parameter.
        numNodes: Input parameter.
        reverse: Input parameter.
        degTree: Input parameter.

    Returns:
        object: Function return value.
    """
    degree = degTree if degTree is not None else degSource

    base_name = f"tree_deg{degSource}_degTree{degree}_path{pathLen}_nodes{numNodes}"
    train_fname = GRAPH_DIR + base_name + "_train_" + str(n_train) + ".txt"
    test_fname = GRAPH_DIR + base_name + "_test_" + str(n_test) + ".txt"

    print(
        f"Generating tree graphs with degSource={degSource}, "
        f"degTree={degree}, pathLen={pathLen}"
    )
    print(f"Training samples: {n_train}, Test samples: {n_test}")

    # Generate training data
    print("Generating n_train training samples...")
    with open(train_fname, "w") as train_file:
        for i in range(n_train):
            try:
                path, edge_list, start, goal = tree_graph(
                    degSource=degSource,
                    pathLen=pathLen,
                    numNodes=numNodes,
                    reverse=reverse,
                    degTree=degree,
                )
                _write_sample(train_file, path, edge_list, start, goal)
                if (i + 1) % 100000 == 0:
                    print(f"Generated {(i + 1)}/{n_train} training samples")
            except ValueError as e:
                print(f"Error generating training sample {i}: {e}")
                raise

    # Generate test data
    print("Generating n_test test samples...")
    with open(test_fname, "w") as test_file:
        for i in range(n_test):
            try:
                path, edge_list, start, goal = tree_graph(
                    degSource=degSource,
                    pathLen=pathLen,
                    numNodes=numNodes,
                    reverse=reverse,
                    degTree=degree,
                )
                _write_sample(test_file, path, edge_list, start, goal)
                if (i + 1) % 100000 == 0:
                    print(f"Generated {(i + 1)}/{n_test} test samples")
            except ValueError as e:
                print(f"Error generating test sample {i}: {e}")
                raise

    print("Successfully generated tree graphs.")

    # Calculate and display expected structure
    if pathLen == 1:
        total_nodes = 1
    elif pathLen == 2:
        total_nodes = 1 + degSource
    else:
        if degree == 1:
            nodes_per_subtree = pathLen - 1
        else:
            nodes_per_subtree = (degree ** (pathLen - 1) - 1) // (degree - 1)
        total_nodes = 1 + degSource * nodes_per_subtree

    print(f"Expected nodes per graph: {total_nodes}")

    return train_fname, test_fname


# --------------------------------------------------------------------------- #
#  DATA LOADING UTILITIES
# --------------------------------------------------------------------------- #
def prefix_target_list(filename=None, reverse=False):
    """Load graphs, split them into prefix and target, and return the list.

    Args:
        filename: Input parameter.
        reverse: Input parameter.

    Returns:
        object: Function return value.
    """
    data_list = []
    with open(filename, "r") as f:
        lines = f.readlines()
    for line in lines:
        prefix, target = line.strip().split("=", maxsplit=1)
        prefix += "[PAUSE]"
        if reverse:
            target = ",".join(target.split(",")[::-1])
        data_list.append((prefix, target))
    return data_list


class Graphs(Dataset):
    """Graphs definition.
    
    Description:
        Encapsulates related behavior and state.
    """
    def __init__(
        self,
        tokenizer,
        n_samples,
        data_path,
        device,
        eval=False,
        teacherless_token=None,
        reverse=False,
    ):
        """  init  .
        
        Args:
            tokenizer: Input parameter.
            n_samples: Input parameter.
            data_path: Input parameter.
            device: Input parameter.
            eval: Input parameter.
            teacherless_token: Input parameter.
            reverse: Input parameter.
        
        Returns:
            object: Function return value.
        """
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.device = device
        self.eval_mode = eval
        self.data_path = data_path
        self.teacherless_token = teacherless_token
        self.reverse = reverse

        self.data_file = prefix_target_list(self.data_path, reverse=reverse)[:n_samples]

        (
            self.tokenized,
            self.num_prefix_tokens,
            self.num_target_tokens,
        ) = tokenizer.tokenize(self.data_file)

        # Total tokens per sequence (prefix + target)
        self.num_tokens = self.num_prefix_tokens + self.num_target_tokens

    def __len__(self):
        """  len  .
        
        Args:
            None: This callable does not take external parameters.
        
        Returns:
            object: Function return value.
        """
        return len(self.data_file)

    def __getitem__(self, idx):
        """  getitem  .
        
        Args:
            idx: Input parameter.
        
        Returns:
            object: Function return value.
        """
        if self.eval_mode:
            # In eval mode return the entire sequence
            return self.tokenized[idx].to(self.device)

        # Create inputs (all but last token)
        x = self.tokenized[idx][:-1].clone()
        if self.teacherless_token is not None:
            # Replace first target token with teacherless token if desired
            x[self.num_prefix_tokens :] = self.teacherless_token
            x = x.to(self.device)

        # Create targets in the form [-1, ..., y_{prefix+1}, ...] where we
        # replace the prefix tokens by -1 so that we can skip their gradient
        # calculation in the loss.
        y_prefix = -torch.ones(self.num_prefix_tokens - 1, dtype=torch.long)
        y_target = self.tokenized[idx][self.num_prefix_tokens :].clone()
        y = torch.cat([y_prefix, y_target])

        return x.to(self.device), y.long().to(self.device)

    def eval(self):
        """Switch to 'eval' mode when generating sequences without teacher-forcing.

        Args:
            None: This callable does not take external parameters.

        Returns:
            object: Function return value.
        """
        self.eval_mode = True

    def train(self):
        """Switch back to 'train' mode for teacher-forcing.

        Args:
            None: This callable does not take external parameters.

        Returns:
            object: Function return value.
        """
        self.eval_mode = False


# --------------------------------------------------------------------------- #
#  HELPER FOR MAPPING TOKENS BACK TO EDGES
# --------------------------------------------------------------------------- #
def get_edge_list(x, num_nodes, path_len):
    """Given the tokenised integer input for the Transformer, map back to edge list.

    Args:
        x: Input parameter.
        num_nodes: Input parameter.
        path_len: Input parameter.

    Returns:
        object: Function return value.
    """
    edge_list = []
    pair = []
    x = x.squeeze().cpu().numpy()

    # Collect edge pairs until sentinel (num_nodes + 2)
    for i, n in enumerate(x):
        if 0 <= n < num_nodes:
            pair.append(int(n))
            if len(pair) == 2:
                edge_list.append(pair)
                pair = []
        if n == num_nodes + 2:
            break

    # i is the index of the sentinel
    start = int(x[i + 1])
    goal = int(x[i + 2])
    path = [int(x[i + j]) for j in range(4, 4 + path_len)]

    return edge_list, start, goal, path


def get_edge_list_byte(x, num_nodes, path_len, decode):
    """Byte-level variant: given tokenised input and a decode table, map back

    Args:
        x: Input parameter.
        num_nodes: Input parameter.
        path_len: Input parameter.
        decode: Input parameter.

    Returns:
        object: Function return value.
    """
    edge_list = []
    x = list(x.squeeze().cpu().numpy())
    dec = [decode[val] for val in x]

    edge = []
    for i, val in enumerate(dec):
        if val not in ["/", "|", "=", "[PAUSE]", "->"]:
            edge.append(val)
            if len(edge) == 2:
                edge_list.append(edge)
                edge = []
        if val == "->":
            break

    # i is at the arrow; next tokens give start / goal / path
    start = dec[i + 1]
    goal = dec[i + 2]
    path = dec[i + 3 : i + 3 + path_len]

    return edge_list, start, goal, path


# --------------------------------------------------------------------------- #
#  SIMPLE DEBUGGING / VISUALIZATION WHEN RUN DIRECTLY
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import types
    from geometric_memory.data import get_dataset
    from geometric_memory.tokenizing import get_tokenizer
    from geometric_memory.utils.device import resolve_default_device
    import matplotlib.pyplot as plt
    import networkx as nx

    # Example: star graphs
    n_train = 200000
    n_test = 20000
    deg = 5
    path_len = 4
    num_nodes = 50
    reverse = True

    generate_and_save(
        n_train=n_train,
        n_test=n_test,
        degSource=deg,
        pathLen=path_len,
        numNodes=num_nodes,
        reverse=reverse,
    )

    # Load data
    device = resolve_default_device()
    args = types.SimpleNamespace(model="gpt", num_nodes=num_nodes)
    args.dataset = "graph"
    args.deg = deg
    args.path_len = path_len
    args.n_train = n_train
    args.n_test = n_test
    args.reverse = reverse
    args.dollar = 11

    tokenizer = get_tokenizer(args)
    trainset, testset = get_dataset(args, tokenizer, device)

    print("Num Tokens:", trainset.num_tokens)
    print("Train Mode:", trainset.__getitem__(10))
    trainset.eval()
    print("Eval Mode:", trainset.__getitem__(10))

    # Visualize one example star graph
    edge_list, start, goal, path = get_edge_list(trainset.__getitem__(10)[0], num_nodes, path_len)
    print("Edge List Len:", len(edge_list))
    print("Path List:", path)
    print("Edge List:", edge_list)
    print("Start:", start, "Goal:", goal)

    G = nx.Graph()
    node_colors = []
    G.add_edges_from(edge_list)
    for node in G.nodes():
        if node == start:
            node_colors.append("green")
        elif node == goal:
            node_colors.append("red")
        elif node in path:
            node_colors.append("yellow")
        else:
            node_colors.append("royalblue")

    nx.draw(G, with_labels=True, node_color=node_colors)
    plt.show()
