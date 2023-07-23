import tensorflow as tf
from tensorflow import Tensor
import heapq


class HuffmanNode:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def count_tensor_values(input_tensor: Tensor):
    input_tensor = tf.reshape(input_tensor, [-1])
    values = {}
    for value in input_tensor:
        if value.numpy() in values:
            values[value.numpy()] += 1
        else:
            values[value.numpy()] = 1

    return values


def build_huffman_tree(counter_dict):
    heap = [HuffmanNode(value, freq) for value, freq in counter_dict.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left_node = heapq.heappop(heap)
        right_node = heapq.heappop(heap)

        merged_node = HuffmanNode(None, left_node.freq + right_node.freq)
        merged_node.left = left_node
        merged_node.right = right_node

        heapq.heappush(heap, merged_node)

    return heap[0]


def generate_codebook(huffman_tree, current_code="", codebook=None):
    if codebook is None:
        codebook = {}

    if huffman_tree is not None:
        if huffman_tree.value is not None:
            codebook[huffman_tree.value] = current_code
        generate_codebook(huffman_tree.left, current_code + "0", codebook)
        generate_codebook(huffman_tree.right, current_code + "1", codebook)

    return codebook


def generate_huffman(value_counter):
    value_frequency_tuples = [(value, freq) for value, freq in value_counter.items()]
    huffman_tree_root = build_huffman_tree(dict(value_frequency_tuples))
    huffman_codebook = generate_codebook(huffman_tree_root)

    print("Huffman Codebook:")
    for value, code in huffman_codebook.items():
        print(f"Value: {value}, Code: {code}")


if __name__ == "__main__":
    vc = count_tensor_values(tf.constant([1, 2, 3, 1, 3, 1, 4], dtype=tf.float32))
    generate_huffman(vc)
