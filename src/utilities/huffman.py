import numpy as np
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


def count_tensor_values(input_tensor):
    #input_tensor = tf.reshape(input_tensor, [-1])
    values = {}
    for value in input_tensor:
        if value in values:
            values[value] += 1
        else:
            values[value] = 1

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


def run_length_encoding(input_tensor: Tensor):
    input_tensor = tf.reshape(input_tensor, [-1])
    i = 0
    n = len(input_tensor)
    run_length_codebook = []

    while i < n:
        count = 1
        while i + 1 < n and input_tensor[i] == input_tensor[i + 1]:
            i += 1
            count += 1
        run_length_codebook.append(f"{input_tensor[i]} {count}")
        i += 1
    return run_length_codebook


def generate_huffman(value_counter):
    value_frequency_tuples = [(value, freq) for value, freq in value_counter.items()]
    huffman_tree_root = build_huffman_tree(dict(value_frequency_tuples))
    huffman_codebook = generate_codebook(huffman_tree_root)
    # print("Huffman Codebook:")
    # for value, code in huffman_codebook.items():
    #     print(f"Value: {value}, Code: {code}")
    return huffman_codebook

def encode_huffman(RLE, huffman_codebook):
    vectorized_map = np.vectorize(huffman_codebook.get)
    encoded_array = vectorized_map(RLE)
    return encoded_array  # "".join(encoded_array)

def decode_huffman(encoded_array, huffman_codebook):
    reverse_codebook = {v: k for k, v in huffman_codebook.items()}
    vectorized_map = np.vectorize(reverse_codebook.get)
    decoded_array = vectorized_map(encoded_array)
    return decoded_array

def decode_rle(decoded_huf):
    original = []
    for dec in decoded_huf:
        split = dec.split(" ")
        times = int(split[1])
        value = float(split[0])
        for t in range(times):
            original.append(value)

    return original


if __name__ == "__main__":
    # vc = count_tensor_values(tf.constant([1, 2, 3, 1, 3, 1, 4], dtype=tf.float32))
    # generate_huffman(vc)
    rle = run_length_encoding(tf.constant([[1, 1, 3, 1, 3, 1, 4], [1, 2, 3, 1, 3, 1, 1]], dtype=tf.float32))
    print("RLE:", rle)
    vc = count_tensor_values(rle)
    huf = generate_huffman(vc)
    enc = encode_huffman(rle, huf)
    print(14*32/(len("".join(enc))+len(str(huf))))
    dec_huf = decode_huffman(enc, huf)
    print(decode_rle(dec_huf))
    print(len(huf), huf)
