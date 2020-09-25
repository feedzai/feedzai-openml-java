"""
This script serves to add copies of trees to the model.
Use it to grow model files to arbitrary size to benchmark model read/write speed.

Usage:
 script <LightGBM_model.txt> <number of trees to add> <output_model_path>

Author: Alberto Ferreira
Copyright: 2020, Feedzai
License: Same as the repo.
"""

import sys
import argparse
import random

class ModelBlocks:
    def __init__(self, filepath):
        self.raw_blocks = ModelBlocks.parse_file_blocks(filepath)

        self.header_block = self.raw_blocks[0]
        self.tree_blocks = list(filter(ModelBlocks.is_tree_block, self.raw_blocks))
        self.end_blocks = self.raw_blocks[
            self.find_end_of_trees_block(self.raw_blocks):
        ]
        del self.raw_blocks

        self.N_ORIGINAL_TREES = len(self.tree_blocks)
        self.check_model_blocks(filepath)

    def check_model_blocks(self, filepath):
        "Asserts that the file output will be the same as the input before any changes are made."
        input_lines  = open(filepath).readlines()
        output_lines = list(self.yield_model_lines()) # list of output lines

        assert len(input_lines) == len(output_lines), "Bug: Corrupted output!"
        
    @staticmethod
    def parse_file_blocks(filepath):
        at_block = False
        blocks = []
        with open(filepath) as f:
            block = []
            for line in f:
                if line == "\n":
                    block.append(line)
                    at_block = False
                else:
                    if not at_block: # starting new block
                        if block:
                            blocks.append(block)
                        block = []
                        at_block = True
                    block.append(line)
            if block:
                blocks.append(block)
        return blocks

    @staticmethod
    def is_tree_block(block):
        return block and block[0].startswith("Tree=")

    @staticmethod
    def tree_size(block):
        # We don't store the second blank line in the tree (add +1):
        return sum(map(len, block))

    def find_end_of_trees_block(self, blocks):
        for b, block in enumerate(blocks):
            if len(block) == 2 and block[0] == "end of trees\n":
                return b
        raise Exception("Corrupt file: Coudln't find end of trees block!")

    def add_sampled_tree(self):
        """
        Adds a copy of an existing tree to the model as the last tree.
        Returns the index of the sampled tree.
        """
        sample_tree_idx = random.randrange(1, self.N_ORIGINAL_TREES)
        new_tree = self.tree_blocks[sample_tree_idx][:] # force copy
        N_trees = len(self.tree_blocks)
        new_tree[0] = f"Tree={N_trees}\n"
        self.tree_blocks.append(new_tree)
        return sample_tree_idx

    def add_sampled_trees(self, n):
        "Adds sampled trees and returns a list of the sampled trees"
        return [self.add_sampled_tree() for i in range(n)]
        
    def gen_tree_sizes_line(self):
        ssv = " ".join([
            str(ModelBlocks.tree_size(tree_block)) for tree_block in self.tree_blocks
        ])
        return f"tree_sizes={ssv}\n"
    
    def yield_model_lines(self):

        for line in self.header_block:
            if not line.startswith("tree_sizes="):
                yield line
            else:
                yield self.gen_tree_sizes_line()

        for block in self.tree_blocks + self.end_blocks:
            for line in block:
                yield line

    def write_to(self, filepath):
        "Dump model to disk"
        with open(filepath, "w") as out_file:
            out_file.writelines(self.yield_model_lines())
            

                

def grow_model(filepath, new_trees:int, output_path):
    print("Parsing model file...")
    model_blocks = ModelBlocks(filepath)
    print(f"Adding {new_trees} sampled trees...")
    model_blocks.add_sampled_trees(new_trees)
    print("Writing to disk...")
    model_blocks.write_to(output_path)
    print("DONE!\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("number_trees", type=int)
    parser.add_argument("output_path")
    args = parser.parse_args()

    assert args.input_path != args.output_path
    grow_model(args.input_path, args.number_trees, args.output_path)
