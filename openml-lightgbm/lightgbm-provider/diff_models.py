"""
This script compares two folders with LGBM models.
It compares models with the same name across the two folders.
If a file with the same name is found in both folders it is compared.
Any differences in any of such compared files prints a report and triggers exit(1).
The report focuses on differences in the parameters of each line.

Usage:
 script <reference_models_folder> <new_models_folder>

Exit code:
 0 if no differences in matching files were found
 1 if any file has differences

Author: Alberto Ferreira
Copyright: 2020, Feedzai
License: Same as the repo.
"""

import argparse
import filecmp
import sys
from pathlib import Path

try:
    from termcolor import colored
    _HAS_TERMCOLOR = True
except ModuleNotFoundError:
    _HAS_TERMCOLOR = False
    def colored(msg, color):
        "Poor replacement for colored."
        if color == "red":
            return f"Original=<{msg}>"
        elif color == "green":
            return f"Current=<{msg}>"
        else:
            return msg


def file_diff_report(a, b, delim=" "):
    """
    Reports the file diff by printing different lines and different elements in them.
    """
    a_lines = open(a).readlines()
    b_lines = open(b).readlines()

    if len(a_lines) != len(b_lines):
        print("Different file line sizes!")
        sys.exit(1)

    for l, _ in enumerate(a_lines):
        a_elems = a_lines[l].split(delim)
        b_elems = b_lines[l].split(delim)

        if a_elems != b_elems:
            print(
                colored(f"\n>>> Different line contents between lines\n", "yellow"),
                colored(a_lines[l], "red"),
                colored(b_lines[l], "green"),
                sep=""
            )
            print(colored("\n=> Different elements in lines\n", "yellow"))
            if len(a_elems) != len(b_elems):
                print("Different line sizes!")
                sys.exit(1)
            for e in range(len(a_elems)):
                a_elem = a_elems[e]
                b_elem = b_elems[e]
                if a_elem != b_elem:
                    print(
                        colored(a_elems[e], "red"),
                        "\n",
                        colored(b_elems[e], "green"),
                        sep=""
                    )

def compare_folders_with_model_files(folder_ref, folder_new):
    """
    Compares two folders with lgbm .txt models inside.

    If a file with the same name is found in the two folders is compared.

    Any difference triggers an exit(1).
    """
    dir_ref = Path(folder_ref)
    dir_new = Path(folder_new)

    diff_filenames = filecmp.dircmp(dir_ref, dir_new).diff_files

    if diff_filenames:
        if not _HAS_TERMCOLOR:
            print("Warning: Please install Python's 'termcolor' library to see detailed colored report!")

        
        for filename in diff_filenames:
            print(colored(f"Found mismatches in file {filename}", "yellow"))
            file_diff_report(dir_ref/filename, dir_new/filename)

        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_models_folder")
    parser.add_argument("new_models_folder")
    args = parser.parse_args()

    compare_folders_with_model_files(
        args.reference_models_folder,
        args.new_models_folder
    )
