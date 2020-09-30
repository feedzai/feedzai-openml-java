"""
This script compares two folders with LGBM models.
It compares models with the same name across the two folders.
If a file with the same name is found in both folders it is compared.
Any differences in any of such compared files prints a report and triggers exit(1).
The report focuses on differences in the parameters of each line.

Requirements:
 - Python 3
 - (Optional) Python's termcolor for best results (`pip install termcolor`)

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
    def warn_missing_libs():
        pass
except ModuleNotFoundError:
    def warn_missing_libs():
        print("Warning: Please install Python's 'termcolor' library to see detailed colored model diff!")
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
        return

    for a_line, b_line in zip(a_lines, b_lines):
        a_elems = a_line.split(delim)
        b_elems = b_line.split(delim)

        if a_elems != b_elems:
            print(
                colored(f"\n\n\n>>> Different line contents between lines\n\n", "yellow"),
                colored(a_line, "red"),
                colored(b_line, "green"),
                sep=""
            )
            print(colored("\n>> Different elements in lines\n", "yellow"))
            if len(a_elems) != len(b_elems):
                print("Different line sizes!")
                return
            for a_elem, b_elem in zip(a_elems, b_elems):
                if a_elem != b_elem:
                    print(
                        colored(a_elem, "red"),
                        "\n",
                        colored(b_elem, "green"),
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
        warn_missing_libs()

        for filename in diff_filenames:
            print(colored(f"Found mismatches in file {filename}", "yellow"))
            file_diff_report(dir_ref/filename, dir_new/filename)

        warn_missing_libs()
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
