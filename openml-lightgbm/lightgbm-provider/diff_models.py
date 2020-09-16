#import argparse
import filecmp
import sys
import logging
from pathlib import Path
from termcolor import colored

dir_std = Path("standard_code_models")
dir_new = Path("new_code_models")

diff_filenames = filecmp.dircmp(dir_std, dir_new).diff_files

def file_diff_report(a, b, delim=" "):
    """Reports the diff but simplifies by elements"""
    a_lines = open(a).readlines()
    b_lines = open(b).readlines()

    if len(a_lines) != len(b_lines):
        print("Different file line sizes!")
        sys.exit(1)
        
    for l in range(len(a_lines)):
        a_elems = a_lines[l].split(delim)
        b_elems = b_lines[l].split(delim)

        if a_elems != b_elems:
            print(
                colored(f"\nDifferent line contents between lines\n", "yellow"),
                colored(a_lines[l], "red"),
                colored(b_lines[l], "green"),
                sep=""
            )
            if len(a_elems) != len(b_elems):
                print("Different line sizes!")
                sys.exit(1)
            for e in range(len(a_elems)):
                a_elem = a_elems[e]
                b_elem = b_elems[e]
                if a_elem != b_elem:
                    print(colored(a_elems[e], "red"), "\n", colored(b_elems[e], "green"), sep="")




if diff_filenames:

    for filename in diff_filenames:
        print(colored(f"Found mismatches in file {filename}", "yellow"))
        file_diff_report(dir_std/filename, dir_new/filename)

    sys.exit(1)
