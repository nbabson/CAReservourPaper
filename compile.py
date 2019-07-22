#!/usr/bin/python

import subprocess, sys

commands = [
    ['pdflatex', sys.argv[1]],
    ['bibtex', sys.argv[1]],
    ['pdflatex', sys.argv[1]],
    ['pdflatex', sys.argv[1]]
]

for c in commands:
    subprocess.call(c)
