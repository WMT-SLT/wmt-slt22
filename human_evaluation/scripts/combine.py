#!/usr/bin/env python3

# Combine multiple XML files
#   Usage: python3 combine.py -i src.xml -o out.xml sys1.xml [sys2.xml ...]
#

import argparse
import logging
import sys


import lxml.etree as ET

LOG = logging.getLogger(__name__)

DEFAULT_TRANSLATOR = "DEFAULT"


def combine(
    source_xml_file,
    xml_files,
    missing_message="NO TRANSLATION AVAILABLE",
    system_suffix="",
):
    """
    Combine multiple XML files in WMT format

    note: a single language is assumed for each of sources, refs and hyps
    """
    src_tree = ET.parse(source_xml_file)
    trees = [ET.parse(xml_file) for xml_file in xml_files]

    for src_doc in src_tree.getroot().findall(".//doc"):
        src_segs = src_doc.findall(".//src//seg")
        src_docid = src_doc.attrib['id']
        hyp_segs = []

        hypo_count = 0
        for tree in trees:
            for hypo in tree.findall(".//doc[@id='{}']//hyp".format(src_docid)):
                hypo.attrib['system'] = hypo.attrib['system'] + system_suffix
                src_doc.append(hypo)
                hypo_count += 1
        sys.stderr.write(f"Added {hypo_count} hypotheses for document ID {src_docid}\n")

    return ET.tostring(
        src_tree, pretty_print=True, xml_declaration=True, encoding='utf-8'
    ).decode()


def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(name)s:  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-file",
        type=argparse.FileType('r'),
        default=sys.stdin,
        help="Input XML file with source and references",
    )
    parser.add_argument(
        "xml_files",
        type=argparse.FileType('r'),
        nargs='+',
        help="XML files to combine into the input XML file",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=argparse.FileType('w'),
        default=sys.stdout,
        help="Output XML file",
    )
    parser.add_argument(
        "-m",
        "--missing-translation-message",
        default="NO TRANSLATION AVAILABLE",
        help="Message to insert when translations are missing",
    )
    parser.add_argument(
        "--suffix",
        default="",
    )
    args = parser.parse_args()

    output = combine(args.input_file, args.xml_files, args.missing_translation_message, args.suffix)
    args.output_file.write(output)


if __name__ == "__main__":
    main()
