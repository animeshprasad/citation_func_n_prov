"""
This script contains all file operations.
"""

import os
import math
from lxml import etree
from xml.etree import ElementTree as ET

TXT_EXTENSION = '.txt'
XML_EXTENSION = '.xml'
PDF_EXTENSION = '.pdf'
JSON_EXTENSION = '.json'


def is_xml(filepath):
    """
    Given a file path, check whether the file is of XML format.
    """
    return filepath.lower().endswith(XML_EXTENSION)


def is_pdf(filepath):
    return filepath.lower().endswith(PDF_EXTENSION)


def get_xml_path(filepath):
    """
    Given a PDF file path, return its corresponding XML file path.
    """
    if is_pdf(filepath):
        return os.path.splitext(filepath)[0] + XML_EXTENSION
    else:
        raise Exception(FILE_NOT_PDF_EXCEPTION)


def get_pdf_path(filepath):
    if is_xml(filepath):
        return os.path.splitext(filepath)[0] + PDF_EXTENSION
    else:
        raise Exception(FILE_NOT_PDF_EXCEPTION)


def remove_ext(filepath):
    """
    Removes the extension from file path.
    """
    return os.path.splitext(filepath)[0]


def get_dirs_in_dir(directory):
    """
    Return the list of all directories in a given directory.
    """
    for root, dirs, files in os.walk(directory):
        return [f for f in dirs]


def get_files_in_dir(directory):
    """
    Return the list of all files in a given directory.
    """
    for root, dirs, files in os.walk(directory):
        return [f for f in files if f != '.DS_Store']


def in_dir(f, directory):
    """
    Return true if the file is in the specified directory.
    """
    return f.strip() in get_files_in_dir(directory)


def file_exists(filepath):
    return os.path.exists(filepath)


def delete_files(files):
    """
    Delete a list of files.
    """
    for f in files:
        if os.path.isfile(f):
            os.remove(f)


def splits(pages, page_lim):
    """
    Return the number of splits needed to decompose the big file.
    """
    return int(math.ceil(pages / float(page_lim)))


def write_xmlstr(s, filepath):
    """
    Write XML string to as an XML file to the path.
    """
    print s
    with open(filepath, 'w') as f:
        parser = etree.XMLParser(recover=True)
        tree = etree.fromstring(s, parser=parser)
        f.write(ET.tostring(tree))
