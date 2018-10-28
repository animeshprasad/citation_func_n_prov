NUMBER = r'(\d)+'
DECIMAL = r'\d+(\.\d+)?%?'
AUTHOR = r'[A-Za-z\- ]+'
YEAR = r'((19|20)(\d){2})'
CITATION_AUTHOR = r'([A-Za-z\-&],? *)+(et al\.)?'
CITATION_YEAR = r'((19|20)(\d){2}[a-z]?)'  # Year (with or without letter)

# Pattern 1: [2], [3, 4], [11-16]
CITATION_PATTERN_1 = r'\[((\d+, *)*|(\d+-))\d+\]'

# Pattern 2: Cole and Cole, 1971; Hirsch, 2005b; (1998), with or without the bracket
CITATION_PATTERN_2 = r'(\(|\[)(' + CITATION_YEAR + r'(, )?)+\)' + r'|' + \
                     '\(?' + CITATION_AUTHOR + ', *(' + CITATION_YEAR + r'(, *)?)+(\)|\])?'


SENT_START = r'<S ?(sid ?= ?"\d+")? ?(ssid ?= ?"\d+")?>'
SENT_END = r'</S>'
