"""
Logger to both print and save outputs.
"""
import sys


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('result.txt', 'a')

    def write(self, message):
        self.terminal.write(message)
        if '- loss:' in message or '- acc:' in message or '==>' in message or '/1144' in message:
            return
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger()
