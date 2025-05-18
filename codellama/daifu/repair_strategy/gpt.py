import requests
import json
import uuid
from pprint import pprint
import re

class GPTRepairBotProxy:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8000" 

    def extract_code_blocks(self, text):
        code_pattern = r"```(.*?)```"
        code_blocks = re.findall(code_pattern, text, re.DOTALL)
        return '\n'.join(max(code_blocks, key=len).split('\n')[1:])
    
    def diagnose(self, tb, suspected_functions):
        data = {'tb':tb,
                'suspected_functions':suspected_functions}
        response = requests.post(self.base_url + "/chain_diagnose", json=data)
        print(response.json())
        return response.json()
    
    def action_code(self, code):
        data = {'code':code}
        response = requests.post(self.base_url + "/chain_action_code", json=data)
        result = self.extract_code_blocks(response.json())
        return result

    def surgery(self, masked_code, exception, faulty_lines):
        data = {'masked_code':masked_code,
                'exception':exception,
                'faulty_lines':faulty_lines}
        response = requests.post(self.base_url + "/chain_surgery", json=data)
        result = self.extract_code_blocks(response.json())
        return result
    
    def surgery_continue_to_fix_indentation(self, masked_code, exception, faulty_lines):
        data = {'masked_code':masked_code,
                'exception':exception,
                'faulty_lines':faulty_lines}
        response = requests.post(self.base_url + "/chain_surgery_continue_to_fix_indentation", json=data)
        result = self.extract_code_blocks(response.json())
        return result

