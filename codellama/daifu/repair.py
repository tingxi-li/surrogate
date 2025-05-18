from .context import CT_MANAGER
from .transform import TRANSFORM_REGISTRY
from . import transform
import traceback
import pdb
import inspect
import multiprocessing
from pathlib import Path
import shutil
from difflib import SequenceMatcher
from bidict import bidict
import types
import code
import sys
import os
from loguru import logger
import better_exceptions
from ._better_exceptions import ExceptionFormatter
import ast
from daifu import astor_us as astor
from .repair_strategy.gpt import GPTRepairBotProxy
import pickle

class Repair():
    def __init__(self, frame_summary, exception):
        self.frame_summary = frame_summary
        self.exception = exception
        self.plays = []

    def record_play(self, play):
        self.plays.append(play)

class EmptyContextManager:
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

_stdin = [None]
repair_lock = EmptyContextManager()
#_stdin_lock = multiprocessing.Manager().Lock()
try:
    _stdin_fd = sys.stdin.fileno()
except Exception:
    _stdin_fd = None

class DaiFuDebugger(pdb.Pdb):
    def _cmdloop(self):
        stdin_bak = sys.stdin
        # with _stdin_lock:
        try:
            if _stdin_fd is not None:
                if not _stdin[0]:
                    _stdin[0] = os.fdopen(_stdin_fd)
                sys.stdin = _stdin[0]
            self.cmdloop()
        finally:
            sys.stdin = stdin_bak

    def do_describe(self, arg):
        original_func_name = '_'.join(
            RP_MANAGER.get_current().frame_summary.name.split('_')[:-3])

        original_func_file = Path.cwd()/'daifu_workspace' / \
            ('(original)'+original_func_name+'.py')

        with original_func_file.open('r') as f:
            lines = f.readlines()

        faulty_cell_name = RP_MANAGER.get_current().frame_summary.name
        faulty_lineno = RP_MANAGER.get_current().frame_summary.lineno

        faulty_cell_name_items = faulty_cell_name.split('_')
        if faulty_cell_name_items[-2] == 'rest':
            faulty_cell_name_items[-2] = 'cell'
            faulty_cell_name = '_'.join(faulty_cell_name_items)
            faulty_lineno -= 3

        mapping_error_lineno_in_origin_file = TRANSFORM_REGISTRY[original_func_name]['mapping'].inverse[(
            faulty_cell_name, faulty_lineno)][1]

        for lineno, line in enumerate(lines, 1):
            s = str(lineno).rjust(3)
            if len(s) < 4:
                s += ' '
            s += ' '
            if lineno == mapping_error_lineno_in_origin_file:
                s += '>>'
            self.message(s + '\t' + line.rstrip())

        s = 'Err  '
        self.message(s + '\t' + RP_MANAGER.get_current().exception.__repr__())
        #traceback.print_tb(RP_MANAGER.get_current().exception.__traceback__)
        #print(better_exceptions.formatter.ExceptionFormatter().format_traceback(RP_MANAGER.get_current().exception.__traceback__)[0])
        current_exception = RP_MANAGER.get_current().exception
        #print(''.join(better_exceptions.format_exception(type(current_exception), current_exception, current_exception.__traceback__)))
        exc_formatter = ExceptionFormatter(first_filename=TRANSFORM_REGISTRY[original_func_name]['fakename'], first_lineno=faulty_lineno, first_function=faulty_cell_name)
        print(''.join(exc_formatter.format_exception(type(current_exception), current_exception, current_exception.__traceback__)))
        #print(exc_formatter.latest_meta_info)
        #print(traceback.print_exception(type(current_exception), current_exception, current_exception.__traceback__))

    def do_focus(self, arg):
        original_func_name = '_'.join(
            RP_MANAGER.get_current().frame_summary.name.split('_')[:-3])

        original_func_file = Path.cwd()/'daifu_workspace' / \
            ('(original)'+original_func_name+'.py')

        surgery_func_file = Path.cwd()/'daifu_workspace' / \
            ('(surgery-'+str(RP_MANAGER.get_queue_len())+')'+original_func_name+'.py')

        shutil.copyfileobj(original_func_file.open('r'),
                           surgery_func_file.open('w'))

        action_func_file = Path.cwd()/'daifu_workspace' / \
            ('(action-'+str(RP_MANAGER.get_queue_len())+')'+original_func_name+'.py')

        with action_func_file.open('w') as f:
            f.write('')

        self.message(str(Path.cwd()/'daifu_workspace' /
                     ('(surgery-'+str(RP_MANAGER.get_queue_len())+')'+original_func_name+'.py')) + ' generaterd.')
        self.message(str(Path.cwd()/'daifu_workspace' /
                     ('(action-'+str(RP_MANAGER.get_queue_len())+')'+original_func_name+'.py')) + ' generaterd.')
        
    def do_pass(self, arg):
        RP_MANAGER.get_current().record_play('Pass')
        logger.info("Repairing... Pass")


    def do_action(self, arg):
        original_func_name = '_'.join(
            RP_MANAGER.get_current().frame_summary.name.split('_')[:-3])

        action_func_file = Path.cwd()/'daifu_workspace' / \
            ('(action-'+str(RP_MANAGER.get_queue_len())+')'+original_func_name+'.py')

        try:
            with action_func_file.open('r') as f:
                action_code = f.read()
        except:
            self.message('Action Fails!')
            self.message('Reason: You should create focus first!')
            return

        code.InteractiveInterpreter(transform.globals_envs['_'.join(RP_MANAGER.get_current(
        ).frame_summary.name.split('_')[:-3])]).runsource(action_code, filename=action_func_file, symbol='exec')

        RP_MANAGER.get_current().record_play('Action')
        logger.info("Repairing... Action")

    def do_surgery(self, arg):
        original_func_name = '_'.join(
            RP_MANAGER.get_current().frame_summary.name.split('_')[:-3])

        surgery_func_file = Path.cwd()/'daifu_workspace' / \
            ('(surgery-'+str(RP_MANAGER.get_queue_len())+')'+original_func_name+'.py')

        try:
            with surgery_func_file.open('r') as f:
                surgery_code = f.read()
        except:
            self.message('Surgery Fails!')
            self.message('Reason: You should create focus first!')
            return

        faulty_cell_name = RP_MANAGER.get_current().frame_summary.name
        faulty_lineno = RP_MANAGER.get_current().frame_summary.lineno

        faulty_cell_name_items = faulty_cell_name.split('_')
        if faulty_cell_name_items[-2] == 'rest':
            faulty_cell_name_items[-2] = 'cell'
            faulty_cell_name = '_'.join(faulty_cell_name_items)
            faulty_lineno -= 3

        transform.update_code(original_func_name, surgery_code,
                             faulty_cell_name, faulty_lineno)
        
        RP_MANAGER.get_current().record_play('Surgery')
        logger.info("Repairing... Surgery")

    def do_broadcast(self, arg):
        original_func_name = '_'.join(
            RP_MANAGER.get_current().frame_summary.name.split('_')[:-3])
        faulty_cell_name = RP_MANAGER.get_current().frame_summary.name
        faulty_lineno = RP_MANAGER.get_current().frame_summary.lineno

        exception_name = repr(RP_MANAGER.get_current().exception)

        broadcast_origin = (original_func_name, faulty_cell_name, faulty_lineno, exception_name, RP_MANAGER.get_current().plays)

        broadcast_origin_file = Path.cwd()/'daifu_workspace' / \
            (str(RP_MANAGER.get_queue_len())+'.broadcast')
        
        with broadcast_origin_file.open('wb') as f:
            pickle.dump(broadcast_origin, f)


    def do_exit(self, arg):
        sys.exit('Deliberately Exit This Program.')

def retrieve_statement_from_line(code, line_number):
    try:
        # Parse the Python code into an Abstract Syntax Tree (AST)
        tree = ast.parse(code)
        
        # Find the node corresponding to the specified line number
        target_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AST) and hasattr(node, 'lineno') and node.lineno == line_number:
                target_node = node
                break
        
        # Reconstruct the statement from the target node and its ancestors
        if target_node:
            statement = astor.to_source(target_node)
            return statement
        
        return None  # Line not found in the AST
    
    except SyntaxError:
        return None  # Invalid Python code

class RepairManager():
    def __init__(self):
        self.queue = []
        self.debugger = DaiFuDebugger(nosigint=True)
        self.debugger.prompt = '(DaiFu) '
        self.newest_action = 0

        self.same_location_time = 0
        self.last_original_func_name = None
        self.last_faulty_cell_name = None
        self.last_faulty_lineno = None
        self.last_exception_name = None

    def analyze_current_fault(self):
        current_exception = CT_MANAGER.get_current().exception
        #current_exception = sys.exc_info()[1]
        frame_summaries = traceback.extract_tb(
            current_exception.__traceback__)
        target = 0
        for i, frame_summary in enumerate(frame_summaries):
            if '_'.join(frame_summary.name.split('_')[:-1]).endswith('daifu_cell') or '_'.join(frame_summary.name.split('_')[:-1]).endswith('daifu_rest'):
                target = i
        return Repair(frame_summaries[target], current_exception)

    def get_current(self):
        return self.queue[-1]
    
    def get_queue_len(self):
        return len(self.queue)
    
    def automated_diagnose(self, original_func_name, faulty_cell_name, faulty_lineno):
        current_exception = RP_MANAGER.get_current().exception

        exc_formatter = ExceptionFormatter(first_filename=TRANSFORM_REGISTRY[original_func_name]['fakename'], first_lineno=faulty_lineno, first_function=faulty_cell_name)
        tb=''.join(exc_formatter.format_exception(type(current_exception), current_exception, current_exception.__traceback__))
        print(tb)
        meta_info = exc_formatter.latest_meta_info

        suspected_functions = [info['function'] for info in meta_info[:exc_formatter.len_considered_frames]]
        print('Suspected Functions:', suspected_functions)

        #if len(suspected_functions) == 1:
        #    print('Culprit Function:')
        #    print(suspected_functions[0])
        #    return 'surgery', None

        culprit_function, traceback_explanation, culprit_explaination = GPTRepairBotProxy().diagnose(tb, suspected_functions)
        print('Traceback Explanation:')
        print(traceback_explanation)
        print('Culprit Function:')
        print(culprit_function)
        print('Culprit Explaination:')
        print(culprit_explaination)
        
        if len(suspected_functions) == 0:
            return None, None
        if culprit_function == suspected_functions[0]:
            return 'surgery', None
        else:
            for info in meta_info:
                if info['function'] == culprit_function:
                    return 'action', info['f_code']
            return None, None
                
    def automated_action(self, original_func_name, located_f_code):
        def retrieve_corressponding_functions(f_code):
            import gc
            retrieved_result = []
            for f in gc.get_referrers(f_code):
                if isinstance(f, types.FunctionType) and f.__code__ == f_code:
                    retrieved_result.append(f)
            return retrieved_result
        
        def clean_indent(src):
            """Clean up indentation from src.
            Any whitespace that can be uniformly removed from the first line
            onwards is removed."""
            try:
                lines = src.expandtabs().split('\n')
            except UnicodeError:
                return None
            else:
                # Find minimum indentation of any non-blank lines after first line.
                margin = sys.maxsize
                for line in lines:
                    content = len(line.lstrip())
                    if content:
                        indent = len(line) - content
                        margin = min(margin, indent)
                # Remove indentation.
                if lines:
                    lines[0] = lines[0].lstrip()
                if margin < sys.maxsize:
                    for i in range(1, len(lines)):
                        lines[i] = lines[i][margin:]
                return "\n".join(lines)
            
        def extract_function_name(code):
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        
        located_functions = retrieve_corressponding_functions(located_f_code)

        located_code = clean_indent(inspect.getsource(located_f_code))

        fixed_code = GPTRepairBotProxy().action_code(located_code)

        print('Located Code: ')
        print(located_code)

        print('Fixed Code: ')
        print(fixed_code)

        f_name = extract_function_name(fixed_code)

        temp_globals = {}
        exec(fixed_code, temp_globals)

        for located_function in located_functions:
            located_function.__code__ = temp_globals[f_name].__code__

        print('-'*50)

        action_func_file = Path.cwd()/'daifu_workspace' / \
            ('(action-'+str(RP_MANAGER.get_queue_len())+')'+original_func_name+'.py')
        
        record_of_this_action = '#ONLY FOR RECORD, CANNOT BE EXECUTED DIRECTLY!\n#located code:\n'+located_code+'\n#corressponding fixed code:\n'+fixed_code
        
        with action_func_file.open('w') as f:
            f.write(record_of_this_action)
        
        logger.info("Repairing... Action")

            
    def automated_surgery(self, original_func_name, faulty_cell_name, faulty_lineno):
        original_func_file = Path.cwd()/'daifu_workspace' / \
            ('(original)'+original_func_name+'.py')

        with original_func_file.open('r') as f:
            lines = f.readlines()

        faulty_cell_name_items = faulty_cell_name.split('_')
        if faulty_cell_name_items[-2] == 'rest':
            faulty_cell_name_items[-2] = 'cell'
            faulty_cell_name = '_'.join(faulty_cell_name_items)
            faulty_lineno -= 3
        
        print('Faulty_cell_name and lineno', faulty_cell_name, faulty_lineno)
        mapping_error_lineno_in_origin_file = TRANSFORM_REGISTRY[original_func_name]['mapping'].inverse[(faulty_cell_name, faulty_lineno)][1]
        print('Mapping Error Line No. in Origin File: ', mapping_error_lineno_in_origin_file)

        target_code = ''.join(lines)
        target_exception = RP_MANAGER.get_current().exception.__repr__()

        retrieved_statement = retrieve_statement_from_line(target_code, mapping_error_lineno_in_origin_file)
        len_retrieved_statement = len(retrieved_statement.split('\n'))
        print('Retrieved Statement: ')
        print(retrieved_statement)
        print('Length of Retrieved Statement: ', len_retrieved_statement)
        target_masked_code = ''.join(lines[:mapping_error_lineno_in_origin_file-1]) + '[MASK]\n' + ''.join(lines[mapping_error_lineno_in_origin_file+len_retrieved_statement-2:])

        target_code = '\n'.join(target_code.split('\n')[1:])
        target_masked_code = '\n'.join(target_masked_code.split('\n')[1:])
        target_faulty_lines = ''.join(lines[mapping_error_lineno_in_origin_file-1:mapping_error_lineno_in_origin_file+len_retrieved_statement-2])

        print('Target Code: ')
        print(target_code)
        print('Target Exception: ')
        print(target_exception)
        print('Target Masked Code: ')
        print(target_masked_code)
        print('Target Faulty Lines: ')
        print(target_faulty_lines)


        surgery_answer = GPTRepairBotProxy().surgery(target_masked_code, target_exception, target_faulty_lines)

        print('Surgery Answer:')
        print(surgery_answer)

        # Check Indentation
        surgery_answer_first_row = surgery_answer.split('\n')[0]
        original_fault_line = lines[mapping_error_lineno_in_origin_file-1]
        surgery_indent_num = len(surgery_answer_first_row) - len(surgery_answer_first_row.lstrip())
        original_indent_num = len(original_fault_line) - len(original_fault_line.lstrip())
        while surgery_indent_num != original_indent_num:
            print('Indentation Error!')
            surgery_answer = GPTRepairBotProxy().surgery(target_masked_code, target_exception, target_faulty_lines)
            print('New Surgery Answer:')
            print(surgery_answer)
            surgery_answer_first_row = surgery_answer.split('\n')[0]
            original_fault_line = lines[mapping_error_lineno_in_origin_file-1]
            surgery_indent_num = len(surgery_answer_first_row) - len(surgery_answer_first_row.lstrip())
            original_indent_num = len(original_fault_line) - len(original_fault_line.lstrip())

        surgery_code = ''.join(lines[:mapping_error_lineno_in_origin_file-1]) + surgery_answer + ''.join(lines[mapping_error_lineno_in_origin_file+len_retrieved_statement-2:])

        print('Surgery Code: ')
        print(surgery_code)

        print('-'*50)

        surgery_func_file = Path.cwd()/'daifu_workspace' / \
            ('(surgery-'+str(RP_MANAGER.get_queue_len())+')'+original_func_name+'.py')

        
        with surgery_func_file.open('w') as f:
            f.write(surgery_code)

    def repair(self):
        print('Try Repairing...')
        try:
            with repair_lock:
                print('Enter Repairing...')
                try:
                    self.queue.append(self.analyze_current_fault())
                    logger.info("Repair " + str(len(self.queue)) + " Begin")

                    original_func_name = '_'.join(
                        RP_MANAGER.get_current().frame_summary.name.split('_')[:-3])
                    faulty_cell_name = RP_MANAGER.get_current().frame_summary.name
                    faulty_lineno = RP_MANAGER.get_current().frame_summary.lineno
                    exception_name = repr(RP_MANAGER.get_current().exception)

                    transform.init_rest(original_func_name, faulty_cell_name, faulty_lineno)

                    #logger.debug(traceback.format_tb(RP_MANAGER.get_current().exception.__traceback__))
                    
                    if self.last_original_func_name == original_func_name and self.last_faulty_cell_name == faulty_cell_name and self.last_faulty_lineno == faulty_lineno and self.last_exception_name == exception_name:
                        self.same_location_time +=1
                        if self.same_location_time > 1:
                            need_manual = True
                        else:
                            need_manual = False
                    else:
                        self.same_location_time = 0
                        need_manual = False

                    #ONLY For Repeated Experiments
                    is_ok = False
                    if 'REPEAT_ACTIONS' in TRANSFORM_REGISTRY:
                        self.debugger.reset()
                        action = TRANSFORM_REGISTRY['REPEAT_ACTIONS'][self.newest_action]
                        if action == 'Action':
                            self.debugger.do_action(None)
                        elif action == 'Surgery':
                            self.debugger.do_surgery(None)
                        elif isinstance(action, list):
                            for action_per in action:
                                if action_per == 'Action':
                                    self.debugger.do_action(None)
                                elif action_per == 'Surgery':
                                    self.debugger.do_surgery(None)
                        elif action == 'Pass':
                            self.debugger.do_pass(None)
                        self.newest_action += 1
                        is_ok = True
                    elif TRANSFORM_REGISTRY['DIRECTLY_RETRY']:
                        is_ok = True
                    elif TRANSFORM_REGISTRY['DIRECTLY_RETRY_FIRST'] and not need_manual and self.same_location_time == 0:
                        is_ok = True
                    else:
                        broadcast_origin_file = Path.cwd()/'daifu_workspace' / \
                            (str(RP_MANAGER.get_queue_len())+'.broadcast')
                        if broadcast_origin_file.exists():
                            with broadcast_origin_file.open('rb') as f:
                                broadcast_origin = pickle.load(f)
                            if broadcast_origin[0] == original_func_name and broadcast_origin[1] == faulty_cell_name and broadcast_origin[2] == faulty_lineno and broadcast_origin[3] == exception_name:
                                self.debugger.reset()
                                for play in broadcast_origin[4]:
                                    if play == 'Action':
                                        self.debugger.do_action(None)
                                    elif play == 'Surgery':
                                        self.debugger.do_surgery(None)
                                    elif play == 'Pass':
                                        self.debugger.do_pass(None)
                                is_ok = True
                    if not is_ok:
                        if TRANSFORM_REGISTRY['IS_AUTOMATIC'] and not need_manual:
                            next_action, f_code = self.automated_diagnose(original_func_name, faulty_cell_name, faulty_lineno)
                            
                            if next_action == 'action':
                                self.automated_action(original_func_name, f_code)
                            elif next_action == 'surgery':
                                self.automated_surgery(original_func_name, faulty_cell_name, faulty_lineno)
                                self.debugger.reset()
                                self.debugger.do_surgery(None)
                            else:
                                print('No Action! Directly Retry and Continue!')
                        else:
                            self.debugger.reset()
                            self.debugger.interaction(
                                None, self.get_current().exception.__traceback__)

                    #faulty_cell_name = RP_MANAGER.get_current().frame_summary.name
                    #faulty_cell_name_items = faulty_cell_name.split('_')
                    #faulty_cell_name_items[-2] = 'rest'
                    #rest_cell_name = '_'.join(faulty_cell_name_items)

                    self.last_original_func_name = original_func_name
                    self.last_faulty_cell_name = faulty_cell_name
                    self.last_faulty_lineno = faulty_lineno
                    self.last_exception_name = exception_name

                    logger.info("Repair " + str(len(self.queue)) + " End")
                    #return transform.globals_envs['_'.join(RP_MANAGER.get_current().frame_summary.name.split('_')[:-3])][rest_cell_name]
                except Exception as e:
                    print('Exception in RepairManager.repair():', e)
                    print(better_exceptions.formatter.ExceptionFormatter().format_traceback(e.__traceback__)[0])
                    print('Exception in Vaccinated Code:', RP_MANAGER.get_current().exception)
                    print(better_exceptions.formatter.ExceptionFormatter().format_traceback(RP_MANAGER.get_current().exception.__traceback__)[0])
                    logger.info("Repair Fails")
                    sys.exit('A Bug in RepairManager.repair() Cause an Exit!')
        except Exception as e:
            print(e)
            sys.exit('A Bug in RepairManager.repair() Cause an Exit!')


RP_MANAGER = RepairManager()
