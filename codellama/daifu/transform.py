import inspect
import ast
import code
import sys
#import os
#sys.path.append(os.getcwd())
from daifu import astor_us as astor
import dis
import functools
import types
from pathlib import Path
from bidict import bidict
import copy
from difflib import SequenceMatcher
import difflib
import traceback
import multiprocessing
import os
import builtins
# sys.path.append('..')
# import daifu

globals_envs = {}
TRANSFORM_REGISTRY = {}

TRANSFORM_REGISTRY['IS_WARN'] = False
TRANSFORM_REGISTRY['IS_AUTOMATIC'] = False
TRANSFORM_REGISTRY['DIRECTLY_RETRY'] = False
TRANSFORM_REGISTRY['DIRECTLY_RETRY_FIRST'] = False

IN_ADDED_CODE = '''daifu_store = {}
for daifu_dataname in daifu.TRANSFORM_REGISTRY['%s']['dataname_list']:
    if daifu_dataname in daifu.TRANSFORM_REGISTRY['%s']['local_variables_list'] and daifu_dataname in globals():
        daifu_store[daifu_dataname] = globals()[daifu_dataname]
    if daifu_dataname in locals():
        globals()[daifu_dataname] = locals()[daifu_dataname]
    elif daifu_dataname not in globals():
        globals()[daifu_dataname] = None'''

IN_ADDED_CODE_WARN = '''daifu_store = {}
for daifu_dataname in daifu.TRANSFORM_REGISTRY['%s']['dataname_list']:
    if daifu_dataname in daifu.TRANSFORM_REGISTRY['%s']['local_variables_list'] and daifu_dataname in globals():
        daifu_store[daifu_dataname] = globals()[daifu_dataname]
        print('Warning:', daifu_dataname, 'is reused, please check.')
    if daifu_dataname in locals():
        globals()[daifu_dataname] = locals()[daifu_dataname]
    elif daifu_dataname not in globals():
        globals()[daifu_dataname] = None'''

OUT_ADDED_CODE = '''for daifu_dataname in daifu_store:
    globals()[daifu_dataname] = daifu_store[daifu_dataname]
'''


class LinenoRecorder(ast.NodeTransformer):
    def generic_visit(self, node):
        super(LinenoRecorder, self).generic_visit(node)
        if hasattr(node, 'lineno'):
            node.recorded_lineno = - node.lineno
        return node


class Return2BreakTransformer(ast.NodeTransformer):
    def __init__(self):
        super(Return2BreakTransformer, self).__init__()

    def visit_Return(self, node):
        node = ast.Break(original_return_value=node.value,
                         recorded_lineno=node.recorded_lineno)
        return node


class TryExceptTransformer(ast.NodeTransformer):
    def __init__(self, dl_func_name, dataname_list):
        super(TryExceptTransformer, self).__init__()
        self.cell_num = 0
        self.dataname_list = dataname_list
        self.is_first_visit_FunctionDef = True
        self.dl_func_name = dl_func_name

    def visit_FunctionDef(self, node):
        if self.is_first_visit_FunctionDef:
            self.is_first_visit_FunctionDef = False
            self.cell_num += 1
            cell_num = self.cell_num
            self.generic_visit(node)
            node.body = [ast.Global(names=self.dataname_list)] + node.body
            node.body = [ast.Try(body=node.body,
                                handlers=[ast.ExceptHandler(type=ast.Name(id='Exception'),
                                                            name=None,
                                                            daifu_added=True,
                                                            body=[ast.Pass()])],
                                orelse=[],
                                finalbody=[],
                                cell_num=cell_num)]
            return node
        else:
            sccode = astor.to_source(node)
            #print(sccode)
            code.InteractiveInterpreter(
                globals_envs[self.dl_func_name]).runsource(sccode)
            


    def visit_For(self, node):
        self.cell_num += 1
        cell_num = self.cell_num
        self.generic_visit(node)
        node.body = [ast.Global(names=self.dataname_list)] + node.body
        node.body = [ast.Try(body=node.body,
                             handlers=[ast.ExceptHandler(type=ast.Name(id='Exception'),
                                                         name=None,
                                                         daifu_added=True,
                                                         body=[ast.Pass()])],
                             orelse=[],
                             finalbody=[],
                             cell_num=cell_num)]
        return node

    def visit_While(self, node):
        self.cell_num += 1
        cell_num = self.cell_num
        self.generic_visit(node)
        node.body = [ast.Global(names=self.dataname_list)] + node.body
        node.body = [ast.Try(body=node.body,
                             handlers=[ast.ExceptHandler(type=ast.Name(id='Exception'),
                                                         name=None,
                                                         daifu_added=True,
                                                         body=[ast.Pass()])],
                             orelse=[],
                             finalbody=[],
                             cell_num=cell_num)]
        return node

    def visit_With(self, node):
        self.cell_num += 1
        cell_num = self.cell_num
        self.generic_visit(node)
        node.body = [ast.Global(names=self.dataname_list)] + node.body
        node.body = [ast.Try(body=node.body,
                             handlers=[ast.ExceptHandler(type=ast.Name(id='Exception'),
                                                         name=None,
                                                         daifu_added=True,
                                                         body=[ast.Pass()])],
                             orelse=[],
                             finalbody=[],
                             cell_num=cell_num,
                             is_with=True)]
        return node


class NameAccumulator(ast.NodeVisitor):
    def __init__(self):
        super(NameAccumulator, self).__init__()
        self.name_set = set()
        self.argname_set = set()
        self.funcname_set = set()
        self.assignname_set = set()

    def visit_Name(self, node):
        self.name_set.add(node.id)
        self.generic_visit(node)

    def visit_arg(self, node):
        self.argname_set.add(node.arg)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.funcname_set.add(node.func.id)
        self.generic_visit(node)

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name):
            self.assignname_set.add(node.targets[0].id)
        if isinstance(node.targets[0], ast.Tuple):
            for v in node.targets[0].elts:
                if isinstance(v, ast.Name):
                    self.assignname_set.add(v.id)
        self.generic_visit(node)

    def get_dataname_set(self):
        return (((self.name_set - self.funcname_set) | self.argname_set) - set(dir(builtins))) | self.assignname_set
        #return (self.name_set - self.funcname_set) | self.argname_set | self.assignname_set


class Break2ReturnTransformer(ast.NodeTransformer):
    def __init__(self, cell_name):
        super(Break2ReturnTransformer, self).__init__()
        self.cell_name = cell_name

    def visit_Break(self, node):
        if hasattr(node, 'no_return'):
            return node
        elif hasattr(node, 'daifu_return_tag'):
            node = ast.Return(value=ast.Tuple(elts=[ast.Name(id='daifu_return_tag'), ast.Name(
                id='daifu_return_item')]))
        elif hasattr(node, 'original_return_value'):
            global TRANSFORM_REGISTRY
            dl_func_name = '_'.join(self.cell_name.split('_')[:-3])

            if node.original_return_value is None:
                node.original_return_value = ast.NameConstant(value=None)

            TRANSFORM_REGISTRY[dl_func_name]['final_return'][self.cell_name] = node.original_return_value

            node = ast.Return(value=ast.Tuple(elts=[ast.Str(s=self.cell_name), node.original_return_value]),
                              recorded_lineno=node.recorded_lineno)
        else:
            node = ast.Return(value=ast.Tuple(elts=[ast.Str(s='break'), ast.NameConstant(value=None)]),
                              recorded_lineno=node.recorded_lineno)
        return node

    def visit_Continue(self, node):
        if hasattr(node, 'no_return'):
            return node
        else:
            node = ast.Return(value=ast.Tuple(elts=[ast.NameConstant(value=None), ast.NameConstant(value=None)]),
                              recorded_lineno=node.recorded_lineno)
        return node


class ReturnAccumulator(ast.NodeVisitor):
    def __init__(self):
        super(ReturnAccumulator, self).__init__()
        self.return_set = set()

    def visit_Return(self, node):
        self.return_set.add(node.value.elts[0])

    def visit_ExceptHandler(self, node):
        if hasattr(node, 'daifu_added'):
            pass
        else:
            self.generic_visit(node)

    def get_return_set(self):
        return self.return_set


class LineMapper(ast.NodeTransformer):
    def generic_visit(self, node):
        super(LineMapper, self).generic_visit(node)
        if hasattr(node, 'recorded_lineno'):
            node.lineno = node.recorded_lineno
        return node


def create_TransformMapping(dl_func_name, transform_block_name, sccode_with_lineno):
    global TRANSFORM_REGISTRY

    code_lines = [line.strip() for line in sccode_with_lineno.split('\n')]

    current_line_num = 1
    for line in code_lines:
        if not line.startswith('#'):
            current_line_num += 1
        elif line.startswith('# line: -'):
            TRANSFORM_REGISTRY[dl_func_name]['mapping'][(dl_func_name, int(
                line.split('-')[-1]))] = (transform_block_name, current_line_num)


class Block2FuncTransformer(ast.NodeTransformer):
    def __init__(self, dl_func_name, is_update):
        super(Block2FuncTransformer, self).__init__()
        self.name_set = set()
        self.funcname_set = set()
        self.dl_func_name = dl_func_name
        self.is_update = is_update

    def visit_Name(self, node):
        self.name_set.add(node.id)
        self.generic_visit(node)
        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.funcname_set.add(node.func.id)
        self.generic_visit(node)
        return node

    def visit_arg(self, node):
        self.name_set.add(node.arg)
        self.generic_visit(node)
        return node

    def get_dataname_set(self):
        return self.name_set - self.funcname_set

    def visit_Try(self, node):
        #before_set = self.get_dataname_set()
        # print('beforeset', before_set)
        self.generic_visit(node)

        global TRANSFORM_REGISTRY

        if hasattr(node, 'cell_num'):
            sctree = ast.FunctionDef(name=self.dl_func_name+'_daifu_cell_'+str(node.cell_num),
                                     args=ast.arguments(args=[],
                                                        vararg=None,
                                                        kwonlyargs=[],
                                                        kw_defaults=[],
                                                        kwarg=None,
                                                        defaults=[]),
                                     body=[ast.Try(body=node.body,
                                                   handlers=[ast.ExceptHandler(type=ast.Name(id='Exception'),
                                                                               name=self.dl_func_name+'_exception_' +
                                                                               str(node.cell_num),
                                                                               daifu_added=True,
                                                                               body=[ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Attribute(value=ast.Name(id='daifu'),
                                                                                                                                                    attr='CT_MANAGER'),
                                                                                                                                attr='save'),
                                                                                                             args=[ast.Call(func=ast.Name(
                                                                                                                 id='locals'), args=[], keywords=[])],
                                                                                                             keywords=[])),
                                                                                     ast.Raise(exc=None,
                                                                                               cause=None)])],
                                                   orelse=[ast.Return(value=ast.Tuple(elts=[ast.NameConstant(value=None), ast.NameConstant(
                                                       value=None)]))],
                                                   finalbody=[])],
                                     decorator_list=[],
                                     returns=None)

            sctree = ast.fix_missing_locations(
                Break2ReturnTransformer(self.dl_func_name+'_daifu_cell_'+str(node.cell_num)).visit(sctree))
            # sctree = ast.fix_missing_locations(sctree)
            # print(astor.dump_tree(sctree))
            # print(astor.to_source(sctree))
            # print(astor.dump_tree(sctree))

            sccode = astor.to_source(sctree)
            TRANSFORM_REGISTRY[self.dl_func_name]['transformed'][self.dl_func_name +
                                                                 '_daifu_cell_'+str(node.cell_num)] = sccode

            sctree = LineMapper().visit(sctree)
            sccode_with_lineno = astor.to_source(
                sctree, add_line_information=True)
            create_TransformMapping(self.dl_func_name, self.dl_func_name +
                                    '_daifu_cell_'+str(node.cell_num), sccode_with_lineno)

            scfile = Path(TRANSFORM_REGISTRY[self.dl_func_name]['workspace'])/(
                '(transformed)' + self.dl_func_name + '_daifu_cell_'+str(node.cell_num)+'.py')
            if "MainProcess" in multiprocessing.current_process().name or self.is_update:
                with scfile.open('w') as f:
                    f.write(sccode)
                    # f.write(sccode_with_lineno)

            return_accumulator = ReturnAccumulator()
            return_accumulator.visit(node)
            return_set = list(return_accumulator.get_return_set())

            node.body = [ast.Assign(targets=[ast.Tuple(elts=[ast.Name(id='daifu_return_tag'), ast.Name(id='daifu_return_item')])],
                                    value=ast.Call(func=ast.Name(id=self.dl_func_name+'_daifu_cell_'+str(node.cell_num)),
                                                   args=[],
                                                   keywords=[]))]

            # if not (len(return_set) == 1 and isinstance(return_set[0], ast.NameConstant) and return_set[0].value==None):
            if len(return_set) > 0 and not (len(return_set) == 1 and isinstance(return_set[0], ast.NameConstant) and return_set[0].value == None) and not hasattr(node, 'is_with'):
                node.body.append(ast.If(test=ast.Compare(left=ast.Name(id='daifu_return_tag'),
                                                         ops=[ast.IsNot()],
                                                         comparators=[ast.NameConstant(value=None)]),
                                        body=[ast.If(test=ast.Compare(left=ast.Name(id='daifu_return_tag'),
                                                                      ops=[
                                            ast.Eq()],
                                            comparators=[ast.Str(s='break')]),
                                            body=[
                                            ast.Break(no_return=True)],
                                            orelse=[ast.Break(daifu_return_tag=True)])],
                                        orelse=[]))

            # node.body.append(ast.Expr(value=ast.Call(func=ast.Name(
            #    id='print'), args=[ast.Str(s='NEW!!!!!')], keywords=[])))

            node.handlers = [ast.ExceptHandler(type=ast.Name(id='Exception'),
                                               name=None,
                                               daifu_added=True,
                                               body=[ast.While(test=ast.NameConstant(value=True),
                                                               body=[ast.Try(body=[ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Attribute(value=ast.Name(id='daifu'),
                                                                                                                                                  attr='RP_MANAGER'),
                                                                                                                              attr='repair'),
                                                                                                           args=[],
                                                                                                           keywords=[])),
                                                                                   ast.Assign(targets=[ast.Tuple(elts=[ast.Name(id='daifu_return_tag'), ast.Name(id='daifu_return_item')])],
                                                                                              value=ast.Call(func=ast.Name(id=self.dl_func_name+'_daifu_rest_'+str(node.cell_num)),
                                                                                                             args=[],
                                                                                                             keywords=[])),
                                                                                   ast.If(test=ast.Compare(left=ast.Name(id='daifu_return_tag'),
                                                                                                           ops=[
                                                                                                               ast.IsNot()],
                                                                                                           comparators=[ast.NameConstant(value=None)]),
                                                                                          body=[ast.If(test=ast.Compare(left=ast.Name(id='daifu_return_tag'),
                                                                                                                        ops=[
                                                                                                                            ast.Eq()],
                                                                                                                        comparators=[ast.Str(s='break')]),
                                                                                                       body=[
                                                                                                           ast.Break(no_return=True)],
                                                                                                       orelse=[ast.Break(daifu_return_tag=True)])],
                                                                                          orelse=[]),
                                                                                   ast.Break(no_return=True)],
                                                                             handlers=[ast.ExceptHandler(type=ast.Name(id='Exception'),
                                                                                                         name=None,
                                                                                                         daifu_added=True,
                                                                                                         body=[ast.Continue(no_return=True)])],
                                                                             orelse=[],
                                                                             finalbody=[])],
                                                               orelse=[])])]

            if not hasattr(node, 'is_with'):
                node.handlers[0].body.append(ast.If(test=ast.Compare(left=ast.Name(id='daifu_return_tag'),
                                                                     ops=[
                                                                         ast.Eq()],
                                                                     comparators=[ast.Str(s='break')]),
                                                    body=[
                                                        ast.Break(no_return=True)],
                                                    orelse=[],
                                                    mark_remove_if_outer=True))

        return node


class IfRemoveTransformer(ast.NodeTransformer):
    def __init__(self):
        super(IfRemoveTransformer, self).__init__()

    def visit_If(self, node):
        return
        ## global TRANSFORM_REGISTRY
        # if len(TRANSFORM_REGISTRY[self.dl_func_name]['final_return']) == 0 or hasattr(node, 'mark_remove_if_outer'):
        # return
        ## node = ast.Return(value=ast.Name(id='daifu_return_item'))

        # node = ast.If()
        # cur_node = node
        # return_list = list(
        #     TRANSFORM_REGISTRY[self.dl_func_name]['final_return'].items())
        # for cell_name, return_values in return_list[:-1]:
        #     cur_node.test = ast.Compare(left=ast.Name(id='daifu_return_tag'), ops=[
        #                                 ast.Eq()], comparators=[ast.Str(s=cell_name)])
        #     cur_node.body = [return_values]
        #     cur_node.orelse = [ast.If()]
        #     cur_node = cur_node.orelse[0]
        # cell_name, return_values = return_list[-1]
        # cur_node.test = ast.Compare(left=ast.Name(id='daifu_return_tag'), ops=[
        #                             ast.Eq()], comparators=[ast.Str(s=cell_name)])
        # cur_node.body = [return_values]
        # cur_node.orelse = []

        # cur_node = node
        # node = ast.Try(body=[cur_node],
        #                handlers=[ast.ExceptHandler(type=None,
        #                                            name=None,
        #                                            daifu_added=True,
        #                                            body=[ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Attribute(value=ast.Name(id='daifu'),
        #                                                                                                                 attr='CT_MANAGER'),
        #                                                                                             attr='save'),
        #                                                                          args=[ast.Call(func=ast.Name(
        #                                                                              id='locals'), args=[], keywords=[])],
        #                                                                          keywords=[]))])],
        #                orelse=[],
        #                finalbody=[])

        # return node


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


def get_code(co_consts_tuple):
    for item in co_consts_tuple:
        if isinstance(item, types.CodeType):
            return item
    raise Exception('No code in co_consts')

def get_local_variables(function_code):
    tree = ast.parse(function_code)

    local_variables = set()
    global_variables = set()

    class LocalVariableVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            local_variables.update(arg.arg for arg in node.args.args)
            self.generic_visit(node)

        def visit_For(self, node):
            if isinstance(node.target, ast.Name):
                local_variables.add(node.target.id)
            self.generic_visit(node)

        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    local_variables.add(target.id)
            self.generic_visit(node)

        def visit_Global(self, node):
            global_variables.update(name for name in node.names)
            self.generic_visit(node)

    visitor = LocalVariableVisitor()
    visitor.visit(tree)

    local_variables -= global_variables

    return list(local_variables)

def transform_code(dl_func):
    dl_func_name = dl_func.__name__

    ((Path.cwd()/'daifu_workspace')).mkdir(parents=True, exist_ok=True)

    global TRANSFORM_REGISTRY
    TRANSFORM_REGISTRY[dl_func_name] = {}

    # round-trip processing to force a unified format
    original_dl_func_tree = ast.parse(clean_indent(inspect.getsource(dl_func)))
    original_dl_func_code = astor.to_source(original_dl_func_tree)
    TRANSFORM_REGISTRY[dl_func_name]['original'] = original_dl_func_code
    TRANSFORM_REGISTRY[dl_func_name]['fakename'] = str(Path.cwd()/'the_crash_for_analysis.py')

    TRANSFORM_REGISTRY[dl_func_name]['local_variables_list'] = get_local_variables(original_dl_func_code)

    TRANSFORM_REGISTRY[dl_func_name]['transformed'] = {}
    TRANSFORM_REGISTRY[dl_func_name]['workspace'] = str(
        ((Path.cwd()/'daifu_workspace')))
    TRANSFORM_REGISTRY[dl_func_name]['mapping'] = bidict()
    TRANSFORM_REGISTRY[dl_func_name]['final_return'] = {}

    #TRANSFORM_REGISTRY[dl_func_name]['cell_arg'] = {}
    #TRANSFORM_REGISTRY[dl_func_name]['cell_return'] = {}
    TRANSFORM_REGISTRY[dl_func_name]['history'] = []

    rawfile = (Path.cwd()/'daifu_workspace')/('(original)'+dl_func_name+'.py')
    if "MainProcess" in multiprocessing.current_process().name:
        with rawfile.open('w') as f:
            f.write(original_dl_func_code)

    # round-trip processing to force a unified format (for lineno)
    original_dl_func_tree = ast.parse(original_dl_func_code)

    name_accumulator = NameAccumulator()
    # print(astor.dump_tree(node))
    name_accumulator.visit(original_dl_func_tree)
    # print(astor.dump_tree(node))
    dataname_set = name_accumulator.get_dataname_set()
    dataname_list = sorted(list(dataname_set))

    if 'daifu' in dataname_list:
        dataname_list.remove('daifu')

    TRANSFORM_REGISTRY[dl_func_name]['dataname_list'] = dataname_list

    original_dl_func_tree = LinenoRecorder().visit(original_dl_func_tree)

    transformed_dl_func_tree = ast.fix_missing_locations(
        TryExceptTransformer(dl_func_name, dataname_list).visit(original_dl_func_tree))

    transformed_dl_func_tree = ast.fix_missing_locations(
        Return2BreakTransformer().visit(transformed_dl_func_tree))

    # print('Transformed ->')
    transformed_dl_func_tree = ast.fix_missing_locations(
        Block2FuncTransformer(dl_func_name, is_update=False).visit(transformed_dl_func_tree))

    transformed_dl_func_tree = ast.fix_missing_locations(
        IfRemoveTransformer().visit(transformed_dl_func_tree))

    if TRANSFORM_REGISTRY['IS_WARN']:
        in_added_code = IN_ADDED_CODE_WARN % (dl_func_name, dl_func_name)
    else:
        in_added_code = IN_ADDED_CODE % (dl_func_name, dl_func_name)

    transformed_dl_func_tree.body[0].body = ast.parse(
        in_added_code).body + transformed_dl_func_tree.body[0].body

    transformed_dl_func_tree.body[0].body.append(
        ast.parse(OUT_ADDED_CODE).body[0])

    if len(TRANSFORM_REGISTRY[dl_func_name]['final_return']) > 0:
        transformed_dl_func_tree.body[0].body.append(
            ast.parse('return daifu_return_item').body[0])

    # print(astor.dump_tree(transformed_dl_func_tree))

    global globals_envs
    for cell_name in TRANSFORM_REGISTRY[dl_func_name]['transformed']:
        sccode = TRANSFORM_REGISTRY[dl_func_name]['transformed'][cell_name]
        scfile = Path(
            TRANSFORM_REGISTRY[dl_func_name]['workspace'])/('(transformed)' + cell_name + '.py')
        code.InteractiveInterpreter(
            globals_envs[dl_func_name]).runsource(sccode, filename=scfile)

    # print(astor.to_source(transformed_dl_func_tree))
    transformed_dl_func_code = astor.to_source(transformed_dl_func_tree)
    TRANSFORM_REGISTRY[dl_func_name]['transformed'][dl_func_name] = transformed_dl_func_code

    # sctree = LineMapper().visit(transformed_dl_func_tree)
    # sccode_with_lineno = astor.to_source(
    #     sctree, add_line_information=True)
    # create_TransformMapping(dl_func_name, dl_func_name, sccode_with_lineno)

    transformed_dl_func_file = Path(
        TRANSFORM_REGISTRY[dl_func_name]['workspace'])/('(transformed)'+dl_func_name + '.py')
    if "MainProcess" in multiprocessing.current_process().name:
        with transformed_dl_func_file.open('w') as f:
            f.write(transformed_dl_func_code)

    transformed_code = code.compile_command(
        transformed_dl_func_code, transformed_dl_func_file, 'single')

    # print(transformed_code.co_consts)
    return get_code(transformed_code.co_consts)


def print_code_diff(old_code, new_code):
    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()

    differ = difflib.Differ()

    diff = list(differ.compare(old_lines, new_lines))

    for line in diff:
        print(line)

def update_code(dl_func_name, dl_func_code, updated_cell_name, faulty_lineno):
    print(updated_cell_name)
    global globals_envs, TRANSFORM_REGISTRY
    try:
        TRANSFORM_REGISTRY[dl_func_name]['history'].append(
            copy.deepcopy(TRANSFORM_REGISTRY[dl_func_name]))

        # round-trip processing to force a unified format
        original_dl_func_tree = ast.parse(clean_indent(dl_func_code))
        original_dl_func_code = astor.to_source(original_dl_func_tree)
        TRANSFORM_REGISTRY[dl_func_name]['original'] = original_dl_func_code

        TRANSFORM_REGISTRY[dl_func_name]['transformed'] = {}
        TRANSFORM_REGISTRY[dl_func_name]['workspace'] = str(
            ((Path.cwd()/'daifu_workspace')))
        TRANSFORM_REGISTRY[dl_func_name]['mapping'] = bidict()
        TRANSFORM_REGISTRY[dl_func_name]['final_return'] = {}

        rawfile = (Path.cwd()/'daifu_workspace')/('(original)'+dl_func_name+'.py')
        with rawfile.open('w') as f:
            f.write(original_dl_func_code)

        # round-trip processing to force a unified format (for lineno)
        original_dl_func_tree = ast.parse(original_dl_func_code)

        original_dl_func_tree = LinenoRecorder().visit(original_dl_func_tree)

        transformed_dl_func_tree = ast.fix_missing_locations(
            TryExceptTransformer(dl_func_name, TRANSFORM_REGISTRY[dl_func_name]['dataname_list']).visit(original_dl_func_tree))

        transformed_dl_func_tree = ast.fix_missing_locations(
            Return2BreakTransformer().visit(transformed_dl_func_tree))

        # print('Transformed ->')
        transformed_dl_func_tree = ast.fix_missing_locations(
            Block2FuncTransformer(dl_func_name, is_update=True).visit(transformed_dl_func_tree))

        transformed_dl_func_tree = ast.fix_missing_locations(
            IfRemoveTransformer().visit(transformed_dl_func_tree))

        if TRANSFORM_REGISTRY['IS_WARN']:
            in_added_code = IN_ADDED_CODE_WARN % (dl_func_name, dl_func_name)
        else:
            in_added_code = IN_ADDED_CODE % (dl_func_name, dl_func_name)

        transformed_dl_func_tree.body[0].body = ast.parse(
            in_added_code).body + transformed_dl_func_tree.body[0].body

        transformed_dl_func_tree.body[0].body.append(
            ast.parse(OUT_ADDED_CODE).body[0])

        if len(TRANSFORM_REGISTRY[dl_func_name]['final_return']) > 0:
            transformed_dl_func_tree.body[0].body.append(
                ast.parse('return daifu_return_item').body[0])

        other_changed_cells = []

        for cell_name in TRANSFORM_REGISTRY[dl_func_name]['transformed']:
            if cell_name == updated_cell_name:
                sccode = TRANSFORM_REGISTRY[dl_func_name]['transformed'][cell_name]
                scfile = Path(
                    TRANSFORM_REGISTRY[dl_func_name]['workspace'])/('(transformed)' + cell_name + '.py')

                # try to throw exception if syntax error
                code.compile_command(sccode)

                code.InteractiveInterpreter(
                    globals_envs[dl_func_name]).runsource(sccode, filename=scfile)
            else:
                if TRANSFORM_REGISTRY[dl_func_name]['transformed'][cell_name] != TRANSFORM_REGISTRY[dl_func_name]['history'][-1]['transformed'][cell_name]:
                    print(cell_name, 'change,', 'only',
                          updated_cell_name, 'can change')
                    print_code_diff(TRANSFORM_REGISTRY[dl_func_name]['history'][-1]['transformed'][cell_name], TRANSFORM_REGISTRY[dl_func_name]['transformed'][cell_name])
                    other_changed_cells.append(cell_name)

        # print(astor.to_source(transformed_dl_func_tree))
        transformed_dl_func_code = astor.to_source(transformed_dl_func_tree)
        TRANSFORM_REGISTRY[dl_func_name]['transformed'][dl_func_name] = transformed_dl_func_code
        if TRANSFORM_REGISTRY[dl_func_name]['transformed'][dl_func_name] != TRANSFORM_REGISTRY[dl_func_name]['history'][-1]['transformed'][dl_func_name]:
            print(dl_func_name, 'change')
            other_changed_cells.append(dl_func_name)

        if len(other_changed_cells):
            raise Exception(
                'Should not change control flow or change cells except the cell throwing the exception!')

        # sctree = LineMapper().visit(transformed_dl_func_tree)
        # sccode_with_lineno = astor.to_source(
        #     sctree, add_line_information=True)
        # create_TransformMapping(dl_func_name, dl_func_name, sccode_with_lineno)

        updated_cell_name_items = updated_cell_name.split('_')
        if updated_cell_name_items[-2] == 'rest':
            updated_cell_name_items[-2] = 'cell'
            updated_cell_name = '_'.join(updated_cell_name_items)
            faulty_lineno -= 3
        updated_cell_name_items[-2] = 'rest'
        rest_name = '_'.join(updated_cell_name_items)

        original_cell_lines = TRANSFORM_REGISTRY[dl_func_name][
            'history'][-1]['transformed'][updated_cell_name].expandtabs().split('\n')
        subject_cell_lines = TRANSFORM_REGISTRY[dl_func_name]['transformed'][updated_cell_name].expandtabs(
        ).split('\n')

        opcodes = SequenceMatcher(
            None, original_cell_lines, subject_cell_lines).get_opcodes()

        for tag, i1, i2, j1, j2 in opcodes:
            if tag != 'equal':
                indent_num = len(
                    subject_cell_lines[j1]) - len(subject_cell_lines[j1].lstrip())
                subject_cell_lines.insert(
                    j1, ' ' * indent_num + 'label .restart')
                break
        else:
            indent_num = len(
                subject_cell_lines[faulty_lineno-1]) - len(subject_cell_lines[faulty_lineno-1].lstrip())
            subject_cell_lines.insert(
                faulty_lineno-1, ' ' * indent_num + 'label .restart')

        subject_cell_lines.insert(0, '@daifu.with_goto')
        subject_cell_lines[1] = subject_cell_lines[1].replace(
            updated_cell_name, rest_name)

        indent_num = len(subject_cell_lines[4]) - \
            len(subject_cell_lines[4].lstrip())
        subject_cell_lines.insert(4, ' ' * indent_num + 'goto .restart')

        rest_code = '\n'.join(subject_cell_lines)

        rest_file = Path(
            TRANSFORM_REGISTRY[dl_func_name]['workspace'])/('(transformed)' + rest_name + '.py')

        with rest_file.open('w') as f:
            f.write(rest_code)

        code.compile_command(rest_code)
        code.InteractiveInterpreter(globals_envs[dl_func_name]).runsource(
            rest_code, filename=rest_file)

        print('Repair Success!')

    except Exception as e:
        TRANSFORM_REGISTRY[dl_func_name]['original'] = TRANSFORM_REGISTRY[dl_func_name]['history'][-1]['original']
        TRANSFORM_REGISTRY[dl_func_name]['transformed'] = TRANSFORM_REGISTRY[dl_func_name]['history'][-1]['transformed']
        TRANSFORM_REGISTRY[dl_func_name]['workspace'] = TRANSFORM_REGISTRY[dl_func_name]['history'][-1]['workspace']
        TRANSFORM_REGISTRY[dl_func_name]['mapping'] = TRANSFORM_REGISTRY[dl_func_name]['history'][-1]['mapping']
        TRANSFORM_REGISTRY[dl_func_name]['final_return'] = TRANSFORM_REGISTRY[dl_func_name]['history'][-1]['final_return']
        TRANSFORM_REGISTRY[dl_func_name]['history'].pop()

        rawfile = (Path.cwd()/'daifu_workspace')/('(original)'+dl_func_name+'.py')
        with rawfile.open('w') as f:
            f.write(TRANSFORM_REGISTRY[dl_func_name]['original'])

        for cell_name in TRANSFORM_REGISTRY[dl_func_name]['transformed']:
            if cell_name == dl_func_name:
                continue
            sccode = TRANSFORM_REGISTRY[dl_func_name]['transformed'][cell_name]
            scfile = Path(
                TRANSFORM_REGISTRY[dl_func_name]['workspace'])/('(transformed)' + cell_name + '.py')
            with scfile.open('w') as f:
                f.write(sccode)
            code.InteractiveInterpreter(
                globals_envs[dl_func_name]).runsource(sccode, filename=scfile)

        print('Repair Fails!')
        print('Reason:', e)
        traceback.print_tb(e.__traceback__)


def init_rest(dl_func_name, updated_cell_name, faulty_lineno):
    global globals_envs, TRANSFORM_REGISTRY
    try:
        updated_cell_name_items = updated_cell_name.split('_')
        if updated_cell_name_items[-2] == 'rest':
            updated_cell_name_items[-2] = 'cell'
            updated_cell_name = '_'.join(updated_cell_name_items)
            faulty_lineno -= 3
        updated_cell_name_items[-2] = 'rest'
        rest_name = '_'.join(updated_cell_name_items)

        subject_cell_lines = TRANSFORM_REGISTRY[dl_func_name]['transformed'][updated_cell_name].expandtabs(
        ).split('\n')

        indent_num = len(subject_cell_lines[faulty_lineno-1]) - \
            len(subject_cell_lines[faulty_lineno-1].lstrip())
        subject_cell_lines.insert(
            faulty_lineno-1, ' ' * indent_num + 'label .restart')

        subject_cell_lines.insert(0, '@daifu.with_goto')
        subject_cell_lines[1] = subject_cell_lines[1].replace(
            updated_cell_name, rest_name)

        indent_num = len(subject_cell_lines[4]) - \
            len(subject_cell_lines[4].lstrip())
        subject_cell_lines.insert(4, ' ' * indent_num + 'goto .restart')

        rest_code = '\n'.join(subject_cell_lines)

        rest_file = Path(
            TRANSFORM_REGISTRY[dl_func_name]['workspace'])/('(transformed)' + rest_name + '.py')

        with rest_file.open('w') as f:
            f.write(rest_code)

        code.compile_command(rest_code)
        code.InteractiveInterpreter(globals_envs[dl_func_name]).runsource(
            rest_code, filename=rest_file)

        print('Rest Init Success!')

    except Exception as e:
        print('Rest Init Fails!')
        print('Reason:', e)
        traceback.print_tb(e.__traceback__)


def transform(target_globals = None):
    def transform_decorator(dl_func):
        global globals_envs
        if target_globals is None:
            globals_envs[dl_func.__name__] = dl_func.__globals__
        else:
            globals_envs[dl_func.__name__] = target_globals

        return functools.update_wrapper(
            types.FunctionType(
                transform_code(dl_func),
                dl_func.__globals__,
                dl_func.__name__,
                dl_func.__defaults__,
                dl_func.__closure__,
            ),
            dl_func
        )
    return transform_decorator
