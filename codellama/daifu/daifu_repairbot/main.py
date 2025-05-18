import custom_llm

from fastapi import FastAPI
from pydantic import BaseModel


from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.chains import LLMChain

BASE_URL = "http://127.0.0.1:1079"

class Fault(BaseModel):
    code: str = None
    masked_code: str = None
    exception: str = None
    faulty_lines: str = None

class Traceback_Info(BaseModel):
    tb: str = None
    suspected_functions: list = None

app = FastAPI()
diagnose_and_action_llm=None
surgery_llm=None

HUMAN_PROMPT_TEMPLATE_FOR_DIAGNOSIS ='''suspected functions: {suspected_functions}
Which one in the suspected functions is the culprit? Your answer should follow the above schema.
'''

@app.post("/chain_diagnose/")
async def diagnose_code(traceback_info: Traceback_Info):
    global diagnose_and_action_llm
    diagnose_and_action_llm=custom_llm.CustomChatGPT(base_url=BASE_URL)

    traceback_explanation_prompt = PromptTemplate.from_template(
        "Please explain the following traceback:\n{traceback}"
    )
    traceback_explanation_chain = LLMChain(prompt=traceback_explanation_prompt, llm=diagnose_and_action_llm)
    traceback_explanation = traceback_explanation_chain.run(traceback=traceback_info.tb)
    
    diagnosis_prompt = PromptTemplate(
        template="Then, based on the above explanation, is {question_function} faulty and thus the exception is caused by the bug in it? You should first explicitly answer 'Yes' or 'No' and then explain your reasoning.",
        input_variables=["question_function"]
    )

    diagnosis_chain = LLMChain(prompt=diagnosis_prompt, llm=diagnose_and_action_llm)
    culprit_function = None
    culprit_explaination = None
    for question_function in traceback_info.suspected_functions[::-1]:
        output = diagnosis_chain.run({"question_function": question_function})
        if 'Yes' in output:
            culprit_function = question_function
            culprit_explaination = output
            break
        elif 'No' in output:
            continue
        else:
            print('Unhandled output in chain_diagnose:')
            print(output)
            break
    return culprit_function, traceback_explanation, culprit_explaination

@app.post("/chain_action_code/")
async def action_code(fault: Fault):
    code_action_prompt = PromptTemplate.from_template(
        "Then, based on the above explanation, rewrite this function to fix it:\n```\n{fauty_code}```\nYou should remain all the original functionality of this function, and should not change any other lines except the faulty lines. Your answer should only contain all of the code in the fixed version, surround by the leading and trailing '```python' and '```'."
    )
    code_action_chain = LLMChain(prompt=code_action_prompt, llm=diagnose_and_action_llm)
    response = code_action_chain.run(fauty_code=fault.code)
    return response

SURGERY_EXAMPLE_INPUT1='''## Source code (Python) with faulty line(s) removed and then masked by '[MASK]':
```
def main(numbers):
    result = 0
    for num in numbers:
[MASK]
        do_something(result)
    return result
```

## Exception:
ZeroDivisionError: division by zero

## Original faulty line(s):
```
        result += 10 / num
```

'''
SURGERY_EXAMPLE_OUTPUT1='''## Revised [MASK]:
To address this exception, the original faulty line(s) masked by '[MASK]' should be revised into:
```
        if num != 0:
            result += 10 / num
```


'''

SURGERY_EXAMPLE_INPUT2='''## Source code (Python) with faulty line(s) removed and then masked by '[MASK]':
```
def process_list(my_list):
    for i in range(len(my_list)):
        if my_list[i] == 0:
            my_list[i] = 1
[MASK]
    return my_list
```

## Exception:
IndexError: list index out of range

## Original faulty line(s):
```
        my_list[i + 1] = my_list[i] * 2
```

'''
SURGERY_EXAMPLE_OUTPUT2='''## Revised [MASK]:
To address this exception, the original faulty line(s) masked by '[MASK]' should be revised into:
```
        if i + 1 < len(my_list):
            my_list[i + 1] = my_list[i] * 2
```


'''

SURGERY_EXAMPLE_INPUT3='''## Source code (Python) with faulty line(s) removed and then masked by '[MASK]':
```
def square_numbers(numbers):
    squared = []
    for num in numbers:
        num = str(num)
        print(''.join(['The square of ', num, ' is:'])
[MASK]
        print(squared[-1])
    return squared
```

## Exception:
TypeError: unsupported operand type(s) for ** or pow(): 'str' and 'int'

## Original faulty line(s):
```
        squared.append(num ** 2)
```

'''
SURGERY_EXAMPLE_OUTPUT3='''## Revised [MASK]:
To address this exception, the original faulty line(s) masked by '[MASK]' should be revised into:
```
        squared.append(float(num) ** 2)
```


'''

surgery_examples = [
    {"input": SURGERY_EXAMPLE_INPUT1, "output": SURGERY_EXAMPLE_OUTPUT1},
    {"input": SURGERY_EXAMPLE_INPUT2, "output": SURGERY_EXAMPLE_OUTPUT2},
    {"input": SURGERY_EXAMPLE_INPUT3, "output": SURGERY_EXAMPLE_OUTPUT3},
]

# This is a prompt template used to format each individual example.
surgery_example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt_for_surgery = FewShotChatMessagePromptTemplate(
    example_prompt=surgery_example_prompt,
    examples=surgery_examples,
)

HUMAN_PROMPT_TEMPLATE_FOR_SURGERY ='''## Source code (Python) with faulty line(s) removed and then masked by '[MASK]':
```
{masked_code}```

## Exception:
{exception}

## Original faulty line(s):
```
{faulty_lines}```

'''

surgery_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an Automated Program Repair Tool specialized in fixing code errors. Your task is to correct the given Python source code and address the exception. You should avoid adding try-except logic into the code."),
        few_shot_prompt_for_surgery,
        ("human", "{input}"),
        ("ai", "##  Revised [MASK]:\nTo address this exception, the original faulty line(s) masked by '[MASK]' should be revised into:\n"),
    ]
)

surgery_prompt_continue_to_fix_indentation = ChatPromptTemplate.from_messages(
    [
        ("system", "Please pay attention to the indentation and then give your revised answer."),
        ("human", "{input}"),
    ]
)

@app.post("/chain_surgery/")
async def surgery_code(fault: Fault):
    global surgery_llm
    surgery_llm=custom_llm.CustomChatGPT(base_url=BASE_URL)
    surgery_chain = LLMChain(prompt=surgery_prompt, llm=surgery_llm)
    input= HUMAN_PROMPT_TEMPLATE_FOR_SURGERY.format(masked_code=fault.masked_code, exception=fault.exception, faulty_lines=fault.faulty_lines)
    response = surgery_chain.run({"input":input})
    return response

@app.post("/chain_surgery_continue_to_fix_indentation/")
async def surgery_code_continue_to_fix_indentation(fault: Fault):
    continue_surgery_chain = LLMChain(prompt=surgery_prompt_continue_to_fix_indentation, llm=surgery_llm)
    input= HUMAN_PROMPT_TEMPLATE_FOR_SURGERY.format(masked_code=fault.masked_code, exception=fault.exception, faulty_lines=fault.faulty_lines)
    response = continue_surgery_chain.run({"input":input})
    return response
