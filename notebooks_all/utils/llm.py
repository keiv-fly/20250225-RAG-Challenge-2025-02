from openai import AsyncOpenAI
from dotenv import load_dotenv
import os, asyncio
import json
from pydantic import BaseModel, Field, constr
from typing import Callable, List, Union, Literal
import re

load_dotenv()
LLM_CLIENT = "openrouter"
LLM_MODEL = "openai/gpt-4o"

model_fast = "google/gemini-2.0-flash-001"
model_image = "google/gemini-2.0-flash-001"
# model_fast = "google/gemma-3-27b-it"
# model_fast = "deepseek/deepseek-r1-distill-llama-70b"
# model_fast = "microsoft/phi-4-multimodal-instruct"
# model_fast = "meta-llama/llama-4-maverick"
# model_deepseek = "deepseek/deepseek-r1-distill-llama-70b"


def get_llm_client(provider_name: str = "openai"):
    match provider_name:
        case "openai":
            return AsyncOpenAI(
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        case "openrouter":
            return AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_KEY"),
            )


def extract_json_from_markdown(ans: str) -> str:
    """
    Extract JSON string from markdown code block.

    Args:
        ans: The input string that may contain a JSON code block

    Returns:
        The extracted JSON string or the original input if no JSON block is found
    """
    json_str = ans
    if (
        isinstance(ans, str)
        and "```json" in ans
        and "```" in ans[ans.find("```json") + 7 :]
    ):
        start_idx = ans.find("```json") + 7
        end_idx = ans.find("```", start_idx)
        json_str = ans[start_idx:end_idx].strip()

    json_str = json_str.replace("n/a", "N/A")
    # Find currency values in the format "currency_of_values_in_table": "xxx" and convert to uppercase
    # This regex works by:
    # 1. Matching the exact key "currency_of_values_in_table"
    # 2. Allowing for any whitespace (\s*) between the key, colon, and value
    # 3. Capturing the 3-letter currency code ([a-zA-Z]{3}) within quotes
    # 4. Using a lambda function to replace the entire match with the same format
    #    but converting the captured currency code to uppercase using .upper()
    if isinstance(json_str, str):
        json_str = re.sub(
            r'"currency_of_values_in_table"\s*:\s*"([a-zA-Z]{3})"',
            lambda m: f'"currency_of_values_in_table": "{m.group(1).upper()}"',
            json_str,
        )
    return json_str


async def rejson(text: str, str_ans_class: str) -> dict:
    client = get_llm_client(provider_name="openrouter")
    model = model_fast
    messages = [
        {
            "role": "system",
            "content": """
You are a financial analyst analyzing financial statements. You have an input text. There is some broken json data in the text. Yout task is to fix the issues with the json data and provide the correct json according to the pydantic class Answer.

---
Output format:
The output should not contain any characters before or after the JSON. If the result is empty it should still correspond to the JSON schema and the result should have N/A in result.
The output is a JSON that corresponds to the following pydantic class:
""".strip()
            + str_ans_class,
        },
        {
            "role": "user",
            "content": f"{text}",
        },
    ]
    try:
        response = await client.chat.completions.create(
            model=model, messages=messages, temperature=0.0
        )
    except Exception as e:
        ans_d = {
            "result": text,
            "result_error": True,
            "result_error_type": "json_error_request_error",
            "result_error_message": str(e),
            "str_ans_class": str_ans_class,
        }
        return ans_d
    ans = response.choices[0].message.content.lower().strip()
    ans_extracted = extract_json_from_markdown(ans)
    try:
        ans_d = json.loads(ans_extracted)
        _ = Answer(**ans_d)
    except Exception as e:
        ans_d = {
            "result": ans_extracted,
            "result_error": True,
            "result_error_type": "json_error",
            "result_error_message": str(e),
            "str_ans_class": str_ans_class,
        }
    return ans_d


async def run_tasks_with_retries(
    run_one_task: Callable, in_params: list[dict], retries: int = 20
):

    tasks_done = ["0"] * len(in_params)
    tasks_tries = {}
    tasks = set()
    print(f"Sending for one question started")
    for i, param in enumerate(in_params):
        tasks_tries[i] = 1
        tasks.add(asyncio.create_task(run_one_task(i, param)))

    tasks_run = []
    while tasks:
        done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for completed in done:
            i, in_params, ans_d = await completed
            if "result_error" in ans_d:

                if tasks_tries[i] < retries:
                    if tasks_tries[i] > 5:
                        await asyncio.sleep(1)
                    print(f"Task {i} failed. Total retries: {tasks_tries[i]}. Retrying")
                    tasks_tries[i] = tasks_tries[i] + 1
                    tasks.add(asyncio.create_task(run_one_task(i, in_params)))
                else:
                    print(
                        f"Task {i} failed. Total retries: {tasks_tries[i]}. Stop retrying. Returning the current answer with error."
                    )
                    tasks_done[i] = "2"
                    tasks_run.append((i, in_params, ans_d))
                    print("".join(tasks_done))
            else:
                tasks_done[i] = "1"
                tasks_run.append((i, in_params, ans_d))
                print("".join(tasks_done))
        tasks_run_sorted = sorted(tasks_run, key=lambda tup: tup[0])
    return tasks_run_sorted


async def find_balance_sheet_page(i_task: int, in_params: dict) -> dict:
    text = in_params["text"]
    client = get_llm_client(provider_name="openrouter")
    model = model_fast
    str_ans_class = r"""
class PageAnswer(BaseModel):
    page_number: int = Field(..., description="The page number of this page that looks like this: '{page_number}------------------------------------------------'")
    has_table: bool = Field(..., description="Whether the page contains a markdown table. The balance sheet page should always have a table.")
    has_numbers_in_table: bool = Field(..., description="Whether the page contains numbers in the table. The balance sheet page should always have numbers in the table.")
    has_two_columns_in_table: bool = Field(..., description="Whether the page contains two columns in the table")
    has_consolidated_numbers_and_not_devisions_as_columns: bool = Field(..., description="Whether the page contains consolidated numbers and not devisions as columns. ")
    has_total_assets: bool = Field(..., description="Whether the page contains a total assets line item. Some balance sheets call them just 'Assets', but it is still a sum of the above numbers.")
    has_cash_and_cash_equivalents: bool = Field(..., description="Whether the page contains a cash and cash equivalents phrase in the text. It could be called differently.")
    why_cash_and_cash_equivalents: str = Field(..., description="If has_cash_and_cash_equivalents is true then answer why the page contains a cash and cash equivalents phrase in the text. If has_cash_and_cash_equivalents is false then answer why the page does not contain a cash and cash equivalents phrase in the text.")
    why_cash_and_cash_equivalents_is_relevant: str = Field(..., description="Why is why_cash_and_cash_equivalents relevant to the asked question for the item.")
    has_total_liabilities: bool = Field(..., description="Whether the page contains a total liabilities line item. It could be called differently.")
    list_of_years_in_columns: List[int] = Field(..., description="List of years in the columns of the table. If the page does not have a table, return an empty list.")
    scale_of_values_in_table: Literal["units", "thousands", "millions", "billions", "N/A"] = Field(..., description="Scale of all the values in the table, except for dividends. If the data is not available, return 'N/A'")
    currency_of_values_in_table: Union[str, Literal["N/A"]] = Field(..., description="Currency of all the values in the table, except for dividends. If the data is not available, return 'N/A'")
    total_assets_of_last_year: Union[float, Literal["N/A"]] = Field(..., description="Total assets of the last year in the table. Provide the number in the same scale as the values in the table. If the data is not available, return 'N/A'")
    total_assets_of_the_year_before_last: Union[float, Literal["N/A"]] = Field(..., description="Total assets of the year before last in the table. Provide the number in the same scale as the values in the table. If the data is not available, return 'N/A'")
    cash_and_cash_equivalents_of_last_year: Union[float, Literal["N/A"]] = Field(..., description="Cash and cash equivalents of the last year in the table. Provide the number in the same scale as the values in the table. If the data is not available, return 'N/A'")
    cash_and_cash_equivalents_of_the_year_before_last: Union[float, Literal["N/A"]] = Field(..., description="Cash and cash equivalents of the year before last in the table. Provide the number in the same scale as the values in the table. If the data is not available, return 'N/A'")
    total_liabilities_of_last_year: Union[float, Literal["N/A"]] = Field(..., description="Total liabilities of the last year in the table. Provide the number in the same scale as the values in the table. If the data is not available, return 'N/A'")
    total_liabilities_of_the_year_before_last: Union[float, Literal["N/A"]] = Field(..., description="Total liabilities of the year before last in the table. Provide the number in the same scale as the values in the table. If the data is not available, return 'N/A'")
    is_balance_sheet: bool = Field(..., description="Whether the page is a consolidated balance sheet. ")
    why_balance_sheet: str = Field(..., description="If is_balance_sheet is true then answer why the page is this page a consolidated balance sheet. If is_balance_sheet is false then answer why the page is not a consolidated balance sheet.")
    
class Answer(BaseModel):
    all_page_numbers_in_input_text: List[int] = Field(..., description=r"List of all page numbers in the input text. The page numbers are not sequential. The page numbers are marked with a page number marker that has this regular expression: r'\{(\d+)\}-{48,}'")
    pages: List[PageAnswer] = Field(..., description="List of pages from the input text. Those are all pages from the input text. They are all checked for the presence of a balance sheet."
                                                     r"The start of the page is marked with a page number marker that has this regular expression: r'\{(\d+)\}-{48,}'"
                                                     "The page ends with the next page number marker or the end of the text.")
    result: Union[int, Literal["N/A"]] = Field(..., description="The page number where the balance sheet is situated. If the data is not available, return 'N/A'")
    result_page_answer: PageAnswer = Field(..., description="The page answer that corresponds to the result page number.")
    """

    # Define the classes in the global scope
    global PageAnswer, Answer
    exec(str_ans_class, globals())

    messages = [
        {
            "role": "system",
            "content": """
You are a financial analyst analyzing financial statements.
Your task is to find the consolidated balance sheet page in the financial statements. The text of the pages will be provided by the user.
All pages start with a page number that looks like this: '{page_number}------------------------------------------------'
Where `page_number` is a number of the page. The page numbers are not sequential.
The consolidated balance sheet page should contain:
1. consolidated numbers in the table. If there are devisions and no consolidated numbers then it is not a balance sheet page. 
2. a total assets line item. The total assets line item could be called just 'Assets'.
3. a cash and cash equivalents line item
4. a total liabilities line item


Provide the answers in the JSON format with the format following the pydantic class Answer.

---
Output format:
The output should start with "```json" and end with "```". If the result is empty it should still correspond to the JSON schema and the result should have N/A in result.
The output is a JSON that corresponds to the following pydantic class:
""".strip()
            + str_ans_class,
        },
        {
            "role": "user",
            "content": f"{text}",
        },
    ]
    try:
        response = await client.chat.completions.create(
            model=model, messages=messages, temperature=0.0
        )
    except Exception as e:
        ans_d = {
            "result": "",
            "result_error": True,
            "result_error_type": "request_error",
            "result_error_message": str(e),
        }
        return i_task, in_params, ans_d
    ans = response.choices[0].message.content.lower().strip()
    ans_extracted = extract_json_from_markdown(ans)
    try:
        ans_d = json.loads(ans_extracted)
        _ = Answer(**ans_d)
        return i_task, in_params, ans_d
    except:
        res = await rejson(ans_extracted, str_ans_class)
        return i_task, in_params, res


async def what_is_on_pgn(i_task: int, in_params: dict) -> dict:
    png_image = in_params["png_image"]
    client = get_llm_client(provider_name="openrouter")
    model = model_image
    str_ans_class = r"""
class Answer(BaseModel):
    result: Union[str, Literal["N/A"]] = Field(..., description="What is on the page? If the data is not available, return 'N/A'")
    """

    # Define the classes in the global scope
    global Answer
    exec(str_ans_class, globals())

    messages = [
        {
            "role": "system",
            "content": """
You need to analyze the image and provide a description of what is on the page.

Provide the answers in the JSON format with the format following the pydantic class Answer.

---
Output format:
The output should start with "```json" and end with "```". If the result is empty it should still correspond to the JSON schema and the result should have N/A in result.
The output is a JSON that corresponds to the following pydantic class:
""".strip()
            + str_ans_class,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{png_image}"},
                },
            ],
        },
    ]
    try:
        response = await client.chat.completions.create(
            model=model, messages=messages, temperature=0.0
        )
    except Exception as e:
        ans_d = {
            "result": "",
            "result_error": True,
            "result_error_type": "request_error",
            "result_error_message": str(e),
        }
        return i_task, in_params, ans_d
    ans = response.choices[0].message.content.lower().strip()
    ans_extracted = extract_json_from_markdown(ans)
    try:
        ans_d = json.loads(ans_extracted)
        _ = Answer(**ans_d)
        return i_task, in_params, ans_d
    except:
        res = await rejson(ans_extracted, str_ans_class)
        return i_task, in_params, res


async def balance_sheet_pgn_extraction(i_task: int, in_params: dict) -> dict:
    png_image = in_params["png_image"]
    client = get_llm_client(provider_name="openrouter")
    model = model_image
    str_ans_class = r"""
class LineItem(BaseModel):
    name: str = Field(..., description="The name of the line item")
    value: Union[float, Literal["N/A"]] = Field(..., description="The value of the line item in the last period/year. If the data is not available, return 'N/A'")

class Answer(BaseModel):
    list_of_years_in_columns: List[int] = Field(..., description="List of years in the columns of the table. If the page does not have a table, return an empty list.")
    last_year: int = Field(..., description="The biggest year in list_of_years_in_columns")
    list_of_periods_in_columns: List[str] = Field(..., description="List of periods in the columns of the table. If the page does not have a table, return an empty list.")
    last_period: str = Field(..., description="The latest period in list_of_periods_in_columns")
    scale_of_values_in_table: Union[Literal["units"], Literal["thousands"], Literal["millions"], Literal["billions"], Literal["N/A"], str] = Field(..., description="Scale of all the values in the table, except for dividends. If the provided scale is not in the list ('units', 'thousands', 'millions', 'billions'), return the scale as is. If the data is not available, return 'N/A'")
    currency_of_values_in_table: Union[constr(pattern=r'^[A-Z]{3}$'), Literal["N/A"]] = Field(..., description="Currency of all the values in the table in three letter code. If the data is not available, return 'N/A'")
    total_assets: Union[float, Literal["N/A"]] = Field(..., description="Total assets of the last period in the table. Provide the number in the same scale as the values in the table. If the data is not available, return 'N/A'")
    cash_and_cash_equivalents: Union[float, Literal["N/A"]] = Field(..., description="Cash and cash equivalents of the last period in the table. Provide the number in the same scale as the values in the table. If the data is not available, return 'N/A'")
    total_liabilities: Union[float, Literal["N/A"]] = Field(..., description="Total liabilities of the last period in the table. Provide the number in the same scale as the values in the table. If the data is not available, return 'N/A'")
    total_equity: Union[float, Literal["N/A"]] = Field(..., description="Total equity of the last period in the table. Provide the number in the same scale as the values in the table. If the data is not available, return 'N/A'")
    total_deposits: Union[float, Literal["N/A"]] = Field(..., description="Total deposits of the last period in the table. This line number should be present for financial institutions. Provide the number in the same scale as the values in the table. If the data is not available, return 'N/A'")
    loans_outstanding: Union[float, Literal["N/A"]] = Field(..., description="Loans outstanding of the last period in the table. This line number should be present for financial institutions. Provide the number in the same scale as the values in the table. If the data is not available, return 'N/A'")
    result: bool = Field(..., description="Did you manage to find the total assets in the table? If yes, return True, otherwise return False.")
    # balance_sheet_markdown_table: Union[str, Literal["N/A"]] = Field(..., description="Convert the balance sheet table to markdown table and return it here. If the data is not available, return 'N/A'")
    line_items: List[LineItem] = Field(..., description="List of line items in the table. If the data is not available, return an empty list.")
    """

    # Define the classes in the global scope
    global Answer
    exec(str_ans_class, globals())

    messages = [
        {
            "role": "system",
            "content": """
You need to analyze the image and provide the answers to the questions in the pydantic class Answer.

Provide the answers in the JSON format with the format following the pydantic class Answer.

---
Output format:
The output should start with "```json" and end with "```". If the result is empty it should still correspond to the JSON schema and the result should have N/A in result.
The output is a JSON that corresponds to the following pydantic class:
""".strip()
            + str_ans_class,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{png_image}"},
                },
            ],
        },
    ]
    try:
        response = await client.chat.completions.create(
            model=model, messages=messages, temperature=0.0
        )
    except Exception as e:
        ans_d = {
            "result": "",
            "result_error": True,
            "result_error_type": "request_error",
            "result_error_message": str(e),
        }
        return i_task, in_params, ans_d
    ans = response.choices[0].message.content.lower().strip()
    ans_extracted = extract_json_from_markdown(ans)
    try:
        ans_d = json.loads(ans_extracted)
        ans_d["currency_of_values_in_table"] = ans_d[
            "currency_of_values_in_table"
        ].upper()
        _ = Answer(**ans_d)
        return i_task, in_params, ans_d
    except:
        res = await rejson(ans_extracted, str_ans_class)
        return i_task, in_params, res
