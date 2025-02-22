from datetime import datetime
import random
from tempfile import TemporaryDirectory
from langchain_openai import ChatOpenAI
from langchain import hub
from typing_extensions import Annotated, TypedDict
import pandas as pd
from langchain_community.document_loaders import DuckDBLoader
from langchain_experimental.utilities.python import PythonREPL
import streamlit as st
from PIL import Image

llm = None

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")


class State(TypedDict):
    question: str | None
    query: str | None
    result: str | None
    answer: str | None
    file_path: str | None
    api_key: str | None


class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]


def read_csv_headers(file_path: str):
    """Read CSV file headers."""
    return pd.read_csv(file_path, nrows=0).columns.tolist()


def write_query(state: State):
    """Generate SQL query to fetch information."""
    file_path = state["file_path"]
    csv_columns = read_csv_headers(file_path)
    prompt = query_prompt_template.invoke(
        {
            "dialect": "duckdb",
            "top_k": 100,
            "table_info": f"""
                Table name: read_csv_auto({file_path})
                Columns: {csv_columns}
                All text values are in UPPER CASE
                The table has not values about biomes, only about brazil country, its states and cities
                Queries related to biomes must solve to use the states that belong to the biome
            """,
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return result


def execute_query(state: State):
    """Execute SQL query."""
    data = DuckDBLoader(state["query"]).load()
    return {"result": data}


def generate_answer(state: State):
    """Generate answer based on query result."""
    prompt = (
        "Given the following user question, corresponding SQL query, and query result, "
        "generate an answer that is relevant to the user question."
        f"\n\nUser question: {state['question']}"
        f"\n\nSQL query: {state['query']}"
        f"\n\nQuery result: {state['result']}"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}


def write_plot_code(state: State):
    """Generate Python code to plot data."""
    plot_output = (
        f"./{random.randint(1000, 9999)}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    )
    prompt = (
        "Based on the inputs provided, generate a complete and executable Python script that creates a plot from a SQL query result. "
        "The script must be fully auto-generated without requiring any manual modifications.\n\n"
        f"User question: {state['question']}\n\n"
        f"SQL query: {state['query']}\n\n"
        f"Query result: {state['result']}\n\n"
        "Requirements:\n"
        "1. Generate a plot that effectively visualizes the query result.\n"
        f"2. Save the plot to the following file path: {plot_output}\n"
        "3. Ensure the script runs without any errors.\n"
        "4. Include a print statement at the end of the script to output the file path of the saved plot.\n"
        "5. The output must be valid Python code without any markdown formatting or introductory comments or backticks.\n"
    )

    response = llm.invoke(prompt)
    return response.content, plot_output


def get_response(state: State):
    global llm
    api_key = state["api_key"]
    llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
    sql_query = write_query(state)
    result = execute_query(sql_query)
    final_state = {**state, **sql_query, **result}
    answer = generate_answer(final_state)
    python_code, plot_output = write_plot_code(final_state)
    python_repl = PythonREPL()
    python_repl.run(python_code)
    return answer, plot_output


def output_response_with_plot(state: State):
    answer, plot_output = get_response(state)
    st.info(answer["answer"])
    image = Image.open(plot_output)
    st.image(image, caption="Plot generated based on the SQL query result.")


def main():
    title = "Fire Guru 🔥"
    st.set_page_config(page_title=title, page_icon="🔥")
    st.title(title)

    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if file is not None:
        st.sidebar.markdown("File uploaded successfully!")
        st.sidebar.markdown("---")
        st.sidebar.markdown("File Information:")
        st.sidebar.markdown(f"**Filename:** {file.name}")
        st.sidebar.markdown(f"**Filetype:** {file.type}")
        st.sidebar.markdown(f"**File size:** {file.size} bytes")

    with st.form("my_form"):
        text = st.text_area(
            "Enter text:", "How many fires occurred in the Amazon rainforest in 2024?"
        )
        submitted = st.form_submit_button("Submit", disabled=file is None)
        if file is None:
            st.warning("Please upload a CSV file.")
        if api_key is None:
            st.warning("Please enter your OpenAI API key.")
        if (
            submitted
            and file is not None
            and api_key is not None
            and api_key.startswith("sk-")
        ):
            with TemporaryDirectory(delete=False) as temp_dir, open(
                temp_dir + "/temp.csv", "wb"
            ) as f:
                file.seek(0)
                f.write(file.getvalue())
                file_path = temp_dir + "/temp.csv"
            state = State(
                question=text,
                file_path=file_path,
                api_key=api_key,
            )
            output_response_with_plot(state)


if __name__ == "__main__":
    main()
