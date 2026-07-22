from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
from werkzeug.datastructures import FileStorage

from multi_modes import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    HARD_PROMPT_CAP,
    _llm_chat_mode,
)


DATA_ANALYSIS_PROMPT_BUDGET = max( 800, HARD_PROMPT_CAP - 500)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

MAX_DATASET_ROWS = 100_000
MAX_PREVIEW_ROWS = 3
MAX_TEXT_VALUES = 5
MAX_CONTEXT_CHARACTERS = 8_000
MAX_CORRELATION_PAIRS = 12
MAX_CATEGORICAL_COLUMNS = 6


DATA_ANALYSIS_SYSTEM_PROMPT = """
You are a careful and practical data analysis assistant.

The application provides a summary calculated with pandas.

Your responsibilities:
- Explain the dataset structure.
- Explain column names and data types.
- Explain missing values.
- Explain descriptive statistics.
- Compare columns when the necessary statistics are available.
- Identify possible data-quality issues.
- Describe visible patterns conservatively.
- Suggest suitable next analysis steps.

Rules:
- Use only information present in the supplied dataset summary.
- Never invent values, correlations, trends, causes, predictions, or statistical significance.
- Never claim that code was executed by you.
- The calculations were performed by the application using pandas.
- If the supplied summary is insufficient, state which additional calculation is required.
- Keep the response concise and clear.
- Use bullet points when helpful.
""".strip()


def get_file_extension(filename: str) -> str:
    """
    Return a lowercase file extension, including the leading dot.

    Example:
        sales.csv -> .csv
    """

    return os.path.splitext(filename or "")[1].lower()


def validate_uploaded_file(uploaded_file: Optional[FileStorage]) -> None:
    """
    Validate that an uploaded file exists and uses a supported extension.

    Raises:
        ValueError: If the file is missing or unsupported.
    """

    if uploaded_file is None:
        raise ValueError("No dataset file was uploaded.")

    filename = (uploaded_file.filename or "").strip()

    if not filename:
        raise ValueError("Please choose a CSV or Excel file.")

    extension = get_file_extension(filename)

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Please upload CSV, XLSX, or XLS."
        )


def load_dataset(uploaded_file: FileStorage) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a pandas DataFrame.

    The uploaded file is read directly from memory and is not saved permanently.
    """

    validate_uploaded_file(uploaded_file)

    filename = uploaded_file.filename or ""
    extension = get_file_extension(filename)

    try:
        uploaded_file.stream.seek(0)

        if extension == ".csv":
            dataframe = _read_csv(uploaded_file)

        elif extension in {".xlsx", ".xls"}:
            dataframe = pd.read_excel(uploaded_file)

        else:
            raise ValueError("Unsupported dataset format.")

    except ValueError:
        raise

    except Exception as exc:
        raise ValueError(
            f"Could not read the dataset: {exc}"
        ) from exc

    if dataframe.empty:
        raise ValueError("The uploaded dataset is empty.")

    if len(dataframe) > MAX_DATASET_ROWS:
        raise ValueError(
            f"The dataset contains {len(dataframe):,} rows. "
            f"The maximum supported size is {MAX_DATASET_ROWS:,} rows."
        )

    # Make all column names safe and readable.
    dataframe.columns = [
        str(column).strip() or f"column_{index + 1}"
        for index, column in enumerate(dataframe.columns)
    ]

    return dataframe


def _read_csv(uploaded_file: FileStorage) -> pd.DataFrame:
    """
    Read a CSV using a few common encodings.
    """

    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_error: Optional[Exception] = None

    for encoding in encodings:
        try:
            uploaded_file.stream.seek(0)

            return pd.read_csv(
                uploaded_file,
                encoding=encoding,
                low_memory=False,
            )

        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(
        "The CSV encoding could not be detected."
    ) from last_error


def dataframe_metadata(
    dataframe: pd.DataFrame,
    filename: str,
) -> Dict[str, Any]:
    """
    Return frontend-friendly dataset metadata.
    """

    return {
        "filename": filename,
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_names": [
            str(column) for column in dataframe.columns
        ],
    }


def build_dataset_context(
    dataframe: pd.DataFrame,
    filename: str,
) -> str:
    """
    Build a compact, deterministic summary for the local LLM.

    pandas performs the calculations. The LLM only explains the resulting
    summary.
    """

    sections: List[str] = []

    sections.append(
        "\n".join(
            [
                f"Dataset name: {filename}",
                f"Number of rows: {dataframe.shape[0]}",
                f"Number of columns: {dataframe.shape[1]}",
                f"Duplicate rows: {int(dataframe.duplicated().sum())}",
            ]
        )
    )

    sections.append(
        "COLUMN INFORMATION\n"
        + build_column_information(dataframe)
    )

    sections.append(
        "MISSING VALUES\n"
        + build_missing_values_summary(dataframe)
    )

    sections.append(
        "DATA PREVIEW\n"
        + dataframe.head(MAX_PREVIEW_ROWS).to_string(index=False)
    )

    numeric_summary = build_numeric_summary(dataframe)

    if numeric_summary:
        sections.append(
            "NUMERIC DESCRIPTIVE STATISTICS\n"
            + numeric_summary
        )

    categorical_summary = build_categorical_summary(dataframe)

    if categorical_summary:
        sections.append(
            "CATEGORICAL COLUMN SUMMARY\n"
            + categorical_summary
        )

    correlation_summary = build_correlation_summary(dataframe)

    if correlation_summary:
        sections.append(
            "NUMERIC CORRELATIONS\n"
            + correlation_summary
        )

    context = "\n\n".join(sections)

    # Prevent a very wide dataset from making the model prompt too large.
    if len(context) > MAX_CONTEXT_CHARACTERS:
        context = (
            context[:MAX_CONTEXT_CHARACTERS]
            + "\n\n[Dataset summary truncated because it was too large.]"
        )

    return context


def build_column_information(dataframe: pd.DataFrame) -> str:
    """
    Return data types and non-null counts for every column.
    """

    lines: List[str] = []

    total_rows = len(dataframe)

    for column in dataframe.columns:
        non_null_count = int(dataframe[column].notna().sum())
        unique_count = int(dataframe[column].nunique(dropna=True))

        lines.append(
            f"- {column}: "
            f"type={dataframe[column].dtype}, "
            f"non-null={non_null_count}/{total_rows}, "
            f"unique={unique_count}"
        )

    return "\n".join(lines)


def build_missing_values_summary(
    dataframe: pd.DataFrame,
) -> str:
    """
    Return missing counts and percentages.
    """

    total_rows = len(dataframe)
    lines: List[str] = []

    for column in dataframe.columns:
        missing_count = int(dataframe[column].isna().sum())

        if total_rows:
            missing_percentage = (
                missing_count / total_rows
            ) * 100
        else:
            missing_percentage = 0.0

        lines.append(
            f"- {column}: "
            f"{missing_count} missing "
            f"({missing_percentage:.2f}%)"
        )

    return "\n".join(lines)


def build_numeric_summary(
    dataframe: pd.DataFrame,
) -> str:
    """
    Calculate descriptive statistics for numeric columns.
    """

    numeric_dataframe = dataframe.select_dtypes(
        include="number"
    )

    if numeric_dataframe.empty:
        return ""

    summary = numeric_dataframe.describe().transpose()

    summary = summary.rename(
        columns={
            "count": "non_null_count",
            "mean": "mean",
            "std": "standard_deviation",
            "min": "minimum",
            "25%": "first_quartile",
            "50%": "median",
            "75%": "third_quartile",
            "max": "maximum",
        }
    )

    return summary.round(4).to_string()


def build_categorical_summary(
    dataframe: pd.DataFrame,
) -> str:
    """
    Show frequent values for a limited number of
    categorical columns.
    """

    categorical_dataframe = dataframe.select_dtypes(
        include=[
            "object",
            "string",
            "category",
            "bool",
        ]
    )

    if categorical_dataframe.empty:
        return ""

    sections: List[str] = []

    selected_columns = (
        categorical_dataframe.columns[
            :MAX_CATEGORICAL_COLUMNS
        ]
    )

    for column in selected_columns:
        value_counts = (
            categorical_dataframe[column]
            .astype("string")
            .fillna("<missing>")
            .value_counts(dropna=False)
            .head(MAX_TEXT_VALUES)
        )

        values = [
            f"{value}: {int(count)}"
            for value, count
            in value_counts.items()
        ]

        sections.append(
            f"{column}\n"
            + "\n".join(
                f"- {value}"
                for value in values
            )
        )

    if (
        categorical_dataframe.shape[1]
        > MAX_CATEGORICAL_COLUMNS
    ):
        omitted_count = (
            categorical_dataframe.shape[1]
            - MAX_CATEGORICAL_COLUMNS
        )

        sections.append(
            f"[{omitted_count} additional categorical "
            "columns omitted from the compact summary.]"
        )

    return "\n\n".join(sections)


def build_correlation_summary(
    dataframe: pd.DataFrame,
) -> str:
    """
    Return the strongest unique Pearson correlations.

    Self-correlations and duplicate pairs are removed.
    """

    numeric_dataframe = dataframe.select_dtypes(
        include="number"
    )

    if numeric_dataframe.shape[1] < 2:
        return ""

    correlation_matrix = numeric_dataframe.corr(
        method="pearson"
    )

    pairs = []

    columns = list(correlation_matrix.columns)

    for left_index in range(len(columns)):
        for right_index in range(
            left_index + 1,
            len(columns),
        ):
            left_column = columns[left_index]
            right_column = columns[right_index]

            correlation = correlation_matrix.loc[
                left_column,
                right_column,
            ]

            if pd.isna(correlation):
                continue

            pairs.append(
                (
                    abs(float(correlation)),
                    left_column,
                    right_column,
                    float(correlation),
                )
            )

    pairs.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    strongest_pairs = pairs[
        :MAX_CORRELATION_PAIRS
    ]

    if not strongest_pairs:
        return ""

    return "\n".join(
        (
            f"- {left_column} vs {right_column}: "
            f"{correlation:.4f}"
        )
        for (
            _,
            left_column,
            right_column,
            correlation,
        ) in strongest_pairs
    )


def make_upload_reply(
    dataframe: pd.DataFrame,
    filename: str,
) -> str:
    """
    Create the initial response displayed after an upload.
    """

    numeric_count = dataframe.select_dtypes(
        include="number"
    ).shape[1]

    categorical_count = dataframe.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).shape[1]

    missing_count = int(dataframe.isna().sum().sum())
    duplicate_count = int(dataframe.duplicated().sum())

    return (
        f"Dataset loaded successfully.\n\n"
        f"File: {filename}\n"
        f"Rows: {dataframe.shape[0]:,}\n"
        f"Columns: {dataframe.shape[1]:,}\n"
        f"Numeric columns: {numeric_count}\n"
        f"Text/categorical columns: {categorical_count}\n"
        f"Missing cells: {missing_count:,}\n"
        f"Duplicate rows: {duplicate_count:,}\n\n"
        "You can now ask me to summarize the dataset, explain "
        "missing values, inspect numerical statistics, compare "
        "columns, or identify possible data-quality issues."
    )


def data_analysis_chat(
    model,
    tokenizer,
    device: str,
    history: Optional[List[str]],
    user_input: str,
    dataset_context: str,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_new_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate a natural-language explanation of the pandas dataset summary.
    """

    safe_history = list(history) if history else []
    cleaned_input = (user_input or "").strip()
    cleaned_context = (dataset_context or "").strip()

    if not cleaned_input:
        return {
            "reply": "",
            "history": safe_history,
            "skipped": True,
            "reason": "empty_input",
        }

    if not cleaned_context:
        return {
            "reply": (
                "Please upload a CSV or Excel dataset before "
                "asking data-analysis questions."
            ),
            "history": safe_history,
            "skipped": True,
            "reason": "no_dataset",
        }

    dataset_token_budget = max(
        700,
        HARD_PROMPT_CAP - 700,
    )

    cleaned_context = trim_context_to_token_budget(
        context=cleaned_context,
        tokenizer=tokenizer,
        max_tokens=dataset_token_budget,
    )


    model_input = (
        "PANDAS DATASET SUMMARY\n"
        "======================\n"
        f"{cleaned_context}\n\n"
        "USER QUESTION\n"
        "=============\n"
        f"{cleaned_input}\n\n"
        "Answer only from the supplied pandas summary."
    )

    return _llm_chat_mode(
        mode_name="data_analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        history=safe_history,
        system_prompt=DATA_ANALYSIS_SYSTEM_PROMPT,
        user_input_for_prompt=model_input,
        raw_user_text_for_history=cleaned_input,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        hard_prompt_cap=HARD_PROMPT_CAP,
        history_trim_rounds=2,
    )


def trim_context_to_token_budget(
    context: str,
    tokenizer,
    max_tokens: int,
) -> str:
    """
    Truncate dataset context using the model tokenizer.
    """

    encoded = tokenizer(
        context,
        add_special_tokens=False,
        truncation=False,
    )

    token_ids = encoded["input_ids"]

    if len(token_ids) <= max_tokens:
        return context

    trimmed_ids = token_ids[:max_tokens]

    trimmed_context = tokenizer.decode(
        trimmed_ids,
        skip_special_tokens=True,
    ).strip()

    return (
        trimmed_context
        + "\n\n[Dataset summary truncated to fit "
        "the model prompt budget.]"
    )