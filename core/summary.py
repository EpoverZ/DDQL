# =============================#
#       Dataset Summary        #
# =============================#

def get_dataset_summary(df, file_format: str = "tabular"):
    """
    Returns structured dataset summary dictionary, including file format.
    """
    rows, cols = df.shape
    return {
        "file_format": file_format,
        "rows": int(rows),
        "columns": int(cols),
        "column_types": {str(col): str(dtype) for col, dtype in df.dtypes.items()}
    }


def print_dataset_summary(df, file_format: str = "tabular"):
    """
    Prints basic dataset information.
    """
    summary = get_dataset_summary(df, file_format=file_format)

    print("\nDataset Summary")
    print("---------------")
    print(f"Format: {summary['file_format']}")
    print(f"Rows: {summary['rows']}")
    print(f"Columns: {summary['columns']}")

    print("\nColumns & DataTypes:")
    for col, dtype in summary["column_types"].items():
        print(f"{col} ---> {dtype}")
