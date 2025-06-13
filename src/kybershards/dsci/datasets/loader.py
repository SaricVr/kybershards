import pandas as pd
from pathlib import Path


script_dir = Path(__file__).parent


def load_workers_smoking_habits() -> pd.DataFrame:
    """Load and return the workers smoking habits dataset.

    This is a classical dataset for correspondence analysis introduced by
    Greenacre (1984). It contains data on the smoking habits of different
    employee categories in a company.

    Dataset Characteristics:

    | Feature | Values |
    |---------|--------|
    | Employee Categories (rows) | Senior-Managers, Junior-Managers, Senior-Employees, Junior-Employees, Secretaries |
    | Smoking Habit Levels (columns) | None, Light, Medium, Heavy |
    | Total Observations | 193 employees |
    | Data Type | Count (frequency) |

    Returns:
        A pandas DataFrame with employee categories as rows
            and smoking habit levels as columns.

            The values in the DataFrame represent the frequency counts of
            employees in each category with each smoking habit level.

    References:
        Greenacre, M. J. (1984). Theory and Applications of Correspondence Analysis.
               London: Academic Press.

    Examples:
        >>> data = load_workers_smoking_habits()
        >>> print(data)
                          None  Light  Medium  Heavy
        Senior-Managers      4      2       3      2
        Junior-Managers      4      3       7      4
        Senior-Employees    25     10      12      4
        Junior-Employees    18     24      33     13
        Secretaries         10      6       7      2
        >>> data.sum().sum()  # Total number of employees
        193
    """
    return pd.read_csv(script_dir / "data" / "workers_smoking_habits.csv", index_col=0)
