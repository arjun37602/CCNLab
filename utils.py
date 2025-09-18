import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

# define the column names b/c the dataset doesn't have a header row
USER_ID = "user_id"  # anonymized user ID
DATE = "date"  # diary date
ENTRIES = "entries"  # List of food entries and nutrients (as JSON objects)
SUMMARY = "summary"  # Daily aggregate of nutrient intake and goal (as JSON objects)
COLUMNS = [USER_ID, DATE, ENTRIES, SUMMARY]

# ENTRIES
MEAL = "meal"
DISHES = "dishes"
SEQUENCE = "sequence"
NUTRITIONS = "nutritions"
NAME = "name"  # name of dish
VALUE = "value"  # value of each of dish's nutrition categories

# self created
CALORIES_GOAL = "Calories_goal"
CALORIES_TOTAL = "Calories_total"
MEAL_COUNT = "meal_count"

# SUMMARY
TOTAL = "total"
GOAL = "goal"
MEAL = "meal"

# other constants
COUNT = "count"
LEN = "len"
NUM_LOGS = "num_logs"
INDEX = "index"

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "mfp-diaries.tsv"

# mapping dict to merge with df.describe() tables
# TODO: turn into static variables (all caps)
courses_mapping_dict = {
    "id": ("unique identifier of a course (primary key)", "string"),
    "maker": ("username of player who created", "string"),
    "difficulty": ("4 levels - easy, normal, expert, superExpert", "string"),
    "gameStyle": ("4 types – marioBrosU, marioBros, marioWorld, marioBros3", "string"),
    "title": ("name", "string"),
    "thumbnail": ("URL", "string"),
    "image": ("URL", "string"),
    "creation": ("course creation timestamp", "string"),
}

course_meta_mapping_dict = {
    "id": ("unique identifier of a course (primary key)", "string"),
    "firstClear": ("username of player who first cleared", "string"),
    "tag": ('theme, e.g. "traditional", "speedrun"', "string"),
    "stars": ("total number of stars given by players", "int"),
    "players": ("number of players who played the course", "int"),
    "tweets": ("total number of tweets players made", "int"),
    "clears": ("total nubmer of times players cleared", "int"),
    "attempts": ("total number of attempts", "int"),
    "clearRate": ("ratio of clears to attempts", "float (64-bit)"),
    "catch": ("date/time of the data capture", "string"),
}

plays_clears_likes_mapping_dict = {
    "id": ("unique identifier of a course (foreign key)", "string"),
    "players": ("username of player", "string"),
    "catch": ("date/time of the data capture", "string"),
}

players_mapping_dict = {
    "id": ("unique identifier of a course (primary key)", "string"),
    "image": ("URL", "string"),
    "flag": ("nationality (2-letter abbreviation)", "string"),
    "name": ("username of player", "string"),
}

records_mapping_dict = {
    "id": ("unique identifier of a course (foreign key)", "string"),
    "players": ("username of player", "string"),
    "catch": ("date/time of the data capture", "string"),
    "timeRecord": ("time record (in milliseconds)", "int"),
}

mfp_mapping_dict = {
    "user_id": ("unique identifier of each user", "int"),
    "date": ("date of log (no 24-hour clock time)", "string"),
    "entries": ("meals & dishes (with nutrition info) the user consumed", "string (can be loaded as JSON)"),
    "summary": ("goal and total (true consumption)", "string (can be loaded as JSON)"),
}


def output_variables_table(df_pl: pl.DataFrame, mapping_dict: dict[str, tuple[str, str]]) -> pl.DataFrame:
    # TODO: drop "min", "max" if all columsn are type string, currently being handled maunally in `.ipynb`
    assert all(len(v) == 2 for v in mapping_dict.values()), "Each value in mapping_dict must be a tuple of length 2."
    base_describe_df = df_pl.describe()
    temp2 = pl.DataFrame(base_describe_df.columns).transpose()
    temp2.columns = base_describe_df.columns
    combined_df = base_describe_df.transpose()
    combined_df.columns = combined_df.row(0)
    combined_df = combined_df.tail(-1)
    statistic_column = pl.Series("statistic", temp2.columns[1:])
    combined_df.insert_column(0, statistic_column)

    mapping_df = pl.DataFrame(
        {
            "statistic": list(mapping_dict.keys()),
            "description": [value[0] for value in mapping_dict.values()],
            "type": [value[1] for value in mapping_dict.values()],
        }
    )

    final_df = combined_df.join(mapping_df, on="statistic", how="left")
    final_df = final_df[[s.name for s in final_df if not (s.null_count() == final_df.height)]]
    # move the "type" column to the 2nd position (index 1)
    original_columns = final_df.columns
    new_order = [original_columns[0]] + [original_columns[-1]] + original_columns[1:-1]
    df_reordered = final_df.select(new_order)
    return df_reordered


def print_table(df: pl.DataFrame) -> None:
    if "ipykernel" in sys.modules:
        from IPython.display import display

        display_func = display
    else:
        display_func = print
    with pl.Config() as cfg:
        cfg.set_verbose(True)
        cfg.set_fmt_str_lengths(
            100
        )  # randomly chosen integer for now, can create a function to find the max number of chars in all cells
        display_func(df)


# TODO: turn the functions below into an SMMNet object
def try_parse_datetime(s: str | datetime):
    if isinstance(s, datetime):
        return s
    formats = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"cannot convert {s} to a datetime object.")


def convert_catch_datetime(df: pl.DataFrame, datetime_col: str) -> pl.DataFrame:
    return df.with_columns(
        [df[datetime_col].map_elements(lambda x: try_parse_datetime(x), return_dtype=datetime).alias(datetime_col)]
    )


class MFP:
    def __init__(self) -> None:
        self.setup_df()
        self.entries_vc = self.get_df_vc(ENTRIES)
        self.summary_vc = self.get_df_vc(SUMMARY)
        self.filtered_df = None

    def setup_df(self) -> None:
        assert DATA_PATH.exists()
        self.df = pl.read_csv(DATA_PATH, has_header=False, separator="\t")
        self.df.columns = COLUMNS
        self.df = self.convert_catch_datetime(DATE)

    def get_df(self) -> pl.DataFrame:
        return self.df

    def get_df_columns(self):
        return self.df.columns

    def get_len_df_columns(self):
        return len(self.df.columns)

    def get_df_vc(self, col_name: str) -> pl.DataFrame:
        return self.df[col_name].value_counts().sort(by=COUNT, descending=True)

    def try_parse_datetime(self, s: str | datetime):
        if isinstance(s, datetime):
            return s
        formats = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
        for fmt in formats:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        raise ValueError(f"cannot convert {s} to a datetime object.")

    def convert_catch_datetime(self, datetime_col: str) -> pl.DataFrame:
        return self.get_df().with_columns(
            [
                self.get_df()[datetime_col]
                .map_elements(lambda x: self.try_parse_datetime(x), return_dtype=datetime)
                .alias(datetime_col)
            ]
        )

    def filter_most_freq_users(self, top_k: int) -> pl.DataFrame:
        top_k_users = self.df[USER_ID].value_counts().sort(COUNT, descending=True)[USER_ID][:top_k].to_list()
        return self.get_df().filter(pl.col(USER_ID).is_in(top_k_users))

    def plot_value_counts_on_axis(
        self, ax, labels: list, values: list, col_name: str, top_n: int, x_axis_title: str = "", y_axis_title: str = ""
    ):
        ax.set_title(col_name, fontsize=18)
        sorted_vals = sorted(values, reverse=True)
        set_sorted_vals = set(sorted_vals)
        if len(set_sorted_vals) == 1 and 1 in set_sorted_vals:
            print(f"{col_name}: All values are unique")
            ax.text(
                0.5,
                0.5,
                f"{col_name}: All values are unique",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            return
        else:
            print(f"{col_name}: {sorted_vals[:50]}")
        # choose to do bar (qualitative labels) or hist (quantitative labels)
        match col_name:
            case _ if col_name in [USER_ID, ENTRIES, SUMMARY]:
                ax.boxplot(values)
            case _:
                ax.hist(values, density=True, bins=20)
                ax.set_xlabel(x_axis_title, fontsize=16)
                ax.set_ylabel(y_axis_title, fontsize=16)
        ax.grid(True)
        ax.tick_params(labelrotation=45, labelsize=14)

    def plot_all_columns(self, top_n: int) -> None:
        num_columns_to_plot = self.get_len_df_columns()
        grid_size = int(np.ceil(np.sqrt(num_columns_to_plot)))  # Creating a square grid that fits all plots

        _, axs = plt.subplots(grid_size, grid_size, figsize=(15, 10))
        axs = axs.flatten()  # Flatten the array of axes to easily iterate over it

        ax_index = 0
        for col in self.get_df_columns():
            value_counts_pl = self.get_df().group_by(col).agg(pl.len()).sort(LEN, descending=True)
            labels = value_counts_pl[col].to_list()
            values = value_counts_pl[LEN].to_list()

            self.plot_value_counts_on_axis(axs[ax_index], labels, values, col, top_n, "Values", "Density")
            ax_index += 1
            if ax_index >= num_columns_to_plot:
                break

        plt.tight_layout()
        plt.show()

    def plot_earliest_latest_logs(self, df) -> None:
        df["earliest_log"] = pd.to_datetime(df["earliest_log"])
        df["latest_log"] = pd.to_datetime(df["latest_log"])

        # Convert to numerical values
        df["earliest_num"] = mdates.date2num(df["earliest_log"])
        df["latest_num"] = mdates.date2num(df["latest_log"])

        # Calculate the min and max values
        min_num = df["earliest_num"].min()
        max_num = df["latest_num"].max()

        # Generate 10 evenly spaced tick values
        ticks = np.linspace(min_num, max_num, 10)

        # Calculate the length of the DataFrame
        num_users = len(df)

        # Generate y-ticks at intervals of 10, ensuring the last index is included if length < 10
        if num_users < 10:
            y_ticks = np.arange(0, num_users)
        else:
            y_ticks = np.arange(0, num_users, 10)
            if num_users - 1 not in y_ticks:
                y_ticks = np.append(y_ticks, num_users - 1)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.barh(df.index, df["latest_num"] - df["earliest_num"], left=df["earliest_num"], color="skyblue")

        # Format the x-axis to show datetime values
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.gca().set_xticks(ticks)
        plt.gca().set_xticklabels([mdates.num2date(tick).strftime("%Y-%m-%d %H:%M") for tick in ticks])

        # Set y-ticks at intervals, ensuring the last index is included if length < 10
        plt.gca().set_yticks(y_ticks)
        plt.gca().set_yticklabels(y_ticks)

        plt.xlabel("Time")
        plt.ylabel("Index")
        plt.title(f"Time intervals of the earliest and latest log of the top {num_users} users")
        plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

        plt.show()

    def expand_entries(self, entry_elem: str, user_id: int, date: datetime) -> pd.DataFrame:
        """
        Loads an entry element (from the ENTRIES column of the main df).
        Returns a df with each data point, e.g. name, nutritional category expanded.
        """
        assert user_id > 0
        json_entry_elem = json.loads(entry_elem)
        data = []
        for entry in json_entry_elem:
            for item in entry[DISHES]:
                row = {USER_ID: user_id, DATE: date, MEAL: entry[MEAL], NAME: item[NAME]}
                for nutrient in item[NUTRITIONS]:
                    row[nutrient[NAME]] = nutrient[VALUE]
                data.append(row)

        return pd.DataFrame(data)

    def expand_summary(self, summary_elem: str, user_id: int) -> pd.Series:
        """
        Loads a summary element (from the SUMMARY column of the main dataframe).
        Returns a Series with metrics as indices and their respective 'total' and 'goal' values,
        along with 'user_id', which can be directly used to extend the DataFrame row-wise.
        """
        assert user_id > 0
        obj = json.loads(summary_elem)
        total_df = pd.DataFrame(obj[TOTAL])
        goal_df = pd.DataFrame(obj[GOAL])
        total_df = total_df.set_index(NAME).T.rename(columns=lambda x: x + "_total")
        goal_df = goal_df.set_index(NAME).T.rename(columns=lambda x: x + "_goal")
        combined_df = pd.concat([total_df, goal_df], axis=1)
        combined_series = combined_df.iloc[0]  # Convert the single row DataFrame to a Series
        combined_series[USER_ID] = user_id
        return combined_series
