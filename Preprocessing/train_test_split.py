"""Split an event log into training, validation and test sets."""


def get_train_test_split_point(df, test_ratio, case_id, timestamp):
    """
    Return the timestamp and row index of the first event of the first test case.
    Cases are ordered by start time; the last `test_ratio` share of cases forms the test set.
    """
    case_start_times = df.groupby(case_id)[timestamp].min().sort_values()
    first_test_case_idx = int(len(case_start_times) * (1 - test_ratio))
    first_test_case_id = case_start_times.index[first_test_case_idx]

    train_test_split_idx = df[df[case_id] == first_test_case_id].index[0]
    train_test_split_time = df.loc[train_test_split_idx, timestamp]

    return train_test_split_time, train_test_split_idx


def get_discard_case_list(df, test_ratio, case_id, timestamp):
    """Return the cases that start before and end after the train/test split point."""
    case_times = df.groupby(case_id)[timestamp].agg(Start="min", End="max").reset_index()
    train_test_split_time, _ = get_train_test_split_point(df, test_ratio, case_id, timestamp)

    discard_cases = case_times[(case_times["Start"] < train_test_split_time) &
                               (case_times["End"] >= train_test_split_time)]

    return discard_cases[case_id].tolist()


def create_table_without_discard_case(df, test_ratio, case_id, timestamp):
    """Return the event log without the cases overlapping the split point."""
    discard_case_list = get_discard_case_list(df, test_ratio, case_id, timestamp)
    return df[~df[case_id].isin(discard_case_list)]


def get_case_list_by_ratio(df, ratio, case_id, timestamp):
    """Return two case lists, ordered by case start time, split according to `ratio`."""
    case_start_times = df.groupby(case_id)[timestamp].min().sort_values()
    num_pre_split_cases = int(len(case_start_times) * (1 - ratio))

    pre_split_case_list = case_start_times[:num_pre_split_cases + 1].index.tolist()
    post_split_case_list = case_start_times[num_pre_split_cases + 1:].index.tolist()

    return pre_split_case_list, post_split_case_list
