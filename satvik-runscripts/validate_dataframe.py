import numpy as np
import pandas as pd

columns = [
    "src_gid",
    "target_gid",
    "src_type",
    "target_type",
    "receptor",
    "template_loc",
    "actual_section",
    "segX",
    "weight",
    "delay",
    "lamtha",
    "threshold",
    "gain",
]

positive_columns = ["delay", "lamtha", "gain"] # should have positive values
numerical_columns = ["segX", "weight", "delay", "lamtha", "threshold", "gain"] # should be a numerical value


def validate_Connectivity_df(df, raise_on_error=True):
    # list of errors raised 
    errors = [] 

    #dataframe is empty
    if len(df) == 0:
        errors.append("conn_dataframe is empty.")
        if raise_on_error and errors:
            raise ValueError(f"Connectivity Dataframe is empty")
        # we will have to terminate in between as cant do any operation on empty ddataframe
            return errors

    # dataframe lacks some columns    
    missing_columns = [c for c in columns if c not in df.columns]
    if missing_columns:
        errors.append(f"Missing column(s): {missing_columns}. Expected: {columns}")
        if raise_on_error and errors:
            raise ValueError(f"conn_dataframe check failed ({len(errors)} issue(s)):\n" + "\n".join(f"  - {e}" for e in errors))
        return errors

    #dataframe has extra columns
    extra_columns = [c for c in df.columns if c not in columns]
    if extra_columns:
        errors.append(f"Unexpected column(s): {extra_columns}")

    #dataframe contains Nan (null) values
    null_counts = df[columns].isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts) > 0:
        errors.append("Null values in: " + ", ".join(f"{col} ({count})" for col, count in null_counts.items()))

    #for every numerical coulmn , it confirms whether it is numeric or not
    for col in numerical_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"'{col}' is not numeric (dtype={df[col].dtype})")
            continue
        non_finite_mask = ~np.isfinite(df[col])
        if non_finite_mask.sum() > 0:
            bad_indices = df.index[non_finite_mask].tolist()[:10]
            errors.append(f"'{col}' has {non_finite_mask.sum()} non-finite value(s) at {bad_indices}")

    #checks whether a positive columns doesnt contain negative values
    for column_name in positive_columns:
        if column_name not in df.columns or not pd.api.types.is_numeric_dtype(df[column_name]):
            continue
        negative_rows = df[df[column_name] < 0]
        if len(negative_rows) > 0:
            errors.append(f"'{column_name}' has {len(negative_rows)} negative value(s), e.g. row {negative_rows.index[0]}={negative_rows[column_name].iloc[0]}")

    #segX is out of 0 and 1
    if pd.api.types.is_numeric_dtype(df["segX"]):
        out_of_range_rows = df[(df["segX"] < 0) | (df["segX"] > 1)]
        if len(out_of_range_rows) > 0:
            errors.append(f"'segX' out of [0,1] in {len(out_of_range_rows)} row(s), e.g. row {out_of_range_rows.index[0]}={out_of_range_rows['segX'].iloc[0]}")

    known_cell_types = set(net.cell_types.keys())
    known_src_types = set(net.gid_ranges.keys())  # includes drives

    #
    unknown_src_types = set(df["src_type"].unique()) - known_src_types
    if unknown_src_types:
        errors.append(f"Unknown src_type(s): {sorted(unknown_src_types)}")

    unknown_target_types = set(df["target_type"].unique()) - known_cell_types
    if unknown_target_types:
        errors.append(f"Unknown target_type(s): {sorted(unknown_target_types)} (must be a real cell type)")

    for target_type in df["target_type"].unique():
        if target_type not in net.cell_types:
            continue
        cell = net.cell_types[target_type]["cell_object"]
        valid_receptors = set(cell.synapses.keys())
        rows_for_target = df[df["target_type"] == target_type]
        unknown_receptors = set(rows_for_target["receptor"].unique()) - valid_receptors
        if unknown_receptors:
            errors.append(f"Unknown receptor(s) {sorted(unknown_receptors)} for target_type='{target_type}'")

    for target_type in df["target_type"].unique():
        if target_type not in net.cell_types:
            continue
        cell = net.cell_types[target_type]["cell_object"]
        valid_locations = set(cell.sect_loc.keys()) | set(cell.sections.keys())
        rows_for_target = df[df["target_type"] == target_type]

        unknown_template_locs = set(rows_for_target["template_loc"].unique()) - valid_locations
        if unknown_template_locs:
            errors.append(f"Unknown template_loc {sorted(unknown_template_locs)} for target_type='{target_type}'")

        unknown_actual_sections = set(rows_for_target["actual_section"].unique()) - valid_locations
        if unknown_actual_sections:
            errors.append(f"Unknown actual_section {sorted(unknown_actual_sections)} for target_type='{target_type}'")

    for src_type in df["src_type"].unique():
        if src_type not in net.gid_ranges:
            continue
        valid_range = net.gid_ranges[src_type]
        rows_for_src = df[df["src_type"] == src_type]
        out_of_range_rows = rows_for_src[~rows_for_src["src_gid"].isin(valid_range)]
        if len(out_of_range_rows) > 0:
            bad_gids = sorted(out_of_range_rows["src_gid"].unique())[:10]
            errors.append(f"{len(out_of_range_rows)} row(s) have src_gid outside range for src_type='{src_type}' ({valid_range.start}-{valid_range.stop-1}): {bad_gids}")

    for target_type in df["target_type"].unique():
        if target_type not in net.gid_ranges:
            continue
        valid_range = net.gid_ranges[target_type]
        rows_for_target = df[df["target_type"] == target_type]
        out_of_range_rows = rows_for_target[~rows_for_target["target_gid"].isin(valid_range)]
        if len(out_of_range_rows) > 0:
            bad_gids = sorted(out_of_range_rows["target_gid"].unique())[:10]
            errors.append(f"{len(out_of_range_rows)} row(s) have target_gid outside range for target_type='{target_type}' ({valid_range.start}-{valid_range.stop-1}): {bad_gids}")


    range_starts = [r.start for r in net.gid_ranges.values()]
    range_stops = [r.stop for r in net.gid_ranges.values()]
    full_gid_span = (0, 0) if not range_starts else (min(range_starts), max(range_stops))

    src_type_mismatch_flags = []
    for _, row in df.iterrows():
        if row["src_gid"] in range(*full_gid_span):
            src_type_mismatch_flags.append(net.gid_to_type(row["src_gid"]) != row["src_type"])
        else:
            src_type_mismatch_flags.append(False)
    src_type_mismatch = pd.Series(src_type_mismatch_flags, index=df.index)
    if src_type_mismatch.sum() > 0:
        errors.append(f"{src_type_mismatch.sum()} row(s) have src_gid whose real type != src_type: {df.index[src_type_mismatch].tolist()[:10]}")

    target_type_mismatch_flags = []
    for _, row in df.iterrows():
        if row["target_gid"] in range(*full_gid_span):
            target_type_mismatch_flags.append(net.gid_to_type(row["target_gid"]) != row["target_type"])
        else:
            target_type_mismatch_flags.append(False)
    target_type_mismatch = pd.Series(target_type_mismatch_flags, index=df.index)
    if target_type_mismatch.sum() > 0:
        errors.append(f"{target_type_mismatch.sum()} row(s) have target_gid whose real type != target_type: {df.index[target_type_mismatch].tolist()[:10]}")

    drive_names = set(net.external_drives.keys()) if hasattr(net, "external_drives") else set()
    target_types_that_are_drives = set(df["target_type"].unique()) & drive_names
    if target_types_that_are_drives:
        errors.append(f"target_type contains drive name(s): {sorted(target_types_that_are_drives)}")

    duplicate_row_count = df.duplicated().sum()
    if duplicate_row_count > 0:
        errors.append(f"{duplicate_row_count} exact duplicate row(s) found — likely added twice.")

    if raise_on_error and errors:
        raise ValueError(f"conn_dataframe check failed ({len(errors)} issue(s)):\n" + "\n".join(f"  - {e}" for e in errors))
    return errors