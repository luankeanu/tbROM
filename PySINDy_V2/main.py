from __future__ import annotations

from assets.analyse import run_analysis_stage
from assets.fit import run_fit_stage
from assets.read import run_read_stage
from assets.run import run_prediction_stage


"""
Section 1: Prompt handling helpers.
This section keeps the command-line workflow predictable by validating the
expected y/n answers before the stage logic runs.
"""


def prompt_yes_no(message: str) -> bool:
    while True:
        answer = input(f"{message} [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("[main] please answer with 'y' or 'n'", flush=True)


"""
Section 2: Stage runners.
This section executes the implemented workflow stages and reports the key facts
returned by each stage in a compact way.
"""


def run_read_stage_from_main() -> None:
    results = run_read_stage()
    print(
        "[main] read stage completed with "
        f"{len(results.cases)} cases, "
        f"{len(results.train_table)} training rows, and "
        f"{len(results.confirmation_table)} confirmation rows",
        flush=True,
    )


def run_fit_stage_from_main() -> None:
    results = run_fit_stage()
    print(
        "[main] fit stage completed with "
        f"fit run {results.fit_index}, "
        f"{results.train_case_count} training cases, and "
        f"{results.train_row_count} training rows",
        flush=True,
    )


def run_prediction_stage_from_main() -> None:
    results = run_prediction_stage()
    print(
        "[main] run stage completed with "
        f"run {results.run_index}, "
        f"{results.confirmation_case_count} confirmation cases, and "
        f"{results.confirmation_row_count} confirmation rows",
        flush=True,
    )


def run_analysis_stage_from_main() -> None:
    results = run_analysis_stage()
    print(
        "[main] analyse stage completed with "
        f"{results.displayed_figure_count} displayed figures, "
        f"{results.prepared_case_count} prepared cases, and "
        f"{results.confirmation_case_count} analysed confirmation cases",
        flush=True,
    )


"""
Section 3: Main workflow.
This section presents the four-stage y/n interface that will remain the single
entry point for the project as the later scripts are added.
"""


def main() -> None:
    if prompt_yes_no("Run read stage?"):
        run_read_stage_from_main()
    else:
        print("[main] read stage skipped", flush=True)

    if prompt_yes_no("Run fit stage?"):
        run_fit_stage_from_main()
    else:
        print("[main] fit stage skipped", flush=True)

    if prompt_yes_no("Run run stage?"):
        run_prediction_stage_from_main()
    else:
        print("[main] run stage skipped", flush=True)

    if prompt_yes_no("Run analyse stage?"):
        run_analysis_stage_from_main()
    else:
        print("[main] analyse stage skipped", flush=True)


if __name__ == "__main__":
    main()
