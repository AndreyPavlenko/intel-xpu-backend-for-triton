from typing import Optional
from dataclasses import dataclass, field
import warnings
from pathlib import Path
import re

import pytest
from pytest import PytestWarning, UsageError


class PytestSelectWarning(PytestWarning):  # pylint:disable = R0903
    pass


def pytest_addoption(parser):
    select_group = parser.getgroup(
        "select",
        "Modify the list of collected tests.",
    )
    select_group.addoption(
        "--select-from-file",
        action="store",
        dest="selectfromfile",
        default=None,
        help="Select tests given in file. One line per test name.",
    )
    select_group.addoption(
        "--deselect-from-file",
        action="store",
        dest="deselectfromfile",
        default=None,
        help="Deselect tests given in file. One line per test name.",
    )
    select_group.addoption(
        "--select-fail-on-missing",
        action="store_true",
        dest="selectfailonmissing",
        default=False,
        help=(
            "Fail instead of warn when not all "  # pragma: no mutate
            "(de-)selected tests could be found."  # pragma: no mutate
        ),
    )


@dataclass
class SelectConfig:
    fail_on_missing: bool
    prefix: str
    file_path: str | None
    select_from_file: bool = False
    deselect_from_file: bool = False
    select_text: str = field(init=False)

    @classmethod
    def _get_option_text(cls, prefix: str) -> str:
        return f"{prefix}selectfromfile"

    def __post_init__(self):
        if self.prefix == "de":
            self.deselect_from_file = True
        else:
            self.select_from_file = True
        self.select_text = self._get_option_text(self.prefix)
        if self.file_path and not Path(self.file_path).exists():
            raise UsageError(f"Given selection file '{self.file_path}' doesn't exist.")

    def get_report_header(self) -> list[str]:
        fail_on_missing_suffix = (", failing on missing selection items" if self.fail_on_missing else "")
        report_header = f"select: {self.prefix}selecting tests from '{self.file_path}'{fail_on_missing_suffix}"
        return [report_header]

    @classmethod
    def from_config(cls, config: pytest.Config) -> Optional["SelectConfig"]:
        option_prefix = ""
        file_path: str | None = None
        fail_on_missing = config.getoption("selectfailonmissing")
        if (option := config.getoption(cls._get_option_text(""))) is not None:
            file_path = option
        if (option := config.getoption(cls._get_option_text("de"))) is not None:
            if file_path is not None:
                raise UsageError("'--select-from-file' and '--deselect-from-file' can not be used together.")
            option_prefix = "de"
            file_path = option
        if file_path is None:
            return None
        return SelectConfig(
            fail_on_missing,
            option_prefix,
            file_path,
        )


@pytest.hookimpl(trylast=True)  # pragma: no mutate
def pytest_report_header(config):  # pylint:disable = R1710
    if (select_config := SelectConfig.from_config(config)) is not None:
        return select_config.get_report_header()


def _check_missing_tests(tests, seen_tests, fail_on_missing, prefix):
    missing_test_names = tests - seen_tests
    if missing_test_names:
        # If any items remain in `test_names` those tests either don't exist or
        # have been deselected by another way - warn user
        n_prefix = "" if prefix == "de" else "de"
        message = (f"pytest-select: Not all {prefix}selected tests exist "
                   f"(or have been {n_prefix}selected otherwise).\n"
                   f"Missing {prefix}selected test names:\n  - ")
        message += "\n  - ".join(missing_test_names)
        if fail_on_missing:
            raise UsageError(message)
        warnings.warn(message, PytestSelectWarning)


def _load_test_names(select_file_name) -> set[str]:
    with Path(select_file_name).open("rt", encoding="UTF-8") as selection_file:
        return {test_name.strip() for test_name in selection_file}


def pytest_collection_modifyitems(session, config, items):  # pylint: disable=W0613
    if (select_config := SelectConfig.from_config(config)) is not None:
        seen_test_names = set()
        selected_items = []
        deselected_items = []
        variant_pattern = re.compile(r"^(.*?)\[[^\]]*\]$")

        test_names = _load_test_names(select_config.file_path)

        for item in items:
            variant_match = variant_pattern.findall(item.nodeid)
            if len(variant_match) == 1:
                item_path = variant_match[0]
                seen_test_names.add(item_path)
            else:
                item_path = item.nodeid
            if (item.name in test_names or item.nodeid in test_names or item_path in test_names):
                selected_items.append(item)
            else:
                deselected_items.append(item)

            seen_test_names.add(item.name)
            seen_test_names.add(item.nodeid)

        if select_config.deselect_from_file:
            # We are *de*selecting, flip collections
            selected_items, deselected_items = deselected_items, selected_items

        _check_missing_tests(test_names, seen_test_names, select_config.fail_on_missing, select_config.prefix)

        # Slice assignment is required since `items` needs to be modified in place
        items[:] = selected_items
        config.hook.pytest_deselected(items=deselected_items)
