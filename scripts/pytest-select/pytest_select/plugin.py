from typing import Optional, ClassVar, Pattern
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
        "--select-fail-on-missing", action="store_true", dest="selectfailonmissing", default=False, help=(
            "Fail instead of warn when not all "  # pragma: no mutate
            "(de-)selected tests could be found."  # pragma: no mutate
        ))
    select_group.addoption(
        "--select-skip-instead-of-deselect",
        action="store_true",
        dest="skipinsteadofdeselect",
        default=False,
        help="Mark test as skipped instead of deselecting in collection",
    )


@dataclass
class SelectConfig:  # pylint:disable = R0902
    fail_on_missing: bool
    skip_instead_of_deselect: bool
    prefix: str
    file_path: Optional[str]
    select_from_file: bool = False
    deselect_from_file: bool = False
    test_names: set[str] = field(default_factory=set)
    seen_test_names: set[str] = field(default_factory=set)
    select_text: str = field(init=False)

    variant_pattern: ClassVar[Pattern] = re.compile(r"^(.*?)\[(.+)\]$")
    option_text: ClassVar[str] = "selectfromfile"

    def __post_init__(self):
        if self.prefix == "de":
            self.deselect_from_file = True
        elif self.skip_instead_of_deselect:
            raise UsageError(
                "pytest-select: '--select-skip-instead-of-deselect' can only be used with '--deselect-from-file' option."
            )
        else:
            self.select_from_file = True
        self.select_text = f"{self.prefix}selectfromfile"
        if self.file_path and not Path(self.file_path).exists():
            raise UsageError(f"Given selection file '{self.file_path}' doesn't exist.")
        with Path(self.file_path).open("rt", encoding="UTF-8") as selection_file:
            for test_name_raw in selection_file:
                test_name = test_name_raw.strip()
                if test_name.startswith("#") or test_name == "":
                    continue
                self.test_names.add(test_name)

    def no_test_items_match(self, name, nodeid, mark_as_seen_if_match=True) -> bool:
        variant_match = self.variant_pattern.findall(nodeid)
        item_path = variant_match[0][0] if len(variant_match) == 1 else nodeid
        match = (name in self.test_names or nodeid in self.test_names or item_path in self.test_names)
        if not match:
            return True
        if mark_as_seen_if_match:
            self.seen_test_names.add(name)
            self.seen_test_names.add(nodeid)
            self.seen_test_names.add(item_path)
        return False

    def get_report_header(self) -> list[str]:
        suffix = ", failing on missing selection items" if self.fail_on_missing else ""
        report_header = f"select: {self.prefix}selecting tests from '{self.file_path}'{suffix}"
        return [report_header]

    def check_missing_tests(self):
        missing_test_names = self.test_names - self.seen_test_names
        if missing_test_names:
            # If any items remain in `test_names` those tests either don't exist or
            # have been deselected by another way - warn user
            n_prefix = "" if self.prefix == "de" else "de"
            message = (f"pytest-select: Not all {self.prefix}selected tests exist "
                       f"(or have been {n_prefix}selected otherwise).\n"
                       f"Missing {self.prefix}selected test names:\n  - ")
            message += "\n  - ".join(missing_test_names)
            if self.fail_on_missing:
                raise UsageError(message)
            warnings.warn(message, PytestSelectWarning)

    @classmethod
    def from_config(cls, config: pytest.Config) -> Optional["SelectConfig"]:
        option_prefix = ""
        file_path: str | None = None
        fail_on_missing = config.getoption("selectfailonmissing")
        skip_instead_of_deselect = config.getoption("skipinsteadofdeselect")
        if (option := config.getoption(f"{cls.option_text}")) is not None:
            file_path = option
        if (option := config.getoption(f"de{cls.option_text}")) is not None:
            if file_path is not None:
                raise UsageError("'--select-from-file' and '--deselect-from-file' can not be used together.")
            option_prefix = "de"
            file_path = option
        if file_path is None:
            return None
        return SelectConfig(
            fail_on_missing,
            skip_instead_of_deselect,
            option_prefix,
            file_path,
        )


@pytest.hookimpl(trylast=True)  # pragma: no mutate
def pytest_report_header(config):  # pylint:disable = R1710
    if (select_config := SelectConfig.from_config(config)) is not None:
        return select_config.get_report_header()


def pytest_collection_modifyitems(session, config, items):  # pylint: disable=W0613
    select_config = SelectConfig.from_config(config)
    if select_config is None:
        return
    selected_items = []
    deselected_items = []
    for item in items:
        no_match = select_config.no_test_items_match(item.name, item.nodeid, mark_as_seen_if_match=True)

        if no_match and select_config.select_from_file:
            deselected_items.append(item)
            continue
        if (no_match and (select_config.deselect_from_file or select_config.skip_instead_of_deselect)):
            selected_items.append(item)
            continue

        if select_config.skip_instead_of_deselect:
            item.add_marker(pytest.mark.skip(reason="Deselected by pytest-select."))
        elif select_config.select_from_file:
            selected_items.append(item)
        else:
            deselected_items.append(item)
    select_config.check_missing_tests()
    if select_config.skip_instead_of_deselect:
        return
    items[:] = selected_items
    config.hook.pytest_deselected(items=deselected_items)
