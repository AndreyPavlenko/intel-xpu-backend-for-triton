from enum import Enum
from typing import Optional, ClassVar, Pattern
from dataclasses import dataclass, field
import warnings
from pathlib import Path
import re

import pytest
from pytest import UsageError


class SelectOption(Enum):
    SELECT = "selectfromfile"
    DESELECT = "deselectfromfile"
    SKIP = "skipfromfile"


@dataclass
class SelectConfig:
    select_option: SelectOption
    file_path: Optional[str]
    fail_on_missing: bool
    test_names: set[str] = field(default_factory=set)
    seen_test_names: set[str] = field(default_factory=set)

    variant_pattern: ClassVar[Pattern] = re.compile(r"^(.*?)\[(.+)\]$")

    def __post_init__(self):
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
        action_text = "selecting"
        if self.select_option == SelectOption.DESELECT:
            action_text = "deselecting"
        elif self.select_option == SelectOption.SKIP:
            action_text = "skipping"
        suffix = ", failing on missing selection items" if self.fail_on_missing else ""
        report_header = f"select: {action_text} tests from '{self.file_path}'{suffix}"
        return [report_header]

    def check_missing_tests(self):
        missing_test_names = self.test_names - self.seen_test_names
        if missing_test_names:
            # If any items remain in `test_names` those tests either don't exist or
            # have been deselected by another way - warn user
            if self.select_option == SelectOption.SELECT:
                message = ("\npytest-select: Not all selected tests exist "
                           "(or have been deselected otherwise).\n"
                           "Missing selected test names:\n  - ")
            elif self.select_option == SelectOption.DESELECT:
                message = ("\npytest-select: Not all deselected tests exist "
                           "(or have been selected otherwise).\n"
                           "Missing deselected test names:\n  - ")
            else:
                message = ("\npytest-select: Not all tests to skip exist "
                           "(or have been not skipped otherwise).\n"
                           "Missing test names to skip:\n  - ")
            message += "\n  - ".join(missing_test_names)
            if self.fail_on_missing:
                raise UsageError(message)
            warnings.warn(UserWarning(message))

    @classmethod
    def from_config(cls, config: pytest.Config) -> Optional["SelectConfig"]:
        fail_on_missing = config.getoption("selectfailonmissing")
        file_path = None
        for option in SelectOption:
            if (option_value := config.getoption(option.value)) is not None:
                select_option = option
            else:
                continue
            if file_path is None:
                file_path = option_value
            else:
                raise UsageError(
                    "'--select-from-file', '--deselect-from-file' and '--skip-from-file' cannot be used together.")
        if file_path is None:
            return None
        return SelectConfig(select_option, file_path, fail_on_missing)


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
        help="Fail instead of warn when not all (de-)selected tests could be found.",
    )
    select_group.addoption(
        "--skip-from-file",
        action="store",
        dest="skipfromfile",
        default=None,
        help="Mark tests from file as skipped.",
    )


@pytest.hookimpl(trylast=True)  # pragma: no mutate
def pytest_report_header(config):  # pylint:disable = R1710
    if (select_config := SelectConfig.from_config(config)) is not None:
        return select_config.get_report_header()


class SelectPlugin:

    def __init__(self):
        # This list will hold all strings collected from worker nodes.
        self.seen_test_names = []

    def pytest_collection_modifyitems(self, session, config, items):  # pylint: disable=W0613
        select_config = SelectConfig.from_config(config)
        if select_config is None:
            return
        option = select_config.select_option
        selected_items = []
        deselected_items = []
        for item in items:
            no_match = select_config.no_test_items_match(item.name, item.nodeid, mark_as_seen_if_match=True)
            if (option in [SelectOption.DESELECT, SelectOption.SKIP] and no_match
                    or option in [SelectOption.SELECT] and not no_match):
                selected_items.append(item)
                continue
            if option in [SelectOption.SKIP]:
                item.add_marker(pytest.mark.skip(reason="Deselected by pytest-select"))
            else:
                deselected_items.append(item)

        if hasattr(config, "workerinput"):
            # config.workeroutput is a dict that will be transferred back to the master process
            config.workeroutput = getattr(config, "seen_test_names", {})
            config.workeroutput["seen_test_names"] = select_config.seen_test_names

        if option in [SelectOption.SKIP]:
            return
        items[:] = selected_items
        config.hook.pytest_deselected(items=deselected_items)

    def pytest_testnodedown(self, node, error):  # pylint: disable=W0613
        if not hasattr(node, "workeroutput"):
            return
        worker_output = node.workeroutput
        if worker_output and "seen_test_names" in worker_output:
            # Extend the global list with the names received from the worker.
            self.seen_test_names.extend(worker_output["seen_test_names"])

    def pytest_sessionfinish(self, session, exitstatus):  # pylint: disable=W0613
        # Ensure that it runs only in the master process when using xdist:
        if hasattr(session.config, "workerinput"):
            return
        select_config = SelectConfig.from_config(session.config)
        if select_config is None:
            return
        select_config.seen_test_names = set(self.seen_test_names)
        select_config.check_missing_tests()


def pytest_configure(config):
    config.pluginmanager.register(SelectPlugin(), "my_plugin")
