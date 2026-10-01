#!/usr/bin/env python3
"""Static Dashboard JavaScript/DOM/API contract checks."""

import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path


BIN_DIR = Path(__file__).resolve().parent
HTML_PATH = BIN_DIR / "static" / "index.html"
SERVER_PATH = BIN_DIR / "dashboard_server.py"

REQUIRED_WORKSPACE_IDS = {
    "appSidebar",
    "sectionNav",
    "workspace",
    "currentAdjustmentMode",
}
REQUIRED_WORKSPACE_SECTION_IDS = {
    "quickSection",
    "hardwareSection",
    "bodySection",
    "faceSection",
    "handSection",
    "streamingSection",
    "hmdSection",
    "logSection",
    "modelSection",
}
REQUIRED_WORKSPACE_FUNCTIONS = {
    "selectWorkspaceSection",
    "loadPerformanceRecommendation",
}


class DashboardStructureParser(HTMLParser):
    """Collect the DOM relationships that cannot be checked reliably by regex."""

    VOID_ELEMENTS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.declared_id_list = []
        self.section_nav_targets = []
        self.top_level_workspace_sections = []
        self.idless_top_level_workspace_sections = []
        self._stack = []

    def handle_starttag(self, tag, attrs) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.declared_id_list.append(element_id)

        inside_section_nav = element_id == "sectionNav" or any(
            frame_id == "sectionNav" for _, frame_id in self._stack
        )
        section_target = attributes.get("data-section")
        if inside_section_nav and section_target:
            self.section_nav_targets.append(section_target)

        parent_id = self._stack[-1][1] if self._stack else None
        classes = set((attributes.get("class") or "").split())
        is_workspace_section = tag == "section" or "workspace-section" in classes
        if parent_id == "workspace" and is_workspace_section:
            if element_id:
                self.top_level_workspace_sections.append(element_id)
            else:
                self.idless_top_level_workspace_sections.append(tag)

        if tag not in self.VOID_ELEMENTS:
            self._stack.append((tag, element_id))

    def handle_startendtag(self, tag, attrs) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID_ELEMENTS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag) -> None:
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index][0] == tag:
                del self._stack[index:]
                break


def main() -> int:
    html = HTML_PATH.read_text(encoding="utf-8")
    server = SERVER_PATH.read_text(encoding="utf-8")

    structure = DashboardStructureParser()
    structure.feed(html)
    structure.close()

    onclicks = set(re.findall(r'onclick="\s*(\w+)\s*\(', html))
    functions = re.findall(r'\bfunction\s+(\w+)\s*\(', html)
    function_set = set(functions)
    missing_functions = sorted(onclicks - function_set)

    declared_id_list = structure.declared_id_list
    declared_ids = set(declared_id_list)
    duplicate_ids = sorted(
        element_id for element_id, count in Counter(declared_id_list).items()
        if count > 1
    )
    referenced_ids = set(re.findall(r'getElementById\([\'\"]([^\'\"]+)[\'\"]\)', html))
    missing_ids = sorted(referenced_ids - declared_ids)

    api_paths = set(re.findall(r'\bapi\([\'\"](/api/[^\'\"?]+)', html))
    route_paths = set(re.findall(r'@app\.route\([\'\"]([^\'\"]+)', server))
    missing_routes = sorted(path for path in api_paths if path not in route_paths)

    failures = []
    if missing_functions:
        failures.append(f"Missing onclick functions: {missing_functions}")
    if missing_ids:
        failures.append(f"Missing DOM ids: {missing_ids}")
    if missing_routes:
        failures.append(f"Missing Flask routes: {missing_routes}")
    if duplicate_ids:
        failures.append(f"Duplicate DOM ids: {duplicate_ids}")

    missing_workspace_ids = sorted(REQUIRED_WORKSPACE_IDS - declared_ids)
    missing_workspace_section_ids = sorted(
        REQUIRED_WORKSPACE_SECTION_IDS - declared_ids
    )
    missing_workspace_functions = sorted(
        REQUIRED_WORKSPACE_FUNCTIONS - function_set
    )
    if missing_workspace_ids:
        failures.append(f"Missing workspace DOM ids: {missing_workspace_ids}")
    if missing_workspace_section_ids:
        failures.append(
            f"Missing workspace section DOM ids: {missing_workspace_section_ids}"
        )
    if missing_workspace_functions:
        failures.append(
            f"Missing workspace functions: {missing_workspace_functions}"
        )

    section_nav_targets = set(structure.section_nav_targets)
    missing_section_targets = sorted(section_nav_targets - declared_ids)
    unnavigated_workspace_sections = sorted(
        set(structure.top_level_workspace_sections) - section_nav_targets
    )
    if missing_section_targets:
        failures.append(
            f"Section navigation targets missing from DOM: {missing_section_targets}"
        )
    if structure.idless_top_level_workspace_sections:
        failures.append(
            "Top-level workspace sections must have ids: "
            f"{structure.idless_top_level_workspace_sections}"
        )
    if unnavigated_workspace_sections:
        failures.append(
            "Top-level workspace sections missing from section navigation: "
            f"{unnavigated_workspace_sections}"
        )

    required_preset_ids = {
        "quickPresetSelect",
        "quickPresetStatus",
        "quickConfidence",
        "quickBodyFps",
        "quickFaceFps",
        "quickHandFps",
        "quickStability",
        "quickMouthGain",
        "quickPresetName",
    }
    required_preset_functions = {
        "applyQuickPreset",
        "applySelectedQuickPreset",
        "saveCurrentQuickPreset",
        "deleteSelectedQuickPreset",
        "collectQuickPresetConfig",
    }
    missing_preset_ids = sorted(required_preset_ids - declared_ids)
    missing_preset_functions = sorted(required_preset_functions - function_set)
    if missing_preset_ids:
        failures.append(f"Missing quick preset DOM ids: {missing_preset_ids}")
    if missing_preset_functions:
        failures.append(f"Missing quick preset functions: {missing_preset_functions}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1

    print(
        "Dashboard contract checks passed: "
        f"{len(onclicks)} onclick functions, "
        f"{len(referenced_ids)} DOM ids, {len(api_paths)} API routes, "
        f"{len(section_nav_targets)} workspace navigation targets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
