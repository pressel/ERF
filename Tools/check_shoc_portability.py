#!/usr/bin/env python3
import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SHOC_SRC = REPO / "Source" / "PBL" / "Shoc"
SHOC_TESTS = REPO / "Tests" / "Unit" / "Shoc"
GPU_UTILS = SHOC_SRC / "ERF_ShocGpuUtils.H"

HOST_DEVICE_SIG = re.compile(r"AMREX_GPU_HOST_DEVICE(?:\s+AMREX_FORCE_INLINE)?\s*(?:\n\s*)*[A-Za-z_:<>&*\s]+\(([^)]*)\)")
DISALLOWED_SIG_TYPES = ("ShocColumnData", "FArrayBox", "MultiFab", "Vector<", "amrex::Vector")
ARRAY_CALL = re.compile(r"\.(array|const_array|box|nComp)\s*\(")
DIRECT_SETVAL = re.compile(r"\.setVal\s*(?:<[^>]+>)?\s*\(")
STD_POW = re.compile(r"std::pow\s*\(")
STD_HOST_DEVICE_MATH = re.compile(r"std::(?:max|min|abs)\s*\(")
GPU_LAMBDA_THIS = re.compile(r"AMREX_GPU_DEVICE[^\n]*\[[^\]]*\bthis\b")
ANON_CONST = re.compile(r"namespace\s*\{[\s\S]*?constexpr\s+(?:Real|amrex::Real|bool)\s+[A-Za-z_][A-Za-z0-9_]*", re.MULTILINE)
HOST_CONTAINER_DECL = re.compile(r"\b(?:amrex::Vector|Vector|std::vector)\s*<[^;=]+>\s*([A-Za-z_][A-Za-z0-9_]*)")
FIXTURE_VALUES_BIND = re.compile(r"\b(?:const\s+)?auto(?:\s*&)?\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*shoc_test::fixture_values\s*\(")
EXPECT_DEATH = re.compile(r"\b(?:EXPECT|ASSERT)_DEATH(?:_IF_SUPPORTED)?\s*\(")
PRODUCTION_SYNC = re.compile(
    r"\b(?:sync_if_needed|amrex::Gpu::streamSynchronize|amrex::Gpu::synchronize|"
    r"Gpu::streamSynchronize|Gpu::synchronize|amrex::ParallelDescriptor::Barrier|"
    r"ParallelDescriptor::Barrier)\s*\("
)


def gpu_device_body_ranges(text: str):
    ranges = []
    for match in re.finditer(r"AMREX_GPU_DEVICE", text):
        brace_start = text.find("{", match.start())
        if brace_start == -1:
            continue
        depth = 0
        for idx in range(brace_start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    ranges.append((brace_start, idx + 1))
                    break
    return ranges


def host_container_names(text: str):
    names = {match.group(1) for match in HOST_CONTAINER_DECL.finditer(text)}
    names.update(match.group(1) for match in FIXTURE_VALUES_BIND.finditer(text))
    return names


def host_device_body_ranges(text: str):
    ranges = []
    for match in re.finditer(r"AMREX_GPU_HOST_DEVICE", text):
        brace_start = text.find("{", match.start())
        if brace_start == -1:
            continue
        depth = 0
        for idx in range(brace_start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    ranges.append((brace_start, idx + 1))
                    break
    return ranges


def iter_files(include_tests: bool):
    for root in [SHOC_SRC] + ([SHOC_TESTS] if include_tests else []):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".H", ".h", ".cpp", ".cc", ".cxx"}:
                yield path


def line_no(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def report_issue(issues, path: pathlib.Path, lineno: int, message: str):
    issues.append(f"{path.relative_to(REPO)}:{lineno}: {message}")


def scan_file(path: pathlib.Path, include_tests: bool, issues):
    text = path.read_text(encoding="utf-8", errors="replace")
    container_names = host_container_names(text)

    if path == GPU_UTILS:
        return

    for match in HOST_DEVICE_SIG.finditer(text):
        params = match.group(1)
        if any(token in params for token in DISALLOWED_SIG_TYPES):
            report_issue(issues, path, line_no(text, match.start()),
                         f"disallowed AMREX_GPU_HOST_DEVICE signature parameters: {params.strip()}")

    for start, end in host_device_body_ranges(text):
        body = text[start:end]
        bad = ARRAY_CALL.search(body)
        if bad:
            report_issue(issues, path, line_no(text, start + bad.start()),
                         f"host-only accessor inside AMREX_GPU_HOST_DEVICE body: {bad.group(1)}()")

        for math_match in STD_HOST_DEVICE_MATH.finditer(body):
            report_issue(issues, path, line_no(text, start + math_match.start()),
                         f"std math call inside AMREX_GPU_HOST_DEVICE body: {math_match.group(0).strip()}")

        for pow_match in STD_POW.finditer(body):
            report_issue(issues, path, line_no(text, start + pow_match.start()),
                         f"std::pow call inside AMREX_GPU_HOST_DEVICE body: {pow_match.group(0).strip()}")

    for start, end in gpu_device_body_ranges(text):
        body = text[start:end]

        for math_match in STD_HOST_DEVICE_MATH.finditer(body):
            report_issue(issues, path, line_no(text, start + math_match.start()),
                         f"std math call inside AMREX_GPU_DEVICE body: {math_match.group(0).strip()}")

        for pow_match in STD_POW.finditer(body):
            report_issue(issues, path, line_no(text, start + pow_match.start()),
                         f"std::pow call inside AMREX_GPU_DEVICE body: {pow_match.group(0).strip()}")

        for name in sorted(container_names):
            if re.search(rf"\b{name}\b", body):
                report_issue(issues, path, line_no(text, start),
                             f"host container referenced inside AMREX_GPU_DEVICE body: {name}")

    if path != GPU_UTILS and path.name != "ERF_ShocDriver.cpp":
        for match in DIRECT_SETVAL.finditer(text):
            report_issue(issues, path, line_no(text, match.start()),
                         "direct setVal call outside ERF_ShocGpuUtils.H")

    if path.is_relative_to(SHOC_SRC):
        for match in PRODUCTION_SYNC.finditer(text):
            report_issue(issues, path, line_no(text, match.start()),
                         f"production synchronization helper outside ERF_ShocGpuUtils.H: {match.group(0).strip()}")

        for match in STD_POW.finditer(text):
            report_issue(issues, path, line_no(text, match.start()),
                         f"std::pow call in SHOC source: {match.group(0).strip()}")

        for match in GPU_LAMBDA_THIS.finditer(text):
            report_issue(issues, path, line_no(text, match.start()),
                         "GPU lambda captures this")

        anon = ANON_CONST.search(text)
        if anon:
            report_issue(issues, path, line_no(text, anon.start()),
                         "anonymous namespace constexpr constant found in SHOC source")

    if include_tests and path.is_relative_to(SHOC_TESTS):
        for match in EXPECT_DEATH.finditer(text):
            report_issue(issues, path, line_no(text, match.start()),
                         f"death test in SHOC unit tests: {match.group(0).strip()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SHOC portability hazards")
    parser.add_argument("--include-tests", action="store_true", help="also scan Tests/Unit/Shoc")
    args = parser.parse_args()

    issues = []
    for path in iter_files(args.include_tests):
        scan_file(path, args.include_tests, issues)

    if issues:
        print("SHOC portability issues found:")
        for issue in issues:
            print(issue)
        return 1

    print("No SHOC portability issues found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
