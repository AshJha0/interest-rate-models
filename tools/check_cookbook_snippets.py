#!/usr/bin/env python3
"""Extract every code block from COOKBOOK.md and compile/run it against the
real libraries, so the recipes cannot rot.

Prerequisites: the C++ library built (`bash cpp/build.sh`), the Java
classes built (`bash java/build.sh`), cargo available.  Python blocks run
with `python/src` on the path; C++/Rust/Java blocks are concatenated into
one harness per language (each block in its own scope, `#include`/`use`/
`import` lines hoisted) with `mkt` and `hw` pre-defined for the blocks that
build on recipe 9.  Snippets that need `../data` run from `python/`.

Usage:  python3 tools/check_cookbook_snippets.py [work_dir]
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CURVE = ("{1,2,3,5,10}", "{0.9787294774691476, 0.9550419621907147, 0.9291361457915193, "
         "0.8715343499971578, 0.7046880897187134}")


def run(cmd, cwd, env=None):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def main() -> int:
    work = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(tempfile.mkdtemp())
    work.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "COOKBOOK.md").read_text()
    blocks = re.findall(r"```(python|cpp|rust|java)\n(.*?)```", src, re.S)
    by: dict[str, list[str]] = {}
    for lang, code in blocks:
        by.setdefault(lang, []).append(code)
    print({k: len(v) for k, v in by.items()})
    pydir = ROOT / "python"
    env = {"PYTHONPATH": str(pydir / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin",
           "HOME": str(pathlib.Path.home())}

    # ---- Python: each block is a standalone script -----------------------
    for i, code in enumerate(by["python"]):
        f = work / f"py_{i}.py"
        f.write_text(code)
        run([sys.executable, str(f)], cwd=pydir, env=env)

    # ---- C++ -------------------------------------------------------------
    bodies = []
    for code in by["cpp"]:
        bodies.append("\n".join(l for l in code.split("\n") if not l.startswith("#include")))
    cpp = ('#include "irm/irm.hpp"\n#include <cstdio>\n#include <stdexcept>\n#include <vector>\n'
           "int main() {\n"
           f"    irm::DiscountCurve mkt({CURVE[0]}, {CURVE[1]});\n"
           "    irm::HullWhite hw(0.1, 0.01, mkt);\n"
           + "".join("    {\n" + b + "\n    }\n" for b in bodies)
           + '    std::puts("cpp snippets OK");\n    return 0;\n}\n')
    (work / "snippets.cpp").write_text(cpp)
    run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Wno-unused-variable",
         "-Wno-unused-but-set-variable", f"-I{ROOT / 'cpp' / 'include'}",
         str(work / "snippets.cpp"), str(ROOT / "cpp" / "build" / "libirm.a"),
         "-o", str(work / "snippets_cpp")], cwd=work)
    run([str(work / "snippets_cpp")], cwd=pydir)

    # ---- Rust ------------------------------------------------------------
    bodies = ["\n".join(l for l in code.split("\n") if not l.startswith("use ")) for code in by["rust"]]
    rs = ("#![allow(unused_variables, unused_imports)]\nuse irm::*;\nuse std::path::Path;\n"
          "fn main() -> std::result::Result<(), IrmError> {\n"
          "    let mkt = irm::DiscountCurve::new(&[1.0,2.0,3.0,5.0,10.0],\n"
          "        &[0.9787294774691476, 0.9550419621907147, 0.9291361457915193,\n"
          "          0.8715343499971578, 0.7046880897187134])?;\n"
          "    let hw = irm::HullWhite::new(0.1, 0.01, mkt.clone())?;\n"
          + "".join("    {\n" + b + "\n    }\n" for b in bodies)
          + '    println!("rust snippets OK");\n    Ok(())\n}\n')
    crate = work / "rs_snip"
    (crate / "src").mkdir(parents=True, exist_ok=True)
    (crate / "src" / "main.rs").write_text(rs)
    (crate / "Cargo.toml").write_text(
        '[package]\nname = "snippets"\nversion = "0.1.0"\nedition = "2021"\n'
        f'[dependencies]\nirm = {{ path = "{ROOT / "rust"}" }}\n')
    run(["cargo", "build", "--release", "--quiet"], cwd=crate)
    run([str(crate / "target" / "release" / "snippets")], cwd=pydir)

    # ---- Java ------------------------------------------------------------
    imports: list[str] = []
    bodies = []
    for code in by["java"]:
        body = []
        for line in code.split("\n"):
            if line.startswith("import "):
                if line not in imports:
                    imports.append(line)
            else:
                body.append(line)
        bodies.append("\n".join(body))
    jv = ("import com.quant.irm.*;\n"
          + "\n".join(i for i in imports if i != "import com.quant.irm.*;") + "\n"
          "public class Snippets {\n"
          "    static DiscountCurve mkt = new DiscountCurve(\n"
          f"        new double[]{CURVE[0]}, new double[]{CURVE[1]});\n"
          "    static HullWhite hw = new HullWhite(0.1, 0.01, mkt);\n"
          "    public static void main(String[] args) {\n"
          + "".join("        {\n" + b + "\n        }\n" for b in bodies)
          + '        System.out.println("java snippets OK");\n    }\n}\n')
    jdir = work / "java"
    jdir.mkdir(exist_ok=True)
    (jdir / "Snippets.java").write_text(jv)
    classes = ROOT / "java" / "out" / "main"
    run(["javac", "-Xlint:all", "-Werror", "-cp", str(classes), "-d", str(jdir),
         str(jdir / "Snippets.java")], cwd=work)
    run(["java", "-cp", f"{classes}:{jdir}", "Snippets"], cwd=pydir)
    print("all COOKBOOK snippets compiled and ran")
    if len(sys.argv) <= 1:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
