#!/usr/bin/env python3
"""
Pre-submission validation script to catch syntax errors before submitting jobs.
Always run this before sbatch submissions.

Usage:
    python3 validate_syntax.py [files...]
    
    If no files specified, validates all critical files:
    - metrics_calculator.py
    - complete_cycle.py
    - analyze_metrics.py
    - diagnose_work.py
"""

import sys
import py_compile
import ast
from pathlib import Path


def validate_syntax(file_path):
    """
    Validate Python file syntax.
    
    Returns:
        (is_valid: bool, error_message: str)
    """
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        
        # Try to parse as AST (catches syntax errors)
        ast.parse(code)
        
        # Try to compile (additional check)
        py_compile.compile(file_path, doraise=True)
        
        return True, None
    
    except SyntaxError as e:
        return False, f"Syntax Error at line {e.lineno}: {e.msg}\n  {e.text}"
    except Exception as e:
        return False, f"Error: {type(e).__name__}: {str(e)}"


def validate_indentation(file_path):
    """
    Check for common indentation/formatting issues.
    
    Returns:
        (is_valid: bool, issues: list[str])
    """
    issues = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            # Check for tabs (should be spaces)
            if '\t' in line:
                issues.append(f"Line {i}: Contains tab character (use spaces)")
            
            # Check for trailing whitespace
            if line.rstrip('\n') != line.rstrip('\n').rstrip():
                issues.append(f"Line {i}: Trailing whitespace")
            
            # Check for lines that are suspiciously short (orphaned docstring ends, etc)
            stripped = line.strip()
            if stripped in ['"""', "'''", '""")', "''')"]:
                # Check previous line
                if i > 1 and not lines[i-1].strip().startswith('"""') and not lines[i-1].strip().startswith("'''"):
                    issues.append(f"Line {i}: Orphaned docstring closer: {stripped}")
    
    except Exception as e:
        issues.append(f"Could not read file: {str(e)}")
    
    return len(issues) == 0, issues


def main():
    if len(sys.argv) > 1:
        files_to_check = [Path(f) for f in sys.argv[1:]]
    else:
        # Default critical files
        files_to_check = [
            Path("metrics_calculator.py"),
            Path("complete_cycle.py"),
            Path("analyze_metrics.py"),
            Path("diagnose_work.py"),
        ]
    
    all_valid = True
    results = []
    
    print("=" * 80)
    print("PRE-SUBMISSION SYNTAX VALIDATION")
    print("=" * 80)
    
    for file_path in files_to_check:
        if not file_path.exists():
            print(f"\n⚠️  {file_path}: FILE NOT FOUND")
            all_valid = False
            continue
        
        print(f"\n📄 Checking {file_path}...")
        
        # Syntax check
        is_valid, syntax_error = validate_syntax(file_path)
        
        if not is_valid:
            print(f"  ❌ SYNTAX ERROR:")
            print(f"     {syntax_error}")
            all_valid = False
            results.append((file_path, False))
            continue
        
        # Indentation check
        indent_valid, indent_issues = validate_indentation(file_path)
        
        if not indent_valid:
            print(f"  ⚠️  FORMATTING ISSUES:")
            for issue in indent_issues[:5]:  # Show first 5 issues
                print(f"     - {issue}")
            if len(indent_issues) > 5:
                print(f"     ... and {len(indent_issues) - 5} more issues")
        
        print(f"  ✓ Valid")
        results.append((file_path, True))
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    valid_count = sum(1 for _, v in results if v)
    total_count = len(results)
    
    for file_path, is_valid in results:
        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"{status:8} {file_path}")
    
    print(f"\n{valid_count}/{total_count} files valid")
    
    if all_valid:
        print("\n✅ All files passed validation! Safe to submit.")
        return 0
    else:
        print("\n❌ Validation failed! Do NOT submit until errors are fixed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
