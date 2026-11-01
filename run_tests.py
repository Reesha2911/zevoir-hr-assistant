"""Send every question in test_questions.txt to the running server and print the replies.

Start the server first:  uvicorn app:app --port 8000
Then run:                python run_tests.py   (optional: python run_tests.py http://localhost:8000)
"""
import re
import sys

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
REFUSAL = "couldn't find this information"


def main() -> int:
    section = ""
    failures = 0
    for line in open("test_questions.txt", encoding="utf-8"):
        line = line.strip()
        if line.startswith("##"):
            section = line
            continue
        match = re.match(r"^(\d+)\.\s+(.*?)(\s*\(expect:.*\))?$", line)
        if not match:
            continue
        number, question = match.group(1), match.group(2)
        try:
            reply = requests.post(f"{BASE}/query", json={"message": question}, timeout=180).json()["reply"]
        except Exception as exc:
            print(f"{number}. ERROR {exc}")
            return 1
        refused = REFUSAL in reply.lower()
        if "OUT-OF-SCOPE" in section:
            verdict = "PASS" if refused else "FAIL"
            failures += verdict == "FAIL"
        else:
            verdict = "CHECK" if not refused else "REFUSED"
        print(f"{number}. [{verdict}] {question}\n    -> {reply[:200].replace(chr(10), ' ')}\n")
    print(f"Out-of-scope failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
