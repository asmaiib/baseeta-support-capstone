"""demo_v0 — the chatbot that works on stage.

    python demo_v0.py "What is your return policy for electronics?"

Sixty lines, one afternoon, and it demonstrates beautifully. It is also where
this project's Module 1 lesson starts: read it and list what would break in
production, before ADR 001 names the failure modes for you. Most of them are
findable from the code alone, which is why the list is not printed here.

Six defects are marked `# SMELL`. There are more than six.

Nothing in this file is imported by the application. It exists to be deleted,
and the commit that deletes it is the first commit of the project.
"""

import os
import sys

from openai import OpenAI  # SMELL 1: the application imports the provider SDK

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
)

SYSTEM = """You are Baseeta Support, the assistant for Baseeta Retail's customer-service
portal. Answer questions about orders, returns and shipping. Be helpful and
friendly. Electronics can be returned within 15 days. Home goods within 30
days. Fashion within 14 days. Standard shipping is SAR 25."""
# SMELL 2: the prompt is a string literal in code — unversioned, unreviewable, and
#          the catalog facts are baked into it, so a policy change is a code deploy.

history = []  # SMELL 3: unbounded. Every turn resends everything, forever.


def ask(question: str) -> str:
    history.append({"role": "user", "content": question})
    response = client.chat.completions.create(
        model="retail_support-flagship",  # SMELL 4: a literal model id, in code
        messages=[{"role": "system", "content": SYSTEM}] + history,
        temperature=0.7,
        # SMELL 5: no max_tokens, no timeout, no retry policy. The demo defaults
        #          that become the production incident.
    )
    answer = response.choices[0].message.content
    # SMELL 6: finish_reason is never read. A truncated answer ships silently, and
    #          a refusal arrives as an empty string nobody handles.
    history.append({"role": "assistant", "content": answer})
    return answer


def main() -> None:
    if len(sys.argv) > 1:
        print(ask(" ".join(sys.argv[1:])))
        return
    print("baseeta demo_v0 — Ctrl-C to leave\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question:
            print("baseeta>", ask(question))


if __name__ == "__main__":
    main()
