"""Execute every lab notebook and save the outputs back into the file.

    docker compose --profile notebooks run --rm --entrypoint "" \
      -v "$(pwd)/scripts/run_notebooks.py:/srv/run_notebooks.py" \
      notebooks /opt/venv/bin/python /srv/run_notebooks.py

Run this way, the outputs committed to the repository are the ones the
containerised stack produced against the compose gateway — not a transcription,
and not a laptop-only result. A notebook that fails here fails for a participant
too, which is the whole point of running it in the container rather than trusting
a local venv that has drifted.
"""
import pathlib
import sys

import nbformat
from nbclient import NotebookClient

fails = {}
for nb_path in sorted(pathlib.Path("/srv/labs").glob("lab*.ipynb")):
    nb = nbformat.read(nb_path, as_version=4)
    try:
        NotebookClient(nb, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": "/srv"}},
                       allow_errors=False).execute()
        nbformat.write(nb, nb_path)
        cells = sum(1 for c in nb.cells if c.get("outputs"))
        print(f"  OK   {nb_path.name}  ({cells} cells with output)")
    except Exception as exc:
        fails[nb_path.name] = f"{type(exc).__name__}: {str(exc)[:400]}"
        print(f"  FAIL {nb_path.name}")

for k, v in fails.items():
    print("\n" + k, "->", v)
sys.exit(1 if fails else 0)
