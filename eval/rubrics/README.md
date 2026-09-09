# Rubrics

`groundedness.v1.md` is deliberately vague — no anchors, no evidence requirement,
no don't-know clause. It is where Module 5's calibration starts, and it produces a
judge that agrees with human labels about two thirds of the time (kappa near 0.4),
which is a random-number generator with an opinion.

`groundedness.v2.md` is the same dimension with anchors, the don't-know clause and
a required evidence quote. It clears the course bar (kappa >= 0.6).

The lesson in the difference: when the judge disagrees with the humans, fix the
rubric, not the humans. Re-calibrate whenever the judge model changes — the judge
is part of the system, and it is under version control for that reason.
