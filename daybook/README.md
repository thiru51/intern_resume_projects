# Daybook

A personal day planner for placement season. It mixes a fixed daily routine
(run, gym, temple, meals, tea with Kavin) with changing tasks (company PPTs,
tests, placement prep, lab work) and answers one question: **what should I do right now?**

## What it does

- **Right now card**: shows the current block of your day, time left, and what's next.
  During a work block it picks the most urgent task for you (overdue first, then today, then nearest deadline).
- **Plain-English quick add**: type or dictate things like
  `Infosys PPT tomorrow 4pm`, `DBMS test friday 10am`, `revise DP for 2 hours today`, `lab record due 28th`.
  It works out the date, time, length and type (PPT / Test / Prep / Lab / Life).
- **Hour-by-hour timeline** of the day, with tick boxes for routine habits.
- **Values row** you can tick each day (stay private, hit the gym, be kind, ...).
- **Upcoming** tab: everything grouped by day, with overdue items on top.
- **Routine** tab: edit your daily blocks and values.

## How it runs

`index.html` is a single file with no build step.

- **As a Claude artifact** (the main way to use it): data is stored in the artifact's database, so the
  same tasks show up on your phone and your laptop. Quick add uses Claude to read your note.
- **Opened as a plain file / GitHub Pages**: it falls back to saving in the browser (`localStorage`)
  on that one device, and a built-in rule-based parser reads your notes instead.

## Tech

Vanilla HTML, CSS and JavaScript. No framework. Live sync through the artifact `db` capability
(document store with real-time snapshots), natural-language parsing through the `sample` capability
with a regex fallback.
