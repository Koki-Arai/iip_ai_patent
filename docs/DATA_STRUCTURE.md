# Data Structure

Expected files in `data/`:

- `ap_YYYYs.txt`: application table.
- `applicant_YYYYs.txt`: applicant table.
- `inventor_YYYYs.txt`: inventor table.
- `cc_YYYYs.txt`: citation table.

The application number `ida` is the main merge key. The citation table uses `citing` and `cited`.

Main constructed variables include `grant`, `grant_delay_days`, `claim_reduction`, `ai_core`, `ai_broad`, `backward_cites`, `reject_cites`, `inventor_count`, and `applicant_count`.
