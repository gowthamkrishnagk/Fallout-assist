# Resolution Comment Format (SAC Order-Fallout)

When you resolve an order-fallout ticket, write the fix in this format in the
ticket's resolution comment. The FalloutAssist tool reads these to suggest the
workaround to the next person who hits the same failure — so a clean, consistent
comment today saves the whole team time tomorrow.

## The format

Wrap your fix steps between `=== FIX ===` and `=== END ===`:

```
=== FIX ===
1. <action you took>
2. <action you took>
3. <action you took>
=== END ===
```

That's the whole requirement. **Do not retype the Failed Step or the Error** — the
tool pulls those automatically from the ticket.

Optionally, add a one-line **Root Cause** on top if you know it:

```
=== FIX ===
Root Cause: <one line — why it happened>
1. <action you took>
2. <action you took>
=== END ===
```

## Example

```
=== FIX ===
Root Cause: the order item referenced in the error still had its Action set to
            "Existing" instead of "Disconnect", so it wasn't disconnected.
1. Open the order item referenced in the error.
2. Change its Action field from "Existing" to "Disconnect".
3. Retry the failed orchestration step.
=== END ===
```

## Rules (these are what make it accurate)

1. **Wrap the fix** in `=== FIX ===` / `=== END ===` so the tool grabs exactly your
   resolution and nothing else from the comment thread.
2. **Numbered actions, not narration** — write "Re-trigger the order," not "I went
   and re-triggered it after checking."
3. **No order-specific values in the steps** — no MSISDNs, order numbers, or
   Salesforce record IDs (e.g. `801PI00001...`). Describe the field generically
   ("the order item referenced in the error"), not the specific ID.
4. **Root Cause is optional** — add it only if you actually know it. A half-guessed
   root cause is worse than none.
5. **One fix per block.** If a ticket had two separate failures, write two blocks.

## What NOT to do

- Don't just write **"duplicate, refer to SAC-12345"** as the resolution. That points
  elsewhere instead of describing the fix. If the real fix lives in another ticket,
  put the `=== FIX ===` block on *that* ticket.
- Don't write one-word closers like **"done"**, **"fixed"**, or **"closing"** — they
  carry no workaround.

## Why this matters

FalloutAssist matches a new failure to past resolved tickets by **failed step + error**,
then shows the resolution comment (or, for several matches, synthesizes one). When the
comment is a clean `=== FIX ===` block, the tool shows it **verbatim** (no AI rewriting)
and ranks it above messy comments — so your fix is reused exactly as you wrote it.
