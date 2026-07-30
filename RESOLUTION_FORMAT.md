# Resolution Comment Format (SAC Order-Fallout)

When you resolve an order-fallout ticket, write the fix as the **8-field workaround
table** (adopted 2026-07-28) in the ticket's resolution comment. FalloutAssist reads
these to suggest the workaround to the next person who hits the same failure — so a
clean, consistent comment today saves the whole team time tomorrow.

## You'll get this pre-filled

You rarely have to type the table from scratch. On every inflow ticket FalloutAssist
leaves **two** comments:

1. **💡 Suggested workaround** — the fix as `=== FIX ===` numbered steps. This is the
   one you read and act on. It carries 👍/👎 links; use them, they train the ranking.
2. **📋 Resolution comment format** — this table, pre-filled from that suggested fix and
   @-mentioning you. **Verify every row against what you actually did**, correct it, and
   post it as your resolution comment.

When nothing matched, the second comment still arrives with `BAN CAN` and `MSISDN`
filled in and the rest blank — yours becomes the first indexed resolution for that
failure, which is the only way the next occurrence gets an answer. The same pair shows
up in the app: the steps, then the table with a **⧉ Copy** button.

A pre-filled draft is a starting point, not a finished resolution. The bottom four rows
are the ones that get reused on other tickets, so they're the ones worth your attention.

## The format

Eight rows, label and value, in this order:

```
|Order Type Failed|Change order|
|BAN CAN|962300000|
|MSISDN|17875550100|
|Cause|Device protection was not mapped under the device|
|Category|BYOD Change|
|Solution applied|Mapped device protection on the order item, then completed the step|
|System modified|SF|
|Customer action|Order item and FRLS verified correct — no customer action needed|
```

Use `NA` for a row that genuinely doesn't apply. Don't drop rows and don't invent
values.

## What each row is for

| Row | What goes in it |
| --- | --- |
| **Order Type Failed** | The order type from the description — "Change order", "Sales order", "Disconnect order". |
| **BAN CAN** | The account on the failing order. |
| **MSISDN** | The line on the failing order. |
| **Cause** | One line: *why* it failed. Leave `NA` if you don't actually know. |
| **Category** | The Order Reason from the description ("BYOD Change", "Existing", …). |
| **Solution applied** | What you did to fix it. The single most important row. |
| **System modified** | The system you actually changed — `SF` / `Salesforce`, `Matrixx`, `Aria`, `Nokia`, `Network`, `EDA`. `NA` if you changed nothing. |
| **Customer action** | The follow-up or verification — including "none needed". |

**The four rows that carry your knowledge forward are Cause, Solution applied, System
modified and Customer action.** When your fix is suggested on someone else's ticket, the
top four rows are re-derived from *their* ticket's description (their order type, their
account, their line) — only those four bottom rows are reused. Spend your effort there.

## Rules (these are what make it accurate)

1. **One line per row.** A value that wraps onto a second line breaks that row — the
   parser reads pipe-delimited lines. Keep `Solution applied` under ~500 characters.
2. **Keep the labels.** Common variants are understood (`Order Type`, `Root Cause`,
   `Reason`, `Solution`, `Workaround`, `System`, `Customer`, `Next step`, `BAN-CAN`), but
   the spellings above are the safe ones. At least three recognizable labels are needed
   before a comment is treated as this format.
3. **No account numbers, MSISDNs or record IDs inside Cause / Solution applied /
   Customer action.** Those rows get reused on other orders, where a *wrong* customer's
   account sitting in a "Solution applied" row reads as an instruction. (When your
   comment is reused as-is, long digit runs and Salesforce record IDs are stripped out of
   those rows — so identifiers there are at best deleted.) Identifiers belong in the BAN
   CAN and MSISDN rows, nowhere else. Describe the record generically: "the order item
   referenced in the error".
4. **Actions, not narration.** "Mapped device protection on the order item," not "I went
   and looked at it and then mapped it."
5. **`System modified` is the system you changed**, not the one you looked at. A wrong
   value sends the next engineer to the wrong console.
6. **One table per fix.** If a ticket had two separate failures, post two tables.

Greetings, @mentions and screenshots around the table are fine — they're ignored. So is
quoting the original table in a later reply: the first occurrence of each row wins.

## What NOT to do

- Don't write **"duplicate, refer to SAC-12345"** as the resolution. That points
  elsewhere instead of describing the fix; put the table on *that* ticket instead.
- Don't write one-word closers — **"done"**, **"fixed"**, **"closing"**. They carry no
  workaround and are actively ranked down.
- Don't fill `Cause` with a guess. A half-guessed cause is worse than `NA`.

## `=== FIX ===` blocks

```
=== FIX ===
Root Cause: <one line, if you know it>
1. <action you took>
2. <action you took>
=== END ===
```

This is the shape FalloutAssist **shows** a workaround in — steps are what you follow.
It's also still accepted as a resolution: old blocks are indexed, ranked as good
resolutions, and turned into the 8-field table when suggested elsewhere.

But **close your ticket with the table, not a block.** The table carries four things a
step list has no room for — the cause, the system you changed, the follow-up, and the
order type/reason that scope the fix — and those are what make it matchable later.

## Why this matters

FalloutAssist matches a new failure to past resolved tickets by **failed step + error**,
then renders the matched resolution as steps and drafts the table from the same
generation, so the fix you're shown and the resolution you're asked to post can never
describe different things. A comment that is *already* in the table format is reused
**verbatim** — no AI rewriting — and ranks well above thin prose resolutions, so your fix
gets handed on exactly as you wrote it.

FalloutAssist never learns from its own comments: both are marked and skipped at ingest,
so a suggestion can't be re-indexed as though an engineer had written it.
