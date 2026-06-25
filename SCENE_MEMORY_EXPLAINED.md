# Scene Memory — How It Works (Plain-Language Guide)

This document explains, without jargon, what the new **scene memory** part of the
project does and why it makes the system better. No technical background needed.

---

## The problem we were solving

The camera looks at a shelf and draws boxes around the products it sees. But it
re-does this from scratch on **every single frame** (a video is ~30 frames per
second). The old version had no memory: each frame was a blank slate.

That caused two annoying problems:

1. **The moment your hand covered a product, its box vanished.** The camera
   couldn't see the product anymore, so it forgot it existed — even though it was
   obviously still there behind your hand.
2. **Products had no stable identity.** There was no way to say "that's the same
   bottle I saw a second ago," so you couldn't reliably point at something and
   have the system know which product you meant.

**Scene memory fixes this by giving the system a short-term memory** — it
remembers products from one frame to the next, so they keep their identity and
don't disappear when briefly hidden.

---

## How it decides "is this the same product as before?"

Every frame, the system has to answer one question for each box it sees:
*"Have I seen this product before, or is it new?"*

To do that, it compares the **new box** to the **boxes it remembers** from the
previous frame and measures how much they overlap.

### The overlap measure (called "IoU")

We measure overlap as a percentage: how much do two boxes share, relative to the
total space they cover together. Think of two stickers on a window — if they sit
almost exactly on top of each other, the overlap is high (close to 100%). If they
barely touch at a corner, it's low (close to 0%).

We set the cutoff at **30%**. That number is a balance:
- Too high → boxes flicker and lose their identity the moment a product shifts a little.
- Too low → the system mixes up products sitting next to each other.

---

## The three possible outcomes

For every frame, each box falls into one of three buckets:

### 1. Matched ✅ (same product as before)
A new box overlaps a remembered box by **30% or more**.
→ The system says "same product," keeps its ID number, and updates its memory.
This is what lets a product keep the same identity as it moves slightly or as the
camera wobbles.

### 2. New detection 🆕 (something that wasn't there before)
A new box overlaps everything it remembers by **less than 30%**.
→ The system says "this is new," and gives it a fresh ID number.

### 3. Lost 👻 (a remembered product that isn't visible right now)
A product the system remembers has **no matching box** in the new frame — usually
because your hand is covering it.
→ Instead of instantly forgetting it, the system **keeps it in memory for about a
second** (shown as a gray box). If the product reappears in that time, it
seamlessly picks up its old identity. If it stays gone, it's finally forgotten.

**This third case is the key fix.** It's why a product's box now stays put when you
briefly wave your hand over it, instead of disappearing forever.

> ⚠️ One honest limitation: if a product moves *very* fast, its old and new
> positions might overlap by less than 30%. The system then treats it as the old
> one disappearing AND a new one appearing — so it gets a new ID and forgets its
> history. For slow, deliberate movements (like pointing at a shelf), this isn't a
> problem.

---

## How it decides *what* each product is (the scoring system)

Knowing where a product is, is only half the job. The system also has to
recognize **what** it is (e.g. "Oatly oat milk"). It does this by comparing the
product to a small catalog of known products and picking the closest match.

But a single glance can be wrong — bad lighting, a blurry frame, a weird angle. So
instead of trusting any one frame, the system **votes over time**.

### How the voting works

Every frame a product is clearly recognized, it earns "points" toward that guess.
The product's label is always whatever guess has the **most points so far**.

Example — the system looks at the same bottle over four frames:

| Frame | Best guess   | Points earned | Running total            | Current label |
|-------|--------------|---------------|--------------------------|---------------|
| 1     | Oatly        | 0.85          | Oatly: 0.85              | Oatly         |
| 2     | Oatly        | 0.80          | Oatly: 1.65              | Oatly         |
| 3     | Alpro (wrong)| 0.55          | Oatly: 1.65, Alpro: 0.55 | **Oatly**     |
| 4     | Oatly        | 0.90          | Oatly: 2.55, Alpro: 0.55 | Oatly         |

Notice frame 3 was a mistake — but it didn't change the label, because "Oatly"
had already built up a strong lead. **One bad frame can't override a confident,
repeated answer.** That's the whole benefit: stability.

A couple of details:
- If the system isn't confident enough on a given frame, it casts **no vote** —
  it would rather stay quiet than guess.
- If a product loses its identity (the fast-movement case above), its points reset
  to zero and it starts building confidence again from scratch.

---

## What you should see on screen

| Box color | Meaning |
|---|---|
| 🟥 Red | A product is detected, but not yet recognized (or not in the catalog) |
| 🟩 Green | A product has been confidently recognized |
| 🟧 Orange | The product you are currently pointing at |
| ⬜ Gray | A product that's briefly hidden (e.g. behind your hand) but still remembered |

An orange dot also marks your fingertip when a hand is detected.

---

## In one sentence

**Scene memory gives the system a short-term memory so products keep their
identity, don't vanish when briefly covered, and get recognized reliably by
voting over many frames instead of trusting any single glance.**
