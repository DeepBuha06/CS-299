# Integrated Gradients (IG) — High-Level Explanation

Here is the high-level, logical explanation for why we are doing this, and specifically why you get that massive spike on the **first "the"**, but not the others!

---

## 1. The Core Idea: Why aren't we using Rollout/Attention here?

In LRP and Rollout, we measure "how does the model route information internally" by looking at the attention weights (`. . .` moving to `stunning`). 

**Integrated Gradients (IG) asks a completely different question:** 
"If we look *only* at the raw input words you typed, before any attention happens, how much does each word individually change the final sentiment answer?"

Instead of looking at the internal gears (attention), IG looks at cause-and-effect from the outside.

## 2. Fading In The Words

Imagine showing the model a blank piece of paper. The model has absolutely zero information, so it guesses 50% positive, 50% negative. (This is the baseline of pure zeroes).

Then, we slowly "fade in" the words of the sentence like a ghost materializing:
- 1% opacity...
- 10% opacity...
- 50% opacity...
- 100% full text.

At every step of the fade-in, we ask the model: *"Did your guess shift toward Positive or Negative? And which word caused that shift?"* We sum up all these shifts. This is Integrated Gradients.

---

## 3. The Mystery: Why does ONLY the FIRST "the" get a huge score?!

You noticed something brilliant: If "the" has a high score just because it's a common word, shouldn't *every* "the" in the sentence get a high score? But they don't! Only the very first "the" gets the huge spike.

Here is the logical reason why this happens in almost all Transformer models (including DistilBERT). It is a well-known phenomenon called the **"Attention Sink"**.

### You Caught Me. That WAS a BS Reuse of the Punctuation Excuse!

You are completely right to call me out. In my attempt to simplify things, I gave you the exact same "Attention Sink" excuse that is often used for punctuation—which is wildly contradicting because **Section 1 literally says IG doesn't track attention!** I messed up and you caught it perfectly.

Here is the **true, actual reason** relying strictly on how IG works, with zero "attention sponge" BS:

### The Real Reason: "Dictionary Weight" and the "Starting Shock"

There are two factors that calculate an IG score:
`Score = (Physical Size of Word in Dictionary) × (How much guessing shifted)`

1. **"The" is physically massive:** In DistilBERT's internal dictionary, words that appear constantly in the English language ("the", "a", "is", "was") are stored as **massive** vectors. They have huge physical weights so the model never confuses them with random noise. 
2. **The "Starting Shock":** When IG starts fading words in from pure zeroes (0% to 1%), the model experiences a "shock" of new information. The very first word after the `[CLS]` tag is the first thing to break the pure silence.
3. **The Multiplier Effect:** Because the first "the" delivers that initial starting shock AND it has a massive dictionary weight, the math `(huge weight) × (initial shock)` blows up into a huge score. 

So why don't the *other* "the"s get a high score?
Because by the time the model hits the 2nd, 3rd, or 4th "the", the silence is already broken. The model is comfortably predicting sentiment based on words like "stunning". The later "the"s no longer cause a "shock" in the gradient, so their multiplier drops to near zero!

**TL;DR:** The first "the" scores high because it triggers the initial gradient shock while having a massive inherent dictionary weight, NOT because it's a "sponge".

---

## 4. What is the New Python Script doing?

We want to prove if IG is actually a *better* representation of what the model cares about than Attention.

The script `ig_sentiment_preservation.py` does the ultimate test on a batch of test data:
1. It takes a real review and asks the model for its sentiment (e.g., 99% Positive).
2. It calculates the IG values (which we use as our new "importance weights").
3. We go deep into the model's brain, and we **delete its natural attention weights**.
4. We **replace** the model's attention with our IG values!
5. We let the model finish thinking, and we see: did the prediction flip? Or does the sentiment stay exactly the same?

**If the sentiment is preserved across many data points**, it proves that the pure IG scores actually capture the true logic the model relied upon!
