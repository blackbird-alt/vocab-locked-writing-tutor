# Conversation probe - v2

| check | pass | of |
|---|---|---|
| content_correct | 8 | 23 |
| verdict_given | 7 | 7 |
| refers_back_ok | 4 | 9 |
| on_task | 23 | 23 |
| conversational | 23 | 23 |
| depth_ok | 16 | 21 |
| in_band (mechanical) | 54 | 56 |

## Failures
- **correct_attempt_splice** turn 1 [content_correct]: Incorrectly implies that a comma + 'so' joining two independent clauses is a splice; that's actually correct usage (comma+FANBOYS).
  > That's good you're working. Both fix it, so you're close. The semicolon works fine here too. You can also use a comma, but that would make it a comma splice again, since "so" is joining two full sente
- **correct_attempt_fragment** turn 0 [content_correct]: Definition of fragment as 'no subject and no main idea' contradicts own example (subject+verb present but still fragment).
  > Here's an example: "Because I was late." That looks like a sentence, but it's not complete. It leaves you asking, "What happened?"

A fragment is a group of words with no subject and no main idea. Loo
- **correct_attempt_fragment** turn 1 [content_correct]: Incorrectly claims the original fragment 'running down the street' has subject 'the dog' and verb 'was running' - those belong to the fixed sentence, not the fr
  > Yes, that's right. In "running down the street," the subject is "the dog," and the verb is "was running." That's a full sentence, so the fix needs to add a subject and verb.

The sentence is missing a
- **wrong_attempt_theme** turn 0 [content_correct]: Mischaracterizes theme as 'a word or two'; that's actually more like topic. Theme should be a broader message or insight, not a short phrase.
  > You're asking the right thing. here's an example: a story about a girl who is scared of dogs. The topic is the thing the story is about: "scared of dogs." The theme is the main idea the story is tryin
- **wrong_attempt_theme** turn 1 [content_correct]: Confirms student's correct topic answer, but defines theme as 'why the character is angry,' which just restates plot rather than expressing a thematic message/i
  > Good catch, and that's exactly what you want. Topic is the subject, the thing the story is about. Theme is the message the story is saying about that thing.

So in your example, the topic is "anger," 
- **wrong_attempt_theme** turn 2 [content_correct]: Again affirms an answer where 'theme' is just a plot restatement ('losing his brother') rather than a thematic idea or lesson, perpetuating the same conceptual 
  > Exactly. Topic is the subject, the thing the story is about: "loss." Theme is the message the story is trying to tell: "losing his brother."

That's why topic tells you what happened, and theme tells 
- **meta_grades** turn 1 [content_correct, depth_ok]: 3rd grade is less advanced than middle school, not more—logical error; also the claim that metaphors are 'true or false' is a confusing/incorrect way to disting
  > That's a little more advanced than we're going. Let's take something he might need: metaphors and similes, how to spot the difference, and the tricky part of a metaphor: it can be true, it can be fals
- **refer_back_explain** turn 0 [content_correct]: Example/explanation is muddled; 'the streets are running' claim is nonsensical, and the 'fix' doesn't correct a dangling modifier.
  > A dangling modifier is a phrase that starts a sentence, but it can't do the job alone. Look at this: "Running down the street." The phrase "running" doesn't tell us who is running. It just says "the s
- **refer_back_explain** turn 1 [content_correct, refers_back_ok]: Incorrectly calls 'running' a noun instead of a participle/verbal; contradicts and confuses prior explanation.
  > Let's go deeper. Here's the finer point. In "Running down the street," "running" is a noun, not a verb. A noun describes a thing, so it needs a subject, not a verb. That's why the fix needs "the stree
- **refer_back_explain** turn 2 [content_correct, refers_back_ok]: Incorrectly claims 'street' is plural and conflates subject-verb agreement with dangling modifiers; doesn't answer the actual question about 'real subject'.
  > Let's go further. Look at "The street was running down." Now the verb is past tense, and the subject is "the street," which is plural. That's the problem: the subject has to match the verb's form. If 
- **refer_back_explain** turn 3 [content_correct, refers_back_ok]: Admits confusion but offers a nonsensical 'fix' ('The box under the bed was running') that misuses the earlier example and doesn't clarify the concept.
  > You're right, and I'm not helping you. Let's just fix it, then go deeper. Here's the fixed version: "The box under the bed was running." Now the subject, "under the bed," has the right verb, "was," ma
- **offtask_chat** turn 0 [content_correct, depth_ok]: Incorrectly calls 'blind' a noun needing pluralization; fabricated, confusing grammar rule not matching real usage.
  > That's a strong story with a lot of emotion, but let's get into the grammar part first, since the rest of your weekend was messy. Here's the real part: "the ref was blind." The word "blind" is a noun,
- **offtask_chat** turn 1 [content_correct, depth_ok]: Correctly admits 'blind' is an adjective (validating student), but then contradicts itself with nonsensical 'blinds' plural noun reasoning.
  > You're asking the right thing. "Blind" does describe the ref, so it's an adjective. "Blinds" is just a noun, so it needs to be plural. That's why "the referee was blind" works, and "the referee's blin
- **offtask_chat** turn 2 [content_correct, refers_back_ok, depth_ok]: Claims 'loud' needs to become 'loudness' for no valid reason; also invents a sentence ('soccer game was loud') never present in student's original story.
  > That's a real test, and this one trips up most people: "the soccer game was loud." That's an adjective describing "loud," so it needs a noun form, "loudness." Fixed version: "The soccer game was loud.
- **offtask_chat** turn 3 [content_correct, refers_back_ok, depth_ok]: Repeats same invalid adjective-to-noun claim and still references a sentence not actually written by the student.
  > Here's the tricky one: "The soccer game was loud." "Loud" describes "game," so it needs "loudness." You can also switch the words around: "The game was loud." Try fixing the sentence with "the loudnes
